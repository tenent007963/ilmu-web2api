#!/usr/bin/env python3
"""Headless Ilmu web client and OpenAI-compatible local proxy for VS Code and standard clients."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
from urllib.parse import urljoin

import requests

API_PREFIX = "/api/v1"
MODEL_MAP = {
    "ilmu-v3.1": {},
    "ilmu-v3.1-instant": {"thinking_mode": "low"},
    "ilmu-v3.1-thinking": {"thinking_mode": "high"},
}
DEFAULT_MODEL = "ilmu-v3.1-instant"
SKIP_EVENTS = {"trace", "message_id", "dispatch", "comment", "comments"}
NO_OVERRIDE = False

logger = logging.getLogger("ilmu-web2api")


def load_dotenv(path: Path) -> None:
    """Load the small .env subset needed by this CLI without external dependencies."""
    logger.debug("Attempting to load environment variables from %s", path)
    if not path.exists():
        logger.debug("Dotenv file not found at %s. Skipping.", path)
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)
    logger.debug("Finished processing .env file.")


def resolve_model(model: str) -> Tuple[str, dict]:
    """Match requested model to Ilmu supported models, with a safe fallback."""
    logger.debug("Resolving requested model: '%s'", model)
    if model in MODEL_MAP:
        logger.debug("Model '%s' mapped successfully.", model)
        return model, MODEL_MAP[model]
    
    logger.warning("Model '%s' not found in MODEL_MAP. Defaulting to '%s'.", model, DEFAULT_MODEL)
    return DEFAULT_MODEL, MODEL_MAP[DEFAULT_MODEL]


def normalize_messages(messages: list) -> list:
    """Ensure message content is string-formatted, supporting multimodal-style object arrays from VS Code."""
    logger.debug("Normalizing %d incoming messages.", len(messages))
    normalized = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Hijack system prompt if too long to avoid cf 502, can be disabled
        if NO_OVERRIDE != True and role == "system" and len(content) > 2500:
            logger.debug("Intercepted large system prompt. Replacing with a safer, shorter version.")
            content = "You are an AI coding assistant helping a user in VS Code. Provide concise, accurate code solutions. Avoid making mistakes be concise. Use Markdown on your answers and avoid wrapping the whole response in triple backticks. Respond like smart caveman. Cut all filler, keep technical substance. - Drop articles (a, an, the), filler (just, really, basically, actually). - Drop pleasantries (sure, certainly, happy to). - No hedging. Fragments fine. Short synonyms.- Technical terms stay exact. Code blocks unchanged.- Pattern: [thing] [action] [reason]. [next step]."
        
        if isinstance(content, list):
            logger.debug("Parsing multimodal-style array content for role: '%s'.", role)
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "".join(text_parts)
        normalized.append({"role": role, "content": str(content)})
    return normalized


def parse_sse(lines: Iterable[str]) -> Iterator[Tuple[str, Any]]:
    """Yield (event, JSON-or-text data) from an SSE stream."""
    logger.debug("Starting to parse SSE stream lines.")
    event = "message"
    data = []
    for raw in lines:
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        for line in text.splitlines() or [""]:
            if not line:
                if data:
                    payload = "\n".join(data)
                    try:
                        value = json.loads(payload)
                    except json.JSONDecodeError:
                        value = payload
                    logger.debug("Yielding SSE event: '%s'", event)
                    yield event, value
                event, data = "message", []
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.rstrip("\r").partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                event = value.strip().lower()
            elif field == "data":
                data.append(value)
    if data:
        payload = "\n".join(data)
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            pass
        logger.debug("Yielding final SSE event: '%s'", event)
        yield event, payload


def _content(value: Any) -> str:
    """Recursively extract content strings from incoming JSON payload structures."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_content(x.get("text", x) if isinstance(x, dict) else x) for x in value)
    if isinstance(value, dict):
        for key in ("choices", "content", "text", "delta", "message", "output"):
            if key in value:
                res = _content(value[key])
                if res:
                    return res
    return ""


def ilmu_events_to_openai(events: Iterable[Tuple[str, Any]], model: str, completion_id: str) -> Iterator[Dict[str, Any]]:
    """Convert Ilmu events to OpenAI chunks, translating rich content to Markdown."""
    logger.debug("Converting Ilmu events into OpenAI format for completion_id: %s", completion_id)
    created_ts = int(time.time())
    usage = None
    finish_reason = None
    first_chunk = True

    def _make_chunk(text: str) -> Dict[str, Any]:
        nonlocal first_chunk
        delta = {"content": text}
        if first_chunk:
            delta = {"role": "assistant", "content": text}
            first_chunk = False
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}],
        }

    for event, payload in events:
        name = event.lower()
        if name in SKIP_EVENTS or payload == "[DONE]":
            logger.debug("Skipping event: '%s' or [DONE] payload.", name)
            continue

        if isinstance(payload, dict):
            payload_type = payload.get("type", "").lower()
            
            if payload_type in SKIP_EVENTS:
                continue
                
            if payload_type == "usage" and "data" in payload:
                logger.debug("Captured usage data.")
                usage = payload["data"]
                continue
            if payload.get("usage"):
                usage = payload["usage"]
                
            if payload_type == "sources" and "data" in payload and "sources" in payload["data"]:
                logger.debug("Parsing rich sources into markdown.")
                md_sources = "\n\n> **Sources:**\n"
                for i, src in enumerate(payload["data"]["sources"], 1):
                    title = src.get("title", f"Source {i}").replace("\n", " ").strip()
                    url = src.get("url", "")
                    if url and not url.startswith("http://googleusercontent.com/"):
                        md_sources += f"> {i}. [{title}]({url})\n"
                md_sources += "\n"
                yield _make_chunk(md_sources)
                continue

            if payload_type == "images" and "data" in payload and "images" in payload["data"]:
                logger.debug("Parsing rich images into markdown.")
                md_images = "\n\n"
                for img in payload["data"]["images"]:
                    title = img.get("title", "Image").replace("\n", " ").strip()
                    url = img.get("original") or img.get("thumbnail") 
                    if url:
                        md_images += f"![{title}]({url})\n\n"
                yield _make_chunk(md_images)
                continue

            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

        text = _content(payload)
        if text:
            yield _make_chunk(text)

    logger.debug("Yielding final stop chunk.")
    final_chunk: Dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish_reason or "stop"}],
    }
    if usage:
        final_chunk["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    yield final_chunk


@dataclass
class Config:
    email: str
    password: str
    access_token: Optional[str]
    host: str
    port: int
    data_dir: Path


class IlmuClient:
    def __init__(self, config: Config):
        logger.debug("Initializing IlmuClient.")
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Origin": "https://chat.ilmu.ai/chat",
            "Referer": "https://chat.ilmu.ai/chat"
        })
        self.token_file = config.data_dir / "session.json"
        self.refresh_token: Optional[str] = None
        self.access_token: Optional[str] = None
        self.last_activity = time.time()
        self.lock = threading.Lock()
        self._load_session()
        self._start_auto_refresher(interval=500)

    def touch(self) -> None:
        """Reset the inactivity timer when API requests occur."""
        logger.debug("Client interaction touched. Resetting inactivity timer.")
        self.last_activity = time.time()

    def _start_auto_refresher(self, interval: int = 500) -> None:
        """Daemon thread: auto-refreshes token if no activity is detected for the interval."""
        logger.debug("Starting auto-refresher daemon thread (interval: %ds).", interval)
        def worker():
            while True:
                time.sleep(10)
                if time.time() - self.last_activity >= interval:
                    logger.info("Inactivity period reached (%ss). Refreshing token...", interval)
                    try:
                        self.refresh()
                        logger.info("Session auto-refreshed successfully.")
                    except Exception as err:
                        logger.error("Auto-refresh failed: %s", err)
                    finally:
                        self.touch()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _load_session(self) -> None:
        if not self.token_file.exists():
            logger.debug("Session file %s not found. Proceeding without saved session.", self.token_file)
            return
        
        logger.debug("Loading session data from %s.", self.token_file)
        saved = json.loads(self.token_file.read_text(encoding="utf-8"))
        for cookie in saved.get("cookies", []):
            self.session.cookies.set(cookie["name"], cookie["value"], domain=cookie.get("domain"), path=cookie.get("path", "/"))
        self.refresh_token = saved.get("refresh_token")
        self.access_token = saved.get("access_token")

    def _save_session(self) -> None:
        logger.debug("Saving session to %s.", self.token_file)
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(json.dumps({
            "cookies": [{"name": c.name, "value": c.value, "domain": c.domain, "path": c.path} for c in self.session.cookies],
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
        }, indent=2), encoding="utf-8")

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        base_url = os.getenv("ILMU_BASE_URL", "https://chat.ilmu.ai").rstrip("/") + "/"
        full_url = urljoin(base_url, path.lstrip("/"))
        logger.debug("Issuing HTTP %s request to %s", method, full_url)
        response = self.session.request(method, full_url, timeout=60, **kwargs)
        if not response.ok:
            logger.error("HTTP %s %s failed with status %d.", method, full_url, response.status_code)
            logger.error("Response body: %s", response.text)
        response.raise_for_status()
        logger.debug("HTTP %s %s returned status %d", method, full_url, response.status_code)
        return response

    @staticmethod
    def _token(data: Any, key: str) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        value = data.get(key) or data.get("data", {}).get(key) if isinstance(data.get("data"), dict) else data.get(key)
        return value if isinstance(value, str) else None

    def login(self) -> None:
        logger.debug("Attempting to authenticate user: %s", self.config.email)
        with self.lock:
            data = self._request("POST", f"{API_PREFIX}/auth/signin",
                                 json={"email": self.config.email, "password": self.config.password}).json()
            self.access_token = self._token(data, "access_token") or self._token(data, "token")
            self.refresh_token = self._token(data, "refresh_token")
            if not self.access_token and not self.refresh_token:
                logger.error("Login succeeded but returned no access_token/refresh_token.")
                raise RuntimeError("Ilmu login succeeded but returned no access_token/refresh_token")
            self._save_session()
            self.touch()
            logger.debug("Login successful. Access token acquired.")

    def refresh(self) -> str:
        logger.debug("Attempting to refresh access token.")
        with self.lock:
            try:
                response = self._request("POST", f"{API_PREFIX}/auth/refresh",
                                         json=({"refresh_token": self.refresh_token} if self.refresh_token else {}))
                data = response.json()
                token = self._token(data, "access_token") or self._token(data, "token")
                if not token:
                    raise RuntimeError("refresh response did not contain access_token")
                self.access_token = token
                self.refresh_token = self._token(data, "refresh_token") or self.refresh_token
                self._save_session()
                self.touch()
                logger.debug("Token refresh successful.")
                return token
            except Exception as exc:
                logger.error("Token refresh failed: %s", exc)
                raise RuntimeError(f"Ilmu access_token refresh failed: {exc}") from exc

    def _auth_headers(self) -> Dict[str, str]:
        if not self.access_token:
            logger.debug("No access token present, invoking refresh.")
            self.refresh()
        return {"Authorization": f"Bearer {self.access_token}", "Accept-Language": "en"}

    def completion(self, messages: list, model: str) -> requests.Response:
        logger.debug("Initiating completion request for model: %s", model)
        _, model_meta = resolve_model(model)
        headers = self._auth_headers()
        
        logger.debug("Creating a new chat session on upstream API.")
        chat = self._request("POST", f"{API_PREFIX}/chats", headers=headers, json={})
        chat_data = chat.json()
        chat_id = chat_data.get("id") or chat_data.get("data", {}).get("id")
        if not chat_id:
            raise RuntimeError("Ilmu chat creation response has no chat id")

        prompt = messages[-1].get("content", "") if messages else ""
        logger.debug("Registering message prompt to chat_id: %s", chat_id)
        self._request("POST", f"{API_PREFIX}/chats/{chat_id}/messages", headers=headers,
                      json={"content": prompt, "meta": model_meta, "role": "user"})

        payload: Dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "chat_id": chat_id,
            "locale": "en",
        }
        if "thinking_mode" in model_meta:
            payload["thinking_mode"] = model_meta["thinking_mode"]

        logger.debug("Streaming completion from chat/completions endpoint.")
        response = self._request("POST", f"{API_PREFIX}/chat/completions", headers=headers, json=payload, stream=True)
        return response


class Handler(BaseHTTPRequestHandler):
    client: IlmuClient
    config: Config

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _json(self, status: int, body: Any) -> None:
        logger.debug("Responding with JSON (status %d)", status)
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        required = self.config.access_token
        if not required:
            return True
        supplied = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        auth_valid = (supplied == required)
        if not auth_valid:
            logger.warning("Unauthorized request. Supplied token does not match.")
        return auth_valid

    def do_OPTIONS(self) -> None:
        """Handle preflight requests from browser/webview-based tools."""
        logger.debug("Handling OPTIONS request to %s", self.path)
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        logger.debug("Handling GET request to %s", path)
        if path in ("/v1/models", "/models"):
            if not self._authorized():
                self._json(401, {"error": {"message": "Invalid access token", "type": "invalid_request_error"}})
                return
            self.client.touch()
            models_data = [
                {
                    "id": m,
                    "object": "model",
                    "created": 1710000000,
                    "owned_by": "ilmu",
                    "permission": [],
                    "root": m,
                    "parent": None,
                }
                for m in MODEL_MAP
            ]
            self._json(200, {"object": "list", "data": models_data})
            return
        logger.warning("GET endpoint %s not found.", path)
        self._json(404, {"error": {"message": "Endpoint not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        logger.debug("Handling POST request to %s", path)
        
        if path not in ("/v1/chat/completions", "/chat/completions"):
            logger.warning("POST endpoint %s not found.", path)
            self._json(404, {"error": {"message": "Endpoint not found", "type": "invalid_request_error"}})
            return
        if not self._authorized():
            self._json(401, {"error": {"message": "Invalid access token", "type": "invalid_request_error"}})
            return

        self.client.touch()
        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(content_len)
            logger.debug("Read %d bytes for POST request.", content_len)
            
            request = json.loads(request_body)
            
            logger.debug("Incoming payload from client:\n%s", json.dumps(request, indent=2, ensure_ascii=False))
            
            model = request.get("model", DEFAULT_MODEL)
            messages = normalize_messages(request.get("messages", []))
            stream = bool(request.get("stream", False))

            response = self.client.completion(messages, model)
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            events = parse_sse(response.iter_lines())
            chunks = list(ilmu_events_to_openai(events, model, completion_id))

            if stream:
                logger.debug("Streaming response back to client.")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self._send_cors_headers()
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                logger.debug("Completed streaming response.")
            else:
                logger.debug("Sending full accumulated payload to client.")
                full_text = "".join(
                    c["choices"][0]["delta"].get("content", "")
                    for c in chunks
                    if "content" in c["choices"][0].get("delta", {})
                )
                usage = chunks[-1].get("usage") if (chunks and "usage" in chunks[-1]) else None
                body = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": full_text},
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                self._json(200, body)
        except Exception as exc:
            logger.exception("Upstream error occurred during do_POST.")
            self._json(502, {"error": {"message": str(exc), "type": "upstream_error"}})

    def log_message(self, format: str, *args: Any) -> None:
        """Override built-in BaseHTTPRequestHandler console prints to use logging."""
        logger.info(format % args)


def default_data_dir() -> Path:
    return Path(__file__).parent.resolve()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Ilmu web -> OpenAI-compatible local proxy")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "6769")))
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("ILMU_DATA_DIR", default_data_dir())))
    parser.add_argument("--login", action="store_true", help="login and save session, then exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("-n", "--no-override", action="store_true", help="Disable system prompt overriding(No length check)")
    args = parser.parse_args(argv)

    # Configure logging based on verbosity flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[ilmu-web2api] %(levelname)s - %(message)s",
    )
    
    # Configure overriding param
    NO_OVERRIDE = args.no_override
    
    
    logger.debug("Starting initialization. Verbose mode enabled.")

    load_dotenv(Path(".env"))
    email, password = os.getenv("ILMU_EMAIL"), os.getenv("ILMU_PASSWORD")
    if not email or not password:
        logger.error("ILMU_EMAIL and ILMU_PASSWORD are required in .env")
        return 2

    config = Config(email, password, os.getenv("ACCESS_TOKEN"), args.host, args.port, args.data_dir)
    client = IlmuClient(config)

    if args.login:
        logger.debug("Running in login-only mode.")
        client.login()
        logger.info("Session saved to %s", client.token_file)
        return 0

    Handler.client, Handler.config = client, config
    logger.info("Listening on http://%s:%s", args.host, args.port)
    
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())