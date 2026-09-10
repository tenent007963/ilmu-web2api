#!/usr/bin/env python3
"""Headless Ilmu web client and OpenAI-compatible local proxy."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
from urllib.parse import urljoin

import requests


API_PREFIX = "/api/v1"
MODEL_MAP = {"ilmu-v3.1": {}, "ilmu-v3.1-instant": {"thinking_mode": "low"}, "ilmu-v3.1-thinking": {"thinking_mode": "high"}} # map to json content
SKIP_EVENTS = {"trace", "message_id", "dispatch", "comment", "comments"}


def load_dotenv(path: Path) -> None:
    """Load the small .env subset needed by this CLI without another dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def model_mapping(model: str) -> dict:
    if model not in MODEL_MAP:
        raise ValueError(f"unsupported model '{model}'; use one of: {', '.join(MODEL_MAP)}")
    return MODEL_MAP[model] # Removed json.dumps()


def parse_sse(lines: Iterable[str]) -> Iterator[Tuple[str, Any]]:
    """Yield (event, JSON-or-text data) from an SSE stream, tolerating CRLF and comments."""
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
        yield event, payload


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_content(x.get("text", x) if isinstance(x, dict) else x) for x in value)
    if isinstance(value, dict):
        # Added "choices" to the tuple below so it traverses the root object
        for key in ("choices", "content", "text", "delta", "message", "output"):
            if key in value:
                result = _content(value[key])
                if result:
                    return result
    return ""


def ilmu_events_to_openai(events: Iterable[Tuple[str, Any]], model: str, stream: bool) -> Iterator[Dict[str, Any]]:
    """Convert Ilmu events to OpenAI chunks, dropping internal protocol events."""
    usage = None
    finish_reason = None
    for event, payload in events:
        name = event.lower()
        if name in SKIP_EVENTS or payload == "[DONE]":
            continue
        if isinstance(payload, dict):
            if payload.get("usage") is not None:
                usage = payload["usage"]
            finish_reason = payload.get("finish_reason") or payload.get("finishReason") or finish_reason
            if payload.get("type", "").lower() in SKIP_EVENTS:
                continue
        text = _content(payload)
        if text:
            yield {
                "id": "chatcmpl-ilmu",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
    if stream:
        yield {
            "id": "chatcmpl-ilmu",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}],
            **({"usage": usage} if usage else {}),
        }


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
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ilmu-web2api/1.0", "Accept-Language": "en"})
        self.token_file = config.data_dir / "session.json"
        self.refresh_token: Optional[str] = None
        self.access_token: Optional[str] = None
        self._load_session()

    def _load_session(self) -> None:
        if not self.token_file.exists():
            return
        saved = json.loads(self.token_file.read_text(encoding="utf-8"))
        for cookie in saved.get("cookies", []):
            self.session.cookies.set(cookie["name"], cookie["value"], domain=cookie.get("domain"), path=cookie.get("path", "/"))
        self.refresh_token = saved.get("refresh_token")
        self.access_token = saved.get("access_token")

    def _save_session(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(json.dumps({
            "cookies": [{"name": c.name, "value": c.value, "domain": c.domain, "path": c.path} for c in self.session.cookies],
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
        }, indent=2), encoding="utf-8")

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        base_url = os.getenv("ILMU_BASE_URL", "https://chat.ilmu.ai").rstrip("/") + "/"
        response = self.session.request(method, urljoin(base_url, path.lstrip("/")), timeout=60, **kwargs)
        response.raise_for_status()
        return response

    @staticmethod
    def _token(data: Any, key: str) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        value = data.get(key) or data.get("data", {}).get(key) if isinstance(data.get("data"), dict) else data.get(key)
        return value if isinstance(value, str) else None

    def login(self) -> None:
        # The web client uses /auth/signin; /auth/login is not an Ilmu endpoint.
        data = self._request("POST", f"{API_PREFIX}/auth/signin",
                             json={"email": self.config.email, "password": self.config.password}).json()
        self.access_token = self._token(data, "access_token") or self._token(data, "token")
        self.refresh_token = self._token(data, "refresh_token")
        if not self.access_token and not self.refresh_token:
            raise RuntimeError("Ilmu login succeeded but returned no access_token/refresh_token")
        self._save_session()

    def refresh(self) -> str:
        try:
            response = self._request("POST", f"{API_PREFIX}/auth/refresh", json=({"refresh_token": self.refresh_token} if self.refresh_token else {}))
            data = response.json()
            token = self._token(data, "access_token") or self._token(data, "token")
            if not token:
                raise RuntimeError("refresh response did not contain access_token")
            self.access_token = token
            self.refresh_token = self._token(data, "refresh_token") or self.refresh_token
            self._save_session()
            return token
        except Exception as exc:
            raise RuntimeError(f"Ilmu access_token refresh failed: {exc}") from exc

    def _auth_headers(self) -> Dict[str, str]:
        if not self.access_token:
            self.refresh()
        return {"Authorization": f"Bearer {self.access_token}", "Accept-Language": "en"}

    def completion(self, messages: list, model: str, stream: bool) -> requests.Response:
        ilmu_model_meta = model_mapping(model)
        headers = self._auth_headers()
        
        # 1. Create chat
        chat = self._request("POST", f"{API_PREFIX}/chats", headers=headers, json={})
        chat_data = chat.json()
        chat_id = chat_data.get("id") or chat_data.get("data", {}).get("id")
        if not chat_id:
            raise RuntimeError("Ilmu chat creation response has no chat id")
            
        prompt = messages[-1].get("content", "") if messages else ""
        
        # 2. Send message (meta is now passed correctly as a dict)
        self._request("POST", f"{API_PREFIX}/chats/{chat_id}/messages", headers=headers, 
                      json={"content": prompt, "meta": ilmu_model_meta, "role": "user"})
        
        # 3. Request completions (matches the exact fetch sample structure)
        payload = {
            "messages": messages,
            "stream": True,
            "chat_id": chat_id,
            "locale": "en"
        }
        
        # Apply thinking_mode at the root level if the model specifies it
        if "thinking_mode" in ilmu_model_meta:
            payload["thinking_mode"] = ilmu_model_meta["thinking_mode"]

        response = self._request("POST", f"{API_PREFIX}/chat/completions", headers=headers,
                                 json=payload, stream=True)
        self.refresh()  # required as last step
        return response


class Handler(BaseHTTPRequestHandler):
    client: IlmuClient
    config: Config

    def _json(self, status: int, body: Any) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        required = self.config.access_token
        if not required:
            return True
        supplied = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        return supplied == required

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            if not self._authorized():
                self._json(401, {"error": {"message": "invalid access token", "type": "invalid_request_error"}})
                return
            self._json(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "ilmu"} for m in MODEL_MAP]})
            return
        self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        if not self._authorized():
            self._json(401, {"error": {"message": "invalid access token", "type": "invalid_request_error"}})
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            model, messages, stream = request["model"], request["messages"], bool(request.get("stream", False))
            response = self.client.completion(messages, model, stream)
            events = parse_sse(response.iter_lines())
            chunks = list(ilmu_events_to_openai(events, model, stream))
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
                usage = next((c["usage"] for c in chunks if "usage" in c), None)
                body = {"id": "chatcmpl-ilmu", "object": "chat.completion", "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
                if usage:
                    body["usage"] = usage
                self._json(200, body)
            try:
                self.client.refresh()  # refresh immediately after each completion
            except RuntimeError as exc:
                if stream:
                    self.wfile.write(f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n".encode())
                else:
                    raise
        except Exception as exc:
            self._json(502, {"error": {"message": str(exc), "type": "upstream_error"}})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[ilmu-web2api] {format % args}", file=sys.stderr)


def default_data_dir() -> Path:
    #root = os.getenv("LOCALAPPDATA") or os.getenv("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    #return Path(root) / "ilmu-web2api"
    root = Path(__file__).parent.resolve()
    return Path(root)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Ilmu web -> OpenAI-compatible local proxy")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("ILMU_DATA_DIR", default_data_dir())))
    parser.add_argument("--login", action="store_true", help="login and save session, then exit")
    args = parser.parse_args(argv)
    load_dotenv(Path(".env"))
    email, password = os.getenv("ILMU_EMAIL"), os.getenv("ILMU_PASSWORD")
    if not email or not password:
        print("ILMU_EMAIL and ILMU_PASSWORD are required in .env", file=sys.stderr)
        return 2
    config = Config(email, password, os.getenv("ACCESS_TOKEN"), args.host, args.port, args.data_dir)
    client = IlmuClient(config)
    if args.login:
        client.login()
        print(f"Session saved to {client.token_file}")
        return 0
    Handler.client, Handler.config = client, config
    print(f"Listening on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
