# Ilmu Web-to-API Proxy

A headless web client and local proxy that bridges the Ilmu AI web interface (`chat.ilmu.ai`) to an OpenAI-compatible API. 

This project allows you to seamlessly integrate Ilmu's language models into IDEs like VS Code, standard OpenAI-compatible API clients, or custom shell/Python automation scripts. It handles session authentication, automatic token refreshing, Server-Sent Events (SSE) streaming, and payload formatting natively.

---

##  Features

*   **OpenAI API Compatibility:** Exposes standard `/v1/models` and `/v1/chat/completions` endpoints.
*   **Automatic Session Management:** Authenticates via email/password, saves session cookies, and features a background daemon to auto-refresh access tokens before they expire.
*   **Streaming Support:** Fully supports SSE (Server-Sent Events) for real-time text generation.
*   **Rich Content Parsing:** Translates Ilmu's rich markdown (sources, images) into standard markdown text.
*   **Containerized:** Easy deployment using an Alpine-based Python 3.11 Docker environment.

---

##  Supported Models

The proxy maps standard requests to the following Ilmu models:
*   `ilmu-v3.1`
*   `ilmu-v3.1-instant` *(Default)*
*   `ilmu-v3.1-thinking` (High thinking mode)

*Note: If an unsupported model is requested, the proxy safely defaults to `ilmu-v3.1-instant`*

---

##  Project Structure

*   `ilmu_web2api.py`: The core Python proxy application handling HTTP requests, authentication, and translation.
*   `Dockerfile`: Packages the script into a lightweight `python:3.11-alpine` image with the `requests` library.
*   `docker-compose.yml`: Orchestrates the container, maps ports, and manages environment variables and volume persistence.

---

##  Configuration & Setup

### 1. Prerequisites
*   Docker and Docker Compose installed.
*   An active Ilmu AI account.

### 2. Environment Variables
Create a `.env` file in the root of the project directory. The application requires your Ilmu credentials to fetch the initial access tokens.

```env
# Required Authentication
ILMU_EMAIL=your_registered_email@example.com
ILMU_PASSWORD=your_secure_password

# Optional Overrides (Handled automatically by docker-compose)
# ACCESS_TOKEN=your_manual_token
```

### 3. Deployment with Docker Compose
To build and start the proxy in the background, run:
```bash
docker-compose up -d --build
```

###  Usage Example
Once the container is running, you can interact with it just like the standard OpenAI API.

cURL Example:

```Bash
curl http://localhost:5678/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ilmu-v3.1-instant",
    "messages": [{"role": "user", "content": "Write a bash script to parse system logs."}],
    "stream": true
  }'
```  

###️ Advanced: CLI Flags (Standalone Python)
If you prefer to run ilmu_web2api.py directly without Docker, it accepts the following command-line arguments:

`--host`: IP to bind to (Defaults to 0.0.0.0).

`--port`: Port to listen on (Defaults to 5678).

`--data-dir`: Custom path to store session.json.

`--login`: Authenticate, save the session file, and exit immediately (useful for manual token generation).

`-v`, `--verbose`: Enable verbose debug logging for troubleshooting upstream errors.

`-n`, `--no-override`: Disables the automatic overriding/shortening of large system prompts (Useful if you want to bypass the system prompt hijacking designed to prevent 502 errors).