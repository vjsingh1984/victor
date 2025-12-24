# Victor Web UI (Vite + React)

Lightweight chat UI for the Victor backend with WebSocket streaming.

## Quick start

```bash
cd web/ui
npm install
VITE_WS_URL=ws://localhost:8000/ws npm run dev
```

Optional auth for secured servers:
- Set `VITE_API_TOKEN=<your_api_key>` to include `api_key` on WebSocket connects and prefetch a signed `session_token`.
- Tokens are never persisted to localStorage; the UI fetches fresh tokens per run.

## Config
- `VITE_WS_URL` points to the backend WebSocket endpoint.
- `VITE_API_TOKEN` (optional) must match `VICTOR_SERVER_API_KEY` on the server.

## Notes
- Session tokens are issued by the server; the UI reuses them for reconnects.
- Stored chat sessions exclude tokens by design (privacy/safety).
