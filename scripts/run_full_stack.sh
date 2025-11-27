#!/usr/bin/env bash
# Purpose: Spin up the FastAPI backend and Vite frontend together for local dev.
# Rationale:
#  - Keep commands side-by-side so you don't have to remember arguments.
#  - Restrict uvicorn reload to server/UI source to avoid reloads on tool-created files.
#  - Auto-install UI deps on first run to reduce setup friction.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Backend command with scoped reload paths and explicit host/port.
BACKEND_PORT=8000
FRONTEND_PORT=5173

BACKEND_CMD=(
  uvicorn web.server.main:app
  --reload
  --reload-dir "$ROOT/web/server"
  --reload-dir "$ROOT/web/ui/src"
  --host 127.0.0.1
  --port "$BACKEND_PORT"
)

# Frontend command; --host allows access from other devices on the network if needed.
FRONTEND_CMD=(
  npm run dev -- --host --port "$FRONTEND_PORT"
)

ask_port_action() {
  local port="$1"
  local desc="$2"
  if ! lsof -ti tcp:"$port" >/dev/null 2>&1; then
    return 0
  fi

  echo "[warn] Port $port in use for $desc."
  echo "Choose: [k]ill existing, [s]kip starting $desc, [a]bort script"
  read -r -p "> " choice </dev/tty || choice="a"
  case "$choice" in
    k|K)
      lsof -ti tcp:"$port" | xargs kill -9 || true
      echo "[info] Killed processes on port $port."
      ;;
    s|S)
      echo "[info] Skipping start of $desc due to port conflict."
      return 1
      ;;
    *)
      echo "[info] Aborting."
      exit 1
      ;;
  esac
}

# Install frontend deps if missing to avoid manual npm install step.
if [ ! -d "$ROOT/web/ui/node_modules" ]; then
  echo "[setup] Installing frontend dependencies..."
  (cd "$ROOT/web/ui" && npm install)
fi

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

echo "[start] Backend: ${BACKEND_CMD[*]}"
if ask_port_action "$BACKEND_PORT" "backend"; then
  (cd "$ROOT" && "${BACKEND_CMD[@]}") &
  pids+=($!)
fi

echo "[start] Frontend: ${FRONTEND_CMD[*]}"
if ask_port_action "$FRONTEND_PORT" "frontend"; then
  (cd "$ROOT/web/ui" && "${FRONTEND_CMD[@]}") &
  pids+=($!)
fi

echo "[info] Processes running. Backend on http://127.0.0.1:8000, Frontend on http://localhost:5173"
echo "[info] Press Ctrl+C to stop both."

# Portable wait: loop until any child dies (avoids bash 5-only wait -n).
while :; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[info] Process $pid exited; shutting down the other."
      exit 0
    fi
  done
  sleep 1
done
