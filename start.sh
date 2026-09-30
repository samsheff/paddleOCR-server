#!/usr/bin/env bash
# Start the OCR server. Configuration comes from .env and/or OCR_* variables.
#
#   ./start.sh                            # localhost:8088 (safe default)
#   OCR_HOST=0.0.0.0 ./start.sh           # reachable from the LAN (opt-in)
#   OCR_PORT=9000 ./start.sh
#   OCR_MAX_PIXELS=627200 ./start.sh       # faster, slightly less detail
#
# Precedence: shell environment > .env file > built-in defaults.
# (pydantic-settings reads .env itself; the same precedence applies there.)
set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
PY="$VENV/bin/python"

[ -x "$PY" ] || { echo "ERROR: $VENV not found. Run ./setup.sh first." >&2; exit 1; }

# Read a key from .env without exporting it, so shell-provided values win.
env_from_file() {
  local key="$1"
  [ -f .env ] || return 0
  local line
  line="$(grep -E "^[[:space:]]*${key}=" .env | tail -n 1 || true)"
  [ -n "$line" ] || return 0
  line="${line#*=}"
  line="$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/")"
  printf '%s' "$line"
}

resolve() {
  local key="$1" fallback="$2" from_file
  if [ -n "${!key:-}" ]; then
    printf '%s' "${!key}"
  else
    from_file="$(env_from_file "$key")"
    if [ -n "$from_file" ]; then
      printf '%s' "$from_file"
    else
      printf '%s' "$fallback"
    fi
  fi
}

HOST="$(resolve OCR_HOST 127.0.0.1)"
PORT="$(resolve OCR_PORT 8088)"
WORKERS="$(resolve OCR_WORKERS 1)"
MODEL="$(resolve OCR_MODEL mlx-community/PaddleOCR-VL-1.5-4bit)"
CONCURRENCY="$(resolve OCR_CONCURRENCY 1)"
KEEPALIVE="$(resolve OCR_REQUEST_TIMEOUT 300)"
DEBUG_FLAG="$(resolve OCR_DEBUG false)"

# Make the resolved values visible to pydantic-settings (it re-reads .env, but
# explicit env vars take precedence there too).
export OCR_HOST="$HOST" OCR_PORT="$PORT"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1

if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ] && [ "$HOST" != "::1" ]; then
  echo "WARNING: listening on $HOST -- reachable by other machines on your network." >&2
fi

echo "PaddleOCR-VL OCR server"
echo "  model      : $MODEL"
if [ "$HOST" = "0.0.0.0" ]; then
  LAN_IP="$(ipconfig getifaddr en1 2>/dev/null || ipconfig getifaddr en0 2>/dev/null || true)"
  echo "  endpoint   : http://127.0.0.1:$PORT (local)"
  [ -n "$LAN_IP" ] && echo "  LAN         : http://$LAN_IP:$PORT"
else
  echo "  endpoint   : http://$HOST:$PORT"
fi
echo "  docs       : http://$HOST:$PORT/docs"
echo "  concurrency: $CONCURRENCY"

LOG_LEVEL="info"
[ "$DEBUG_FLAG" = "true" ] && LOG_LEVEL="debug"

exec "$PY" -m uvicorn app.server:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --timeout-keep-alive "$KEEPALIVE" \
  --no-access-log \
  --log-level "$LOG_LEVEL"
