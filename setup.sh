#!/usr/bin/env bash
# One-time setup: create an isolated venv, install pinned deps, fetch the model.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-.venv}"

info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- pick an installer ------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  INSTALLER="uv"
elif command -v pip3 >/dev/null 2>&1; then
  INSTALLER="pip"
else
  die "need uv or pip3 on PATH (brew install uv)"
fi

# --- platform guard ---------------------------------------------------
[ "$(uname -m)" = "arm64" ] || die "this service requires Apple Silicon (arm64); found $(uname -m)"

# --- venv -------------------------------------------------------------
if [ ! -d "$VENV" ]; then
  info "creating virtualenv ($VENV, python $PYTHON_VERSION)"
  if [ "$INSTALLER" = "uv" ]; then
    uv venv --python "$PYTHON_VERSION" "$VENV"
  else
    python3 -m venv "$VENV"
  fi
else
  info "reusing existing virtualenv ($VENV)"
fi

PY="$VENV/bin/python"

info "upgrading build tooling"
if [ "$INSTALLER" = "uv" ]; then
  VIRTUAL_ENV="$VENV" uv pip install --upgrade pip setuptools wheel >/dev/null
else
  "$PY" -m pip install --upgrade pip setuptools wheel >/dev/null
fi

info "installing requirements"
if [ "$INSTALLER" = "uv" ]; then
  VIRTUAL_ENV="$VENV" uv pip install -r requirements.txt
else
  "$PY" -m pip install -r requirements.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  info "created .env from .env.example"
else
  info ".env already exists, leaving it alone"
fi

MODEL="$(grep -E '^OCR_MODEL=' .env | cut -d= -f2- | tr -d '"' | tr -d "'")"
MODEL="${MODEL:-mlx-community/PaddleOCR-VL-1.5-4bit}"

info "prefetching model: $MODEL"
if [ "$INSTALLER" = "uv" ]; then
  VIRTUAL_ENV="$VENV" uv pip install -q huggingface_hub >/dev/null
fi
HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" - <<PYEOF
from huggingface_hub import snapshot_download
path = snapshot_download("$MODEL", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.py"])
print("model cached at:", path)
PYEOF

info "verifying Metal / GPU availability"
"$PY" - <<'PYEOF'
import mlx.core as mx
device = str(mx.default_device())
metal = mx.metal.is_available() if hasattr(mx, "metal") else False
print("mlx device:", device)
print("metal available:", metal)
if not metal or "gpu" not in device:
    raise SystemExit("MLX is not using the GPU. Check the mlx wheel for this macOS version.")
PYEOF

info "setup complete. Start the server with ./start.sh"
