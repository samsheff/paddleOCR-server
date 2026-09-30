#!/usr/bin/env bash
# Sweep resolution settings across a fresh server instance each time.
#
#   ./scripts/sweep.sh                    # pixel budget sweep at 200 dpi
#   ./scripts/sweep.sh --dpi 150 200 300  # DPI sweep
#
# Writes bench_sweep_px<PX>_dpi<DPI>.json per configuration.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/bin/python"
URL="http://127.0.0.1:8089"
PIXELS=(392000 627200 1003520 1605632 2510800)
DPIS=(200)

if [ "${1:-}" = "--dpi" ]; then
  shift
  DPIS=("$@")
fi

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for dpi in "${DPIS[@]}"; do
  for px in "${PIXELS[@]}"; do
    echo "### OCR_MAX_PIXELS=$px  OCR_PDF_DPI=$dpi"
    OCR_MAX_PIXELS="$px" OCR_PDF_DPI="$dpi" OCR_PORT=8089 OCR_HOST=127.0.0.1 \
      PYTHONPATH=. "$PY" -m uvicorn app.server:app --host 127.0.0.1 --port 8089 \
      --log-level warning > /tmp/ocr_sweep.log 2>&1 &
    SERVER_PID=$!

    for _ in $(seq 1 90); do
      curl -fsS "$URL/health" >/dev/null 2>&1 && break
      sleep 1
    done

    "$PY" scripts/benchmark.py --url "$URL" --pages 1-3 --repeats 2 \
      --out "bench_sweep_px${px}_dpi${dpi}.json" 2>&1 | tail -20
    "$PY" scripts/accuracy.py --url "$URL" 2>&1 | tail -6

    cleanup
    SERVER_PID=""
    sleep 2
  done
done
