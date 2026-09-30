#!/usr/bin/env bash
# Smoke-test a running server against an image or PDF.
#
#   ./test.sh                          # bundled sample page
#   ./test.sh samples/sample_page.png
#   ./test.sh sample.pdf               # first 3 pages
#   ./test.sh samples/sample.pdf --pages 1-5
#   ./test.sh samples/sample.pdf --pages all --max-tokens 2048
#   ./test.sh samples/sample.pdf --out results/   # also write per-page .txt
#
# Start the server first with ./start.sh
set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
PY="$VENV/bin/python"
BASE_URL="${BASE_URL:-http://${OCR_HOST:-127.0.0.1}:${OCR_PORT:-8088}}"

TARGET="${1:-samples/sample_page.png}"
PAGES=""
MAX_TOKENS=""
OUTDIR=""

shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --pages)      PAGES="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --out)        OUTDIR="$2"; shift 2 ;;
    --url)        BASE_URL="$2"; shift 2 ;;
    -h|--help)    sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -x "$PY" ] || { echo "ERROR: $VENV not found. Run ./setup.sh first." >&2; exit 1; }
[ -f "$TARGET" ] || { echo "ERROR: file not found: $TARGET" >&2; exit 1; }

bold() { printf '\033[1m%s\033[0m\n' "$*"; }

bold "waiting for $BASE_URL/health ..."
for _ in $(seq 1 120); do
  curl -fsS "$BASE_URL/health" >/dev/null 2>&1 && break
  sleep 2
done

bold "GET /health"
curl -fsS "$BASE_URL/health" | "$PY" scripts/show.py --health || {
  echo "server not reachable at $BASE_URL (is ./start.sh running?)" >&2; exit 1; }

bold "GET /info"
curl -fsS "$BASE_URL/info" | "$PY" scripts/show.py --health

# --- decide which pages to OCR -----------------------------------------
IS_PDF=0
if head -c 5 "$TARGET" | grep -q '%PDF'; then IS_PDF=1; fi

if [ "$IS_PDF" = "1" ]; then
  NPAGES="$("$PY" - "$TARGET" <<'PYEOF'
import sys, pypdfium2 as pdfium
d = pdfium.PdfDocument(sys.argv[1])
print(len(d)); d.close()
PYEOF
)"
  [ -z "$PAGES" ] && PAGES="1-$(( NPAGES < 3 ? NPAGES : 3 ))"
  [ "$PAGES" = "all" ] && PAGES="1-$NPAGES"
  PAGE_LIST="$("$PY" - "$PAGES" <<'PYEOF'
import sys
out = []
for part in sys.argv[1].split(","):
    if "-" in part:
        a, b = part.split("-", 1)
        out.extend(range(int(a), int(b) + 1))
    else:
        out.append(int(part))
print(" ".join(str(p) for p in out))
PYEOF
)"
  echo "PDF with $NPAGES page(s); OCRing pages: $PAGE_LIST"
else
  PAGE_LIST="1"
fi

[ -n "$OUTDIR" ] && mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.tsv"
[ -n "$OUTDIR" ] && printf 'page\tchars\ttime_ms\ttok_per_s\tloop_guard\n' > "$SUMMARY"

for p in $PAGE_LIST; do
  URL="$BASE_URL/ocr?page=$p"
  [ -n "$MAX_TOKENS" ] && URL="$URL&max_tokens=$MAX_TOKENS"

  printf '\n'
  bold "POST /ocr  file=$(basename "$TARGET")  page=$p"
  RESPONSE="$(curl -fsS -X POST "$URL" -F "file=@$TARGET")" || {
    echo "request failed" >&2; continue; }
  printf '%s' "$RESPONSE" | "$PY" scripts/show.py

  if [ -n "$OUTDIR" ]; then
    RESP_FILE="$(mktemp)"
    printf '%s' "$RESPONSE" > "$RESP_FILE"
    "$PY" - "$RESP_FILE" "$OUTDIR" "$p" <<'PYEOF'
import json, sys
resp_file, outdir, page = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(resp_file) as fh:
    d = json.load(fh)
with open(f"{outdir}/page{page:03d}.txt", "w") as fh:
    fh.write(d.get("text", ""))
with open(f"{outdir}/summary.tsv", "a") as fh:
    fh.write(
        f"{d.get('page')}\t{len(d.get('text',''))}\t{d.get('processing_time_ms')}"
        f"\t{d.get('tokens_per_second')}\t{d.get('loop_guard_triggered')}\n"
    )
PYEOF
    rm -f "$RESP_FILE"
  fi
done

if [ -n "$OUTDIR" ]; then
  printf '\n'
  bold "summary ($OUTDIR/summary.tsv)"
  column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
fi
echo
