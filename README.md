# PaddleOCR-VL-1.5 — local OCR server (MLX / Apple Silicon)

A FastAPI service that transcribes images and PDF pages to plain text using
[PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) running
on the Apple GPU through [MLX](https://github.com/ml-explore/mlx) / MLX-VLM.

Everything is local. No cloud APIs, no external inference services, no telemetry.

- **Measured on an M4 Max (64 GB):** ~1.28 s/page end to end at the default
  resolution, 100% character accuracy against known ground truth on the bundled
  samples, 1.7 GB peak GPU memory, ~610 generated tokens/sec during decode.
- **Model loads once at startup** and stays resident. Requests are serialised
  (`concurrency=1`) because the Apple GPU is a shared resource.

---

## 1. Quick start

```bash
cd paddle-ocr-server

./setup.sh                 # venv + deps + model download + GPU check
./start.sh                 # http://127.0.0.1:8088
./test.sh samples/sample.pdf --pages 1-3
```

Interactive API docs: <http://127.0.0.1:8088/docs>

### Requirements

- macOS on Apple Silicon (arm64). Not supported on Intel or Linux.
- Python 3.12 (3.11/3.13 also fine), `uv` recommended.
- ~2 GB disk for the model + venv.
- `setup.sh` refuses to run on non-arm64 and fails loudly if MLX is not on the GPU.

---

## 2. API

Base URL: `http://127.0.0.1:8088`

### `GET /health`

Liveness plus model readiness. Poll this before sending work.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "mlx-community/PaddleOCR-VL-1.5-4bit",
  "uptime_s": 118.4,
  "requests_served": 21
}
```

`status` is `"loading"` while the model is still being loaded at startup and
`"ok"` afterwards. Requests to `/ocr` return `503` until it is `"ok"`.

### `GET /info`

Engine, hardware and configuration details. Useful for verifying the GPU is in
use and for debugging a client's expectations.

```json
{
  "model": "mlx-community/PaddleOCR-VL-1.5-4bit",
  "prompt": "OCR:",
  "backend": "mlx",
  "mlx_device": "Device(gpu, 0)",
  "gpu_available": true,
  "metal_available": true,
  "model_loaded": true,
  "model_load_time_s": 0.96,
  "concurrency": 1,
  "max_tokens": 4096,
  "max_pixels": 1003520,
  "min_pixels": 112896,
  "max_image_dim": 4096,
  "pdf_dpi": 200,
  "sanitize": true,
  "loop_guard": true,
  "gpu_memory_gb": 0.71,
  "gpu_peak_memory_gb": 1.68,
  "host_memory_mb": 1202.0,
  "api": { "version": "1.0.0", "endpoints": ["/health", "/info", "/ocr", "/ocr/json", "/docs"] },
  "config": { "...": "all OCR_* settings" }
}
```

### `POST /ocr` — multipart upload

| Field    | Type   | Default | Notes                                              |
| -------- | ------ | ------- | -------------------------------------------------- |
| `file`   | file   | —       | PNG, JPEG, TIFF, BMP, WebP **or** a PDF            |
| `page`   | int    | `1`     | 1-based PDF page. Ignored for images. `422` if < 1 |
| `max_tokens` | int | server default | Generation cap for this request        |

```bash
curl -X POST http://127.0.0.1:8088/ocr -F "file=@page.png"

# PDF, page 3, with a tighter token budget
curl -X POST "http://127.0.0.1:8088/ocr?page=3&max_tokens=2048" \
     -F "file=@document.pdf"
```

### `POST /ocr/json` — base64 body

For clients that cannot do multipart. Accepts a bare base64 string or a
`data:` URL. Exactly one of the two fields must be present.

```jsonc
{
  "image_base64": "iVBORw0KGgo...",   // or "data:image/png;base64,iVBORw0KGgo..."
  "pdf_base64":   null,
  "page": 1,
  "max_tokens": 4096                  // optional
}
```

```bash
curl -X POST http://127.0.0.1:8088/ocr/json \
     -H 'Content-Type: application/json' \
     -d "{\"image_base64\": \"$(base64 -i page.png)\"}"
```

### Response schema

Both OCR endpoints return the same object. The first five fields are the stable
contract; everything after `page` is diagnostic and safe to ignore.

```json
{
  "text": "Quarterly Operations Report\nPrepared for the Board of Directors - March 2026\n\n1. Summary\n...",
  "processing_time_ms": 1437,
  "model": "mlx-community/PaddleOCR-VL-1.5-4bit",
  "image_width": 842,
  "image_height": 1191,

  "prompt_tokens": 1273,
  "generation_tokens": 299,
  "generation_tps": 621.66,
  "tokens_per_second": 208.01,
  "effective_tps": 208.01,
  "prompt_tokens_per_second": 1373.09,
  "preprocess_time_ms": 37.9,
  "inference_time_ms": 1399.1,
  "peak_gpu_memory_gb": 1.683,
  "finish_reason": "stop",
  "loop_guard_triggered": false,
  "page": 1,
  "source_width": 1654,
  "source_height": 2339,
  "pdf_dpi": 200
}
```

| Field | Meaning |
| ----- | ------- |
| `text` | Plain-text transcription. Line breaks and paragraph/table structure preserved. No summaries or rewrites. |
| `processing_time_ms` | Whole request handling time on the server, integer ms. |
| `model` | Model id that produced the text. |
| `image_width` / `image_height` | Dimensions **fed to the model**, after downscaling. |
| `source_width` / `source_height` | Dimensions as uploaded / as rendered from PDF. |
| `page` | 1-based PDF page, or `null` for an image upload. |
| `pdf_dpi` | Render DPI used for PDFs, `null` for images. |
| `preprocess_time_ms` | Image decode + PDF render + resize. |
| `inference_time_ms` | Prefill + decode (preprocess excluded). |
| `prompt_tokens` | Input tokens, including vision patch tokens. Scales with image area. |
| `generation_tokens` | Output tokens. |
| `generation_tps` | Decode-only throughput (mlx-vlm's measure, excludes prefill). |
| `tokens_per_second` / `effective_tps` | Output tokens / full inference time. Use this for per-page budgeting. |
| `peak_gpu_memory_gb` | MLX peak memory for that request. |
| `finish_reason` | `"stop"` (EOS) or `"length"` (`max_tokens` reached). |
| `loop_guard_triggered` | `true` if the model began repeating and output was cut at the restart. See below. |

### Errors

| Status | Meaning |
| ------ | ------- |
| `400` | Undecodable image, invalid base64, or PDF page out of range (`detail` explains which). |
| `413` | Upload exceeds `OCR_MAX_UPLOAD_MB`. |
| `422` | Bad query parameter (e.g. `page=0`), or `/ocr/json` with neither base64 field. |
| `503` | Model still loading at startup. |
| `500` | Unexpected error. The detail is logged server-side, not returned. |

---

## 3. Output guarantees

`text` is a transcription, not a description:

- No summarising, rewriting, translating, correcting or "cleaning up".
- Paragraphs are separated by blank lines; table rows stay on their own lines
  (aligned plain text at lower resolutions, Markdown pipes at higher ones — the
  model chooses, and both are left intact).
- Word-wrapped body text is joined into paragraph flow. That is the model's own
  document-parsing behaviour, not a post-processing step.
- The only post-processing is mechanical: `<|...|>` control/layout tokens such as
  `<|LOC_94|>` are stripped and runs of 3+ blank lines are clamped to one. Set
  `OCR_SANITIZE=false` to disable it entirely.

**Repetition guard.** At some resolution budgets PaddleOCR-VL occasionally
finishes a page and then starts it again, looping until the token budget runs
out. One sample page did this: 521 tokens, 2.7 s, and the page returned three
times over. The server detects the restart, stops decoding, and returns only the
first pass — `loop_guard_triggered` is `true` when this happens, so a client can
detect a possibly truncated page. Set `OCR_LOOP_GUARD=false` to disable.

---

## 4. Configuration

Every setting is an `OCR_`-prefixed environment variable, read from the process
environment or `.env`. Precedence: shell environment > `.env` > defaults.

Copy the annotated template first:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `OCR_MODEL` | `mlx-community/PaddleOCR-VL-1.5-4bit` | Any mlx-vlm PaddleOCR-VL build or a local path. Also `...-1.6-4bit`, `...-4bit`. |
| `OCR_PROMPT` | `OCR:` | Task prompt. **Leave alone** — free-form instructions flip the model into spotting mode, which emits `<\|LOC_n\|>` tokens. |
| `OCR_MAX_TOKENS` | `4096` | Default generation cap. |
| `OCR_TEMPERATURE` | `0.0` | `0` = greedy, deterministic. |
| `OCR_MAX_PIXELS` | `1003520` | Vision budget in pixels, rounded down to a multiple of 784 (=28×28). This is the main speed/accuracy dial. |
| `OCR_MIN_PIXELS` | `112896` | Lower bound; small images are upscaled to reach it. |
| `OCR_MAX_IMAGE_DIM` | `4096` | Hard ceiling on the longest edge (0 disables). |
| `OCR_PDF_DPI` | `200` | PDF rasterisation DPI. 150–200 is the useful range. |
| `OCR_HOST` | `127.0.0.1` | `0.0.0.0` exposes on the LAN. |
| `OCR_PORT` | `8088` | |
| `OCR_CONCURRENCY` | `1` | Parallel inferences. Keep at 1 for a shared GPU. |
| `OCR_REQUEST_TIMEOUT` | `300` | Seconds; also the HTTP keep-alive. |
| `OCR_MAX_UPLOAD_MB` | `32` | |
| `OCR_DEBUG` | `false` | Verbose server logging. |
| `OCR_WARMUP` | `true` | One tiny generation at startup so the first real request is not paying for kernel compilation. |
| `OCR_SANITIZE` | `true` | Strip control tokens / clamp blank lines. |
| `OCR_LOOP_GUARD` | `true` | Stop decoding when output restarts from the top. |

### Resolution / speed trade-off (measured on this M4 Max)

A4 page, 200 DPI, mean of 6 requests. Accuracy is character similarity against
the known text of the sample pages.

| `OCR_MAX_PIXELS` | Vision size | ms/page | decode tok/s | char accuracy |
| --------------- | ----------- | ------- | ------------ | ------------- |
| 392 000 | 526×744 | 509 | 700 | 0.970 |
| 627 200 | 665×941 | 796 | 626 | **1.000** |
| **1 003 520** (default) | 842×1191 | 1253 | 586 | **1.000** |
| 1 605 632 | 1065×1506 | 2291 | 590 | 0.901 (looped page) |
| 2 510 800 | 1332×1884 | 4017 | 531 | 0.995 |

Latency is dominated by the vision encoder, not by token decoding — prefill is
~73% of request time. So halving `OCR_MAX_PIXELS` roughly halves latency.

**Default choice:** `1003520` (= `1280 * 28 * 28`), the value the PaddleOCR-VL
model card recommends for the `ocr` task. It was exact on every sample page.

**Faster choice:** `OCR_MAX_PIXELS=627200` gives 1.6× throughput at the same
accuracy on these clean synthetic pages. Choose it for large volumes of
good-quality scans; stay at the default for small print, dense tables, faxes and
phone photos.

---

## 5. Scripts

| Command | What it does |
| ------- | ------------ |
| `./setup.sh` | Creates `.venv`, installs `requirements.txt`, copies `.env.example` to `.env`, downloads the model, asserts MLX is on the GPU. |
| `./start.sh` | Starts uvicorn. `OCR_HOST=0.0.0.0 ./start.sh` for LAN access. |
| `./test.sh samples/sample.pdf` | Health + info + OCR the first 3 pages, with timings. |
| `./test.sh doc.pdf --pages 1-20 --out results/` | OCR a page range and write `pageNNN.txt` plus a `summary.tsv`. |
| `./test.sh page.png --max-tokens 1024` | Override the token budget per request. |

Utilities:

```bash
# regenerate the bundled sample page / PDF
.venv/bin/python scripts/make_sample.py
.venv/bin/python scripts/make_pdf_sample.py

# benchmark: model load, preprocess, inference, tok/s, peak memory
.venv/bin/python scripts/benchmark.py --pages 1-3 --repeats 5

# accuracy against the known text of the sample pages
.venv/bin/python scripts/accuracy.py

# resolution sweep (restarts a server per setting)
./scripts/sweep.sh

# pretty-print any response from the API
curl -s -X POST localhost:8088/ocr -F file=@page.png | .venv/bin/python scripts/show.py
```

### Tests

```bash
.venv/bin/python -m pytest tests/test_unit.py tests/test_api.py   # no GPU needed
.venv/bin/python tests/test_live.py                              # needs ./start.sh
```

`test_api.py` stubs the model, so the HTTP contract is verified in under a
second with no weights loaded. `test_live.py` drives a real running server and
checks transcription fidelity, GPU acceleration, PDF handling, and that the
model is not reloaded between requests.

---

## 6. LAN access

Localhost is the default and is what you want unless another machine needs to
call it.

```bash
OCR_HOST=0.0.0.0 ./start.sh
# PaddleOCR-VL OCR server
#   endpoint   : http://127.0.0.1:8088 (local)
#   LAN         : http://192.168.68.60:8088
```

Or set `OCR_HOST=0.0.0.0` in `.env` persistently. The service has **no
authentication** — only do this on a trusted network. To expose it to the wider
internet, put a reverse proxy with TLS and auth in front of it.

---

## 7. How it works

```
app/config.py        Settings (pydantic-settings, OCR_* env vars)
app/ocr.py           OCREngine: loads the model once, pinned inference thread,
                     repetition guard, per-request metrics
app/preprocessing.py image decode (EXIF-aware), PDF rasterisation via pypdfium2,
                     pixel/dimension ceilings
app/server.py        FastAPI routes, request validation, response shaping
```

**Model residency.** `OCREngine.load()` runs in the application lifespan, before
the server accepts traffic, and holds the weights for the whole process. Nothing
in the request path reloads anything; `GET /info` reports the one-time
`model_load_time_s`.

**Pinned inference thread.** MLX keeps its Metal streams thread-local. Running
inference from asyncio's shared thread pool meant requests landed on different
threads, and the first request after an idle gap paid for stream recreation:
2 796 ms instead of 1 438 ms, at 165 tok/s instead of 620. The engine now owns a
single-thread executor that every request goes through, which removed the
penalty entirely. CPU work (image decode, PDF render) stays on the shared pool
since it holds no MLX state.

**API used.** Current `mlx-vlm` (0.7.x), not the older samples:

```python
from mlx_vlm import load, prompt_utils
from mlx_vlm.utils import load_config
from mlx_vlm.generate import dispatch

model, processor = load("mlx-community/PaddleOCR-VL-1.5-4bit")
config = load_config("mlx-community/PaddleOCR-VL-1.5-4bit")
prompt = prompt_utils.apply_chat_template(processor, config, "OCR:", num_images=1)
result = dispatch.generate(model, processor, prompt, [image], max_tokens=4096,
                          temperature=0.0)
result.text, result.generation_tokens, result.generation_tps, result.peak_memory
```

Older snippets used `model.generate(...)` or a `transformers`-based pipeline.
Those still work but do not expose the same timing or peak-memory fields. The
prompt is cached, and the vision pixel budget is pinned once on the processor so
no config file is re-read per request.

**Prompt matters.** `"OCR:"` is the task prompt that yields clean plain text. A
natural-language instruction ("transcribe this exactly...") is *not* equivalent:
it switches the model to text spotting and the output fills up with `<|LOC_n|>`
polygon tokens. Measured directly — that is why `OCR_PROMPT` is a single value
rather than a free field.

**PDF handling.** `pypdfium2` rasterises the requested page at `OCR_PDF_DPI`
(white background, annotations and form fields drawn), then the image is
downscaled so the vision tower never sees more than `OCR_MAX_PIXELS`. The model's
own `smart_resize` does the final patch-aligned resize, so there is exactly one
resampling pass at each stage and no duplicated work.

---

## 8. Measured performance

Hardware: Apple M4 Max, 64 GB unified memory, macOS 27.0.1, arm64.
Model: `mlx-community/PaddleOCR-VL-1.5-4bit` (0.9 B params, 4-bit, 704 MB).
Server: `OCR_MAX_PIXELS=1003520`, `OCR_PDF_DPI=200`, `OCR_CONCURRENCY=1`.
Workload: `samples/sample.pdf`, 3 pages, 5 iterations = 15 requests, after warmup.

| Metric | Value |
| ------ | ----- |
| Model load time (cold, weights not in page cache) | ~1.3 s |
| Model load time (warm) | **0.76–0.97 s** |
| Package import time | 0.60 s |
| Total per page (mean) | **1.276 s** (0.78 pages/s) |
| Total per page (min / max) | 976 ms / 1453 ms |
| Image + PDF preprocessing (mean) | 39.8 ms (3.1% of total) |
| Inference (mean) | 1235.7 ms |
| — of which prefill / vision encode | ~73% of request time |
| — of which token decode | 324 ms |
| Generated tokens/sec (decode only) | **611 tok/s** |
| Effective tokens/sec (out ÷ full inference) | 140 tok/s |
| Prefill throughput | 1368 tok/s |
| Prompt tokens / page | 1273 (scales with image area) |
| Output tokens / page | 198 (mean) |
| Peak GPU memory | **1.68 GB** |
| Host RSS (server process) | ~1.20 GB |
| Character accuracy vs. ground truth | **1.000** (all 3 pages) |

Per page:

| Page | Content | Total | Preprocess | Tokens out | tok/s |
| ---- | ------- | ----- | ---------- | ---------- | ----- |
| 1 | Title, prose, 4-row table, bullet list (1056 chars) | 1421 ms | 41 ms | 299 | 209 eff / 611 decode |
| 2 | Dense 18 px body text, heading, 6-row table (1167 chars) | 1372 ms | 40 ms | 278 | 200 eff / 611 decode |
| 3 | Two lines (62 chars) | 953 ms | 39 ms | 16 | 17 eff / 636 decode |

### What was optimised

1. **Model loaded once at startup**, resident for the process lifetime.
2. **Pinned inference thread** — removed a 2× worst-case decode penalty after
   idle gaps (2 796 ms → 1 438 ms).
3. **Repetition guard** — one looping page went from 2 740 ms / 2 166 chars to
   1 347 ms / 1 167 chars of correct text.
4. **Prompt template and pixel budget cached** on the processor, so no config is
   re-parsed per request.
5. **Single-pass resampling** — PDF render and preprocessing cap dimensions;
   `smart_resize` alone performs the patch-aligned resize.
6. **Warmup generation at startup** so the first user request is not paying for
   kernel compilation.

### Remaining bottleneck

Prefill (the vision tower forward pass) is ~73% of the time and scales with
image area. The only real lever left is `OCR_MAX_PIXELS`, which trades detail
for latency as shown in the table above. Decode is already fast at ~611 tok/s,
so batching or speculative decoding would buy little here — this is a 0.9 B
model, not a large one.

Note: the Mac Studio's GPU is shared with other workloads. When other GPU
processes are active, decode throughput drops (observed 165–630 tok/s under
contention versus ~610 tok/s idle); prefill is far less affected. The figures
above were measured on an otherwise idle GPU.
