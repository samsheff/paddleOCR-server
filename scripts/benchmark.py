"""Benchmark the OCR service and record the results.

Reports, per page: preprocessing time, inference time, total seconds/page,
generated tokens/sec and peak memory. Optionally sweeps OCR_MAX_PIXELS and
OCR_PDF_DPI so the accuracy/latency trade-off is visible.

    python scripts/benchmark.py
    python scripts/benchmark.py --sweep-pixels
    python scripts/benchmark.py --url http://127.0.0.1:8088 --pages 1-3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def post_ocr(url: str, path: Path, page: int = 1, mime: str = "application/pdf") -> dict:
    boundary = f"----ocrbench{uuid.uuid4().hex}"
    filename = path.name
    with open(path, "rb") as fh:
        blob = fh.read()

    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    body += blob
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{url}/ocr?page={page}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    wall0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    data["client_wall_s"] = round(time.perf_counter() - wall0, 3)
    return data


def get_json(url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{url}{path}", timeout=60) as resp:
        return json.loads(resp.read().decode())


def rss_mb() -> float:
    if psutil is None:
        return -1.0
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def bench(url: str, path: Path, pages: list[int], repeats: int, warm: bool) -> dict:
    if warm:
        post_ocr(url, path, pages[0])
        time.sleep(1.0)

    rows = []
    for i in range(repeats):
        for page in pages:
            d = post_ocr(url, path, page)
            rows.append(
                {
                    "iter": i + 1,
                    "page": d.get("page"),
                    "total_ms": d["processing_time_ms"],
                    "preprocess_ms": d["preprocess_time_ms"],
                    "inference_ms": d["inference_time_ms"],
                    "gen_tokens": d["generation_tokens"],
                    "prompt_tokens": d["prompt_tokens"],
                    "tps": d["tokens_per_second"],
                    "decode_tps": d["generation_tps"],
                    "prefill_tps": d.get("prompt_tokens_per_second", 0),
                    "image": f"{d['image_width']}x{d['image_height']}",
                    "source": f"{d.get('source_width')}x{d.get('source_height')}",
                    "chars": len(d["text"]),
                    "peak_gb": d["peak_gpu_memory_gb"],
                    "client_wall_ms": round(d["client_wall_s"] * 1000, 1),
                }
            )
            print(
                f"  iter {i + 1} page {d.get('page')}: "
                f"{d['processing_time_ms']:>6} ms  "
                f"pre {d['preprocess_time_ms']:>6.1f}  inf {d['inference_time_ms']:>7.1f}  "
                f"{d['tokens_per_second']:>6.1f} eff tok/s  "
                    f"{d['generation_tps']:>6.1f} decode tok/s  "
                f"{len(d['text']):>5} chars  {d['image_width']}x{d['image_height']}  "
                f"peak {d['peak_gpu_memory_gb']} GB"
            )

    totals = [r["total_ms"] for r in rows]
    return {
        "rows": rows,
        "stats": {
            "pages": len(rows),
            "mean_ms": round(statistics.mean(totals), 1),
            "median_ms": round(statistics.median(totals), 1),
            "min_ms": min(totals),
            "max_ms": max(totals),
            "stdev_ms": round(statistics.stdev(totals), 1) if len(totals) > 1 else 0.0,
            "mean_preprocess_ms": round(statistics.mean(r["preprocess_ms"] for r in rows), 1),
            "mean_inference_ms": round(statistics.mean(r["inference_ms"] for r in rows), 1),
            "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows), 1),
            "mean_gen_tokens": round(statistics.mean(r["gen_tokens"] for r in rows), 1),
            # Share of end-to-end latency spent encoding the image (prefill,
            # which includes the vision tower forward pass).
            "prefill_share_pct": round(
                100.0
                * sum(
                    1000.0 * r["prompt_tokens"] / r["prefill_tps"]
                    for r in rows
                    if r["prefill_tps"]
                )
                / sum(r["total_ms"] for r in rows),
                1,
            ),
            # Decode-only time, for contrast with prefill.
            "mean_decode_ms": round(
                statistics.mean(
                    r["gen_tokens"] / r["decode_tps"] * 1000.0
                    for r in rows
                    if r["decode_tps"]
                ),
                1,
            ),
            "mean_seconds_per_page": round(
                statistics.mean(r["total_ms"] for r in rows) / 1000.0, 3
            ),
            "mean_tps_effective": round(statistics.mean(r["tps"] for r in rows), 2),
            "mean_tps_decode": round(statistics.mean(r["decode_tps"] for r in rows), 2),
            "mean_tps_prefill": round(statistics.mean(r["prefill_tps"] for r in rows), 2),
            "peak_gpu_gb": max(r["peak_gb"] for r in rows),
            "pages_per_s": round(1000.0 / statistics.mean(totals), 3),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    ap.add_argument("--file", default=str(ROOT / "samples" / "sample.pdf"))
    ap.add_argument("--pages", default="1-3")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "bench_results.json"))
    ap.add_argument("--sweep-pixels", action="store_true",
                    help="restart the server across OCR_MAX_PIXELS values (slow)")
    args = ap.parse_args()

    if "-" in args.pages:
        a, b = args.pages.split("-", 1)
        pages = list(range(int(a), int(b) + 1))
    else:
        pages = [int(x) for x in args.pages.split(",")]

    path = Path(args.file)
    if not path.is_absolute():
        path = Path.cwd() / path

    print(f"benchmarking {args.url} against {path} pages={pages} repeats={args.repeats}")

    health = get_json(args.url, "/health")
    info = get_json(args.url, "/info")
    print(f"server: {health['status']}  model={info['model']}  device={info['mlx_device']}")
    print(f"metal={info['metal_available']}  max_pixels={info['max_pixels']}  "
          f"pdf_dpi={info['pdf_dpi']}  concurrency={info['concurrency']}\n")

    t_start = time.perf_counter()
    result = bench(args.url, path, pages, args.repeats, warm=not args.no_warm)
    result["server"] = {k: info[k] for k in (
        "model", "backend", "mlx_device", "metal_available", "max_pixels",
        "min_pixels", "pdf_dpi", "max_image_dim", "concurrency", "max_tokens",
        "model_load_time_s", "host_memory_mb",
    )}
    result["wall_clock_s"] = round(time.perf_counter() - t_start, 2)
    result["file"] = str(path)
    # Client-side RSS is meaningless here; ask the server what it uses.
    try:
        result["stats"]["server_rss_mb"] = get_json(args.url, "/info")["host_memory_mb"]
    except Exception:
        pass

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k, v in result["stats"].items():
        print(f"  {k:>22}: {v}")

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(f"cannot reach server: {exc}", file=sys.stderr)
        raise SystemExit(1)
