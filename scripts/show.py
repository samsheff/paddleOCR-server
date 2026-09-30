"""Pretty-print an /ocr response. Used by test.sh and handy on its own.

    curl -s -X POST localhost:8088/ocr -F file=@page.png | python scripts/show.py
    curl -s localhost:8088/health | python scripts/show.py --health
"""

from __future__ import annotations

import argparse
import json
import sys


def show_ocr(d: dict) -> None:
    if "text" not in d:
        print(json.dumps(d, indent=2))
        return
    print(f"  model              : {d.get('model')}")
    print(f"  processing_time_ms : {d.get('processing_time_ms')}")
    print(f"  page               : {d.get('page')}")
    print(
        f"  image              : {d.get('image_width')}x{d.get('image_height')}"
        f"  (source {d.get('source_width')}x{d.get('source_height')}"
        f", dpi {d.get('pdf_dpi')})"
    )
    print(f"  preprocess/infer   : {d.get('preprocess_time_ms')}ms / {d.get('inference_time_ms')}ms")
    print(
        f"  tokens             : {d.get('generation_tokens')} out"
        f" (+{d.get('prompt_tokens')} in)"
    )
    print(
        f"  tok/sec            : {d.get('tokens_per_second')} effective,"
        f" {d.get('generation_tps')} decode"
    )
    print(f"  peak_gpu_memory_gb : {d.get('peak_gpu_memory_gb')}")
    if d.get("loop_guard_triggered"):
        print("  loop_guard_triggered: True (output was repeating; truncated)")
    print(f"  finish_reason      : {d.get('finish_reason')}")
    print("  --- text ---")
    print(d.get("text", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health", action="store_true", help="format /health or /info")
    args = ap.parse_args()

    raw = sys.stdin.read().strip()
    if not raw:
        print("no input on stdin", file=sys.stderr)
        return 1
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"not JSON: {exc}\n{raw}", file=sys.stderr)
        return 1

    if args.health or "status" in d or "mlx_device" in d:
        if "mlx_device" in d:
            for key in (
                "model", "backend", "mlx_device", "metal_available", "gpu_available",
                "model_loaded", "model_load_time_s", "max_pixels", "min_pixels",
                "max_image_dim", "pdf_dpi", "max_tokens", "concurrency",
                "loop_guard", "requests_served", "host_memory_mb",
            ):
                if key in d:
                    print(f"  {key:>20}: {d[key]}")
        else:
            print(json.dumps(d, indent=2))
        return 0

    show_ocr(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
