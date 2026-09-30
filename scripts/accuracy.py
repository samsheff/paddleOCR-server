"""Character-level accuracy of the OCR service against known ground truth.

The bundled sample pages are synthetic, so we know exactly what text they
contain. This compares the server output to that reference so resolution
sweeps can be judged on accuracy, not just latency.

    python scripts/accuracy.py --url http://127.0.0.1:8089
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark import post_ocr  # noqa: E402

# Ground truth for the pages produced by scripts/make_sample.py and
# scripts/make_pdf_sample.py, in reading order.

PAGE1_LINES = [
    "Quarterly Operations Report",
    "Prepared for the Board of Directors - March 2026",
    "1. Summary",
    "During the quarter the platform processed 4,182,905 documents at a median",
    "latency of 318 milliseconds. Throughput improved by 12.4% quarter over",
    "quarter, driven primarily by the new batching scheduler and a 3x increase",
    "in the GPU worker pool. No customer-facing incidents were recorded.",
    "2. Throughput by Region",
    "Region Documents Median ms Error rate",
    "us-east-1 1,905,442 291 0.012%",
    "us-west-2 1,204,338 322 0.009%",
    "eu-central-1 712,905 355 0.014%",
    "ap-southeast-1 360,220 401 0.021%",
    "3. Notes",
    "- Storage costs fell 8.1% after the compaction pass scheduled on 2026-02-14.",
    "- Two upstream dependencies were upgraded without a rollback window.",
    "- Remaining work: shard the audit log table and retire the legacy queue.",
]

PAGE2_LINES = [
    "Appendix A: Methodology Notes",
    "Page images were rasterised at the configured DPI and resized so that the",
    "vision tower received no more than the configured pixel budget. Text was",
    "then transcribed with PaddleOCR-VL-1.5 in MLX 4-bit on Apple Silicon.",
    "Latency was measured end to end, including PDF rendering, image",
    "preprocessing, prefill and token-by-token decoding. No cloud services were",
    "involved at any stage; all inference ran locally on the GPU.",
    "Accuracy notes:",
    "1. Small print below roughly 10 px in the source image is the most",
    "common source of recognition errors.",
    "2. Rotated or heavily skewed scans benefit from a higher render DPI.",
    "3. Tables are emitted as aligned plain text or Markdown depending on",
    "the resolution the model receives.",
    "The service keeps the model resident between requests, so the second and",
    "later calls avoid both disk I/O and weight materialisation. Throughput is",
    "therefore dominated by prefill cost, which scales with image area.",
    "A.1 Measured Latency",
    "Stage ms",
    "PDF render @200dpi 58",
    "Image preprocess 12",
    "Prefill 640",
    "Decode 480",
    "Total 1190",
]

PAGE3_LINES = [
    "3. Acknowledgements",
    "No external inference services were used.",
]

GROUND_TRUTH = {1: PAGE1_LINES, 2: PAGE2_LINES, 3: PAGE3_LINES}

_WS = re.compile(r"\s+")
# Model output occasionally uses markdown table pipes or list markers in place
# of plain spacing; normalise those before scoring.
_PUNCT_MAP = str.maketrans({"`": "", "*": "", "|": " ", "_": ""})


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT_MAP)
    text = text.replace("—", "-").replace("–", "-")
    text = _WS.sub(" ", text)
    return text.strip().lower()


def score(reference_lines: list[str], hypothesis: str) -> dict:
    ref = normalise(" ".join(reference_lines))
    hyp = normalise(hypothesis)
    ratio = difflib.SequenceMatcher(None, ref, hyp).ratio()

    ref_words = ref.split()
    hyp_words = hyp.split()
    matched = sum(
        1 for w in hyp_words if w in ref_words
    )
    word_precision = matched / len(hyp_words) if hyp_words else 0.0
    missing = [w for w in ref_words if w not in hyp_words]

    return {
        "char_similarity": round(ratio, 4),
        "word_recall": round(1 - len(missing) / len(ref_words), 4) if ref_words else 0.0,
        "word_precision": round(word_precision, 4),
        "missing_words": missing[:20],
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    ap.add_argument("--file", default=str(ROOT / "samples" / "sample.pdf"))
    ap.add_argument("--pages", default="1-3")
    ap.add_argument("--dump-text", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    if "-" in args.pages:
        a, b = args.pages.split("-", 1)
        pages = list(range(int(a), int(b) + 1))
    else:
        pages = [int(x) for x in args.pages.split(",")]

    print(f"accuracy check against {path.name} via {args.url}")
    results = []
    for page in pages:
        ref = GROUND_TRUTH.get(page)
        if ref is None:
            print(f"  page {page}: no ground truth, skipping")
            continue
        d = post_ocr(args.url, path, page)
        s = score(ref, d["text"])
        s["page"] = page
        s["ms"] = d["processing_time_ms"]
        s["image"] = f"{d['image_width']}x{d['image_height']}"
        results.append(s)
        print(
            f"  page {page}: char_sim={s['char_similarity']:.4f} "
            f"word_recall={s['word_recall']:.4f} "
            f"word_precision={s['word_precision']:.4f} "
            f"({s['ms']} ms, {s['image']})"
        )
        if s["missing_words"]:
            print(f"      missing: {s['missing_words'][:8]}")
        if args.dump_text:
            print("      ---")
            for line in d["text"].splitlines():
                print("      " + line)

    if results:
        mean_sim = sum(r["char_similarity"] for r in results) / len(results)
        mean_recall = sum(r["word_recall"] for r in results) / len(results)
        mean_ms = sum(r["ms"] for r in results) / len(results)
        print(f"\n  mean char_similarity : {mean_sim:.4f}")
        print(f"  mean word_recall     : {mean_recall:.4f}")
        print(f"  mean latency         : {mean_ms:.0f} ms")
        print(json.dumps({"summary": {
            "char_similarity": round(mean_sim, 4),
            "word_recall": round(mean_recall, 4),
            "mean_ms": round(mean_ms, 1),
        }}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
