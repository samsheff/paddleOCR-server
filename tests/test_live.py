"""Live end-to-end checks against a running server (requires GPU + model).

Skipped automatically when no server is reachable, so `pytest` stays usable
offline. Run ./start.sh first, or pass --url.

    python tests/test_live.py
    python tests/test_live.py --url http://127.0.0.1:8088
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
import urllib.parse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

REQUIRED_FIELDS = (
    "text",
    "processing_time_ms",
    "model",
    "image_width",
    "image_height",
)


def post_ocr(url: str, data: bytes, filename: str, mime: str, page: int = 1,
             timeout: float = 300.0) -> dict:
    boundary = f"----ocrlive{uuid.uuid4().hex}"
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    body += data
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{url}/ocr?page={page}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get(url: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(f"{url}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def png_bytes(w: int, h: int, lines: list[str]) -> bytes:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
        fb = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 36)
    except OSError:
        f = fb = ImageFont.load_default()
    d.text((60, 60), lines[0], fill="black", font=fb)
    y = 130
    for line in lines[1:]:
        d.text((60, y), line, fill="black", font=f)
        y += 38
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class Live:
    def __init__(self, url: str) -> None:
        self.url = url
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed.append(f"{name}: {detail}")

    def run(self) -> int:
        print(f"live checks against {self.url}\n")

        # --- health / info -------------------------------------------------
        h = get(self.url, "/health")
        self.check("GET /health returns ok", h.get("status") == "ok", str(h)[:120])

        info = get(self.url, "/info")
        self.check(
            "hardware acceleration active",
            bool(info.get("gpu_available")) and bool(info.get("metal_available")),
            f"device={info.get('mlx_device')} metal={info.get('metal_available')}",
        )
        self.check("model loaded once at startup", info.get("model_loaded") is True)
        self.check(
            "backend is mlx", info.get("backend") == "mlx", str(info.get("backend"))
        )

        # --- image OCR -----------------------------------------------------
        blob = png_bytes(
            900, 400,
            ["Live Test Document",
             "The quick brown fox jumps over the lazy dog.",
             "Reference number 88213-A appears here."],
        )
        d = post_ocr(self.url, blob, "live.png", "image/png")
        missing = [f for f in REQUIRED_FIELDS if f not in d]
        self.check("POST /ocr response contract", not missing, f"missing={missing}")
        text = d.get("text", "")
        self.check("transcribes headline", "Live Test Document" in text, text[:80])
        self.check(
            "transcribes reference number", "88213" in text.replace(" ", ""), text[:120]
        )
        self.check("preserves line structure", text.count("\n") >= 2, repr(text[:120]))
        self.check(
            "no hallucinated summary",
            "summary" not in text.lower() and "here is" not in text.lower(),
            text[:80],
        )
        self.check(
            "reports image size",
            d.get("image_width", 0) > 0 and d.get("image_height", 0) > 0,
            f"{d.get('image_width')}x{d.get('image_height')}",
        )

        # --- jpeg ----------------------------------------------------------
        jbuf = io.BytesIO()
        Image.open(io.BytesIO(blob)).convert("RGB").save(jbuf, "JPEG", quality=92)
        d = post_ocr(self.url, jbuf.getvalue(), "live.jpg", "image/jpeg")
        self.check("accepts JPEG", "Live Test Document" in d.get("text", ""))

        # --- base64 JSON ---------------------------------------------------
        req = urllib.request.Request(
            f"{self.url}/ocr/json",
            data=json.dumps(
                {"image_base64": base64.b64encode(blob).decode()}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            d = json.loads(resp.read().decode())
        self.check("POST /ocr/json base64", "Live Test Document" in d.get("text", ""))

        # --- PDF -----------------------------------------------------------
        pdf = SAMPLES / "sample.pdf"
        if pdf.exists():
            data = pdf.read_bytes()
            d = post_ocr(self.url, data, "doc.pdf", "application/pdf", page=1)
            self.check("PDF page 1 OCR", "Quarterly Operations Report" in d.get("text", ""),
                       d.get("text", "")[:80])
            self.check("PDF reports page number", d.get("page") == 1)
            self.check("PDF reports render dpi", d.get("pdf_dpi") == 200)
            d2 = post_ocr(self.url, data, "doc.pdf", "application/pdf", page=3)
            self.check("PDF page 3 OCR", "Acknowledgements" in d2.get("text", ""),
                       d2.get("text", "")[:80])
            self.check(
                "no duplicated output",
                d2.get("text", "").count("Acknowledgements") <= 1,
            )

        # --- model residency: second request must not re-load ---------------
        before = get(self.url, "/info")
        t0 = time.perf_counter()
        post_ocr(self.url, blob, "live.png", "image/png")
        first_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        post_ocr(self.url, blob, "live.png", "image/png")
        second_ms = (time.perf_counter() - t0) * 1000
        after = get(self.url, "/info")
        self.check(
            "model stays resident (no reload between requests)",
            after["requests_served"] == before["requests_served"] + 2,
            f"{before['requests_served']} -> {after['requests_served']}",
        )
        self.check(
            "steady-state latency (no cold start)",
            second_ms < max(first_ms, 500),
            f"first={first_ms:.0f}ms second={second_ms:.0f}ms",
        )

        # --- error handling -------------------------------------------------
        try:
            post_ocr(self.url, b"definitely not an image", "bad.png", "image/png")
            self.check("rejects garbage input", False, "no error raised")
        except urllib.error.HTTPError as e:
            self.check("rejects garbage input", e.code == 400, f"HTTP {e.code}")

        print(f"\n{self.passed} passed, {len(self.failed)} failed")
        for f in self.failed:
            print(f"  FAILED {f}")
        return 1 if self.failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    try:
        get(url, "/health", timeout=5)
    except Exception as exc:
        print(f"no server at {url}: {exc}")
        print("start it with ./start.sh")
        return 2
    return Live(url).run()


if __name__ == "__main__":
    sys.exit(main())
