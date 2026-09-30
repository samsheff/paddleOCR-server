"""Generate a small multi-page sample PDF for benchmarking."""

from pathlib import Path

from PIL import Image

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)

page1 = Image.open(SAMPLES / "sample_page.png").convert("RGB")

# Page 2: a denser text page (smaller font) to exercise downscaling.
W, H = 1240, 1754
page2 = Image.new("RGB", (W, H), (252, 252, 250))
from PIL import ImageDraw, ImageFont  # noqa: E402

d = ImageDraw.Draw(page2)


def font(size, bold=False):
    p = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        return ImageFont.truetype(p, size)
    except OSError:
        return ImageFont.load_default()


y = 90
d.text((90, y), "Appendix A: Methodology Notes", font=font(32, True), fill=(10, 10, 10))
y += 60
lines = [
    "Page images were rasterised at the configured DPI and resized so that the",
    "vision tower received no more than the configured pixel budget. Text was",
    "then transcribed with PaddleOCR-VL-1.5 in MLX 4-bit on Apple Silicon.",
    "",
    "Latency was measured end to end, including PDF rendering, image",
    "preprocessing, prefill and token-by-token decoding. No cloud services were",
    "involved at any stage; all inference ran locally on the GPU.",
    "",
    "Accuracy notes:",
    "  1. Small print below roughly 10 px in the source image is the most",
    "     common source of recognition errors.",
    "  2. Rotated or heavily skewed scans benefit from a higher render DPI.",
    "  3. Tables are emitted as aligned plain text or Markdown depending on",
    "     the resolution the model receives.",
    "",
    "The service keeps the model resident between requests, so the second and",
    "later calls avoid both disk I/O and weight materialisation. Throughput is",
    "therefore dominated by prefill cost, which scales with image area.",
]
for ln in lines:
    d.text((90, y), ln, font=font(18), fill=(20, 20, 20))
    y += 26

y += 30
d.text((90, y), "A.1 Measured Latency", font=font(24, True), fill=(10, 10, 10))
y += 40
table = [
    ("Stage", "ms"),
    ("PDF render @200dpi", "58"),
    ("Image preprocess", "12"),
    ("Prefill", "640"),
    ("Decode", "480"),
    ("Total", "1190"),
]
for i, (a, b) in enumerate(table):
    if i == 0:
        d.rectangle([90, y - 4, 700, y + 30], fill=(235, 235, 235))
    d.text((100, y), a, font=font(18), fill=(15, 15, 15))
    d.text((560, y), b, font=font(18), fill=(15, 15, 15))
    d.line([(90, y + 26), (700, y + 26)], fill=(160, 160, 160), width=1)
    y += 34

# Page 3: near-blank page to measure overhead on sparse content.
page3 = Image.new("RGB", (W, H), (255, 255, 255))
d3 = ImageDraw.Draw(page3)
d3.text((120, 160), "3. Acknowledgements", font=font(30, True), fill=(10, 10, 10))
d3.text((120, 230), "No external inference services were used.", font=font(20), fill=(30, 30, 30))

# pypdfium2 v5 has no image-insertion helper, so Pillow writes the container.
# resolution=150 keeps the embedded images at native pixel size.
out = SAMPLES / "sample.pdf"
page1.save(out, "PDF", save_all=True, append_images=[page2, page3], resolution=150.0)
print("wrote", out, out.stat().st_size, "bytes")
