"""Generate a realistic synthetic document page for testing (no network)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "samples"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1240, 1754  # ~A4 at 150 DPI
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)


def font(size, bold=False):
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


f_title = font(44, bold=True)
f_h2 = font(28, bold=True)
f_body = font(22)
f_small = font(18)

y = 120
d.text((120, y), "Quarterly Operations Report", font=f_title, fill=(17, 17, 17))
y += 70
d.text((120, y), "Prepared for the Board of Directors - March 2026", font=f_small, fill=(90, 90, 90))
y += 60

d.line([(120, y), (W - 120, y)], fill=(30, 30, 30), width=3)
y += 40

d.text((120, y), "1. Summary", font=f_h2, fill=(17, 17, 17))
y += 48

body = [
    "During the quarter the platform processed 4,182,905 documents at a median",
    "latency of 318 milliseconds. Throughput improved by 12.4% quarter over",
    "quarter, driven primarily by the new batching scheduler and a 3x increase",
    "in the GPU worker pool. No customer-facing incidents were recorded.",
]
for line in body:
    d.text((120, y), line, font=f_body, fill=(20, 20, 20))
    y += 34
y += 24

d.text((120, y), "2. Throughput by Region", font=f_h2, fill=(17, 17, 17))
y += 48

rows = [
    ("Region", "Documents", "Median ms", "Error rate"),
    ("us-east-1", "1,905,442", "291", "0.012%"),
    ("us-west-2", "1,204,338", "322", "0.009%"),
    ("eu-central-1", "712,905", "355", "0.014%"),
    ("ap-southeast-1", "360,220", "401", "0.021%"),
]
col_x = [120, 460, 720, 980]
row_h = 40
for i, row in enumerate(rows):
    if i == 0:
        d.rectangle([120, y - 6, W - 120, y + row_h - 10], fill=(238, 238, 238))
    for cx, cell in zip(col_x, row):
        d.text((cx, y), cell, font=f_small if i else f_small, fill=(15, 15, 15))
    d.line([(120, y + row_h - 12), (W - 120, y + row_h - 12)], fill=(150, 150, 150), width=1)
    y += row_h
y += 30

d.text((120, y), "3. Notes", font=f_h2, fill=(17, 17, 17))
y += 48
for line in [
    "Storage costs fell 8.1% after the compaction pass scheduled on 2026-02-14.",
    "Two upstream dependencies were upgraded without a rollback window.",
    "Remaining work: shard the audit log table and retire the legacy queue.",
]:
    d.text((120, y), "- " + line, font=f_body, fill=(20, 20, 20))
    y += 34

img.save(OUT / "sample_page.png", dpi=(150, 150))
print("wrote", OUT / "sample_page.png", img.size)
