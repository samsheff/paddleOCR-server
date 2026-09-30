"""Tests that need no model: preprocessing, sanitising, loop guard, config."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.ocr import CycleStoppingCriteria, sanitize_ocr_text  # noqa: E402
from app.preprocessing import (  # noqa: E402
    is_pdf,
    pdf_page_count,
    prepare_image,
    render_pdf_page,
)

SAMPLES = ROOT / "samples"


def _png_bytes(w: int = 800, h: int = 600, text: str = "hello") -> bytes:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
    except OSError:
        f = ImageFont.load_default()
    d.text((40, 40), text, fill="black", font=f)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- config
def test_defaults_are_safe():
    s = Settings()
    assert s.host == "127.0.0.1", "must not listen on the LAN by default"
    assert s.concurrency == 1
    assert s.max_pixels % 784 == 0, "pixel budget must align to the 28x28 patch grid"


def test_max_pixels_is_rounded_to_patch_grid():
    s = Settings(max_pixels=1_000_001)
    assert s.max_pixels % 784 == 0
    assert s.max_pixels < 1_000_001


@pytest.mark.parametrize(
    "kwargs",
    [{"max_pixels": 0}, {"concurrency": 0}, {"min_pixels": -1}],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(Exception):
        Settings(**kwargs)


# ------------------------------------------------------------ sanitising
def test_sanitize_strips_control_tokens():
    assert sanitize_ocr_text("a<|LOC_94|>b<|IMAGE_PLACEHOLDER|>c") == "abc"


def test_sanitize_clamps_blank_runs_but_keeps_paragraphs():
    assert sanitize_ocr_text("a\n\n\n\n\nb") == "a\n\nb"
    assert sanitize_ocr_text("a\n\nb") == "a\n\nb"


def test_sanitize_normalises_newlines_and_edges():
    assert sanitize_ocr_text("\r\nline one\r\nline two\r\n") == "line one\nline two"


def test_sanitize_preserves_visible_characters():
    # Markdown pipes, asterisks and list markers are content, not markup.
    raw = "| a | b |\n- one\n- two"
    assert sanitize_ocr_text(raw) == raw


def test_sanitize_empty():
    assert sanitize_ocr_text("") == ""


# ------------------------------------------------------------- loop guard
def test_loop_guard_fires_on_exact_repetition():
    """A page emitted twice must be stopped at the point of the restart."""
    crit = CycleStoppingCriteria([2])  # default min_tokens=200, as in production
    first = list(range(100, 500))
    crit.reset()
    for t in first:
        assert not crit(t), "must not fire during the first pass"
    for t in first:
        if crit(t):
            break
    assert crit.cycles_detected == 1
    assert crit.restart_index == len(first)


def test_loop_guard_does_not_fire_on_short_output():
    """Below min_tokens nothing is checked, so short pages always finish.

    This matters for the real model: a one-line page can legitimately echo a
    short phrase twice, and aborting it would lose the whole transcription.
    """
    crit = CycleStoppingCriteria([2], min_tokens=200)
    crit.reset()
    seq = list(range(10, 90)) * 2  # 160 tokens, under the threshold
    for t in seq:
        assert not crit(t)
    assert crit.cycles_detected == 0
    assert crit.restart_index is None
    assert len(seq) < crit.min_tokens


def test_loop_guard_fires_on_the_real_looped_page():
    """Regression: token stream captured from a page that looped at runtime.

    The model emitted the page, restarted from token 0, and reproduced the
    first 64 tokens exactly before diverging (the repeated pass dropped a
    paragraph). Without the guard this page cost 521 tokens and returned the
    content three times over.
    """
    fixture = Path(__file__).parent / "data" / "looped_page_tokens.json"
    tokens = json.loads(fixture.read_text())
    crit = CycleStoppingCriteria([2])
    crit.reset()
    for t in tokens:
        if crit(t):
            break
    assert crit.cycles_detected == 1
    assert crit.restart_index == 279


def test_loop_guard_keeps_eos_behaviour():
    crit = CycleStoppingCriteria([2])
    crit.reset()
    for t in [10, 11, 12]:
        assert crit(t) is False
    assert crit(2) is True, "EOS must still terminate generation"


def test_loop_guard_ignores_normal_pages():
    crit = CycleStoppingCriteria([2], min_tokens=200)
    crit.reset()
    # 600 unique-ish tokens: no restart, should run to EOS.
    for t in range(1000, 1600):
        if crit(t):
            pytest.fail(f"guard fired spuriously at {t}")
    assert crit.cycles_detected == 0
    assert crit.restart_index is None


def test_loop_guard_tolerates_boilerplate_repetition():
    """Repeated headers/tables are content, not a restart-from-zero loop.

    The guard only fires when the *start of the page* is reproduced, so a page
    whose opening lines reappear later (multi-page headers, repeated row
    labels) must decode to completion.
    """
    crit = CycleStoppingCriteria([2], min_tokens=64)
    crit.reset()
    opening = list(range(100, 140))       # unique page opening
    header = list(range(2000, 2020))      # a table header, recurs later
    body = list(range(300, 900))
    seq = opening + body + header + body + header + body
    for t in seq:
        if crit(t):
            pytest.fail(
                f"guard fired on legitimately repetitive content at {crit.restart_index}"
            )
    assert crit.cycles_detected == 0
    assert crit.restart_index is None


# ----------------------------------------------------------- preprocessing
def test_is_pdf_detects_pdf():
    assert is_pdf(b"%PDF-1.7\nrest") is True
    assert is_pdf(b"\x89PNG\r\n\x1a\n") is False


def test_prepare_image_respects_pixel_budget():
    data = _png_bytes(2000, 1500)
    page = prepare_image(data, max_pixels=500_000, max_dim=4096)
    assert page.width * page.height <= 500_000
    assert (page.source_width, page.source_height) == (2000, 1500)
    assert page.width < page.source_width


def test_prepare_image_respects_max_dim():
    data = _png_bytes(3000, 1200)
    page = prepare_image(data, max_pixels=0, max_dim=1500)
    assert max(page.width, page.height) <= 1500


def test_prepare_image_does_not_upscale():
    data = _png_bytes(100, 80)
    page = prepare_image(data, max_pixels=10_000_000, max_dim=4096)
    assert (page.width, page.height) == (100, 80)


def test_prepare_image_converts_to_rgb():
    img = Image.new("L", (300, 200), 128)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    page = prepare_image(buf.getvalue(), max_pixels=10_000_000, max_dim=0)
    assert page.image.mode == "RGB"


@pytest.mark.skipif(not (SAMPLES / "sample.pdf").exists(), reason="sample.pdf not built")
def test_pdf_page_count():
    assert pdf_page_count((SAMPLES / "sample.pdf").read_bytes()) == 3


@pytest.mark.skipif(not (SAMPLES / "sample.pdf").exists(), reason="sample.pdf not built")
def test_render_pdf_page_shapes_and_dpi():
    data = (SAMPLES / "sample.pdf").read_bytes()
    page = render_pdf_page(data, 0, dpi=200, max_pixels=1_003_520, max_dim=4096)
    assert page.page_number == 1
    assert page.pdf_dpi == 200
    assert page.width * page.height <= 1_003_520
    assert page.image.mode == "RGB"
    assert page.source_width > page.width, "oversized render should be downscaled"


@pytest.mark.skipif(not (SAMPLES / "sample.pdf").exists(), reason="sample.pdf not built")
def test_render_pdf_page_out_of_range():
    data = (SAMPLES / "sample.pdf").read_bytes()
    with pytest.raises(IndexError):
        render_pdf_page(data, 99, dpi=150, max_pixels=1_003_520, max_dim=4096)


@pytest.mark.skipif(not (SAMPLES / "sample.pdf").exists(), reason="sample.pdf not built")
def test_render_pdf_page_is_deterministic():
    data = (SAMPLES / "sample.pdf").read_bytes()
    a = render_pdf_page(data, 1, dpi=150, max_pixels=1_003_520, max_dim=4096)
    b = render_pdf_page(data, 1, dpi=150, max_pixels=1_003_520, max_dim=4096)
    assert a.image.tobytes() == b.image.tobytes()
