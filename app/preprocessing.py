"""Image and PDF preprocessing for the OCR server.

Responsibilities:
  * decode PNG/JPEG/TIFF/... bytes into an RGB PIL image
  * render individual PDF pages at a configurable DPI (pypdfium2)
  * downscale oversized inputs so the vision tower never sees more pixels
    than its ``max_pixels`` budget

The model's own image processor performs the final patch-aligned resize
(``smart_resize``), so we deliberately do not duplicate that here: we only
guard the hard pixel/edge ceilings and keep the cheap path fast.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageFile, ImageOps

# Be forgiving about slightly truncated uploads rather than hard-failing.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Anything Pillow can open. Keeps GIF/WebP out of the hot path ambiguity but
# still handles what users realistically send.
IMAGE_FORMATS = ("PNG", "JPEG", "TIFF", "BMP", "WEBP")

PDF_MAGIC = b"%PDF-"


@dataclass
class Page:
    """A single decoded page ready for the model."""

    image: Image.Image
    width: int
    height: int
    source_width: int
    source_height: int
    page_number: int | None = None
    pdf_dpi: int | None = None


def is_pdf(data: bytes) -> bool:
    return data[:1024].lstrip()[:5] == PDF_MAGIC or data[:5] == PDF_MAGIC


def _open_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    # Honour EXIF rotation so phone photos are not fed sideways.
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _cap_dimensions(img: Image.Image, max_pixels: int, max_dim: int) -> Image.Image:
    """Downscale (never upscale) so the image fits the vision budget.

    Uses a single LANCZOS pass. The model's image processor will still do a
    second, patch-aligned resize afterwards, so we avoid overshooting the
    budget here -- exactly hitting it is the processor's job.
    """
    w, h = img.size
    scale = 1.0

    if max_pixels > 0 and w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5

    if max_dim > 0 and max(w, h) * scale > max_dim:
        scale = min(scale, max_dim / max(w, h))

    if scale >= 0.999:
        return img

    new_w = max(28, int(w * scale))
    new_h = max(28, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def prepare_image(
    data: bytes,
    *,
    max_pixels: int,
    max_dim: int,
) -> Page:
    """Decode an encoded image and apply resolution guards."""
    img = _open_image(data)
    src_w, src_h = img.size
    img = _cap_dimensions(img, max_pixels, max_dim)
    w, h = img.size
    return Page(
        image=img,
        width=w,
        height=h,
        source_width=src_w,
        source_height=src_h,
    )


def pdf_page_count(data: bytes) -> int:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(io.BytesIO(data))
    try:
        return len(doc)
    finally:
        doc.close()


def render_pdf_page(
    data: bytes,
    page_index: int,
    *,
    dpi: int,
    max_pixels: int,
    max_dim: int,
) -> Page:
    """Render one 0-based PDF page to RGB and apply resolution guards."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(io.BytesIO(data))
    try:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(
                f"page {page_index + 1} out of range (document has {len(doc)} pages)"
            )
        pdf_page = doc[page_index]
        try:
            bitmap = pdf_page.render(
                scale=dpi / 72.0,
                draw_annots=True,
                may_draw_forms=True,
                # RGB with white background: pdfium defaults to transparent,
                # which the model reads as a black page.
                fill_color=(255, 255, 255, 255),
                color_scheme=None,
            )
            img = bitmap.to_pil().convert("RGB")
        finally:
            # Free the page eagerly; long documents otherwise accumulate.
            try:
                pdf_page.close()
            except Exception:
                pass

        src_w, src_h = img.size
        img = _cap_dimensions(img, max_pixels, max_dim)
        w, h = img.size
        return Page(
            image=img,
            width=w,
            height=h,
            source_width=src_w,
            source_height=src_h,
            page_number=page_index + 1,
            pdf_dpi=dpi,
        )
    finally:
        doc.close()
