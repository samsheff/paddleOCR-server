"""End-to-end API tests against a running server.

These do not load the model; they exercise the HTTP layer with the engine
stubbed out, so they run in a second and need no GPU.

    pytest tests/test_api.py
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.ocr import OCREngine, OCRResult  # noqa: E402
from app.preprocessing import prepare_image  # noqa: E402
from app.server import create_app  # noqa: E402

SAMPLES = ROOT / "samples"


class StubEngine(OCREngine):
    """Pretends to be a loaded model; returns canned text."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.model = object()
        self.requests: list[int] = []

    def load(self) -> float:
        self.load_time_s = 0.01
        return 0.01

    def warmup(self) -> None:
        return None

    def ocr(self, page, max_tokens=None):  # type: ignore[override]
        self.requests.append(page.page_number or 0)
        return OCRResult(
            text=f"stub text {page.width}x{page.height}",
            processing_time_ms=7,
            model=self.settings.model,
            image_width=page.width,
            image_height=page.height,
            prompt_tokens=100,
            generation_tokens=10,
            generation_tps=100.0,
            effective_tps=90.0,
            prompt_tps=500.0,
            peak_memory_gb=1.0,
            finish_reason="stop",
            page=page.page_number,
            extra={"source_width": page.source_width, "source_height": page.source_height,
                   "pdf_dpi": page.pdf_dpi, "sanitized_chars": 0},
        )


def _png_bytes(w=600, h=400) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    app = create_app(Settings())
    app.state.engine = StubEngine(app.state.settings)
    # The stub needs to be present before lifespan runs.
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "uptime_s" in body


def test_info_reports_gpu(client):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    for key in ("model", "backend", "mlx_device", "max_pixels", "pdf_dpi", "config"):
        assert key in body
    assert body["backend"] == "mlx"


def test_ocr_image(client):
    r = client.post("/ocr", files={"file": ("page.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    # Contract required by clients.
    for key in ("text", "processing_time_ms", "model", "image_width", "image_height"):
        assert key in body, f"missing documented field: {key}"
    assert isinstance(body["text"], str)
    assert isinstance(body["processing_time_ms"], int)
    assert body["image_width"] == 600
    assert body["image_height"] == 400
    assert body["page"] is None


def test_ocr_rejects_empty_upload(client):
    r = client.post("/ocr", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_ocr_rejects_undecodable_input(client):
    r = client.post("/ocr", files={"file": ("bad.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_ocr_rejects_oversized_upload(client):
    big = io.BytesIO()
    Image.new("RGB", (4000, 4000), "white").save(big, "PNG", compress_level=0)
    data = big.getvalue()
    app = client.app
    # max_upload_bytes derives from max_upload_mb, so lower that.
    app.state.settings.max_upload_mb = 0
    try:
        r = client.post("/ocr", files={"file": ("big.png", data, "image/png")})
        assert r.status_code == 413
    finally:
        app.state.settings.max_upload_mb = 32


@pytest.mark.skipif(not (SAMPLES / "sample.pdf").exists(), reason="sample.pdf not built")
def test_ocr_pdf_page(client):
    data = (SAMPLES / "sample.pdf").read_bytes()
    r = client.post(
        "/ocr?page=2", files={"file": ("doc.pdf", data, "application/pdf")}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["page"] == 2
    assert body["pdf_dpi"] == 200
    assert body["source_width"] > 0


@pytest.mark.skipif(not (SAMPLES / "sample.pdf").exists(), reason="sample.pdf not built")
def test_ocr_pdf_page_out_of_range(client):
    data = (SAMPLES / "sample.pdf").read_bytes()
    r = client.post("/ocr?page=99", files={"file": ("doc.pdf", data, "application/pdf")})
    assert r.status_code == 400
    assert "out of range" in r.json()["detail"]


@pytest.mark.skipif(not (SAMPLES / "sample.pdf").exists(), reason="sample.pdf not built")
def test_ocr_pdf_page_zero_rejected(client):
    data = (SAMPLES / "sample.pdf").read_bytes()
    r = client.post("/ocr?page=0", files={"file": ("doc.pdf", data, "application/pdf")})
    assert r.status_code == 422  # FastAPI query validation


def test_ocr_json_base64_image(client):
    payload = {"image_base64": base64.b64encode(_png_bytes()).decode()}
    r = client.post("/ocr/json", json=payload)
    assert r.status_code == 200
    assert r.json()["image_width"] == 600


def test_ocr_json_data_uri(client):
    uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    r = client.post("/ocr/json", json={"image_base64": uri})
    assert r.status_code == 200


@pytest.mark.skipif(not (SAMPLES / "sample.pdf").exists(), reason="sample.pdf not built")
def test_ocr_json_base64_pdf(client):
    data = base64.b64encode((SAMPLES / "sample.pdf").read_bytes()).decode()
    r = client.post("/ocr/json", json={"pdf_base64": data, "page": 3})
    assert r.status_code == 200
    assert r.json()["page"] == 3


def test_ocr_json_requires_a_payload(client):
    r = client.post("/ocr/json", json={})
    assert r.status_code == 422


def test_ocr_json_bad_base64(client):
    r = client.post("/ocr/json", json={"image_base64": "!!!not base64!!!"})
    assert r.status_code in (400, 422)


def test_max_tokens_override(client):
    r = client.post(
        "/ocr?max_tokens=64", files={"file": ("p.png", _png_bytes(), "image/png")}
    )
    assert r.status_code == 200


def test_openapi_documents_the_contract(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert "/ocr" in spec["paths"]
    assert "/health" in spec["paths"]
    assert "/info" in spec["paths"]


def test_model_not_loaded_returns_503():
    app = create_app(Settings())
    app.state.engine = StubEngine(app.state.settings)
    app.state.engine.model = None
    with TestClient(app) as c:
        r = c.post("/ocr", files={"file": ("p.png", _png_bytes(), "image/png")})
        assert r.status_code == 503
        assert c.get("/health").json()["status"] == "loading"
