"""FastAPI HTTP server exposing PaddleOCR-VL (MLX) over a small JSON API.

Endpoints
    GET  /health   liveness + model readiness
    GET  /info     engine/runtime/configuration details
    POST /ocr      OCR one image or one PDF page (multipart)
    POST /ocr/json OCR from base64 or a PDF byte slice (JSON body)

The model is loaded during application startup and stays resident; there is no
per-request loading anywhere in this file.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .ocr import OCREngine
from .preprocessing import is_pdf, prepare_image, render_pdf_page

log = logging.getLogger("ocr.server")

DESCRIPTION = """
Local OCR service built on PaddleOCR-VL running on the Apple GPU via MLX.

* `POST /ocr` accepts a PNG/JPEG image or a PDF (plus an optional `page`).
* `POST /ocr/json` accepts base64 payloads for non-multipart clients.
* Output is a faithful plain-text transcription: no summaries, no rewriting.
""".strip()

TAGS = [
    {"name": "health", "description": "Liveness and model readiness"},
    {"name": "ocr", "description": "OCR images and PDF pages"},
]


class Base64OCRRequest(BaseModel):
    """JSON body alternative to multipart, for clients that prefer base64."""

    image_base64: str | None = Field(
        None, description="Base64-encoded image (PNG/JPEG). Optional data: URL prefix."
    )
    pdf_base64: str | None = Field(
        None, description="Base64-encoded PDF. Optional data: URL prefix."
    )
    page: int = Field(1, ge=1, description="1-based PDF page to OCR")
    max_tokens: int | None = Field(
        None, ge=1, description="Override the default generation limit"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    engine: OCREngine = app.state.engine

    t0 = time.perf_counter()
    await asyncio.to_thread(engine.load)
    if settings.warmup:
        await asyncio.to_thread(engine.warmup)
    log.info(
        "engine ready in %.2fs (pid=%s)", time.perf_counter() - t0, __import__("os").getpid()
    )
    try:
        yield
    finally:
        log.info("shutting down")


def _configure_logging(settings: Settings) -> None:
    """Send the app's own logs to stderr at the configured level.

    Without this, the engine's load/warmup/loop-guard messages would be
    swallowed by uvicorn's logging configuration.
    """
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # Third-party chatter we never need: hub polls, http connection pools,
    # deprecation warnings from model loading.
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if settings.debug else logging.WARNING
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings)

    app = FastAPI(
        title="PaddleOCR-VL (MLX) OCR Server",
        description=DESCRIPTION,
        version="1.0.0",
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = OCREngine(settings)
    app.state.started_at = time.time()

    def engine_dep() -> OCREngine:
        return app.state.engine

    # ------------------------------------------------------------------
    @app.get("/health", tags=["health"])
    async def health(engine: OCREngine = Depends(engine_dep)) -> dict[str, Any]:
        return {
            "status": "ok" if engine.model is not None else "loading",
            "model_loaded": engine.model is not None,
            "model": settings.model,
            "uptime_s": round(time.time() - app.state.started_at, 1),
            "requests_served": engine.requests_served,
        }

    @app.get("/info", tags=["health"])
    async def info(engine: OCREngine = Depends(engine_dep)) -> dict[str, Any]:
        payload = engine.info()
        payload["api"] = {
            "version": app.version,
            "endpoints": ["/health", "/info", "/ocr", "/ocr/json", "/docs"],
            "input_formats": ["image/png", "image/jpeg", "image/tiff", "application/pdf"],
            "concurrency": settings.concurrency,
        }
        payload["config"] = settings.public_dict()
        return payload

    # ------------------------------------------------------------------
    @app.post("/ocr", tags=["ocr"], response_model=None)
    async def ocr_multipart(
        file: UploadFile = File(..., description="Image file or PDF document"),
        page: int = Query(
            1, ge=1, description="1-based PDF page to OCR (ignored for images)"
        ),
        max_tokens: int | None = Query(
            None, ge=1, description="Override the default generation limit"
        ),
        engine: OCREngine = Depends(engine_dep),
    ) -> dict[str, Any]:
        if engine.model is None:
            raise HTTPException(status_code=503, detail="model not loaded yet")

        data = await _read_upload(file, settings.max_upload_bytes)

        t0 = time.perf_counter()
        try:
            if is_pdf(data):
                page_img = await _prep(
                    render_pdf_page,
                    data,
                    page - 1,
                    dpi=settings.pdf_dpi,
                    max_pixels=settings.max_pixels,
                    max_dim=settings.max_image_dim,
                )
            else:
                page_img = await _prep(
                    prepare_image,
                    data,
                    max_pixels=settings.max_pixels,
                    max_dim=settings.max_image_dim,
                )
        except IndexError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"could not decode input: {exc}"
            ) from exc

        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        return await _run(engine, page_img, preprocess_ms, max_tokens)

    # ------------------------------------------------------------------
    @app.post("/ocr/json", tags=["ocr"], response_model=None)
    async def ocr_json(
        payload: Base64OCRRequest, engine: OCREngine = Depends(engine_dep)
    ) -> dict[str, Any]:
        if engine.model is None:
            raise HTTPException(status_code=503, detail="model not loaded yet")
        if not payload.image_base64 and not payload.pdf_base64:
            raise HTTPException(
                status_code=422, detail="provide image_base64 or pdf_base64"
            )

        raw = payload.pdf_base64 or payload.image_base64 or ""
        try:
            data = _decode_base64(raw)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="payload too large")

        is_pdf_input = payload.pdf_base64 is not None or is_pdf(data)
        t0 = time.perf_counter()
        try:
            if is_pdf_input:
                page_img = await _prep(
                    render_pdf_page,
                    data,
                    payload.page - 1,
                    dpi=settings.pdf_dpi,
                    max_pixels=settings.max_pixels,
                    max_dim=settings.max_image_dim,
                )
            else:
                page_img = await _prep(
                    prepare_image,
                    data,
                    max_pixels=settings.max_pixels,
                    max_dim=settings.max_image_dim,
                )
        except IndexError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"could not decode input: {exc}"
            ) from exc

        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        return await _run(engine, page_img, preprocess_ms, payload.max_tokens)

    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled(_request, exc: Exception) -> JSONResponse:  # pragma: no cover
        # Never leak a stack trace or internals to the client; the detail goes
        # to the log instead.
        log.exception("unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500, content={"detail": "internal server error"}
        )

    return app


async def _run(engine: OCREngine, page_img, preprocess_ms: float, max_tokens):
    # Inference goes to the engine's dedicated single-thread executor, not the
    # shared pool: MLX keeps its Metal streams thread-local, so pinning every
    # request to one thread avoids re-creating streams (measured ~2x decode
    # penalty on the first request after a gap otherwise).
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(engine.executor, engine.ocr, page_img, max_tokens)
    body = {
        "text": result.text,
        "processing_time_ms": result.processing_time_ms,
        "model": result.model,
        "image_width": result.image_width,
        "image_height": result.image_height,
        # --- diagnostics, harmless for clients that ignore them ---
        "prompt_tokens": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        # Decode-only rate reported by mlx-vlm.
        "generation_tps": round(result.generation_tps, 2),
        # Output tokens / full inference wall time (includes prefill + vision).
        "tokens_per_second": round(result.effective_tps, 2),
        "effective_tps": round(result.effective_tps, 2),
        "prompt_tokens_per_second": round(result.prompt_tps, 2),
        "inference_time_ms": round(result.processing_time_ms - preprocess_ms, 1),
        "preprocess_time_ms": round(preprocess_ms, 1),
        "peak_gpu_memory_gb": round(result.peak_memory_gb, 3),
        "finish_reason": result.finish_reason,
        "loop_guard_triggered": result.extra.get("loop_guard_triggered", False),
        "page": result.page,
        "source_width": result.extra.get("source_width"),
        "source_height": result.extra.get("source_height"),
        "pdf_dpi": result.extra.get("pdf_dpi"),
    }
    return body


async def _prep(fn, /, *args, **kwargs):
    """Run blocking CPU work (image decode, PDF raster) off the event loop.

    These paths hold no MLX state, so the shared thread pool is fine; only
    model inference needs the engine's pinned executor.
    """
    return await asyncio.to_thread(fn, *args, **kwargs)


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"upload exceeds {max_bytes // (1024 * 1024)} MB"
        )
    return data


def _decode_base64(value: str) -> bytes:
    cleaned = value.strip()
    if cleaned.startswith("data:"):
        _, _, cleaned = cleaned.partition(",")
    try:
        return base64.b64decode(cleaned, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 payload: {exc}") from exc


app = create_app()
