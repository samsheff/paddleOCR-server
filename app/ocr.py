"""PaddleOCR-VL inference engine backed by MLX (Apple Silicon GPU).

The model is loaded exactly once and kept resident for the process lifetime.
Requests are serialized through a lock (default ``concurrency=1``) because the
Apple GPU is a single shared resource for this workload and MLX's graph
execution is not re-entrant across concurrent streams here.

A note on the API: mlx-vlm's current entrypoints are
``mlx_vlm.load()`` for (model, processor), ``mlx_vlm.prompt_utils.
apply_chat_template()`` for the model-specific prompt wrapper, and
``mlx_vlm.generate.dispatch.generate()``, which returns a
``GenerationResult`` carrying token counts, tokens/sec and peak memory. That is
the API implemented here rather than the older ``model.generate`` style samples.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .preprocessing import Page

log = logging.getLogger("ocr.engine")

# Residual control / layout tokens the model can emit. They are markup, not
# transcribed content, so they are stripped. Visible characters are untouched.
_CONTROL_TOKEN_RE = re.compile(r"<\|[A-Za-z_0-9]+\|>")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


# Token-block sizes tried when checking whether decoding restarted from the
# top. Measured against a real looped page: output matched the first 64 tokens
# of the page exactly, then diverged (the repeated pass dropped a paragraph),
# so 64 is the widest window that is reliably stable. Smaller windows are the
# fallback for very short pages.
_GUARD_BLOCKS = (64, 48, 32)


class CycleStoppingCriteria:
    """Stop decoding when the model restarts the page and repeats itself.

    PaddleOCR-VL occasionally loops: it finishes a page, then silently begins
    the page again and keeps going until EOS or the token budget runs out.
    Observed at some resolution budgets -- one sample page was emitted 3x, 521
    tokens, ~2.7 s, and the caller received the content three times over.

    The signature is specific: output beginning again at token 0. So we check
    whether the most recent token block matches the block at the *start* of
    generation, rather than any repeated block, which keeps legitimately
    repetitive documents (repeated table headers, boilerplate) safe.

    This object also performs the model's normal EOS check, because mlx-vlm
    replaces ``tokenizer.stopping_criteria`` wholesale when one is supplied.
    """

    def __init__(
        self,
        eos_token_ids: list[int],
        *,
        min_tokens: int = 200,
    ) -> None:
        self.eos_token_ids = set(int(t) for t in (eos_token_ids or []))
        self.min_tokens = min_tokens
        self.tokens: list[int] = []
        self.cycles_detected = 0
        self.restart_index: int | None = None

    def reset(self) -> None:
        self.tokens.clear()
        self.cycles_detected = 0
        self.restart_index = None

    def _window_matches_start(self, size: int) -> bool:
        """True if the last `size` tokens reproduce the start of the page."""
        block = self.tokens[-size:]
        if block[0] != self.tokens[0]:
            return False
        # A handful of differing tokens is tolerable (whitespace and
        # detokenisation boundaries shift run slightly between passes); an
        # exact 64-token reproduction is otherwise the only realistic trigger.
        budget = max(1, size // 32)
        diff = 0
        for a, b in zip(self.tokens, block):
            if a != b:
                diff += 1
                if diff > budget:
                    return False
        return True

    def __call__(self, token) -> bool:
        t = int(token.item()) if hasattr(token, "item") else int(token)
        self.tokens.append(t)

        if t in self.eos_token_ids:
            return True

        n = len(self.tokens)
        if n < self.min_tokens:
            return False

        # Checked every token: the restart offset is arbitrary, so sampling on
        # a fixed cadence can miss it. Each check is a few integer compares.
        for size in _GUARD_BLOCKS:
            if n > size and self._window_matches_start(size):
                self.cycles_detected += 1
                self.restart_index = n - size
                log.warning(
                    "repetition loop: output restarted at token %d; stopping decode",
                    n - size,
                )
                return True
        return False


def sanitize_ocr_text(text: str) -> str:
    """Remove control/layout markup and clamp blank-line runs."""
    if not text:
        return ""
    cleaned = _CONTROL_TOKEN_RE.sub("", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _BLANK_RUN_RE.sub("\n\n", cleaned)
    return cleaned.strip("\n")


@dataclass
class OCRResult:
    text: str
    processing_time_ms: int
    model: str
    image_width: int
    image_height: int
    prompt_tokens: int = 0
    generation_tokens: int = 0
    generation_tps: float = 0.0
    effective_tps: float = 0.0
    prompt_tps: float = 0.0
    peak_memory_gb: float = 0.0
    finish_reason: str | None = None
    page: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class OCREngine:
    """Loads PaddleOCR-VL once and serves single-image OCR requests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any = None
        self.processor: Any = None
        self.config: Any = None
        self._prompt_cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self.load_time_s: float = 0.0
        self._import_time_s: float = 0.0
        self.requests_served: int = 0
        self._slots = threading.Semaphore(settings.concurrency)
        # MLX keeps its Metal streams thread-local. Running inference from
        # asyncio's shared thread pool means successive requests land on
        # different threads, and each new thread re-creates streams and loses
        # the warm GPU state -- measured at up to 2x slower decode on the
        # first request after a gap. A dedicated single-thread executor keeps
        # every request on the same thread.
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ocr-infer"
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def load(self) -> float:
        t_import = time.perf_counter()
        import mlx.core as mx  # noqa: PLC0415
        from mlx_vlm import load as mlx_load  # noqa: PLC0415
        from mlx_vlm import prompt_utils  # noqa: PLC0415
        from mlx_vlm.generate import dispatch  # noqa: PLC0415
        from mlx_vlm.utils import load_config  # noqa: PLC0415

        self._import_time_s = time.perf_counter() - t_import
        self._mx = mx
        self._dispatch = dispatch
        self._prompt_utils = prompt_utils

        t0 = time.perf_counter()
        model, processor = mlx_load(self.settings.model)
        config = load_config(self.settings.model)
        mx.synchronize()
        self.load_time_s = time.perf_counter() - t0

        # Lock in the vision budget so every request uses the same budget and
        # the processor never re-reads a config file per call.
        processor.image_processor.min_pixels = self.settings.min_pixels
        processor.image_processor.max_pixels = self.settings.max_pixels

        self.model = model
        self.processor = processor
        self.config = config

        # Shared, resettable loop guard. Requests are serialised by self._lock,
        # so one instance is safe to reuse across requests.
        self._stopping = CycleStoppingCriteria([model.config.eos_token_id])

        log.info(
            "loaded %s in %.2fs (imports %.2fs) on %s",
            self.settings.model,
            self.load_time_s,
            self._import_time_s,
            mx.default_device(),
        )
        return self.load_time_s

    def warmup(self) -> None:
        """Run one tiny generation so the first real request is not the one
        paying for Metal kernel compilation and buffer allocation."""
        from PIL import Image  # noqa: PLC0415

        blank = Image.new("RGB", (448, 448), (255, 255, 255))
        t0 = time.perf_counter()
        # Run it on the pinned inference thread, which is where every real
        # request will execute, so the warm state lands where it is needed.
        self.executor.submit(self._run, blank, 1).result()
        log.info("warmup complete in %.2fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _build_prompt(self) -> str:
        key = self.settings.prompt
        cached = self._prompt_cache.get(key)
        if cached is None:
            cached = self._prompt_utils.apply_chat_template(
                self.processor, self.config, key, num_images=1
            )
            self._prompt_cache[key] = cached
        return cached

    def _truncate_at_restart(self, out: Any) -> str:
        """Drop any duplicated tail produced before the loop guard fired.

        The guard stops decoding as soon as a restart is confirmed, so the
        output still contains the tokens generated since the page began to
        repeat. Cut the text back to the first pass.
        """
        text = out.text or ""
        restart = self._stopping.restart_index
        if not restart or not out.token_ids:
            return text

        # Restart index counts generated tokens; decode only the first pass.
        tokens = list(out.token_ids)[:restart]
        try:
            first_pass = self.processor.tokenizer.decode(
                tokens, skip_special_tokens=True
            )
        except Exception:  # noqa: BLE001 - never fail a request over cleanup
            log.exception("could not truncate repeated output")
            return text
        log.info(
            "truncated repeated output: %d -> %d chars",
            len(text),
            len(first_pass),
        )
        return first_pass

    def _run(self, image, max_tokens: int) -> Any:
        prompt = self._build_prompt()
        mx = self._mx
        # Isolate per-request peak-memory accounting from the warmup run.
        mx.reset_peak_memory()
        self._stopping.reset()
        return self._dispatch.generate(
            self.model,
            self.processor,
            prompt,
            [image],
            max_tokens=max_tokens,
            temperature=self.settings.temperature,
            stopping_criteria=self._stopping if self.settings.loop_guard else None,
        )

    def ocr(self, page: Page, max_tokens: int | None = None) -> OCRResult:
        if self.model is None:
            raise RuntimeError("engine not loaded; call load() first")

        mt = max_tokens or self.settings.max_tokens
        t0 = time.perf_counter()

        # concurrency=1 in practice: the executor already serialises inference
        # to one thread, and _slots/_lock bound it to `concurrency` if raised.
        with self._slots:
            with self._lock:
                out = self._run(page.image, mt)
                self._mx.synchronize()

        total_ms = int((time.perf_counter() - t0) * 1000)
        gen_tokens = out.generation_tokens

        raw = self._truncate_at_restart(out)
        text = sanitize_ocr_text(raw) if self.settings.sanitize else raw.strip()
        self.requests_served += 1

        return OCRResult(
            text=text,
            processing_time_ms=total_ms,
            model=self.settings.model,
            image_width=page.width,
            image_height=page.height,
            prompt_tokens=out.prompt_tokens,
            generation_tokens=gen_tokens,
            # Two different rates, both useful:
            #  - generation_tps: pure decode speed as measured by mlx-vlm
            #    (excludes prefill/vision encode, which is most of the cost)
            #  - effective_tps: output tokens / whole inference wall time,
            #    i.e. what a caller actually experiences per page
            generation_tps=out.generation_tps or (gen_tokens / (total_ms / 1000.0)),
            effective_tps=gen_tokens / max(total_ms / 1000.0, 1e-6),
            prompt_tps=out.prompt_tps,
            peak_memory_gb=self._mx.get_peak_memory() / 1e9,
            finish_reason=out.finish_reason,
            page=page.page_number,
            extra={
                "source_width": page.source_width,
                "source_height": page.source_height,
                "pdf_dpi": page.pdf_dpi,
                "sanitized_chars": len(raw) - len(text),
                "loop_guard_triggered": bool(self._stopping.cycles_detected),
            },
        )

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def info(self) -> dict:
        if self.model is None or not hasattr(self, "_mx"):
            # Not loaded yet (or a stub in tests): report what is known.
            return {
                "model": self.settings.model,
                "prompt": self.settings.prompt,
                "backend": "mlx",
                "mlx_device": "uninitialised",
                "gpu_available": False,
                "metal_available": False,
                "model_loaded": self.model is not None,
                "model_load_time_s": round(self.load_time_s, 3),
                "model_import_time_s": round(self._import_time_s, 3),
                "requests_served": self.requests_served,
                "concurrency": self.settings.concurrency,
                "max_tokens": self.settings.max_tokens,
                "max_pixels": self.settings.max_pixels,
                "min_pixels": self.settings.min_pixels,
                "max_image_dim": self.settings.max_image_dim,
                "pdf_dpi": self.settings.pdf_dpi,
                "sanitize": self.settings.sanitize,
                "loop_guard": self.settings.loop_guard,
                "gpu_memory_gb": 0.0,
                "gpu_peak_memory_gb": 0.0,
                "host_memory_mb": _rss_mb(),
            }

        mx = self._mx
        device = str(mx.default_device())
        return {
            "model": self.settings.model,
            "prompt": self.settings.prompt,
            "backend": "mlx",
            "mlx_device": device,
            "gpu_available": device.startswith("gpu") or "gpu" in device,
            "metal_available": bool(getattr(mx.metal, "is_available", lambda: False)()),
            "model_loaded": self.model is not None,
            "model_load_time_s": round(self.load_time_s, 3),
            "model_import_time_s": round(self._import_time_s, 3),
            "requests_served": self.requests_served,
            "concurrency": self.settings.concurrency,
            "max_tokens": self.settings.max_tokens,
            "max_pixels": self.settings.max_pixels,
            "min_pixels": self.settings.min_pixels,
            "max_image_dim": self.settings.max_image_dim,
            "pdf_dpi": self.settings.pdf_dpi,
            "sanitize": self.settings.sanitize,
            "loop_guard": self.settings.loop_guard,
            "gpu_memory_gb": round(mx.get_active_memory() / 1e9, 3),
            "gpu_peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "host_memory_mb": _rss_mb(),
        }


def _rss_mb() -> float:
    try:
        import psutil  # noqa: PLC0415

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return -1.0
