"""Runtime configuration for the PaddleOCR-VL MLX OCR server.

All settings can be overridden with environment variables (see .env.example)
or with an ``OCR_``-prefixed CLI flag when using the module entrypoint.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- model -----------------------------------------------------------
    # MLX port of PaddleOCR-VL-1.5. Other valid ids include
    # mlx-community/PaddleOCR-VL-1.6-4bit or a local path.
    model: str = "mlx-community/PaddleOCR-VL-1.5-4bit"

    # Prompt sent to the VLM. "OCR:" is the task prompt that produces clean
    # plain text. Any free-form instruction flips the model into text
    # spotting mode, which emits <|LOC_n|> layout tokens.
    prompt: str = "OCR:"

    max_tokens: int = 4096
    temperature: float = 0.0

    # --- vision resolution ----------------------------------------------
    # Vision budget in pixels, must be a multiple of 28x28 (=784) for the
    # patch grid. 1003520 == 1280*28*28, the value recommended by the
    # PaddleOCR-VL model card for the "ocr" task.
    max_pixels: int = 1_003_520
    min_pixels: int = 112_896

    # Hard cap on the longest image edge before OCR (0 = disabled).
    max_image_dim: int = 4096

    # --- pdf --------------------------------------------------------------
    pdf_dpi: int = 200
    pdf_max_pages: int = 200

    # Stop decoding if the model restarts a page and repeats itself. Without
    # this, a looped page burns the whole token budget and returns duplicated
    # text. Turn off only if you are OCRing legitimately repetitive documents.
    loop_guard: bool = True

    # --- server ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8088
    concurrency: int = 1
    request_timeout: float = 300.0
    max_upload_mb: int = 32

    # --- misc -------------------------------------------------------------
    debug: bool = False
    warmup: bool = True
    # Drop residual control/layout tokens (<|LOC_n|>, image placeholders)
    # and clamp runs of blank lines. Never touches visible characters.
    sanitize: bool = True

    @field_validator("max_pixels")
    @classmethod
    def _round_to_patch(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_pixels must be positive")
        return max(v // 784 * 784, 784)

    @field_validator("min_pixels")
    @classmethod
    def _check_min(cls, v: int) -> int:
        if v < 0:
            raise ValueError("min_pixels must be >= 0")
        return v

    @field_validator("concurrency")
    @classmethod
    def _check_conc(cls, v: int) -> int:
        if v < 1:
            raise ValueError("concurrency must be >= 1")
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def public_dict(self) -> dict:
        """Settings safe to expose over HTTP (no secrets on this service)."""
        return self.model_dump(mode="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
