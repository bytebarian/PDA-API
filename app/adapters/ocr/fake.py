"""Deterministic fake OCR provider for tests."""

from __future__ import annotations

from pathlib import Path

from app.adapters.ocr.base import (
    OCRPageResult,
    OCRProvider,
    OCRResult,
    OCRUnsupportedMimeTypeError,
    OCRError,
    mime_type_requires_ocr,
    normalize_mime_type,
)


class FakeOCRProvider:
    """Simple in-memory OCR provider used by tests."""

    name = "fake"

    def __init__(
        self,
        *,
        text: str = "fake ocr text",
        confidence: float | None = 99.0,
        engine_version: str | None = "fake-ocr-1.0",
        warnings: list[str] | None = None,
        error: OCRError | None = None,
    ) -> None:
        self._text = text
        self._confidence = confidence
        self._engine_version = engine_version
        self._warnings = list(warnings or [])
        self._error = error

    async def extract_text(
        self,
        file_path: Path,
        *,
        mime_type: str,
        languages: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> OCRResult:
        del file_path, timeout_seconds

        if not mime_type_requires_ocr(mime_type):
            raise OCRUnsupportedMimeTypeError(
                f"Unsupported OCR MIME type: {normalize_mime_type(mime_type) or '<unknown>'}"
            )
        if self._error is not None:
            raise self._error

        return OCRResult(
            provider=self.name,
            engine_version=self._engine_version,
            languages=list(languages or ["eng"]),
            pages=[
                OCRPageResult(
                    page_number=1,
                    text=self._text,
                    confidence=self._confidence,
                    metadata={"mime_type": normalize_mime_type(mime_type)},
                )
            ],
            warnings=list(self._warnings),
        )

    async def healthcheck(self) -> bool:
        return self._error is None


def _assert_protocol(provider: OCRProvider) -> OCRProvider:
    return provider
