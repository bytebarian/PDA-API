"""OCR provider contracts, DTOs, and domain exceptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/jpg", "image/png"})


def normalize_mime_type(mime_type: str | None) -> str:
    """Return a normalised MIME type without parameters."""

    return (mime_type or "").strip().lower().split(";")[0].strip()


def mime_type_requires_ocr(mime_type: str | None) -> bool:
    """Return True when *mime_type* should be routed through OCR."""

    return normalize_mime_type(mime_type) in SUPPORTED_IMAGE_MIME_TYPES


class OCRError(Exception):
    """Base class for OCR-related errors."""


class OCRProviderUnavailableError(OCRError):
    """Raised when the configured OCR provider cannot be used."""


class OCRUnsupportedMimeTypeError(OCRError):
    """Raised when a provider does not support the given MIME type."""


class OCRUnreadableImageError(OCRError):
    """Raised when an image cannot be read by the OCR engine."""


class OCRTimeoutError(OCRError):
    """Raised when OCR takes longer than the configured timeout."""


class OCREmptyResultError(OCRError):
    """Raised when OCR completes but yields no meaningful text."""


class OCRProviderResponseError(OCRError):
    """Raised when the OCR provider returns malformed or unexpected output."""


@dataclass(frozen=True)
class OCRPageResult:
    """OCR output for a single page/image."""

    page_number: int
    text: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRResult:
    """Complete OCR output returned by a provider."""

    provider: str
    engine_version: str | None
    languages: list[str]
    pages: list[OCRPageResult]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class OCRProvider(Protocol):
    """Protocol every OCR provider must satisfy."""

    name: str

    async def extract_text(
        self,
        file_path: Path,
        *,
        mime_type: str,
        languages: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> OCRResult:
        """Extract OCR text from *file_path*."""

    async def healthcheck(self) -> bool:
        """Return True when the provider is ready to serve requests."""
