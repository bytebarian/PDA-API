"""OCR adapters."""

from app.adapters.ocr.base import (
    OCREmptyResultError,
    OCRError,
    OCRPageResult,
    OCRProvider,
    OCRProviderResponseError,
    OCRProviderUnavailableError,
    OCRResult,
    OCRTimeoutError,
    OCRUnreadableImageError,
    OCRUnsupportedMimeTypeError,
    SUPPORTED_IMAGE_MIME_TYPES,
    mime_type_requires_ocr,
    normalize_mime_type,
)
from app.adapters.ocr.fake import FakeOCRProvider
from app.adapters.ocr.tesseract import TesseractOCRProvider

__all__ = [
    "OCREmptyResultError",
    "OCRError",
    "OCRPageResult",
    "OCRProvider",
    "OCRProviderResponseError",
    "OCRProviderUnavailableError",
    "OCRResult",
    "OCRTimeoutError",
    "OCRUnreadableImageError",
    "OCRUnsupportedMimeTypeError",
    "SUPPORTED_IMAGE_MIME_TYPES",
    "FakeOCRProvider",
    "TesseractOCRProvider",
    "mime_type_requires_ocr",
    "normalize_mime_type",
]
