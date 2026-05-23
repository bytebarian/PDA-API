"""Text extraction adapter interface and built-in adapters.

Provides a protocol-based contract for text extraction, a result DTO,
domain exceptions, and deterministic adapters for plain-text and
Markdown files.  A registry/resolver maps MIME types and file extensions
to the correct adapter and exposes a high-level ``extract_text_from_file``
helper for callers.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.models.document import Document

# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class TextExtractionError(Exception):
    """Base class for text extraction errors."""


class UnsupportedTextExtractionTypeError(TextExtractionError):
    """Raised when no adapter supports the requested MIME type or extension."""


class TextExtractionFileNotFoundError(TextExtractionError):
    """Raised when the file to extract text from does not exist."""


class TextExtractionDecodeError(TextExtractionError):
    """Raised when a file cannot be decoded with the expected encoding."""


# ---------------------------------------------------------------------------
# Result DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedTextResult:
    """Holds the result of a text extraction operation."""

    text: str
    metadata: dict[str, Any]
    page_count: int | None = None
    language: str | None = None


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TextExtractionAdapter(Protocol):
    """Protocol that every text extraction adapter must satisfy."""

    supported_mime_types: Collection[str]
    supported_extensions: Collection[str]

    async def extract(
        self,
        file_path: Path,
        *,
        document: Document | None = None,
    ) -> ExtractedTextResult:
        """Extract text from *file_path* and return an :class:`ExtractedTextResult`."""
        ...


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


class PlainTextAdapter:
    """Extracts text from UTF-8 plain-text files."""

    supported_mime_types: Collection[str] = frozenset({"text/plain"})
    supported_extensions: Collection[str] = frozenset({".txt"})

    async def extract(
        self,
        file_path: Path,
        *,
        document: Document | None = None,
    ) -> ExtractedTextResult:
        if not file_path.exists():
            raise TextExtractionFileNotFoundError(
                f"File not found: {file_path}"
            )

        raw = file_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TextExtractionDecodeError(
                f"Cannot decode {file_path} as UTF-8: {exc}"
            ) from exc

        # Normalise line endings to LF.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        ext = file_path.suffix.lower()
        mime = _mime_for_extension(ext) or "text/plain"

        return ExtractedTextResult(
            text=text,
            metadata={
                "extractor": "PlainTextAdapter",
                "source_extension": ext,
                "mime_type": mime,
                "byte_size": len(raw),
                "char_count": len(text),
            },
        )


class MarkdownAdapter:
    """Extracts raw text from Markdown files (no rendering)."""

    supported_mime_types: Collection[str] = frozenset(
        {"text/markdown", "text/x-markdown"}
    )
    supported_extensions: Collection[str] = frozenset({".md", ".markdown"})

    async def extract(
        self,
        file_path: Path,
        *,
        document: Document | None = None,
    ) -> ExtractedTextResult:
        if not file_path.exists():
            raise TextExtractionFileNotFoundError(
                f"File not found: {file_path}"
            )

        raw = file_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TextExtractionDecodeError(
                f"Cannot decode {file_path} as UTF-8: {exc}"
            ) from exc

        # Normalise line endings to LF.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        ext = file_path.suffix.lower()
        mime = _mime_for_extension(ext) or "text/markdown"

        return ExtractedTextResult(
            text=text,
            metadata={
                "extractor": "MarkdownAdapter",
                "source_extension": ext,
                "mime_type": mime,
                "byte_size": len(raw),
                "char_count": len(text),
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXTENSION_TO_MIME: dict[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


def _mime_for_extension(ext: str) -> str | None:
    return _EXTENSION_TO_MIME.get(ext.lower())


# ---------------------------------------------------------------------------
# Registry / resolver
# ---------------------------------------------------------------------------

#: Ordered list of all built-in adapters.  Future adapters should be appended.
_REGISTRY: list[TextExtractionAdapter] = [
    PlainTextAdapter(),
    MarkdownAdapter(),
]


def _resolve_adapter(
    *,
    mime_type: str | None = None,
    filename: str | None = None,
) -> TextExtractionAdapter:
    """Return the first adapter that matches *mime_type* or *filename* extension.

    MIME type takes precedence over extension when both are provided.

    Raises:
        UnsupportedTextExtractionTypeError: when no adapter matches.
    """
    # Normalise inputs.
    norm_mime = (mime_type or "").strip().lower().split(";")[0].strip()
    ext = Path(filename).suffix.lower() if filename else ""

    # Try MIME type first.
    if norm_mime:
        for adapter in _REGISTRY:
            if norm_mime in {m.lower() for m in adapter.supported_mime_types}:
                return adapter

    # Fall back to file extension.
    if ext:
        for adapter in _REGISTRY:
            if ext in {e.lower() for e in adapter.supported_extensions}:
                return adapter

    label = mime_type or filename or "<unknown>"
    raise UnsupportedTextExtractionTypeError(
        f"No text extraction adapter found for: {label}"
    )


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------


async def extract_text_from_file(
    file_path: Path,
    *,
    mime_type: str | None = None,
    filename: str | None = None,
    document: Document | None = None,
) -> ExtractedTextResult:
    """Resolve the appropriate adapter and extract text from *file_path*.

    Args:
        file_path: Absolute or relative path to the file.
        mime_type: Optional MIME type hint (used for adapter resolution).
        filename: Optional filename hint used when *mime_type* is absent or
            insufficient to determine the adapter.
        document: Optional :class:`~app.models.document.Document` passed
            through to the adapter for contextual metadata.

    Returns:
        :class:`ExtractedTextResult` populated by the chosen adapter.

    Raises:
        UnsupportedTextExtractionTypeError: when no adapter matches.
        TextExtractionFileNotFoundError: when *file_path* does not exist.
        TextExtractionDecodeError: when the file cannot be decoded.
    """
    # Derive a filename hint from the path when not supplied.
    resolved_filename = filename or file_path.name or None

    adapter = _resolve_adapter(mime_type=mime_type, filename=resolved_filename)
    return await adapter.extract(file_path, document=document)
