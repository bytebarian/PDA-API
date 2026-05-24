"""Deterministic, configurable document chunking service.

Provides a pure text-splitting algorithm, settings validation, and a
high-level ``chunk_document`` helper that loads persisted chunking
configuration, runs the algorithm, and persists results into the
``document_chunks`` table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.app_settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    AppSettings,
)
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class ChunkingError(Exception):
    """Base class for chunking errors."""


class ChunkingValidationError(ChunkingError):
    """Raised when chunking settings are invalid."""


class ChunkingEmptyTextError(ChunkingError):
    """Raised when the text to chunk is empty or whitespace-only."""


# ---------------------------------------------------------------------------
# Result DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkResult:
    """Represents a single produced chunk before DB persistence."""

    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    page_number: int | None = None
    source_section: str | None = None
    metadata: dict[str, Any] | None = field(default=None)


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunking parameters or raise :class:`ChunkingValidationError`."""
    if chunk_size <= 0:
        raise ChunkingValidationError(f"chunkSize must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ChunkingValidationError(f"chunkOverlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ChunkingValidationError(
            f"chunkOverlap ({chunk_overlap}) must be < chunkSize ({chunk_size})"
        )


# ---------------------------------------------------------------------------
# Pure chunking algorithm
# ---------------------------------------------------------------------------

_BOUNDARY_WINDOW = 200


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _find_best_boundary(text: str, start: int, target_end: int) -> int:
    """Return the best split position at or before *target_end*.

    Looks backward within a fixed window from *target_end*, preferring
    boundaries in this priority order:

    1. Paragraph boundary (``\\n\\n``)
    2. Line boundary (``\\n``)
    3. Sentence boundary (``. ``, ``! ``, ``? ``)
    4. Whitespace boundary (`` ``)
    5. Hard split at *target_end*
    """
    if target_end >= len(text):
        return len(text)

    # Never go before start+1 to guarantee forward progress.
    search_start = max(start + 1, target_end - _BOUNDARY_WINDOW)
    segment = text[search_start:target_end]

    # 1. Paragraph boundary
    idx = segment.rfind("\n\n")
    if idx >= 0:
        return search_start + idx + 2

    # 2. Line boundary
    idx = segment.rfind("\n")
    if idx >= 0:
        return search_start + idx + 1

    # 3. Sentence boundary
    for marker in (". ", "! ", "? "):
        idx = segment.rfind(marker)
        if idx >= 0:
            return search_start + idx + len(marker)

    # 4. Whitespace boundary
    idx = segment.rfind(" ")
    if idx >= 0:
        return search_start + idx + 1

    # 5. Hard split
    return target_end


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[ChunkResult]:
    """Split *text* into ordered, boundary-aware, overlapping chunks.

    The same *text* + *chunk_size* + *chunk_overlap* always produces identical
    output (deterministic).

    Args:
        text: Raw or pre-normalised source text.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Number of characters adjacent chunks share.

    Returns:
        Ordered list of :class:`ChunkResult` objects.  Returns an empty list
        when the normalised text is empty.

    Raises:
        ChunkingValidationError: When *chunk_size* or *chunk_overlap* is invalid.
    """
    validate_chunk_settings(chunk_size, chunk_overlap)

    normalized = _normalize_line_endings(text).strip()
    if not normalized:
        return []

    chunks: list[ChunkResult] = []
    start = 0

    while start < len(normalized):
        target_end = min(start + chunk_size, len(normalized))
        end = _find_best_boundary(normalized, start, target_end)

        content = normalized[start:end].strip()
        if content:
            chunks.append(
                ChunkResult(
                    chunk_index=len(chunks),
                    content=content,
                    start_offset=start,
                    end_offset=end,
                )
            )

        if end >= len(normalized):
            break

        # Advance with overlap; +1 guard prevents infinite loops.
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


async def _load_chunk_settings(db: AsyncSession) -> tuple[int, int]:
    """Return ``(chunk_size, chunk_overlap)`` from the most recently updated AppSettings row.

    Falls back to module-level defaults when no row exists.
    """
    result = await db.execute(
        select(AppSettings).order_by(AppSettings.updated_at.desc()).limit(1)
    )
    settings = result.scalar_one_or_none()
    if settings is not None:
        return settings.chunk_size, settings.chunk_overlap
    return DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


async def replace_document_chunks(
    db: AsyncSession,
    document: Document,
    chunks: list[ChunkResult],
) -> None:
    """Delete stale chunks for *document* and insert the fresh set.

    Also updates ``document.chunk_count`` in-place (the caller is responsible
    for committing the transaction).
    """
    await db.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
    )

    for chunk in chunks:
        metadata: dict[str, Any] | None = chunk.metadata
        if chunk.source_section is not None:
            metadata = dict(metadata or {})
            metadata["source_section"] = chunk.source_section

        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                source_start_offset=chunk.start_offset,
                source_end_offset=chunk.end_offset,
                page_number=chunk.page_number,
                metadata_jsonb=metadata,
            )
        )

    document.chunk_count = len(chunks)


# ---------------------------------------------------------------------------
# High-level service entry point
# ---------------------------------------------------------------------------


async def chunk_document(
    db: AsyncSession,
    document: Document,
) -> list[ChunkResult]:
    """Chunk a document's extracted text and persist the results.

    Loads chunking settings from the database (or uses defaults), validates
    them, splits the text, replaces any stale chunk records, and updates
    ``document.chunk_count``.

    Args:
        db: Active async database session.
        document: Document whose ``extracted_text`` will be chunked.

    Returns:
        Ordered list of :class:`ChunkResult` produced for the document.

    Raises:
        ChunkingEmptyTextError: When ``document.extracted_text`` is absent or
            whitespace-only.
        ChunkingValidationError: When the persisted settings are invalid.
    """
    raw_text = document.extracted_text or ""
    normalized = _normalize_line_endings(raw_text).strip()
    if not normalized:
        raise ChunkingEmptyTextError("No extractable text available for chunking")

    chunk_size, chunk_overlap = await _load_chunk_settings(db)
    validate_chunk_settings(chunk_size, chunk_overlap)

    chunks = chunk_text(normalized, chunk_size, chunk_overlap)
    await replace_document_chunks(db, document, chunks)
    return chunks
