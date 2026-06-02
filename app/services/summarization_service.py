"""Summarization service for the PDA document-processing pipeline.

Responsibilities
----------------
1. Load a document by ID.
2. Obtain normalized/extracted text (``Document.extracted_text``), or build
   fallback text from ordered chunks when the field is absent.
3. Guard against empty source text (return a *skipped* result).
4. Truncate input to ``Settings.summary_max_input_chars`` to avoid overloading
   the local model.  For very large documents the first ``max_input_chars``
   characters are used (deterministic truncation strategy).  A map-reduce
   approach is left as a future enhancement; current documents are typically
   small enough for single-pass summarization.
5. Call the configured local summarization provider.
6. Validate and normalize provider output.
7. Persist the summary and summary metadata on the Document row.
8. Return a typed ``DocumentSummaryResult``.

The service does **not** update processing job stage history – that is the
responsibility of the orchestrator stage runner (``_run_summary_generation_stage``
in ``processing_orchestrator.py``).

Privacy
-------
Raw document text is **never** logged.  Log lines contain only document id,
provider name, model name, status, input/output lengths, and sanitized error
reasons.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.summarization.base import (
    SummarizationError,
    SummarizationProvider,
)
from app.core.config import Settings, get_settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

_FALLBACK_NO_TEXT = "Summary unavailable: no extracted text was found."
_SUMMARY_STATUS_PENDING = "pending"
_SUMMARY_STATUS_PROCESSING = "processing"
_SUMMARY_STATUS_READY = "ready"
_SUMMARY_STATUS_FAILED = "failed"
_SUMMARY_STATUS_SKIPPED = "skipped"


class SummarizationDocumentNotFoundError(LookupError):
    """Raised when the target document cannot be found."""


class SummarizationNoTextError(ValueError):
    """Raised when the document has no usable text for summarization."""


@dataclass(frozen=True)
class DocumentSummaryResult:
    """Result of a summarization run."""

    document_id: uuid.UUID
    summary: str | None
    summary_status: str
    summary_model: str | None
    summary_generated_at: datetime | None
    summary_error: str | None
    skipped: bool = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_summarization_provider(
    *,
    provider_name: str | None = None,
    settings: Settings | None = None,
) -> SummarizationProvider:
    """Resolve and return a summarization provider instance.

    Falls back to the *ollama* provider when *provider_name* is ``None``.
    """
    from app.adapters.summarization.mock import MockSummarizationProvider
    from app.adapters.summarization.ollama import OllamaSummarizationProvider

    resolved_settings = settings or get_settings()
    name = provider_name or resolved_settings.summarization_provider

    if name == "mock":
        return MockSummarizationProvider()
    if name == "ollama":
        return OllamaSummarizationProvider(
            base_url=resolved_settings.ollama_base_url,
            timeout_seconds=resolved_settings.ollama_timeout_seconds,
        )
    # Unknown provider – fall back to Ollama with a warning.
    logger.warning(
        "Unknown summarization provider %r; falling back to ollama",
        name,
    )
    return OllamaSummarizationProvider(
        base_url=resolved_settings.ollama_base_url,
        timeout_seconds=resolved_settings.ollama_timeout_seconds,
    )


async def _get_chunk_text(db: AsyncSession, document_id: uuid.UUID) -> str:
    """Return ordered chunk text concatenated with newlines, or empty string."""
    result = await db.execute(
        select(DocumentChunk.content)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = result.scalars().all()
    return "\n".join(c for c in chunks if c)


async def summarize_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    provider: SummarizationProvider | None = None,
    settings: Settings | None = None,
) -> DocumentSummaryResult:
    """Generate and persist a summary for the given document.

    Parameters
    ----------
    db:
        Active async database session.
    document_id:
        UUID of the document to summarize.
    provider:
        Optional summarization provider override (useful for testing).
    settings:
        Optional settings override; falls back to ``get_settings()``.

    Returns
    -------
    DocumentSummaryResult
        Typed result carrying the generated summary and metadata.

    Raises
    ------
    SummarizationDocumentNotFoundError
        When *document_id* does not match any document in the database.
    """
    resolved_settings = settings or get_settings()
    resolved_provider = provider or get_summarization_provider(settings=resolved_settings)

    document = await db.get(Document, document_id)
    if document is None:
        raise SummarizationDocumentNotFoundError(
            f"Document not found: {document_id}"
        )

    # --- Mark processing -------------------------------------------------
    document.summary_status = _SUMMARY_STATUS_PROCESSING
    document.summary_error = None
    await db.flush()

    # --- Obtain source text ----------------------------------------------
    source_text = (document.extracted_text or "").strip()

    if not source_text:
        # Try to build text from stored chunks as a fallback.
        source_text = (await _get_chunk_text(db, document_id)).strip()

    if not source_text:
        logger.info(
            "summarize_document: document_id=%s has no usable text; skipping",
            document_id,
        )
        document.summary = _FALLBACK_NO_TEXT
        document.summary_status = _SUMMARY_STATUS_SKIPPED
        document.summary_model = None
        document.summary_generated_at = _utcnow()
        document.summary_error = "no_extracted_text"
        return DocumentSummaryResult(
            document_id=document_id,
            summary=_FALLBACK_NO_TEXT,
            summary_status=_SUMMARY_STATUS_SKIPPED,
            summary_model=None,
            summary_generated_at=document.summary_generated_at,
            summary_error="no_extracted_text",
            skipped=True,
        )

    # --- Truncate input --------------------------------------------------
    max_input = resolved_settings.summary_max_input_chars
    if len(source_text) > max_input:
        logger.debug(
            "summarize_document: document_id=%s text truncated from %d to %d chars",
            document_id,
            len(source_text),
            max_input,
        )
        source_text = source_text[:max_input]

    # --- Call provider ---------------------------------------------------
    model = resolved_settings.summarization_model
    max_output = resolved_settings.summary_max_output_chars

    try:
        result = await resolved_provider.summarize(
            source_text,
            model=model,
            max_output_chars=max_output,
        )
    except SummarizationError as exc:
        error_reason = str(exc) or exc.__class__.__name__
        logger.warning(
            "summarize_document: document_id=%s provider=%s model=%s failed: %s",
            document_id,
            resolved_provider.name,
            model,
            error_reason,
        )
        document.summary_status = _SUMMARY_STATUS_FAILED
        document.summary_error = error_reason
        document.summary_generated_at = _utcnow()
        return DocumentSummaryResult(
            document_id=document_id,
            summary=None,
            summary_status=_SUMMARY_STATUS_FAILED,
            summary_model=model,
            summary_generated_at=document.summary_generated_at,
            summary_error=error_reason,
        )

    # --- Persist ---------------------------------------------------------
    summary_text = result.summary.strip()
    if len(summary_text) > max_output:
        summary_text = summary_text[:max_output].rsplit(" ", 1)[0]

    document.summary = summary_text
    document.summary_model = result.model
    document.summary_generated_at = _utcnow()
    document.summary_status = _SUMMARY_STATUS_READY
    document.summary_error = None

    logger.info(
        "summarize_document: document_id=%s provider=%s model=%s "
        "input_chars=%d output_chars=%d status=ready",
        document_id,
        result.provider,
        result.model,
        result.input_char_count,
        result.output_char_count,
    )

    return DocumentSummaryResult(
        document_id=document_id,
        summary=summary_text,
        summary_status=_SUMMARY_STATUS_READY,
        summary_model=result.model,
        summary_generated_at=document.summary_generated_at,
        summary_error=None,
    )
