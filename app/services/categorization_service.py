"""Categorization service for the PDA document-processing pipeline.

Responsibilities
----------------
1. Load a document by ID.
2. Obtain normalized/extracted text (``Document.extracted_text``) and summary.
3. Guard against absent text (return a *skipped* result).
4. Truncate input to ``Settings.categorization_max_input_chars``.
5. Call the configured local categorization provider.
6. Validate the returned category against the allowed vocabulary.
7. Persist category and categorization metadata on the Document row.
8. Return a typed ``DocumentCategorizationResult``.

The service does **not** update processing job stage history – that is the
responsibility of the orchestrator stage runner
(``_run_category_assignment_stage`` in ``processing_orchestrator.py``).

Privacy
-------
Raw document text is **never** logged.  Log lines contain only document id,
provider name, model name, category, confidence, status, and sanitized error
reasons.

Manual category protection
--------------------------
If ``document.category_source`` is ``"manual"``, the service will **not**
overwrite the existing category unless ``force`` is set to ``True``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.categorization.base import (
    ALLOWED_CATEGORIES,
    CategorizationError,
    CategorizationProvider,
    CategorizationSource,
    DocumentCategory,
)
from app.core.config import Settings, get_settings
from app.models.document import Document

logger = logging.getLogger(__name__)

_STATUS_PENDING = "pending"
_STATUS_PROCESSING = "processing"
_STATUS_READY = "ready"
_STATUS_FAILED = "failed"
_STATUS_SKIPPED = "skipped"


class CategorizationDocumentNotFoundError(LookupError):
    """Raised when the target document cannot be found."""


@dataclass(frozen=True)
class DocumentCategorizationResult:
    """Result of a categorization run."""

    document_id: uuid.UUID
    category: str | None
    category_status: str
    category_source: str | None
    category_confidence: float | None
    category_reason: str | None
    category_model: str | None
    category_generated_at: datetime | None
    category_error: str | None
    skipped: bool = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_categorization_provider(
    *,
    provider_name: str | None = None,
    settings: Settings | None = None,
) -> CategorizationProvider:
    """Resolve and return a categorization provider instance."""
    from app.adapters.categorization.mock import MockCategorizationProvider
    from app.adapters.categorization.ollama import OllamaCategorizationProvider
    from app.adapters.categorization.rules import RulesCategorizationProvider

    resolved_settings = settings or get_settings()
    name = provider_name or resolved_settings.categorization_provider

    if name == "rules":
        return RulesCategorizationProvider()
    if name == "mock":
        return MockCategorizationProvider()
    if name == "ollama":
        return OllamaCategorizationProvider(
            base_url=resolved_settings.ollama_base_url,
            timeout_seconds=resolved_settings.ollama_timeout_seconds,
        )
    # Unknown provider – fall back to rules with a warning.
    logger.warning(
        "Unknown categorization provider %r; falling back to rules",
        name,
    )
    return RulesCategorizationProvider()


async def categorize_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    provider: CategorizationProvider | None = None,
    settings: Settings | None = None,
    force: bool = False,
) -> DocumentCategorizationResult:
    """Assign and persist a category for the given document.

    Parameters
    ----------
    db:
        Active async database session.
    document_id:
        UUID of the document to categorize.
    provider:
        Optional categorization provider override (useful for testing).
    settings:
        Optional settings override; falls back to ``get_settings()``.
    force:
        When ``True``, overwrite even a manually assigned category.

    Returns
    -------
    DocumentCategorizationResult
        Typed result carrying the assigned category and metadata.

    Raises
    ------
    CategorizationDocumentNotFoundError
        When *document_id* does not match any document in the database.
    """
    resolved_settings = settings or get_settings()
    resolved_provider = provider or get_categorization_provider(settings=resolved_settings)

    document = await db.get(Document, document_id)
    if document is None:
        raise CategorizationDocumentNotFoundError(
            f"Document not found: {document_id}"
        )

    # --- Protect manually assigned categories ----------------------------
    if not force and document.category_source == CategorizationSource.manual.value:
        logger.info(
            "categorize_document: document_id=%s has manual category; skipping",
            document_id,
        )
        return DocumentCategorizationResult(
            document_id=document_id,
            category=document.category,
            category_status=document.category_status,
            category_source=document.category_source,
            category_confidence=document.category_confidence,
            category_reason=document.category_reason,
            category_model=document.category_model,
            category_generated_at=document.category_generated_at,
            category_error=None,
            skipped=True,
        )

    # --- Mark processing -------------------------------------------------
    document.category_status = _STATUS_PROCESSING
    document.category_error = None
    await db.flush()

    # --- Obtain source text ----------------------------------------------
    source_text = (document.extracted_text or "").strip()

    if not source_text:
        logger.info(
            "categorize_document: document_id=%s has no usable text; skipping",
            document_id,
        )
        document.category = DocumentCategory.other.value
        document.category_status = _STATUS_SKIPPED
        document.category_source = CategorizationSource.fallback.value
        document.category_confidence = 0.0
        document.category_reason = "no_extracted_text"
        document.category_model = None
        document.category_generated_at = _utcnow()
        document.category_error = "no_extracted_text"
        return DocumentCategorizationResult(
            document_id=document_id,
            category=DocumentCategory.other.value,
            category_status=_STATUS_SKIPPED,
            category_source=CategorizationSource.fallback.value,
            category_confidence=0.0,
            category_reason="no_extracted_text",
            category_model=None,
            category_generated_at=document.category_generated_at,
            category_error="no_extracted_text",
            skipped=True,
        )

    # --- Truncate input --------------------------------------------------
    max_input = resolved_settings.categorization_max_input_chars
    if len(source_text) > max_input:
        logger.debug(
            "categorize_document: document_id=%s text truncated from %d to %d chars",
            document_id,
            len(source_text),
            max_input,
        )
        source_text = source_text[:max_input]

    # --- Call provider ---------------------------------------------------
    try:
        result = await resolved_provider.categorize(
            source_text,
            filename=document.filename,
            summary=document.summary,
            metadata=document.metadata_jsonb,
        )
    except CategorizationError as exc:
        error_reason = str(exc) or exc.__class__.__name__
        logger.warning(
            "categorize_document: document_id=%s provider=%s failed: %s",
            document_id,
            resolved_provider.name,
            error_reason,
        )
        document.category = DocumentCategory.other.value
        document.category_status = _STATUS_FAILED
        document.category_source = CategorizationSource.fallback.value
        document.category_confidence = 0.0
        document.category_reason = "provider_error"
        document.category_model = None
        document.category_generated_at = _utcnow()
        document.category_error = error_reason
        return DocumentCategorizationResult(
            document_id=document_id,
            category=DocumentCategory.other.value,
            category_status=_STATUS_FAILED,
            category_source=CategorizationSource.fallback.value,
            category_confidence=0.0,
            category_reason="provider_error",
            category_model=None,
            category_generated_at=document.category_generated_at,
            category_error=error_reason,
        )

    # --- Validate category -----------------------------------------------
    category = result.category.strip().lower() if result.category else ""
    if category not in ALLOWED_CATEGORIES:
        logger.warning(
            "categorize_document: document_id=%s provider=%s returned invalid "
            "category %r; falling back to other",
            document_id,
            resolved_provider.name,
            result.category,
        )
        category = DocumentCategory.other.value

    # --- Apply minimum confidence threshold ------------------------------
    confidence = max(0.0, min(1.0, result.confidence))
    min_conf = resolved_settings.categorization_min_confidence
    if confidence < min_conf and category != DocumentCategory.other.value:
        logger.info(
            "categorize_document: document_id=%s confidence %.4f < threshold %.4f; "
            "category demoted to other",
            document_id,
            confidence,
            min_conf,
        )
        category = DocumentCategory.other.value

    # --- Persist ---------------------------------------------------------
    document.category = category
    document.category_status = _STATUS_READY
    document.category_source = result.source
    document.category_confidence = confidence
    document.category_reason = (result.reason or "")[:500]
    document.category_model = result.model
    document.category_generated_at = _utcnow()
    document.category_error = None

    logger.info(
        "categorize_document: document_id=%s provider=%s category=%s "
        "confidence=%.4f status=ready",
        document_id,
        resolved_provider.name,
        category,
        confidence,
    )

    return DocumentCategorizationResult(
        document_id=document_id,
        category=category,
        category_status=_STATUS_READY,
        category_source=result.source,
        category_confidence=confidence,
        category_reason=document.category_reason,
        category_model=result.model,
        category_generated_at=document.category_generated_at,
        category_error=None,
    )
