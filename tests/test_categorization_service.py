"""Unit tests for the categorization service.

All tests use MockCategorizationProvider to avoid requiring a live Ollama
instance or any internet access.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.categorization.base import (
    CategorizationProviderUnavailableError,
    DocumentCategory,
)
from app.adapters.categorization.mock import MockCategorizationProvider
from app.models.document import Document
from app.services.categorization_service import (
    CategorizationDocumentNotFoundError,
    DocumentCategorizationResult,
    categorize_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_document(
    db: AsyncSession,
    *,
    filename: str = "test.pdf",
    extracted_text: str | None = "Sample document text.",
    category: str | None = None,
    category_source: str | None = None,
    category_status: str = "pending",
) -> Document:
    doc = Document(
        filename=filename,
        status="awaiting",
        extracted_text=extracted_text,
        category=category,
        category_source=category_source,
        category_status=category_status,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


# ---------------------------------------------------------------------------
# Basic categorization
# ---------------------------------------------------------------------------


async def test_categorize_document_returns_typed_result(db_session: AsyncSession) -> None:
    """categorize_document must return a DocumentCategorizationResult."""
    doc = await _insert_document(db_session)
    provider = MockCategorizationProvider(category=DocumentCategory.invoice.value, confidence=0.9)

    result = await categorize_document(db_session, doc.id, provider=provider)

    assert isinstance(result, DocumentCategorizationResult)
    assert result.document_id == doc.id
    assert result.category == DocumentCategory.invoice.value
    assert result.category_status == "ready"
    assert result.skipped is False


async def test_categorize_document_persists_category(db_session: AsyncSession) -> None:
    """Assigned category must be persisted in the Document row."""
    doc = await _insert_document(db_session, extracted_text="Invoice text.")
    provider = MockCategorizationProvider(category=DocumentCategory.invoice.value, confidence=0.95)

    await categorize_document(db_session, doc.id, provider=provider)
    await db_session.commit()

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.category == DocumentCategory.invoice.value
    assert refreshed.category_status == "ready"
    assert refreshed.category_source == "mock"
    assert refreshed.category_confidence == 0.95
    assert refreshed.category_generated_at is not None
    assert refreshed.category_error is None


# ---------------------------------------------------------------------------
# Document not found
# ---------------------------------------------------------------------------


async def test_categorize_document_raises_when_not_found(db_session: AsyncSession) -> None:
    """CategorizationDocumentNotFoundError must be raised for a missing document."""
    with pytest.raises(CategorizationDocumentNotFoundError):
        await categorize_document(db_session, uuid.uuid4())


# ---------------------------------------------------------------------------
# No text / skipped path
# ---------------------------------------------------------------------------


async def test_categorize_document_skipped_when_no_text(db_session: AsyncSession) -> None:
    """When the document has no extracted text the result must be skipped."""
    doc = await _insert_document(db_session, extracted_text=None)
    provider = MockCategorizationProvider()

    result = await categorize_document(db_session, doc.id, provider=provider)
    await db_session.commit()

    assert result.skipped is True
    assert result.category_status == "skipped"
    assert result.category == DocumentCategory.other.value

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.category_status == "skipped"
    assert refreshed.category_error == "no_extracted_text"


async def test_categorize_document_skipped_when_empty_text(db_session: AsyncSession) -> None:
    """When the document has empty extracted text the result must be skipped."""
    doc = await _insert_document(db_session, extracted_text="   ")
    provider = MockCategorizationProvider()

    result = await categorize_document(db_session, doc.id, provider=provider)

    assert result.skipped is True
    assert result.category_status == "skipped"


# ---------------------------------------------------------------------------
# Provider error path
# ---------------------------------------------------------------------------


async def test_categorize_document_persists_failure_on_provider_error(
    db_session: AsyncSession,
) -> None:
    """Provider error must be captured and persisted as failed/fallback, not raised."""
    doc = await _insert_document(db_session, extracted_text="Some document text.")
    failing_provider = MockCategorizationProvider(
        error=CategorizationProviderUnavailableError("Ollama unavailable")
    )

    result = await categorize_document(db_session, doc.id, provider=failing_provider)
    await db_session.commit()

    assert result.category_status == "failed"
    assert result.category == DocumentCategory.other.value
    assert result.category_error is not None

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.category_status == "failed"
    assert refreshed.category_error == "Ollama unavailable"
    assert refreshed.category == DocumentCategory.other.value


# ---------------------------------------------------------------------------
# Invalid category from provider – validation
# ---------------------------------------------------------------------------


async def test_categorize_document_rejects_invalid_category(
    db_session: AsyncSession,
) -> None:
    """If the provider returns a category outside the allowed vocabulary, it must
    be replaced with 'other'."""
    from app.adapters.categorization.base import CategorizationResult, CategorizationSource

    class _BadProvider:
        name = "bad"

        async def categorize(self, text: str, **_kw: object) -> CategorizationResult:
            return CategorizationResult(
                category="totally_invalid_category",
                confidence=0.99,
                reason="bad provider output",
                source=CategorizationSource.mock.value,
            )

        async def healthcheck(self) -> bool:
            return True

    doc = await _insert_document(db_session, extracted_text="Some text.")
    result = await categorize_document(db_session, doc.id, provider=_BadProvider())  # type: ignore[arg-type]

    assert result.category == DocumentCategory.other.value


# ---------------------------------------------------------------------------
# Minimum confidence threshold
# ---------------------------------------------------------------------------


async def test_categorize_document_demotes_low_confidence_to_other(
    db_session: AsyncSession,
) -> None:
    """Category with confidence below the threshold must be demoted to 'other'."""
    from app.core.config import Settings

    doc = await _insert_document(db_session, extracted_text="Some text.")
    # Provider returns invoice with low confidence
    provider = MockCategorizationProvider(
        category=DocumentCategory.invoice.value, confidence=0.1
    )
    # Set min_confidence to 0.55 (the default)
    settings = Settings(
        categorization_provider="mock",
        categorization_min_confidence=0.55,
    )

    result = await categorize_document(db_session, doc.id, provider=provider, settings=settings)

    assert result.category == DocumentCategory.other.value


async def test_categorize_document_keeps_high_confidence_category(
    db_session: AsyncSession,
) -> None:
    """Category with confidence at or above threshold must be kept."""
    from app.core.config import Settings

    doc = await _insert_document(db_session, extracted_text="Some text.")
    provider = MockCategorizationProvider(
        category=DocumentCategory.invoice.value, confidence=0.8
    )
    settings = Settings(
        categorization_provider="mock",
        categorization_min_confidence=0.55,
    )

    result = await categorize_document(db_session, doc.id, provider=provider, settings=settings)

    assert result.category == DocumentCategory.invoice.value


async def test_categorize_document_passes_configured_model_to_local_provider(
    db_session: AsyncSession,
) -> None:
    from app.adapters.categorization.base import CategorizationResult, CategorizationSource
    from app.core.config import Settings

    class _LocalProvider:
        name = CategorizationSource.local_model.value

        def __init__(self) -> None:
            self.model: object = None

        async def categorize(self, text: str, **kw: object) -> CategorizationResult:
            self.model = kw.get("model")
            return CategorizationResult(
                category=DocumentCategory.invoice.value,
                confidence=0.9,
                reason="configured model used",
                source=CategorizationSource.local_model.value,
                model=str(self.model),
            )

        async def healthcheck(self) -> bool:
            return True

    doc = await _insert_document(db_session, extracted_text="Invoice text.")
    provider = _LocalProvider()
    settings = Settings(
        categorization_provider="ollama",
        categorization_model="custom-categorizer:latest",
    )

    result = await categorize_document(db_session, doc.id, provider=provider, settings=settings)  # type: ignore[arg-type]

    assert provider.model == "custom-categorizer:latest"
    assert result.category_model == "custom-categorizer:latest"


# ---------------------------------------------------------------------------
# Manual category protection
# ---------------------------------------------------------------------------


async def test_categorize_document_does_not_overwrite_manual_category(
    db_session: AsyncSession,
) -> None:
    """A manually assigned category must NOT be overwritten by the service."""
    doc = await _insert_document(
        db_session,
        extracted_text="Invoice text.",
        category=DocumentCategory.contract.value,
        category_source="manual",
        category_status="ready",
    )
    provider = MockCategorizationProvider(category=DocumentCategory.invoice.value)

    result = await categorize_document(db_session, doc.id, provider=provider)

    assert result.skipped is True
    assert result.category == DocumentCategory.contract.value
    assert result.category_source == "manual"


async def test_categorize_document_force_overwrites_manual_category(
    db_session: AsyncSession,
) -> None:
    """With force=True, a manually assigned category must be overwritten."""
    doc = await _insert_document(
        db_session,
        extracted_text="Invoice text.",
        category=DocumentCategory.contract.value,
        category_source="manual",
        category_status="ready",
    )
    provider = MockCategorizationProvider(category=DocumentCategory.invoice.value, confidence=0.9)

    result = await categorize_document(db_session, doc.id, provider=provider, force=True)

    assert result.skipped is False
    assert result.category == DocumentCategory.invoice.value


# ---------------------------------------------------------------------------
# Category survives DB session restart (durability)
# ---------------------------------------------------------------------------


async def test_category_survives_db_session_restart(db_session: AsyncSession) -> None:
    """Category must be readable after expiring all objects."""
    doc = await _insert_document(db_session, extracted_text="Contract text.")
    doc_id = doc.id
    provider = MockCategorizationProvider(category=DocumentCategory.contract.value, confidence=0.88)

    await categorize_document(db_session, doc_id, provider=provider)
    await db_session.commit()

    db_session.expire_all()

    refreshed = await db_session.get(Document, doc_id)
    assert refreshed is not None
    assert refreshed.category == DocumentCategory.contract.value
    assert refreshed.category_status == "ready"
