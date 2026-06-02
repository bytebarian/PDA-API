"""Unit tests for the summarization service.

All tests use MockSummarizationProvider to avoid requiring a live Ollama
instance or any internet access.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.summarization.base import (
    SummarizationProviderUnavailableError,
)
from app.adapters.summarization.mock import MockSummarizationProvider
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.summarization_service import (
    DocumentSummaryResult,
    SummarizationDocumentNotFoundError,
    summarize_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_document(
    db: AsyncSession,
    *,
    filename: str = "test.pdf",
    extracted_text: str | None = "Sample document text.",
    summary_status: str = "pending",
) -> Document:
    doc = Document(
        filename=filename,
        status="awaiting",
        extracted_text=extracted_text,
        summary_status=summary_status,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


# ---------------------------------------------------------------------------
# Basic summarization
# ---------------------------------------------------------------------------


async def test_summarize_document_returns_typed_result(db_session: AsyncSession) -> None:
    """summarize_document must return a DocumentSummaryResult."""
    doc = await _insert_document(db_session)
    provider = MockSummarizationProvider(summary="This is a mock summary.")

    result = await summarize_document(db_session, doc.id, provider=provider)

    assert isinstance(result, DocumentSummaryResult)
    assert result.document_id == doc.id
    assert result.summary == "This is a mock summary."
    assert result.summary_status == "ready"
    assert result.skipped is False


async def test_summarize_document_persists_summary(db_session: AsyncSession) -> None:
    """Generated summary must be persisted in the Document row."""
    doc = await _insert_document(db_session, extracted_text="Important contract text.")
    provider = MockSummarizationProvider(summary="Contract summary.")

    await summarize_document(db_session, doc.id, provider=provider)
    await db_session.commit()

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.summary == "Contract summary."
    assert refreshed.summary_status == "ready"
    assert refreshed.summary_model is not None
    assert refreshed.summary_generated_at is not None
    assert refreshed.summary_error is None


async def test_summarize_document_persists_summary_model(db_session: AsyncSession) -> None:
    """summary_model must be saved from the provider result."""
    doc = await _insert_document(db_session)
    provider = MockSummarizationProvider(summary="A summary.")

    from app.core.config import Settings

    settings = Settings(
        summarization_model="llama3.2:3b",
        summarization_provider="mock",
    )
    await summarize_document(db_session, doc.id, provider=provider, settings=settings)
    await db_session.commit()

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.summary_model == "llama3.2:3b"


async def test_summarize_document_persists_generated_at(db_session: AsyncSession) -> None:
    """summary_generated_at must be a timezone-aware datetime after summarization."""
    doc = await _insert_document(db_session)
    provider = MockSummarizationProvider(summary="Summary text.")

    before = datetime.now(timezone.utc)
    await summarize_document(db_session, doc.id, provider=provider)
    after = datetime.now(timezone.utc)
    await db_session.commit()

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.summary_generated_at is not None
    ts = refreshed.summary_generated_at
    if ts.tzinfo is None:
        # SQLite may strip tzinfo; compare naively
        ts = ts.replace(tzinfo=timezone.utc)
    assert before <= ts <= after


# ---------------------------------------------------------------------------
# Input prompt / truncation behavior
# ---------------------------------------------------------------------------


async def test_summarize_document_truncates_long_input(db_session: AsyncSession) -> None:
    """Input text longer than summary_max_input_chars must be truncated."""
    long_text = "A" * 20000
    doc = await _insert_document(db_session, extracted_text=long_text)
    provider = MockSummarizationProvider(summary="Truncated summary.")

    from app.core.config import Settings

    settings = Settings(summary_max_input_chars=500)
    # Capture the text that the provider receives.
    received: list[str] = []

    original_summarize = provider.summarize

    async def recording_summarize(text: str, *, model: str, max_output_chars: int = 600):  # type: ignore[override]
        received.append(text)
        return await original_summarize(text, model=model, max_output_chars=max_output_chars)

    provider.summarize = recording_summarize  # type: ignore[method-assign]

    await summarize_document(db_session, doc.id, provider=provider, settings=settings)

    assert len(received) == 1
    assert len(received[0]) <= 500


async def test_summarize_document_enforces_max_output_chars(db_session: AsyncSession) -> None:
    """Summary must be trimmed to summary_max_output_chars."""
    long_summary = "word " * 300  # 1500 chars
    doc = await _insert_document(db_session)
    provider = MockSummarizationProvider(summary=long_summary)

    from app.core.config import Settings

    settings = Settings(summary_max_output_chars=100)
    result = await summarize_document(db_session, doc.id, provider=provider, settings=settings)

    assert result.summary is not None
    assert len(result.summary) <= 100


# ---------------------------------------------------------------------------
# Empty / missing text handling
# ---------------------------------------------------------------------------


async def test_summarize_document_empty_text_produces_skipped_result(
    db_session: AsyncSession,
) -> None:
    """A document with empty extracted_text must produce a skipped result."""
    doc = await _insert_document(db_session, extracted_text="")
    provider = MockSummarizationProvider()

    result = await summarize_document(db_session, doc.id, provider=provider)

    assert result.skipped is True
    assert result.summary_status == "skipped"
    assert result.summary is not None  # fallback message
    assert "unavailable" in (result.summary or "").lower()


async def test_summarize_document_none_text_produces_skipped_result(
    db_session: AsyncSession,
) -> None:
    """A document with None extracted_text must produce a skipped result."""
    doc = await _insert_document(db_session, extracted_text=None)
    provider = MockSummarizationProvider()

    result = await summarize_document(db_session, doc.id, provider=provider)

    assert result.skipped is True
    assert result.summary_status == "skipped"


async def test_summarize_document_empty_text_persists_skipped_status(
    db_session: AsyncSession,
) -> None:
    """summary_status must be 'skipped' in the DB when no text is available."""
    doc = await _insert_document(db_session, extracted_text="")
    provider = MockSummarizationProvider()

    await summarize_document(db_session, doc.id, provider=provider)
    await db_session.commit()

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.summary_status == "skipped"
    assert refreshed.summary_error is not None


async def test_summarize_document_uses_chunk_text_when_extracted_missing(
    db_session: AsyncSession,
) -> None:
    """When extracted_text is absent, chunks must be used as fallback source."""
    doc = await _insert_document(db_session, extracted_text=None)

    # Add chunks to the document.
    chunk1 = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="First chunk text.",
        source_start_offset=0,
    )
    chunk2 = DocumentChunk(
        document_id=doc.id,
        chunk_index=1,
        content="Second chunk text.",
        source_start_offset=17,
    )
    db_session.add_all([chunk1, chunk2])
    await db_session.commit()

    received_texts: list[str] = []
    provider = MockSummarizationProvider(summary="Chunk-based summary.")
    original = provider.summarize

    async def capture(text: str, *, model: str, max_output_chars: int = 600):  # type: ignore[override]
        received_texts.append(text)
        return await original(text, model=model, max_output_chars=max_output_chars)

    provider.summarize = capture  # type: ignore[method-assign]

    result = await summarize_document(db_session, doc.id, provider=provider)

    assert result.skipped is False
    assert result.summary_status == "ready"
    assert len(received_texts) == 1
    assert "First chunk text." in received_texts[0]


# ---------------------------------------------------------------------------
# Provider error handling
# ---------------------------------------------------------------------------


async def test_summarize_document_provider_error_persists_failed_status(
    db_session: AsyncSession,
) -> None:
    """A provider error must set summary_status='failed' and persist the reason."""
    doc = await _insert_document(db_session)
    provider = MockSummarizationProvider(
        error=SummarizationProviderUnavailableError("Ollama is down")
    )

    result = await summarize_document(db_session, doc.id, provider=provider)
    await db_session.commit()

    assert result.summary_status == "failed"
    assert result.summary is None
    assert "Ollama is down" in (result.summary_error or "")

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.summary_status == "failed"
    assert refreshed.summary_error is not None
    assert "Ollama is down" in refreshed.summary_error


async def test_summarize_document_provider_error_does_not_raise(
    db_session: AsyncSession,
) -> None:
    """SummarizationError from a provider must not propagate; it must be captured."""
    doc = await _insert_document(db_session)
    provider = MockSummarizationProvider(
        error=SummarizationProviderUnavailableError("Network unreachable")
    )

    # Must not raise
    result = await summarize_document(db_session, doc.id, provider=provider)
    assert result.summary_status == "failed"


async def test_summarize_document_provider_error_preserves_extracted_text(
    db_session: AsyncSession,
) -> None:
    """A summarization failure must not corrupt Document.extracted_text."""
    original_text = "Original extracted text that must survive failure."
    doc = await _insert_document(db_session, extracted_text=original_text)
    provider = MockSummarizationProvider(
        error=SummarizationProviderUnavailableError("Timeout")
    )

    await summarize_document(db_session, doc.id, provider=provider)
    await db_session.commit()

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.extracted_text == original_text


# ---------------------------------------------------------------------------
# Reprocessing behavior
# ---------------------------------------------------------------------------


async def test_summarize_document_reprocess_updates_stale_summary(
    db_session: AsyncSession,
) -> None:
    """Calling summarize_document again must replace the previous summary."""
    doc = await _insert_document(db_session, extracted_text="First content.")
    first_provider = MockSummarizationProvider(summary="First summary.")
    await summarize_document(db_session, doc.id, provider=first_provider)
    await db_session.commit()

    # Update extracted_text to simulate reprocessing.
    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    refreshed.extracted_text = "Updated content."
    await db_session.commit()

    second_provider = MockSummarizationProvider(summary="Updated summary.")
    result = await summarize_document(db_session, doc.id, provider=second_provider)
    await db_session.commit()

    assert result.summary == "Updated summary."

    final = await db_session.get(Document, doc.id)
    assert final is not None
    assert final.summary == "Updated summary."
    assert final.summary_status == "ready"


# ---------------------------------------------------------------------------
# Document not found
# ---------------------------------------------------------------------------


async def test_summarize_document_missing_document_raises(db_session: AsyncSession) -> None:
    """summarize_document must raise SummarizationDocumentNotFoundError for unknown ids."""
    provider = MockSummarizationProvider()

    with pytest.raises(SummarizationDocumentNotFoundError):
        await summarize_document(db_session, uuid.uuid4(), provider=provider)
