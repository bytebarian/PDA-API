"""Integration tests for the summarization pipeline stage.

These tests verify that the summary_generation stage runs correctly within the
full processing orchestrator, including stage history recording, non-critical
failure handling, and document API response exposure.

All tests use the mock summarization provider; no live Ollama instance or
internet access is required.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app as fastapi_app

import app.models  # noqa: F401

from app.models.document import Document
from app.models.processing_job import ProcessingJob
from app.services.processing_orchestrator import process_job


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_session(  # type: ignore[override]
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[AsyncSession, None]:
    """In-memory SQLite session with embedding provider stubbed out."""
    monkeypatch.setenv("PDA_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("PDA_EMBEDDING_MODEL", "test-fake-embedding-model")
    monkeypatch.setenv("PDA_EMBEDDING_DIMENSIONS", "1536")
    get_settings.cache_clear()

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()
    get_settings.cache_clear()


@pytest.fixture
def client(tmp_path: Path, db_session: AsyncSession) -> Generator[TestClient, None, None]:
    """TestClient with in-memory DB and temp storage path."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    fastapi_app.dependency_overrides[get_db] = override_get_db

    test_settings = Settings(
        storage_path=tmp_path,  # type: ignore[arg-type]
        _env_file=None,  # type: ignore[call-arg]
    )
    fastapi_app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(fastapi_app) as c:
        yield c

    fastapi_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_document(
    db: AsyncSession,
    *,
    filename: str = "pipeline.pdf",
    extracted_text: str | None = "Pipeline document text.",
    status: str = "awaiting",
) -> Document:
    doc = Document(
        filename=filename,
        status=status,
        extracted_text=extracted_text,
    )
    db.add(doc)
    await db.flush()
    return doc


async def _insert_job(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    stage: str = "queued",
    status: str = "awaiting",
) -> ProcessingJob:
    job = ProcessingJob(
        document_id=document_id,
        status=status,
        stage=stage,
    )
    db.add(job)
    await db.commit()
    return job


# ---------------------------------------------------------------------------
# Fixture: mock summarization provider patched into the service
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_summarization_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the summarization provider resolver with the mock provider."""
    from app.adapters.summarization.mock import MockSummarizationProvider
    import app.services.summarization_service as svc

    monkeypatch.setattr(
        svc,
        "get_summarization_provider",
        lambda **_kwargs: MockSummarizationProvider(summary="Integration test summary."),
    )


# ---------------------------------------------------------------------------
# Full pipeline stage flow includes summary_generation
# ---------------------------------------------------------------------------


async def test_process_job_stage_flow_includes_summary_generation(
    db_session: AsyncSession,
) -> None:
    """summary_generation must appear in the completed stage history."""
    doc = await _insert_document(db_session)
    job = await _insert_job(db_session, doc.id)

    processed = await process_job(db_session, job.id)

    stages = [(e["stage"], e["status"]) for e in processed.stage_history_jsonb]
    assert ("summary_generation", "processing") in stages
    assert ("summary_generation", "completed") in stages


async def test_process_job_summary_persisted_after_pipeline(
    db_session: AsyncSession,
) -> None:
    """After a successful pipeline run the document must have a persisted summary."""
    doc = await _insert_document(db_session, extracted_text="Important contract.")
    job = await _insert_job(db_session, doc.id)

    await process_job(db_session, job.id)

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.summary == "Integration test summary."
    assert refreshed.summary_status == "ready"
    assert refreshed.summary_model is not None
    assert refreshed.summary_generated_at is not None


async def test_process_job_summary_survives_db_session_restart(
    db_session: AsyncSession,
) -> None:
    """Summary must survive after expiring all objects and re-fetching from DB."""
    doc = await _insert_document(db_session, extracted_text="Contract text.")
    doc_id = doc.id  # Save before expire_all
    job = await _insert_job(db_session, doc.id)

    await process_job(db_session, job.id)
    await db_session.commit()

    # Expire all loaded objects to force a real DB read.
    db_session.expire_all()

    refreshed = await db_session.get(Document, doc_id)
    assert refreshed is not None
    assert refreshed.summary == "Integration test summary."
    assert refreshed.summary_status == "ready"


# ---------------------------------------------------------------------------
# Non-critical failure: summarization fails but document becomes ready
# ---------------------------------------------------------------------------


async def test_process_job_summary_failure_does_not_fail_document(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When summarization fails, the document must still become ready (non-critical)."""
    from app.adapters.summarization.base import SummarizationProviderUnavailableError
    from app.adapters.summarization.mock import MockSummarizationProvider
    import app.services.summarization_service as svc

    failing_provider = MockSummarizationProvider(
        error=SummarizationProviderUnavailableError("Ollama unavailable")
    )
    monkeypatch.setattr(
        svc,
        "get_summarization_provider",
        lambda **_kwargs: failing_provider,
    )

    doc = await _insert_document(db_session, extracted_text="Some document text.")
    job = await _insert_job(db_session, doc.id)

    # The pipeline should complete without raising even though summarization failed.
    processed = await process_job(db_session, job.id)

    assert processed.status == "ready"
    assert processed.stage == "completed"

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.status == "ready"
    assert refreshed.summary_status == "failed"
    assert refreshed.summary_error is not None
    assert refreshed.summary is None


async def test_process_job_summary_failure_recorded_in_stage_history(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A summarization failure must be captured in stage history details."""
    from app.adapters.summarization.base import SummarizationProviderUnavailableError
    from app.adapters.summarization.mock import MockSummarizationProvider
    import app.services.summarization_service as svc

    failing_provider = MockSummarizationProvider(
        error=SummarizationProviderUnavailableError("Connection refused")
    )
    monkeypatch.setattr(
        svc,
        "get_summarization_provider",
        lambda **_kwargs: failing_provider,
    )

    doc = await _insert_document(db_session)
    job = await _insert_job(db_session, doc.id)

    processed = await process_job(db_session, job.id)

    summary_entry = next(
        (
            e
            for e in processed.stage_history_jsonb
            if e["stage"] == "summary_generation" and e["status"] == "completed"
        ),
        None,
    )
    assert summary_entry is not None
    assert summary_entry["details"]["summary_status"] == "failed"


# ---------------------------------------------------------------------------
# Skipped (no text) path
# ---------------------------------------------------------------------------


async def test_process_job_no_text_produces_skipped_summary(
    db_session: AsyncSession,
) -> None:
    """A document with no extracted text produces a skipped summary directly via service."""
    from app.adapters.summarization.mock import MockSummarizationProvider
    from app.services.summarization_service import summarize_document

    # Insert a document that already has all pipeline stages done except summarization.
    doc = Document(
        filename="no-text.pdf",
        status="ready",
        extracted_text=None,
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)
    doc_id = doc.id

    provider = MockSummarizationProvider()
    result = await summarize_document(db_session, doc_id, provider=provider)
    await db_session.commit()

    refreshed = await db_session.get(Document, doc_id)
    assert refreshed is not None
    # Either 'skipped' (no text) or 'failed' is acceptable when no text is available.
    assert refreshed.summary_status in {"skipped", "failed"}
    assert result.skipped is True


# ---------------------------------------------------------------------------
# Reprocess refreshes stale summary
# ---------------------------------------------------------------------------


async def test_reprocess_updates_stale_summary(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second processing run must replace the stale summary with a fresh one."""
    from app.adapters.summarization.mock import MockSummarizationProvider
    import app.services.summarization_service as svc

    # First run: inject first-summary provider.
    monkeypatch.setattr(
        svc,
        "get_summarization_provider",
        lambda **_kwargs: MockSummarizationProvider(summary="First summary."),
    )

    doc = await _insert_document(db_session, extracted_text="Original text.")
    job1 = await _insert_job(db_session, doc.id)
    await process_job(db_session, job1.id)

    first_refreshed = await db_session.get(Document, doc.id)
    assert first_refreshed is not None
    assert first_refreshed.summary == "First summary."

    # Second run: update text and inject second-summary provider.
    first_refreshed.status = "awaiting"
    first_refreshed.extracted_text = "Updated document text."
    monkeypatch.setattr(
        svc,
        "get_summarization_provider",
        lambda **_kwargs: MockSummarizationProvider(summary="Updated summary."),
    )

    job2 = ProcessingJob(
        document_id=doc.id,
        status="awaiting",
        stage="queued",
    )
    db_session.add(job2)
    await db_session.commit()

    await process_job(db_session, job2.id)

    final = await db_session.get(Document, doc.id)
    assert final is not None
    assert final.summary == "Updated summary."
    assert final.summary_status == "ready"


# ---------------------------------------------------------------------------
# API response exposes summary fields
# ---------------------------------------------------------------------------


async def test_document_list_exposes_summary_fields(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Document list response must include summary and summary_status."""
    doc = Document(
        filename="api-summary-test.pdf",
        status="ready",
        summary="Exposed summary text.",
        summary_status="ready",
    )
    db_session.add(doc)
    await db_session.commit()

    response = client.get("/documents")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) >= 1
    item = next(i for i in items if i["filename"] == "api-summary-test.pdf")
    assert item["summary"] == "Exposed summary text."
    assert item["summary_status"] == "ready"


async def test_document_detail_exposes_summary_fields(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Document detail response must include all summary-related fields."""
    doc = Document(
        filename="detail-summary.pdf",
        status="ready",
        summary="Detailed summary.",
        summary_status="ready",
        summary_model="llama3.2:3b",
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)
    doc_id = doc.id

    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "Detailed summary."
    assert body["summary_status"] == "ready"
    assert body["summary_model"] == "llama3.2:3b"
    assert "summary_generated_at" in body
    assert "summary_error" in body
