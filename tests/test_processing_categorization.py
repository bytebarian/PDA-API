"""Integration tests for the categorization pipeline stage.

These tests verify that the category_assignment stage runs correctly within
the full processing orchestrator, including stage history recording,
non-critical failure handling, and document API response exposure.

All tests use the mock categorization provider; no live Ollama instance or
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
    extracted_text: str | None = "Pipeline document text. Invoice VAT total amount.",
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
# Patch providers for tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch both summarization and categorization providers with mocks."""
    from app.adapters.categorization.mock import MockCategorizationProvider
    from app.adapters.summarization.mock import MockSummarizationProvider
    import app.services.categorization_service as cat_svc
    import app.services.summarization_service as sum_svc

    monkeypatch.setattr(
        sum_svc,
        "get_summarization_provider",
        lambda **_kwargs: MockSummarizationProvider(summary="Integration test summary."),
    )
    monkeypatch.setattr(
        cat_svc,
        "get_categorization_provider",
        lambda **_kwargs: MockCategorizationProvider(category="invoice", confidence=0.9),
    )


# ---------------------------------------------------------------------------
# Stage flow includes category_assignment
# ---------------------------------------------------------------------------


async def test_process_job_stage_flow_includes_category_assignment(
    db_session: AsyncSession,
) -> None:
    """category_assignment must appear in the completed stage history."""
    doc = await _insert_document(db_session)
    job = await _insert_job(db_session, doc.id)

    processed = await process_job(db_session, job.id)

    stages = [(e["stage"], e["status"]) for e in processed.stage_history_jsonb]
    assert ("category_assignment", "processing") in stages
    assert ("category_assignment", "completed") in stages


async def test_process_job_category_persisted_after_pipeline(
    db_session: AsyncSession,
) -> None:
    """After a successful pipeline run the document must have a persisted category."""
    doc = await _insert_document(db_session, extracted_text="Invoice total amount VAT.")
    job = await _insert_job(db_session, doc.id)

    await process_job(db_session, job.id)

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.category == "invoice"
    assert refreshed.category_status == "ready"
    assert refreshed.category_source is not None
    assert refreshed.category_generated_at is not None


async def test_process_job_category_survives_db_session_restart(
    db_session: AsyncSession,
) -> None:
    """Category must survive after expiring all objects and re-fetching from DB."""
    doc = await _insert_document(db_session, extracted_text="Invoice text.")
    doc_id = doc.id
    job = await _insert_job(db_session, doc.id)

    await process_job(db_session, job.id)
    await db_session.commit()

    db_session.expire_all()

    refreshed = await db_session.get(Document, doc_id)
    assert refreshed is not None
    assert refreshed.category == "invoice"
    assert refreshed.category_status == "ready"


# ---------------------------------------------------------------------------
# Non-critical failure: categorization fails but document becomes ready
# ---------------------------------------------------------------------------


async def test_process_job_categorization_failure_does_not_fail_document(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When categorization fails, the document must still become ready (non-critical)."""
    from app.adapters.categorization.base import CategorizationProviderUnavailableError
    from app.adapters.categorization.mock import MockCategorizationProvider
    import app.services.categorization_service as cat_svc

    failing_provider = MockCategorizationProvider(
        error=CategorizationProviderUnavailableError("Ollama unavailable")
    )
    monkeypatch.setattr(
        cat_svc,
        "get_categorization_provider",
        lambda **_kwargs: failing_provider,
    )

    doc = await _insert_document(db_session, extracted_text="Some document text.")
    job = await _insert_job(db_session, doc.id)

    processed = await process_job(db_session, job.id)

    assert processed.status == "ready"
    assert processed.stage == "completed"

    refreshed = await db_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.status == "ready"
    assert refreshed.category_status == "failed"
    assert refreshed.category_error is not None


async def test_process_job_categorization_failure_recorded_in_stage_history(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A categorization failure must be captured in stage history details."""
    from app.adapters.categorization.base import CategorizationProviderUnavailableError
    from app.adapters.categorization.mock import MockCategorizationProvider
    import app.services.categorization_service as cat_svc

    failing_provider = MockCategorizationProvider(
        error=CategorizationProviderUnavailableError("Connection refused")
    )
    monkeypatch.setattr(
        cat_svc,
        "get_categorization_provider",
        lambda **_kwargs: failing_provider,
    )

    doc = await _insert_document(db_session)
    job = await _insert_job(db_session, doc.id)

    processed = await process_job(db_session, job.id)

    cat_entry = next(
        (
            e
            for e in processed.stage_history_jsonb
            if e["stage"] == "category_assignment" and e["status"] == "completed"
        ),
        None,
    )
    assert cat_entry is not None
    assert cat_entry["details"]["category_status"] == "failed"


# ---------------------------------------------------------------------------
# Skipped (no text) path
# ---------------------------------------------------------------------------


async def test_process_job_no_text_produces_skipped_category(
    db_session: AsyncSession,
) -> None:
    """A document with no extracted text must produce a skipped category via service."""
    from app.adapters.categorization.mock import MockCategorizationProvider
    from app.services.categorization_service import categorize_document

    doc = Document(
        filename="no-text.pdf",
        status="ready",
        extracted_text=None,
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)
    doc_id = doc.id

    provider = MockCategorizationProvider()
    result = await categorize_document(db_session, doc_id, provider=provider)
    await db_session.commit()

    refreshed = await db_session.get(Document, doc_id)
    assert refreshed is not None
    assert refreshed.category_status in {"skipped", "failed"}
    assert result.skipped is True


# ---------------------------------------------------------------------------
# Reprocess refreshes stale category
# ---------------------------------------------------------------------------


async def test_reprocess_updates_stale_category(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second processing run must replace the stale category with a fresh one."""
    from app.adapters.categorization.mock import MockCategorizationProvider
    import app.services.categorization_service as cat_svc

    # First run
    monkeypatch.setattr(
        cat_svc,
        "get_categorization_provider",
        lambda **_kwargs: MockCategorizationProvider(category="invoice", confidence=0.9),
    )

    doc = await _insert_document(db_session, extracted_text="Original invoice text.")
    job1 = await _insert_job(db_session, doc.id)
    await process_job(db_session, job1.id)

    first_refreshed = await db_session.get(Document, doc.id)
    assert first_refreshed is not None
    assert first_refreshed.category == "invoice"

    # Second run with a different category result
    first_refreshed.status = "awaiting"
    first_refreshed.extracted_text = "Updated contract text."
    monkeypatch.setattr(
        cat_svc,
        "get_categorization_provider",
        lambda **_kwargs: MockCategorizationProvider(category="contract", confidence=0.88),
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
    assert final.category == "contract"
    assert final.category_status == "ready"


# ---------------------------------------------------------------------------
# API response exposes category fields
# ---------------------------------------------------------------------------


async def test_document_list_exposes_category_fields(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Document list response must include category and category_status."""
    doc = Document(
        filename="api-category-test.pdf",
        status="ready",
        category="invoice",
        category_status="ready",
        category_source="rules",
        category_confidence=0.87,
    )
    db_session.add(doc)
    await db_session.commit()

    response = client.get("/documents")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) >= 1
    item = next(i for i in items if i["filename"] == "api-category-test.pdf")
    assert item["category"] == "invoice"
    assert item["categoryStatus"] == "ready"
    assert item["categorySource"] == "rules"
    assert item["categoryConfidence"] == pytest.approx(0.87, abs=1e-4)


async def test_document_detail_exposes_category_fields(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Document detail response must include all category-related fields."""
    doc = Document(
        filename="detail-category.pdf",
        status="ready",
        category="contract",
        category_status="ready",
        category_source="rules",
        category_confidence=0.75,
        category_reason="matched 3/11 keywords: contract, termination, clause",
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)
    doc_id = doc.id

    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "contract"
    assert body["categoryStatus"] == "ready"
    assert body["categorySource"] == "rules"
    assert body["categoryConfidence"] == pytest.approx(0.75, abs=1e-4)
    assert body["categoryReason"] is not None
    assert "categoryGeneratedAt" in body
    assert "categoryError" in body


# ---------------------------------------------------------------------------
# No raw text in logs (privacy guard via caplog)
# ---------------------------------------------------------------------------


async def test_no_raw_document_text_in_logs(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """Raw document text must not appear in categorization service log output."""
    from app.adapters.categorization.mock import MockCategorizationProvider as _MockProvider
    from app.services.categorization_service import categorize_document as _cat_doc

    sensitive_text = "SENSITIVE_INVOICE_CONTENT_MUST_NOT_BE_LOGGED VAT total amount"
    doc = await _insert_document(db_session, extracted_text=sensitive_text)
    provider = _MockProvider(category="invoice", confidence=0.9)

    import logging

    with caplog.at_level(logging.DEBUG, logger="app.services.categorization_service"):
        await _cat_doc(db_session, doc.id, provider=provider)

    for record in caplog.records:
        assert sensitive_text not in record.getMessage(), (
            f"Raw document text found in log: {record.getMessage()}"
        )
