"""Tests for POST /documents/{document_id}/reprocess."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app as fastapi_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.processing_job import ProcessingJob

import app.models  # noqa: F401 – register all ORM models


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """In-memory SQLite session with all tables created."""
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


async def _insert_document(
    db: AsyncSession,
    *,
    filename: str = "doc.pdf",
    path: str = "stored/doc.pdf",
    checksum_sha256: str | None = "abc123",
    size: int = 123,
    metadata_jsonb: dict[str, str] | None = None,
    status: str = "ready",
) -> Document:
    document = Document(
        filename=filename,
        mime_type="application/pdf",
        status=status,
        path=path,
        size=size,
        checksum_sha256=checksum_sha256,
        metadata_jsonb=metadata_jsonb,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document


async def _insert_chunk(db: AsyncSession, document_id: uuid.UUID) -> DocumentChunk:
    chunk = DocumentChunk(document_id=document_id, chunk_index=0, content="existing text")
    db.add(chunk)
    await db.commit()
    await db.refresh(chunk)
    return chunk


async def _insert_job(db: AsyncSession, document_id: uuid.UUID) -> ProcessingJob:
    job = ProcessingJob(document_id=document_id, status="ready", stage="completed")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_reprocess_creates_new_job_and_resets_document_status(
    client: TestClient, db_session: AsyncSession
) -> None:
    document = await _insert_document(db_session, status="ready")

    response = client.post(
        f"/documents/{document.id}/reprocess",
        json={"force": True, "reason": "manual retry"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["document_id"] == str(document.id)
    assert body["document_status"] == "awaiting"
    assert body["job_status"] == "awaiting"
    assert body["job_stage"] == "queued"

    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    assert refreshed.status == "awaiting"

    new_job = await db_session.get(ProcessingJob, uuid.UUID(body["job_id"]))
    assert new_job is not None
    assert new_job.document_id == document.id
    assert new_job.status == "awaiting"
    assert new_job.stage == "queued"
    assert new_job.stage_history_jsonb == [{"stage": "queued", "reason": "manual retry"}]


def test_reprocess_missing_document_returns_404(client: TestClient) -> None:
    response = client.post(f"/documents/{uuid.uuid4()}/reprocess")
    assert response.status_code == 404


async def test_reprocess_does_not_mutate_file_fields_or_existing_chunks(
    client: TestClient, db_session: AsyncSession
) -> None:
    document = await _insert_document(
        db_session,
        filename="report.pdf",
        path="stored/report.pdf",
        checksum_sha256="digest",
        size=999,
        metadata_jsonb={"source": "upload"},
        status="failed",
    )
    await _insert_chunk(db_session, document.id)

    response = client.post(f"/documents/{document.id}/reprocess")
    assert response.status_code == 201

    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    assert refreshed.filename == "report.pdf"
    assert refreshed.path == "stored/report.pdf"
    assert refreshed.checksum_sha256 == "digest"
    assert refreshed.size == 999
    assert refreshed.metadata_jsonb == {"source": "upload"}
    assert refreshed.status == "awaiting"

    chunks = (
        await db_session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
    ).scalars().all()
    assert len(chunks) == 1


async def test_reprocess_can_be_called_repeatedly_and_creates_distinct_jobs(
    client: TestClient, db_session: AsyncSession
) -> None:
    document = await _insert_document(db_session, status="ready")
    existing_job = await _insert_job(db_session, document.id)

    first = client.post(f"/documents/{document.id}/reprocess")
    second = client.post(f"/documents/{document.id}/reprocess")

    assert first.status_code == 201
    assert second.status_code == 201

    first_job_id = uuid.UUID(first.json()["job_id"])
    second_job_id = uuid.UUID(second.json()["job_id"])
    assert first_job_id != second_job_id
    assert first_job_id != existing_job.id
    assert second_job_id != existing_job.id

    jobs = (
        await db_session.execute(
            select(ProcessingJob)
            .where(ProcessingJob.document_id == document.id)
            .order_by(ProcessingJob.created_at.asc(), ProcessingJob.id.asc())
        )
    ).scalars().all()
    assert len(jobs) == 3
