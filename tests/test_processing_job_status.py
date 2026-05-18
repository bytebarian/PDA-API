"""Tests for GET /jobs/{job_id}."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from app.models.processing_job import ProcessingJob

import app.models  # noqa: F401 – register all ORM models


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """In-memory SQLite AsyncSession with all tables created."""
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
    status: str = "ready",
) -> Document:
    document = Document(
        filename=filename,
        mime_type="application/pdf",
        status=status,
        path=f"stored/{filename}",
        size=123,
        checksum_sha256="abc123",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document


async def _insert_job(
    db: AsyncSession,
    *,
    document_id: uuid.UUID,
    status: str = "processing",
    stage: str = "ocr",
    attempt_count: int = 1,
    max_attempts: int = 3,
    error_message: str | None = None,
    error_details_jsonb: dict[str, Any] | None = None,
    stage_history_jsonb: list[Any] | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> ProcessingJob:
    job = ProcessingJob(
        document_id=document_id,
        status=status,
        stage=stage,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        error_message=error_message,
        error_details_jsonb=error_details_jsonb,
        stage_history_jsonb=stage_history_jsonb or [],
        started_at=started_at,
        completed_at=completed_at,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_get_job_status_returns_job_detail(client: TestClient, db_session: AsyncSession) -> None:
    document = await _insert_document(db_session)
    started_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    completed_at = datetime(2026, 1, 2, 4, 5, 6, tzinfo=timezone.utc)
    job = await _insert_job(
        db_session,
        document_id=document.id,
        status="failed",
        stage="failed",
        attempt_count=2,
        max_attempts=5,
        error_message="ocr failed",
        error_details_jsonb={"code": "OCR_TIMEOUT", "retryable": True},
        stage_history_jsonb=[
            {"stage": "queued"},
            {"stage": "ocr", "status": "processing"},
            {"stage": "failed", "reason": "timeout"},
        ],
        started_at=started_at,
        completed_at=completed_at,
    )

    response = client.get(f"/jobs/{job.id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(job.id)
    assert body["document_id"] == str(document.id)
    assert body["status"] == "failed"
    assert body["stage"] == "failed"
    assert body["attempt_count"] == 2
    assert body["max_attempts"] == 5
    assert body["error_message"] == "ocr failed"
    assert body["error_details_jsonb"] == {"code": "OCR_TIMEOUT", "retryable": True}
    assert body["stage_history_jsonb"] == [
        {"stage": "queued"},
        {"stage": "ocr", "status": "processing"},
        {"stage": "failed", "reason": "timeout"},
    ]
    assert body["started_at"] is not None
    assert body["completed_at"] is not None
    assert body["created_at"] is not None
    assert body["updated_at"] is not None


def test_get_job_status_missing_returns_404(client: TestClient) -> None:
    response = client.get(f"/jobs/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Processing job not found"


async def test_get_job_status_does_not_mutate_job_or_document(
    client: TestClient, db_session: AsyncSession
) -> None:
    document = await _insert_document(db_session, status="ready")
    job = await _insert_job(
        db_session,
        document_id=document.id,
        status="ready",
        stage="completed",
        attempt_count=1,
        max_attempts=3,
        error_message=None,
        error_details_jsonb={"ok": True},
        stage_history_jsonb=[{"stage": "completed"}],
    )

    before_document_status = document.status
    before_job_snapshot = {
        "status": job.status,
        "stage": job.stage,
        "attempt_count": job.attempt_count,
        "max_attempts": job.max_attempts,
        "error_message": job.error_message,
        "error_details_jsonb": job.error_details_jsonb,
        "stage_history_jsonb": job.stage_history_jsonb,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "updated_at": job.updated_at,
    }

    response = client.get(f"/jobs/{job.id}")
    assert response.status_code == 200

    await db_session.refresh(document)
    await db_session.refresh(job)

    assert document.status == before_document_status
    assert job.status == before_job_snapshot["status"]
    assert job.stage == before_job_snapshot["stage"]
    assert job.attempt_count == before_job_snapshot["attempt_count"]
    assert job.max_attempts == before_job_snapshot["max_attempts"]
    assert job.error_message == before_job_snapshot["error_message"]
    assert job.error_details_jsonb == before_job_snapshot["error_details_jsonb"]
    assert job.stage_history_jsonb == before_job_snapshot["stage_history_jsonb"]
    assert job.started_at == before_job_snapshot["started_at"]
    assert job.completed_at == before_job_snapshot["completed_at"]
    assert job.updated_at == before_job_snapshot["updated_at"]

    jobs = (await db_session.execute(select(ProcessingJob))).scalars().all()
    assert len(jobs) == 1
