"""Tests for DELETE /documents/{document_id}."""

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

import app.models  # noqa: F401 – register all ORM models

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.processing_job import ProcessingJob


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
    path: str | None,
) -> Document:
    """Insert a minimal Document row for delete endpoint tests."""
    document = Document(
        filename=filename,
        mime_type="application/pdf",
        status="awaiting",
        path=path,
        size=1,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document


async def _insert_job(db: AsyncSession, document_id: uuid.UUID) -> ProcessingJob:
    """Insert a ProcessingJob row linked to a document."""
    job = ProcessingJob(document_id=document_id, status="awaiting", stage="upload_received")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _insert_chunk(db: AsyncSession, document_id: uuid.UUID) -> DocumentChunk:
    """Insert a DocumentChunk row linked to a document."""
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=0,
        content="hello world",
        token_count=2,
    )
    db.add(chunk)
    await db.commit()
    await db.refresh(chunk)
    return chunk


async def test_delete_document_removes_db_rows_and_safe_file(
    client: TestClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    stored = tmp_path / "stored.pdf"
    stored.write_bytes(b"document")
    document = await _insert_document(db_session, filename="stored.pdf", path="stored.pdf")
    await _insert_job(db_session, document.id)
    await _insert_chunk(db_session, document.id)

    response = client.delete(f"/documents/{document.id}")

    assert response.status_code == 204
    assert response.content == b""
    assert not stored.exists()
    assert await db_session.get(Document, document.id) is None
    assert (await db_session.execute(select(ProcessingJob))).scalars().all() == []
    assert (await db_session.execute(select(DocumentChunk))).scalars().all() == []


def test_delete_missing_document_returns_404(client: TestClient) -> None:
    response = client.delete(f"/documents/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_delete_document_succeeds_when_stored_file_missing(
    client: TestClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    missing_file = tmp_path / "missing.pdf"
    document = await _insert_document(
        db_session,
        filename="missing.pdf",
        path=str(missing_file),
    )
    await _insert_job(db_session, document.id)

    response = client.delete(f"/documents/{document.id}")

    assert response.status_code == 204
    assert await db_session.get(Document, document.id) is None
    assert (await db_session.execute(select(ProcessingJob))).scalars().all() == []


async def test_delete_document_does_not_remove_file_outside_storage_root(
    client: TestClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    outside_file = tmp_path.parent / "outside-delete.pdf"
    outside_file.write_bytes(b"outside")
    document = await _insert_document(
        db_session,
        filename="outside-delete.pdf",
        path="../outside-delete.pdf",
    )
    await _insert_chunk(db_session, document.id)

    response = client.delete(f"/documents/{document.id}")

    assert response.status_code == 204
    assert outside_file.exists()
    assert await db_session.get(Document, document.id) is None
    assert (await db_session.execute(select(DocumentChunk))).scalars().all() == []
