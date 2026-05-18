"""Tests for PATCH /documents/{document_id} metadata updates."""

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

import app.models  # noqa: F401 – register all ORM models

from app.models.document import Document


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
    path: str = "stored/document.pdf",
    category: str | None = "initial-category",
    file_type: str | None = "pdf",
    summary: str | None = "initial-summary",
    metadata_jsonb: dict[str, str] | None = None,
    status: str = "awaiting",
) -> Document:
    document = Document(
        filename=filename,
        path=path,
        category=category,
        file_type=file_type,
        summary=summary,
        metadata_jsonb=metadata_jsonb,
        status=status,
        size=1,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document


async def test_patch_document_rename_sanitizes_and_keeps_stored_path(
    client: TestClient, db_session: AsyncSession
) -> None:
    document = await _insert_document(db_session, filename="original.pdf", path="stored/original.pdf")

    response = client.patch(
        f"/documents/{document.id}",
        json={"filename": "../../../renamed.pdf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "renamed.pdf"

    updated = await db_session.get(Document, document.id)
    assert updated is not None
    assert updated.path == "stored/original.pdf"


async def test_patch_document_updates_only_provided_fields(
    client: TestClient, db_session: AsyncSession
) -> None:
    document = await _insert_document(db_session)

    response = client.patch(
        f"/documents/{document.id}",
        json={"category": "finance"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "finance"
    assert body["file_type"] == "pdf"
    assert body["summary"] == "initial-summary"

    updated = await db_session.get(Document, document.id)
    assert updated is not None
    assert updated.category == "finance"
    assert updated.file_type == "pdf"
    assert updated.summary == "initial-summary"


async def test_patch_document_replaces_metadata_jsonb(
    client: TestClient, db_session: AsyncSession
) -> None:
    document = await _insert_document(
        db_session,
        metadata_jsonb={"key_a": "one", "key_b": "two"},
    )

    response = client.patch(
        f"/documents/{document.id}",
        json={"metadata_jsonb": {"key_a": "updated"}},
    )

    assert response.status_code == 200
    updated = await db_session.get(Document, document.id)
    assert updated is not None
    assert updated.metadata_jsonb == {"key_a": "updated"}


def test_patch_document_missing_returns_404(client: TestClient) -> None:
    response = client.patch(f"/documents/{uuid.uuid4()}", json={"category": "finance"})
    assert response.status_code == 404


async def test_patch_document_rejects_empty_filename(
    client: TestClient, db_session: AsyncSession
) -> None:
    document = await _insert_document(db_session, filename="before.pdf")

    response = client.patch(f"/documents/{document.id}", json={"filename": "   "})

    assert response.status_code == 422
    unchanged = await db_session.get(Document, document.id)
    assert unchanged is not None
    assert unchanged.filename == "before.pdf"


async def test_patch_document_rejects_disallowed_fields(
    client: TestClient, db_session: AsyncSession
) -> None:
    document = await _insert_document(db_session, status="awaiting")

    response = client.patch(
        f"/documents/{document.id}",
        json={"status": "ready"},
    )

    assert response.status_code == 422
    unchanged = await db_session.get(Document, document.id)
    assert unchanged is not None
    assert unchanged.status == "awaiting"
