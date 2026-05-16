"""Tests for GET /documents/{document_id}/download."""

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
    mime_type: str | None = "application/pdf",
    path: str | None,
) -> Document:
    """Insert a minimal Document row for download endpoint tests."""
    doc = Document(
        filename=filename,
        mime_type=mime_type,
        status="awaiting",
        path=path,
        size=1,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def test_download_document_returns_file_bytes_and_sanitized_headers(
    client: TestClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    data = b"%PDF-1.4 download me"
    stored = tmp_path / "stored.pdf"
    stored.write_bytes(data)
    doc = await _insert_document(
        db_session,
        filename="../../statement.pdf",
        mime_type="application/pdf",
        path=str(stored),
    )

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 200
    assert response.content == data
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment;" in response.headers["content-disposition"]
    assert 'filename="statement.pdf"' in response.headers["content-disposition"]
    assert ".." not in response.headers["content-disposition"]


async def test_download_document_falls_back_to_octet_stream(
    client: TestClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    stored = tmp_path / "notes.txt"
    stored.write_bytes(b"hello")
    doc = await _insert_document(
        db_session,
        filename="notes.txt",
        mime_type=None,
        path=str(stored),
    )

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"


async def test_download_document_sanitizes_control_chars_in_filename(
    client: TestClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    stored = tmp_path / "safe.pdf"
    stored.write_bytes(b"ok")
    doc = await _insert_document(
        db_session,
        filename="bad\nname.pdf",
        path=str(stored),
    )

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 200
    assert 'filename="bad_name.pdf"' in response.headers["content-disposition"]


def test_download_missing_document_returns_404(client: TestClient) -> None:
    response = client.get(f"/documents/{uuid.uuid4()}/download")
    assert response.status_code == 404


async def test_download_missing_file_returns_404(
    client: TestClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    doc = await _insert_document(
        db_session,
        filename="ghost.pdf",
        path=str(tmp_path / "does-not-exist.pdf"),
    )

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 404


async def test_download_empty_document_path_returns_404(
    client: TestClient, db_session: AsyncSession
) -> None:
    doc = await _insert_document(
        db_session,
        filename="empty-path.pdf",
        path="",
    )

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 404


async def test_download_path_outside_storage_root_returns_404(
    client: TestClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    outside_file = tmp_path.parent / "outside.pdf"
    outside_file.write_bytes(b"outside")
    doc = await _insert_document(
        db_session,
        filename="outside.pdf",
        path=str(outside_file),
    )

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 404


async def test_download_traversal_path_returns_404(
    client: TestClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    outside_file = tmp_path.parent / "traversal.pdf"
    outside_file.write_bytes(b"outside")
    doc = await _insert_document(
        db_session,
        filename="traversal.pdf",
        path="../traversal.pdf",
    )

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 404


async def test_download_relative_path_is_resolved_from_storage_root(
    client: TestClient,
    db_session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_file = tmp_path / "foo.pdf"
    root_file.write_bytes(b"root")

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "foo.pdf").write_bytes(b"cwd")
    monkeypatch.chdir(subdir)

    doc = await _insert_document(
        db_session,
        filename="foo.pdf",
        path="foo.pdf",
    )

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 200
    assert response.content == b"root"


async def test_download_storage_root_resolution_oserror_returns_404(
    client: TestClient,
    db_session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = tmp_path / "doc.pdf"
    stored.write_bytes(b"ok")
    doc = await _insert_document(
        db_session,
        filename="doc.pdf",
        path=str(stored),
    )

    original_resolve = Path.resolve

    def _raise_for_storage_root(self: Path, *, strict: bool = False) -> Path:
        if strict and self == tmp_path:
            raise PermissionError("denied")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _raise_for_storage_root)

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 404


async def test_download_candidate_resolution_oserror_returns_404(
    client: TestClient,
    db_session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = tmp_path / "doc.pdf"
    stored.write_bytes(b"ok")
    doc = await _insert_document(
        db_session,
        filename="doc.pdf",
        path=str(stored),
    )

    original_resolve = Path.resolve

    def _raise_for_candidate(self: Path, *, strict: bool = False) -> Path:
        if strict and self == stored:
            raise PermissionError("denied")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _raise_for_candidate)

    response = client.get(f"/documents/{doc.id}/download")
    assert response.status_code == 404
