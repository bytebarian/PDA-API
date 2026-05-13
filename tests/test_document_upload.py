"""Tests for POST /documents/upload."""

from __future__ import annotations

import io
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """TestClient with the DB dependency and storage path overridden."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    # Override the DB dependency so the endpoint uses the in-memory session.
    fastapi_app.dependency_overrides[get_db] = override_get_db

    # Override settings so files land in a temporary directory.
    test_settings = Settings(
        storage_path=tmp_path,  # type: ignore[arg-type]
        _env_file=None,  # type: ignore[call-arg]
    )

    fastapi_app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(fastapi_app) as c:
        yield c

    fastapi_app.dependency_overrides.clear()


def _pdf_bytes() -> bytes:
    """Minimal valid-ish PDF content."""
    return b"%PDF-1.4 fake pdf content for testing"


def _text_bytes() -> bytes:
    return b"Hello, this is plain text."


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_upload_pdf_returns_201(client: TestClient, tmp_path: Path) -> None:
    """A valid PDF upload must return HTTP 201."""
    response = client.post(
        "/documents/upload",
        files={"file": ("document.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")},
    )
    assert response.status_code == 201, response.text


def test_upload_returns_correct_schema(client: TestClient, tmp_path: Path) -> None:
    """Response body must contain the expected fields."""
    response = client.post(
        "/documents/upload",
        files={"file": ("document.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")},
    )
    body = response.json()
    assert "document_id" in body
    assert "job_id" in body
    assert "filename" in body
    assert body["status"] == "awaiting"
    assert body["job_status"] == "awaiting"
    assert body["job_stage"] == "upload_received"


def test_upload_sanitizes_filename(client: TestClient, tmp_path: Path) -> None:
    """Traversal sequences in the filename must be stripped."""
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "../../../etc/passwd",
                io.BytesIO(_pdf_bytes()),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 201
    body = response.json()
    # The returned filename must NOT contain path separators.
    assert "/" not in body["filename"]
    assert "\\" not in body["filename"]
    assert ".." not in body["filename"]


def test_upload_stores_file_on_disk(client: TestClient, tmp_path: Path) -> None:
    """The uploaded file must be persisted under the configured storage path."""
    data = _pdf_bytes()
    response = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", io.BytesIO(data), "application/pdf")},
    )
    assert response.status_code == 201
    stored = list(tmp_path.iterdir())
    assert len(stored) == 1
    assert stored[0].read_bytes() == data


def test_upload_plain_text_returns_201(client: TestClient, tmp_path: Path) -> None:
    """text/plain uploads must also be accepted."""
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(_text_bytes()), "text/plain")},
    )
    assert response.status_code == 201


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_upload_rejects_invalid_mime_type(client: TestClient, tmp_path: Path) -> None:
    """Files with a disallowed MIME type must be rejected with 415."""
    response = client.post(
        "/documents/upload",
        files={"file": ("script.exe", io.BytesIO(b"\x4d\x5a"), "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_rejects_invalid_mime_type_detail(client: TestClient, tmp_path: Path) -> None:
    """The 415 response must contain a descriptive error message."""
    response = client.post(
        "/documents/upload",
        files={"file": ("bad.bin", io.BytesIO(b"data"), "application/octet-stream")},
    )
    body = response.json()
    assert "detail" in body
    assert "application/octet-stream" in body["detail"]


def test_upload_rejects_oversized_file(client: TestClient, tmp_path: Path) -> None:
    """Files larger than the configured limit must be rejected with 413."""
    tiny_limit_settings = Settings(
        storage_path=tmp_path,  # type: ignore[arg-type]
        max_file_size_bytes=10,
        _env_file=None,  # type: ignore[call-arg]
    )
    fastapi_app.dependency_overrides[get_settings] = lambda: tiny_limit_settings

    response = client.post(
        "/documents/upload",
        files={"file": ("big.pdf", io.BytesIO(b"A" * 20), "application/pdf")},
    )
    assert response.status_code == 413


def test_upload_rejects_empty_file(client: TestClient, tmp_path: Path) -> None:
    """An empty file upload must be rejected with 400."""
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Path traversal prevention (unit-level)
# ---------------------------------------------------------------------------


def test_sanitize_filename_strips_traversal() -> None:
    """sanitize_filename must remove path-traversal components."""
    from app.services.file_storage import sanitize_filename

    assert "/" not in sanitize_filename("../../etc/passwd")
    assert ".." not in sanitize_filename("../../etc/passwd")


def test_sanitize_filename_strips_leading_dots() -> None:
    from app.services.file_storage import sanitize_filename

    result = sanitize_filename(".hidden_file.pdf")
    assert not result.startswith(".")


def test_sanitize_filename_empty_falls_back() -> None:
    from app.services.file_storage import sanitize_filename

    result = sanitize_filename("")
    assert result  # must not be empty


def test_sanitize_filename_preserves_safe_name() -> None:
    from app.services.file_storage import sanitize_filename

    assert sanitize_filename("my_document.pdf") == "my_document.pdf"
