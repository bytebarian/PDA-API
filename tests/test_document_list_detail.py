"""Tests for GET /documents and GET /documents/{document_id}.

Request paths use no prefix because the default ``Settings.api_prefix``
normalises to an empty string (""), consistent with the existing upload tests.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone
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
from app.models.processing_job import ProcessingJob

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
    filename: str = "doc.pdf",
    status: str = "awaiting",
    category: str | None = None,
    file_type: str | None = None,
    size: int = 100,
    created_at: datetime | None = None,
) -> Document:
    kwargs: dict = dict(
        filename=filename,
        status=status,
        category=category,
        file_type=file_type,
        size=size,
    )
    if created_at is not None:
        kwargs["created_at"] = created_at
        kwargs["updated_at"] = created_at
    doc = Document(**kwargs)
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def _insert_job(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    status: str = "awaiting",
    stage: str = "upload_received",
    created_at: datetime | None = None,
) -> ProcessingJob:
    kwargs: dict = dict(
        document_id=document_id,
        status=status,
        stage=stage,
    )
    if created_at is not None:
        kwargs["created_at"] = created_at
        kwargs["updated_at"] = created_at
    job = ProcessingJob(**kwargs)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


# ---------------------------------------------------------------------------
# GET /documents – empty list
# ---------------------------------------------------------------------------


def test_list_documents_empty(client: TestClient) -> None:
    """Empty database must return an empty list with correct metadata."""
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["page"] == 1
    assert body["page_size"] == 20
    assert body["total"] == 0


# ---------------------------------------------------------------------------
# GET /documents – populated list
# ---------------------------------------------------------------------------


async def test_list_documents_returns_items(
    client: TestClient, db_session: AsyncSession
) -> None:
    """List endpoint returns inserted documents."""
    await _insert_document(db_session, filename="alpha.pdf")
    await _insert_document(db_session, filename="beta.pdf")

    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2


async def test_list_documents_summary_fields(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Each item in the list must expose the expected summary fields."""
    await _insert_document(db_session, filename="report.pdf", status="ready", size=512)

    response = client.get("/documents")
    item = response.json()["items"][0]
    assert "id" in item
    assert item["filename"] == "report.pdf"
    assert item["status"] == "ready"
    assert item["size"] == 512
    assert "chunk_count" in item
    assert "created_at" in item
    assert "updated_at" in item


# ---------------------------------------------------------------------------
# GET /documents – pagination
# ---------------------------------------------------------------------------


async def test_list_documents_pagination(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Pagination metadata must be correct and items must be sliced."""
    for i in range(5):
        await _insert_document(db_session, filename=f"doc{i}.pdf")

    response = client.get("/documents?page=1&page_size=3")
    body = response.json()
    assert body["total"] == 5
    assert body["page"] == 1
    assert body["page_size"] == 3
    assert len(body["items"]) == 3

    response2 = client.get("/documents?page=2&page_size=3")
    body2 = response2.json()
    assert body2["total"] == 5
    assert len(body2["items"]) == 2


async def test_list_documents_page_beyond_end(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Requesting a page beyond total results must return empty items list."""
    await _insert_document(db_session, filename="only.pdf")

    response = client.get("/documents?page=99&page_size=20")
    body = response.json()
    assert body["total"] == 1
    assert body["items"] == []


# ---------------------------------------------------------------------------
# GET /documents – sorting
# ---------------------------------------------------------------------------


async def test_list_documents_sort_newest(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Default sort (newest) must return the most recently created document first."""
    t1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc)
    doc_a = await _insert_document(db_session, filename="first.pdf", created_at=t1)
    doc_b = await _insert_document(db_session, filename="second.pdf", created_at=t2)

    response = client.get("/documents")
    items = response.json()["items"]
    # newest first – doc_b has a later created_at
    ids = [item["id"] for item in items]
    assert ids[0] == str(doc_b.id)
    assert ids[1] == str(doc_a.id)


async def test_list_documents_sort_oldest(
    client: TestClient, db_session: AsyncSession
) -> None:
    """sort=oldest must return the oldest document first."""
    t1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc)
    doc_a = await _insert_document(db_session, filename="first.pdf", created_at=t1)
    doc_b = await _insert_document(db_session, filename="second.pdf", created_at=t2)

    response = client.get("/documents?sort=oldest")
    items = response.json()["items"]
    ids = [item["id"] for item in items]
    assert ids[0] == str(doc_a.id)
    assert ids[1] == str(doc_b.id)


# ---------------------------------------------------------------------------
# GET /documents – filters
# ---------------------------------------------------------------------------


async def test_list_documents_filter_by_status(
    client: TestClient, db_session: AsyncSession
) -> None:
    """status filter must only return documents with the matching status."""
    await _insert_document(db_session, filename="ready.pdf", status="ready")
    await _insert_document(db_session, filename="waiting.pdf", status="awaiting")

    response = client.get("/documents?status=ready")
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["status"] == "ready"


async def test_list_documents_filter_by_category(
    client: TestClient, db_session: AsyncSession
) -> None:
    """category filter must only return documents with the exact category."""
    await _insert_document(db_session, filename="invoice.pdf", category="finance")
    await _insert_document(db_session, filename="memo.pdf", category="hr")

    response = client.get("/documents?category=finance")
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["filename"] == "invoice.pdf"


async def test_list_documents_filter_by_file_type(
    client: TestClient, db_session: AsyncSession
) -> None:
    """file_type filter must only return documents with the matching file_type."""
    await _insert_document(db_session, filename="a.pdf", file_type="pdf")
    await _insert_document(db_session, filename="b.txt", file_type="txt")

    response = client.get("/documents?file_type=pdf")
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["filename"] == "a.pdf"


async def test_list_documents_filter_by_q(
    client: TestClient, db_session: AsyncSession
) -> None:
    """q filter must perform case-insensitive filename substring matching."""
    await _insert_document(db_session, filename="Invoice_2024.pdf")
    await _insert_document(db_session, filename="Meeting_Notes.pdf")

    response = client.get("/documents?q=invoice")
    body = response.json()
    assert body["total"] == 1
    assert "Invoice" in body["items"][0]["filename"]


async def test_list_documents_combined_filters(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Multiple filters applied together must narrow results correctly."""
    await _insert_document(
        db_session, filename="report.pdf", status="ready", category="finance"
    )
    await _insert_document(
        db_session, filename="summary.pdf", status="ready", category="hr"
    )
    await _insert_document(
        db_session, filename="draft.pdf", status="awaiting", category="finance"
    )

    response = client.get("/documents?status=ready&category=finance")
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["filename"] == "report.pdf"


# ---------------------------------------------------------------------------
# GET /documents/{document_id} – detail success
# ---------------------------------------------------------------------------


async def test_get_document_returns_200(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Detail endpoint must return 200 for an existing document."""
    doc = await _insert_document(db_session, filename="detail.pdf")
    response = client.get(f"/documents/{doc.id}")
    assert response.status_code == 200


async def test_get_document_core_fields(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Detail response must include core document fields."""
    doc = await _insert_document(
        db_session,
        filename="detail.pdf",
        status="ready",
        category="legal",
        file_type="pdf",
        size=2048,
    )
    response = client.get(f"/documents/{doc.id}")
    body = response.json()

    assert body["id"] == str(doc.id)
    assert body["filename"] == "detail.pdf"
    assert body["status"] == "ready"
    assert body["category"] == "legal"
    assert body["file_type"] == "pdf"
    assert body["size"] == 2048
    assert "chunk_count" in body
    assert "created_at" in body
    assert "updated_at" in body


async def test_get_document_no_job_has_null_latest_job(
    client: TestClient, db_session: AsyncSession
) -> None:
    """latest_job must be null when no processing job exists for the document."""
    doc = await _insert_document(db_session, filename="nojob.pdf")
    response = client.get(f"/documents/{doc.id}")
    assert response.status_code == 200
    assert response.json()["latest_job"] is None


async def test_get_document_with_job_includes_job_summary(
    client: TestClient, db_session: AsyncSession
) -> None:
    """latest_job must be populated when a processing job exists."""
    doc = await _insert_document(db_session, filename="withjob.pdf")
    await _insert_job(
        db_session,
        doc.id,
        status="processing",
        stage="ocr",
    )

    response = client.get(f"/documents/{doc.id}")
    body = response.json()
    assert body["latest_job"] is not None
    job = body["latest_job"]
    assert job["status"] == "processing"
    assert job["stage"] == "ocr"
    assert "id" in job
    assert "created_at" in job


async def test_get_document_latest_job_is_most_recent(
    client: TestClient, db_session: AsyncSession
) -> None:
    """latest_job must refer to the most recently created job, not the first."""
    t1 = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    doc = await _insert_document(db_session, filename="multi.pdf")
    await _insert_job(db_session, doc.id, status="failed", stage="ocr", created_at=t1)
    await _insert_job(db_session, doc.id, status="processing", stage="chunking", created_at=t2)

    response = client.get(f"/documents/{doc.id}")
    body = response.json()
    assert body["latest_job"]["stage"] == "chunking"


# ---------------------------------------------------------------------------
# GET /documents/{document_id} – 404
# ---------------------------------------------------------------------------


def test_get_document_not_found_returns_404(client: TestClient) -> None:
    """Requesting a non-existent document UUID must return 404."""
    missing_id = uuid.uuid4()
    response = client.get(f"/documents/{missing_id}")
    assert response.status_code == 404


def test_get_document_not_found_has_detail(client: TestClient) -> None:
    """404 response must contain a 'detail' field."""
    missing_id = uuid.uuid4()
    response = client.get(f"/documents/{missing_id}")
    body = response.json()
    assert "detail" in body


# ---------------------------------------------------------------------------
# Validation – invalid query params
# ---------------------------------------------------------------------------


def test_list_documents_invalid_page_size(client: TestClient) -> None:
    """page_size > 100 must be rejected with 422."""
    response = client.get("/documents?page_size=101")
    assert response.status_code == 422


def test_list_documents_invalid_page(client: TestClient) -> None:
    """page < 1 must be rejected with 422."""
    response = client.get("/documents?page=0")
    assert response.status_code == 422


def test_list_documents_invalid_sort(client: TestClient) -> None:
    """sort values other than 'newest' or 'oldest' must be rejected with 422."""
    response = client.get("/documents?sort=random")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /documents – q wildcard escaping
# ---------------------------------------------------------------------------


async def test_list_documents_q_literal_percent_wildcard(
    client: TestClient, db_session: AsyncSession
) -> None:
    """q=% must match only filenames containing a literal %, not every filename."""
    await _insert_document(db_session, filename="file%report.pdf")
    await _insert_document(db_session, filename="other.pdf")

    # Without escaping, ilike("%%%") matches every non-empty filename.
    # With proper escaping only the file whose name contains a literal % is returned.
    response = client.get("/documents?q=%25")  # %25 is URL-encoded %
    body = response.json()
    assert body["total"] == 1
    assert "%" in body["items"][0]["filename"]


async def test_list_documents_q_literal_underscore_wildcard(
    client: TestClient, db_session: AsyncSession
) -> None:
    """q=_ must match only filenames with a literal _, not act as a single-char wildcard."""
    await _insert_document(db_session, filename="file_name.pdf")
    await _insert_document(db_session, filename="filename.pdf")

    # Without escaping, ilike("%_%") matches any filename with ≥ 1 character (both rows).
    # With escaping only the filename containing a literal _ is returned.
    response = client.get("/documents?q=_")
    body = response.json()
    assert body["total"] == 1
    assert "_" in body["items"][0]["filename"]
