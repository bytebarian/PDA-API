"""Gamma API contract hardening tests for routes and response consistency."""

from __future__ import annotations

import io
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
from app.main import app as fastapi_app, create_app

import app.models  # noqa: F401 - register all ORM models


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
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
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    fastapi_app.dependency_overrides[get_db] = override_get_db
    fastapi_app.dependency_overrides[get_settings] = lambda: Settings(
        storage_path=tmp_path,  # type: ignore[arg-type]
        _env_file=None,  # type: ignore[call-arg]
    )

    with TestClient(fastapi_app) as c:
        yield c

    fastapi_app.dependency_overrides.clear()


def _pdf_bytes() -> bytes:
    return b"%PDF-1.4 contract test pdf"


def test_api_prefix_routes_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDA_API_PREFIX", "/api/v1")
    get_settings.cache_clear()
    prefixed_app = create_app()

    with TestClient(prefixed_app) as client:
        assert client.get("/api/v1/health/live").status_code == 200
        assert client.get("/health/live").status_code == 404
        assert client.get("/api/v1/documents/upload").status_code == 405

    get_settings.cache_clear()


def test_upload_static_route_does_not_collide_with_uuid_document_route(client: TestClient) -> None:
    static_route_response = client.get("/documents/upload")
    dynamic_route_response = client.get("/documents/not-a-uuid")

    assert static_route_response.status_code == 405
    assert dynamic_route_response.status_code == 404


def test_upload_reprocess_and_job_polling_contracts_are_consistent(client: TestClient) -> None:
    upload = client.post(
        "/documents/upload",
        files={"file": ("contract.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    upload_body = upload.json()
    assert {"document_id", "job_id"}.issubset(upload_body)

    document_id = upload_body["document_id"]
    upload_job_id = upload_body["job_id"]

    upload_job = client.get(f"/jobs/{upload_job_id}")
    assert upload_job.status_code == 200, upload_job.text
    assert upload_job.json()["document_id"] == document_id

    detail_after_upload = client.get(f"/documents/{document_id}")
    assert detail_after_upload.status_code == 200, detail_after_upload.text
    assert detail_after_upload.json()["latest_job"]["id"] == upload_job_id

    reprocess = client.post(
        f"/documents/{document_id}/reprocess",
        json={"force": True, "reason": "contract check"},
    )
    assert reprocess.status_code == 201, reprocess.text
    reprocess_body = reprocess.json()
    assert reprocess_body["document_id"] == document_id
    assert "job_id" in reprocess_body

    reprocess_job_id = reprocess_body["job_id"]
    reprocess_job = client.get(f"/jobs/{reprocess_job_id}")
    assert reprocess_job.status_code == 200, reprocess_job.text
    assert reprocess_job.json()["document_id"] == document_id

    detail_after_reprocess = client.get(f"/documents/{document_id}")
    assert detail_after_reprocess.status_code == 200, detail_after_reprocess.text
    latest_job = detail_after_reprocess.json()["latest_job"]
    assert latest_job["id"] == reprocess_job_id
    assert latest_job["status"] == "awaiting"
    assert latest_job["stage"] == "queued"


def test_document_download_reprocess_and_job_routes_are_disjoint(client: TestClient) -> None:
    upload = client.post(
        "/documents/upload",
        files={"file": ("route-check.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    document_id = upload.json()["document_id"]
    job_id = upload.json()["job_id"]

    assert client.get(f"/documents/{document_id}/download").status_code == 200
    assert client.post(f"/documents/{document_id}/reprocess").status_code == 201
    assert client.get(f"/jobs/{job_id}").status_code == 200

    random_document_id = uuid.uuid4()
    random_job_id = uuid.uuid4()
    assert client.get(f"/documents/{random_document_id}/download").status_code == 404
    assert client.get(f"/jobs/{random_job_id}").status_code == 404
