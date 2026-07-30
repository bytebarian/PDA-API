"""Tests for background processing job enqueue/runner."""

from __future__ import annotations

import io
import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app as fastapi_app
from app.workers.processing import enqueue_processing_job, run_processing_job

import app.models  # noqa: F401


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


def test_enqueue_processing_job_schedules_runner() -> None:
    tasks = BackgroundTasks()
    job_id = uuid.uuid4()
    enqueue_processing_job(tasks, job_id)
    assert len(tasks.tasks) == 1
    assert tasks.tasks[0].func is run_processing_job
    assert tasks.tasks[0].args == (job_id,)


def test_upload_enqueues_processing_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[uuid.UUID] = []

    def _capture(background_tasks: BackgroundTasks, job_id: uuid.UUID) -> None:
        captured.append(job_id)

    monkeypatch.setattr("app.api.routers.documents.enqueue_processing_job", _capture)

    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert response.status_code == 201, response.text
    assert len(captured) == 1
    assert str(captured[0]) == response.json()["job_id"]


@pytest.mark.asyncio
async def test_run_processing_job_opens_session_and_calls_process_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = uuid.uuid4()
    session = object()

    class _SessionCM:
        async def __aenter__(self) -> object:
            return session

        async def __aexit__(self, *args: object) -> None:
            return None

    session_factory = MagicMock(return_value=_SessionCM())
    process = AsyncMock()

    monkeypatch.setattr(
        "app.workers.processing.get_session_factory",
        MagicMock(return_value=session_factory),
    )
    monkeypatch.setattr("app.workers.processing.process_job", process)

    await run_processing_job(job_id)

    session_factory.assert_called_once()
    process.assert_awaited_once_with(session, job_id)
