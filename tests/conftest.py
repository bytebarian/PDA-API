"""Shared fixtures for document model and schema tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.db.base import Base
import app.models  # noqa: F401 – ensure all models are registered


@pytest.fixture(autouse=True)
def _disable_background_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep API contract tests deterministic: do not run the pipeline in-request.

    Upload/reprocess still schedule work in production; tests that need the
    orchestrator call ``process_job`` directly.
    """
    monkeypatch.setattr(
        "app.api.routers.documents.enqueue_processing_job",
        lambda background_tasks, job_id: None,
    )


@pytest.fixture
async def db_session(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLite in-memory session with all tables created."""
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
