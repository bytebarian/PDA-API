from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, text

from app.core.config import get_settings


def test_pgvector_extension_available_after_migrations(monkeypatch: pytest.MonkeyPatch) -> None:
    postgres_url = os.getenv("PDA_PGVECTOR_TEST_DATABASE_URL")
    if not postgres_url:
        pytest.skip(
            "Set PDA_PGVECTOR_TEST_DATABASE_URL to run PostgreSQL+pgvector integration checks."
        )

    if not postgres_url.startswith("postgresql+"):
        pytest.fail("PDA_PGVECTOR_TEST_DATABASE_URL must be a SQLAlchemy PostgreSQL URL.")

    monkeypatch.setenv("PDA_DATABASE_URL", postgres_url)
    get_settings.cache_clear()

    alembic_ini = Path(__file__).resolve().parents[1] / "alembic.ini"
    command.upgrade(Config(str(alembic_ini)), "head")

    sync_url = postgres_url.replace("+asyncpg", "")
    engine = create_engine(sync_url)
    try:
        with engine.connect() as connection:
            extension = connection.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
        if extension != "vector":
            pytest.fail(
                "pgvector extension is unavailable in the test database; ensure CREATE EXTENSION vector succeeds."
            )
    finally:
        engine.dispose()
        get_settings.cache_clear()
