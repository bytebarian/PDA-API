"""Migration compatibility tests for the document_chunks table."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, inspect

from app.core.config import get_settings


def test_document_chunk_migration_upgrade_and_downgrade_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New migration should upgrade/downgrade cleanly on SQLite fallback."""
    db_path = tmp_path / "migration.sqlite3"
    monkeypatch.setenv("PDA_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()

    alembic_ini = Path(__file__).resolve().parents[1] / "alembic.ini"
    config = Config(str(alembic_ini))

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        inspector = inspect(engine)
        assert "document_chunks" in inspector.get_table_names()
        column_names = {column["name"] for column in inspector.get_columns("document_chunks")}
        assert "embedding" in column_names

        command.downgrade(config, "1b8bb4d20971")

        inspector_after_downgrade = inspect(engine)
        assert "document_chunks" not in inspector_after_downgrade.get_table_names()
    finally:
        engine.dispose()
        get_settings.cache_clear()
