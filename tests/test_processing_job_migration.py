"""Migration compatibility tests for the processing_jobs table."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, inspect

from app.core.config import get_settings


def test_processing_job_migration_upgrade_and_downgrade_sqlite(
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
        assert "processing_jobs" in inspector.get_table_names()
        column_names = {column["name"] for column in inspector.get_columns("processing_jobs")}
        assert {
            "document_id",
            "status",
            "stage",
            "error_details_jsonb",
            "stage_history_jsonb",
        }.issubset(column_names)
        index_names = {index["name"] for index in inspector.get_indexes("processing_jobs")}
        assert "ix_processing_jobs_document_id" in index_names
        assert "ix_processing_jobs_document_id_status" in index_names

        command.downgrade(config, "93f56f34f1ae")

        inspector_after_downgrade = inspect(engine)
        assert "processing_jobs" not in inspector_after_downgrade.get_table_names()
    finally:
        engine.dispose()
        get_settings.cache_clear()
