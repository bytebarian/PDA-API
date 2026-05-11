"""Migration compatibility tests for the app_settings table."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, inspect

from app.core.config import get_settings


def test_app_settings_migration_upgrade_and_downgrade_sqlite(
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
        assert "app_settings" in inspector.get_table_names()
        column_names = {column["name"] for column in inspector.get_columns("app_settings")}
        assert {
            "storage_path",
            "max_file_size_bytes",
            "allowed_file_types_jsonb",
            "ocr_enabled",
            "chunk_size",
            "chunk_overlap",
            "embedding_dimensions",
            "privacy_local_only",
            "telemetry_enabled",
            "extra_settings_jsonb",
        }.issubset(column_names)

        command.downgrade(config, "c5a2f0423a10")

        inspector_after_downgrade = inspect(engine)
        assert "app_settings" not in inspector_after_downgrade.get_table_names()
    finally:
        engine.dispose()
        get_settings.cache_clear()
