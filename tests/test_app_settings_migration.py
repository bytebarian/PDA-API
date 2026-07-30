"""Migration compatibility tests for the app_settings table."""

from __future__ import annotations

import uuid
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, inspect, text

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

    command.upgrade(config, "d89ec8d9a902")

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


def test_backfill_migration_replaces_obsolete_llm_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Migration e1b2c3d4f5a6 must replace obsolete llm_model values with the canonical default."""
    db_path = tmp_path / "backfill.sqlite3"
    monkeypatch.setenv("PDA_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()

    alembic_ini = Path(__file__).resolve().parents[1] / "alembic.ini"
    config = Config(str(alembic_ini))

    # Apply up to (and including) the table-creation migration.
    command.upgrade(config, "d89ec8d9a902")

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:
            # Insert two rows: one with the obsolete model, one with a valid model.
            conn.execute(
                text(
                    "INSERT INTO app_settings "
                    "(id, storage_path, max_file_size_bytes, allowed_file_types_jsonb, "
                    " ocr_enabled, chunk_size, chunk_overlap, privacy_local_only, "
                    " telemetry_enabled, extra_settings_jsonb, llm_model, created_at, updated_at) "
                    "VALUES "
                    "(:id1, './storage', 10485760, '[]', 1, 1000, 200, 1, 0, '{}', "
                    " 'llama3.1:8b-instruct', datetime('now'), datetime('now')), "
                    "(:id2, './storage', 10485760, '[]', 1, 1000, 200, 1, 0, '{}', "
                    " 'llama3.1:8b', datetime('now'), datetime('now'))"
                ),
                {"id1": str(uuid.uuid4()), "id2": str(uuid.uuid4())},
            )

        # Apply the backfill migration (apply entire head which includes backfill).
        command.upgrade(config, "e1b2c3d4f5a6")

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT llm_model FROM app_settings ORDER BY created_at")
            ).fetchall()

        models = [row[0] for row in rows]
        # Obsolete value must be replaced.
        assert models[0] == "llama3.1:8b-instruct-q8_0"
        # Valid value must be preserved.
        assert models[1] == "llama3.1:8b"
    finally:
        engine.dispose()
        get_settings.cache_clear()


def test_backfill_migration_upgrade_and_downgrade_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Migration e1b2c3d4f5a6 must apply and reverse cleanly on SQLite."""
    db_path = tmp_path / "backfill_ud.sqlite3"
    monkeypatch.setenv("PDA_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()

    alembic_ini = Path(__file__).resolve().parents[1] / "alembic.ini"
    config = Config(str(alembic_ini))

    command.upgrade(config, "e1b2c3d4f5a6")

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        inspector = inspect(engine)
        assert "app_settings" in inspector.get_table_names()

        command.downgrade(config, "c7e3b1a4f9d2")

        inspector_after = inspect(engine)
        # Table should still exist after downgrade (only server default reverted).
        assert "app_settings" in inspector_after.get_table_names()
    finally:
        engine.dispose()
        get_settings.cache_clear()
