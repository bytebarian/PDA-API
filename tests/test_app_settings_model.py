"""Tests for the AppSettings SQLAlchemy ORM model."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.app_settings import AppSettings


def test_app_settings_table_name() -> None:
    """AppSettings model must use the 'app_settings' table name."""
    assert AppSettings.__tablename__ == "app_settings"


def test_app_settings_inherits_base() -> None:
    """AppSettings must inherit from the shared declarative Base."""
    from app.db.base import Base

    assert issubclass(AppSettings, Base)


def test_app_settings_has_expected_columns() -> None:
    """All required app settings columns must be present in the table mapping."""
    expected = {
        "id",
        "storage_path",
        "max_file_size_bytes",
        "allowed_file_types_jsonb",
        "ocr_enabled",
        "ocr_provider",
        "ocr_language",
        "ocr_dpi",
        "chunk_size",
        "chunk_overlap",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "llm_provider",
        "llm_model",
        "privacy_local_only",
        "telemetry_enabled",
        "extra_settings_jsonb",
        "created_at",
        "updated_at",
    }
    actual = {col.name for col in AppSettings.__table__.columns}
    assert expected == actual


def test_app_settings_primary_key_is_id() -> None:
    """The 'id' column must be the primary key."""
    pk_cols = {col.name for col in AppSettings.__table__.primary_key}
    assert pk_cols == {"id"}


def test_app_settings_defaults_declared() -> None:
    """Important defaults must be declared on mapped columns."""
    assert AppSettings.__table__.c.storage_path.default is not None
    assert AppSettings.__table__.c.storage_path.default.arg == "./storage"

    assert AppSettings.__table__.c.max_file_size_bytes.default is not None
    assert AppSettings.__table__.c.max_file_size_bytes.default.arg == 10 * 1024 * 1024

    assert AppSettings.__table__.c.allowed_file_types_jsonb.default is not None
    assert callable(AppSettings.__table__.c.allowed_file_types_jsonb.default.arg)

    assert AppSettings.__table__.c.ocr_enabled.default is not None
    assert AppSettings.__table__.c.ocr_enabled.default.arg is True

    assert AppSettings.__table__.c.chunk_size.default is not None
    assert AppSettings.__table__.c.chunk_size.default.arg == 1000

    assert AppSettings.__table__.c.chunk_overlap.default is not None
    assert AppSettings.__table__.c.chunk_overlap.default.arg == 200

    assert AppSettings.__table__.c.embedding_dimensions.default is not None
    assert AppSettings.__table__.c.embedding_dimensions.default.arg == 1536

    assert AppSettings.__table__.c.llm_provider.default is not None
    assert AppSettings.__table__.c.llm_provider.default.arg == "local"

    assert AppSettings.__table__.c.privacy_local_only.default is not None
    assert AppSettings.__table__.c.privacy_local_only.default.arg is True

    assert AppSettings.__table__.c.telemetry_enabled.default is not None
    assert AppSettings.__table__.c.telemetry_enabled.default.arg is False


def test_app_settings_optional_fields_default_none() -> None:
    """Nullable fields with no default should initialize as None."""
    settings = AppSettings()
    assert settings.embedding_provider is None
    assert settings.embedding_model is None


async def test_app_settings_insert_and_read_defaults(db_session: AsyncSession) -> None:
    """AppSettings row can be inserted and retrieved with defaults."""
    settings = AppSettings()
    db_session.add(settings)
    await db_session.commit()
    await db_session.refresh(settings)

    assert settings.id is not None
    assert settings.storage_path == "./storage"
    assert settings.max_file_size_bytes == 10 * 1024 * 1024
    assert settings.allowed_file_types_jsonb == [
        "application/pdf",
        "text/plain",
        "image/png",
        "image/jpeg",
        "image/jpg",
    ]
    assert settings.ocr_enabled is True
    assert settings.ocr_provider == "tesseract"
    assert settings.ocr_language == "eng"
    assert settings.ocr_dpi == 300
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.embedding_provider is None
    assert settings.embedding_model is None
    assert settings.embedding_dimensions == 1536
    assert settings.llm_provider == "local"
    assert settings.llm_model == "llama3.1:8b-instruct"
    assert settings.privacy_local_only is True
    assert settings.telemetry_enabled is False
    assert settings.extra_settings_jsonb == {}
    assert settings.created_at is not None
    assert settings.updated_at is not None


async def test_app_settings_json_fields_roundtrip(db_session: AsyncSession) -> None:
    """JSON fields should persist in an insert/read cycle."""
    payload = AppSettings(
        allowed_file_types_jsonb=["application/pdf", "text/markdown"],
        extra_settings_jsonb={"feature_flags": {"ocr": True}},
    )
    db_session.add(payload)
    await db_session.commit()
    await db_session.refresh(payload)

    assert payload.allowed_file_types_jsonb == ["application/pdf", "text/markdown"]
    assert payload.extra_settings_jsonb == {"feature_flags": {"ocr": True}}


async def test_app_settings_allowed_file_types_append_persists(
    db_session: AsyncSession,
) -> None:
    """In-place append on allowed_file_types_jsonb should be tracked and persisted."""
    settings = AppSettings(allowed_file_types_jsonb=["application/pdf"])
    db_session.add(settings)
    await db_session.commit()

    settings.allowed_file_types_jsonb.append("text/markdown")
    await db_session.commit()
    await db_session.refresh(settings)

    assert settings.allowed_file_types_jsonb == ["application/pdf", "text/markdown"]


async def test_app_settings_extra_settings_top_level_assignment_persists(
    db_session: AsyncSession,
) -> None:
    """Top-level assignment on extra_settings_jsonb should be tracked and persisted."""
    settings = AppSettings(extra_settings_jsonb={"feature_flags": {"ocr": True}})
    db_session.add(settings)
    await db_session.commit()

    settings.extra_settings_jsonb["beta_enabled"] = True
    await db_session.commit()
    await db_session.refresh(settings)

    assert settings.extra_settings_jsonb == {
        "feature_flags": {"ocr": True},
        "beta_enabled": True,
    }
