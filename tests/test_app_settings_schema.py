"""Tests for AppSettings Pydantic schemas."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.app_settings import (
    DEFAULT_ALLOWED_FILE_TYPES,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    AppSettingsBase,
    AppSettingsCreate,
    AppSettingsRead,
    AppSettingsUpdate,
)


def test_app_settings_base_defaults() -> None:
    """AppSettingsBase must apply correct defaults."""
    settings = AppSettingsBase()
    assert settings.storage_path == "./storage"
    assert settings.max_file_size_bytes == 10 * 1024 * 1024
    assert settings.allowed_file_types_jsonb == list(DEFAULT_ALLOWED_FILE_TYPES)
    assert settings.ocr_enabled is True
    assert settings.ocr_provider == "tesseract"
    assert settings.ocr_language == "eng"
    assert settings.ocr_dpi == 300
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.embedding_provider is None
    assert settings.embedding_model is None
    assert settings.embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    assert settings.llm_provider == DEFAULT_LLM_PROVIDER
    assert settings.llm_model == DEFAULT_LLM_MODEL
    assert settings.privacy_local_only is True
    assert settings.telemetry_enabled is False
    assert settings.extra_settings_jsonb == {}


def test_app_settings_create_inherits_base_defaults() -> None:
    """AppSettingsCreate must inherit defaults from AppSettingsBase."""
    settings = AppSettingsCreate()
    assert settings.chunk_size == 1000
    assert settings.telemetry_enabled is False


def test_app_settings_update_all_optional() -> None:
    """AppSettingsUpdate must allow instantiation with no fields."""
    update = AppSettingsUpdate()
    assert update.storage_path is None
    assert update.telemetry_enabled is None


def test_app_settings_update_partial_payload() -> None:
    """AppSettingsUpdate accepts a subset of fields."""
    update = AppSettingsUpdate(ocr_enabled=False, telemetry_enabled=True)
    assert update.ocr_enabled is False
    assert update.telemetry_enabled is True
    assert update.storage_path is None


def test_app_settings_read_requires_id_and_timestamps() -> None:
    """AppSettingsRead must require id, created_at, and updated_at."""
    now = datetime.now(tz=timezone.utc)
    settings_id = uuid.uuid4()

    settings = AppSettingsRead(
        id=settings_id,
        created_at=now,
        updated_at=now,
    )
    assert settings.id == settings_id
    assert settings.created_at == now
    assert settings.updated_at == now


def test_app_settings_read_from_attributes() -> None:
    """AppSettingsRead must be constructible from ORM model attributes."""
    from app.models.app_settings import AppSettings

    now = datetime.now(tz=timezone.utc)
    orm_obj = AppSettings()
    orm_obj.id = uuid.uuid4()
    orm_obj.storage_path = "./storage"
    orm_obj.max_file_size_bytes = 10 * 1024 * 1024
    orm_obj.allowed_file_types_jsonb = [
        "application/pdf",
        "text/plain",
        "image/png",
        "image/jpeg",
    ]
    orm_obj.ocr_enabled = True
    orm_obj.chunk_size = 1000
    orm_obj.chunk_overlap = 200
    orm_obj.privacy_local_only = True
    orm_obj.telemetry_enabled = False
    orm_obj.extra_settings_jsonb = {}
    orm_obj.created_at = now
    orm_obj.updated_at = now

    schema = AppSettingsRead.model_validate(orm_obj)
    assert schema.id == orm_obj.id
    assert schema.storage_path == "./storage"


def test_app_settings_read_missing_id_raises() -> None:
    """AppSettingsRead must raise when id is missing."""
    now = datetime.now(tz=timezone.utc)
    with pytest.raises(ValidationError):
        AppSettingsRead(  # type: ignore[call-arg]
            created_at=now,
            updated_at=now,
        )
