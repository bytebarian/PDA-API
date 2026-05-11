"""Pydantic schemas/DTOs for AppSettings resources."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.app_settings import (
    DEFAULT_ALLOWED_FILE_TYPES,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    DEFAULT_OCR_PROVIDER,
    DEFAULT_STORAGE_PATH,
)


class AppSettingsBase(BaseModel):
    """Fields shared across create/read/update operations."""

    storage_path: str = DEFAULT_STORAGE_PATH
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    allowed_file_types_jsonb: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_FILE_TYPES))
    ocr_enabled: bool = True
    ocr_provider: str | None = DEFAULT_OCR_PROVIDER
    ocr_language: str | None = DEFAULT_OCR_LANGUAGE
    ocr_dpi: int | None = DEFAULT_OCR_DPI
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = DEFAULT_EMBEDDING_DIMENSIONS
    llm_provider: str | None = DEFAULT_LLM_PROVIDER
    llm_model: str | None = DEFAULT_LLM_MODEL
    privacy_local_only: bool = True
    telemetry_enabled: bool = False
    extra_settings_jsonb: dict[str, Any] = Field(default_factory=dict)


class AppSettingsCreate(AppSettingsBase):
    """Schema for creating app settings."""


class AppSettingsUpdate(BaseModel):
    """Schema for partially updating application settings."""

    storage_path: str | None = None
    max_file_size_bytes: int | None = None
    allowed_file_types_jsonb: list[str] | None = None
    ocr_enabled: bool | None = None
    ocr_provider: str | None = None
    ocr_language: str | None = None
    ocr_dpi: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    privacy_local_only: bool | None = None
    telemetry_enabled: bool | None = None
    extra_settings_jsonb: dict[str, Any] | None = None


class AppSettingsRead(AppSettingsBase):
    """Schema for reading app settings, including DB-generated fields."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
