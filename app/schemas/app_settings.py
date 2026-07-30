"""Pydantic schemas/DTOs for AppSettings resources."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_STORAGE_PATH = "./storage"
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_ALLOWED_FILE_TYPES = (
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "text/markdown",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
)
DEFAULT_OCR_PROVIDER = "tesseract"
DEFAULT_OCR_LANGUAGE = "eng"
DEFAULT_OCR_DPI = 300
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_DIMENSIONS = 384
DEFAULT_LLM_PROVIDER = "local"
DEFAULT_LLM_MODEL = "llama3.1:8b-instruct-q8_0"

SUPPORTED_LLM_MODELS: frozenset[str] = frozenset(
    {
        "llama3.1:8b-instruct-q8_0",
        "llama3.1:8b",
        "llama3.2:3b",
        "gemma3:1b",
    }
)


def _default_allowed_file_types_jsonb() -> list[str]:
    """Return a fresh copy of the default allowed file types."""

    return list(DEFAULT_ALLOWED_FILE_TYPES)


class AppSettingsBase(BaseModel):
    """Fields shared across create/read/update operations."""

    storage_path: str = DEFAULT_STORAGE_PATH
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    allowed_file_types_jsonb: list[str] = Field(default_factory=_default_allowed_file_types_jsonb)
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

    @field_validator("llm_model")
    @classmethod
    def llm_model_must_be_supported(cls, value: str | None) -> str | None:
        """Reject model identifiers not in the supported allow-list."""
        if value is not None and value not in SUPPORTED_LLM_MODELS:
            sorted_models = sorted(SUPPORTED_LLM_MODELS)
            raise ValueError(
                f"Unsupported llm_model '{value}'. "
                f"Supported values: {sorted_models}"
            )
        return value


class AppSettingsRead(AppSettingsBase):
    """Schema for reading app settings, including DB-generated fields."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
