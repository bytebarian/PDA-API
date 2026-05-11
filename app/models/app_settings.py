"""SQLAlchemy ORM model for the app_settings table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Uuid, Boolean, func, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

DEFAULT_STORAGE_PATH = "./storage"
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_ALLOWED_FILE_TYPES = (
    "application/pdf",
    "text/plain",
    "image/png",
    "image/jpeg",
)
DEFAULT_OCR_PROVIDER = "tesseract"
DEFAULT_OCR_LANGUAGE = "eng"
DEFAULT_OCR_DPI = 300
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_LLM_PROVIDER = "local"
DEFAULT_LLM_MODEL = "llama3.1:8b-instruct"


def _default_allowed_file_types() -> list[str]:
    return list(DEFAULT_ALLOWED_FILE_TYPES)


class AppSettings(Base):
    """Persistence model for application-level settings."""

    __tablename__ = "app_settings"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    storage_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default=DEFAULT_STORAGE_PATH,
    )
    max_file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=DEFAULT_MAX_FILE_SIZE_BYTES,
    )
    allowed_file_types_jsonb: Mapped[list[str]] = mapped_column(
        JSON().with_variant(postgresql.JSONB(), "postgresql"),
        nullable=False,
        default=_default_allowed_file_types,
    )
    ocr_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    ocr_provider: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        default=DEFAULT_OCR_PROVIDER,
    )
    ocr_language: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        default=DEFAULT_OCR_LANGUAGE,
    )
    ocr_dpi: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=DEFAULT_OCR_DPI,
    )
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, default=DEFAULT_CHUNK_SIZE)
    chunk_overlap: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=DEFAULT_CHUNK_OVERLAP,
    )
    embedding_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_dimensions: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
    )
    llm_provider: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        default=DEFAULT_LLM_PROVIDER,
    )
    llm_model: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        default=DEFAULT_LLM_MODEL,
    )
    privacy_local_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    telemetry_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    extra_settings_jsonb: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(postgresql.JSONB(), "postgresql"),
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
