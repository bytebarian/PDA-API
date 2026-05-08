"""Pydantic schemas/DTOs for Document resources."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DocumentBase(BaseModel):
    """Fields shared across create/read/update operations."""

    filename: str
    category: str | None = None
    file_type: str | None = None
    mime_type: str | None = None
    status: str = "awaiting"
    path: str | None = None
    size: int = 0
    checksum_sha256: str | None = None
    metadata_jsonb: dict[str, Any] | None = None
    extracted_text: str | None = None
    summary: str | None = None
    chunk_count: int = 0
    embedding_model: str | None = None
    last_indexed_at: datetime | None = None


class DocumentCreate(DocumentBase):
    """Schema for creating a new document record."""


class DocumentUpdate(BaseModel):
    """Schema for partially updating an existing document (all fields optional)."""

    filename: str | None = None
    category: str | None = None
    file_type: str | None = None
    mime_type: str | None = None
    status: str | None = None
    path: str | None = None
    size: int | None = None
    checksum_sha256: str | None = None
    metadata_jsonb: dict[str, Any] | None = None
    extracted_text: str | None = None
    summary: str | None = None
    chunk_count: int | None = None
    embedding_model: str | None = None
    last_indexed_at: datetime | None = None


class DocumentRead(DocumentBase):
    """Schema for reading document data, including DB-generated fields."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
