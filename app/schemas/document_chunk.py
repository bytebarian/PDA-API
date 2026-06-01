"""Pydantic schemas/DTOs for DocumentChunk resources."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DocumentChunkBase(BaseModel):
    """Fields shared across create/read/update operations."""

    document_id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int = 0
    page_number: int | None = None
    source_start_offset: int | None = None
    source_end_offset: int | None = None
    metadata_jsonb: dict[str, Any] | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_provider: str | None = None
    embedding_dimension: int | None = None
    embedding_created_at: datetime | None = None


class DocumentChunkCreate(DocumentChunkBase):
    """Schema for creating a new document chunk."""


class DocumentChunkUpdate(BaseModel):
    """Schema for partially updating an existing chunk (all fields optional)."""

    chunk_index: int | None = None
    content: str | None = None
    token_count: int | None = None
    page_number: int | None = None
    source_start_offset: int | None = None
    source_end_offset: int | None = None
    metadata_jsonb: dict[str, Any] | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_provider: str | None = None
    embedding_dimension: int | None = None
    embedding_created_at: datetime | None = None


class DocumentChunkRead(DocumentChunkBase):
    """Schema for reading chunk data, including DB-generated fields."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
