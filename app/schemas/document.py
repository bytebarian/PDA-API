"""Pydantic schemas/DTOs for Document resources."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.domain.status import DocumentStatus, ProcessingJobStage, ProcessingJobStatus


class DocumentBase(BaseModel):
    """Fields shared across create/read/update operations."""

    filename: str
    category: str | None = None
    file_type: str | None = None
    mime_type: str | None = None
    status: DocumentStatus = DocumentStatus.awaiting
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

    model_config = ConfigDict(extra="forbid")

    filename: str | None = None
    category: str | None = None
    file_type: str | None = None
    metadata_jsonb: dict[str, Any] | None = None
    summary: str | None = None


class DocumentRead(DocumentBase):
    """Schema for reading document data, including DB-generated fields."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# List / detail response schemas
# ---------------------------------------------------------------------------


class DocumentSummary(BaseModel):
    """Stable summary fields returned in the list endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    category: str | None
    file_type: str | None
    status: DocumentStatus
    size: int
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class ProcessingJobSummary(BaseModel):
    """Condensed view of the most recent processing job for a document."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: ProcessingJobStatus
    stage: ProcessingJobStage
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class DocumentDetail(BaseModel):
    """Full document detail including optional processing job summary."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    category: str | None
    file_type: str | None
    mime_type: str | None
    status: DocumentStatus
    size: int
    checksum_sha256: str | None
    summary: str | None
    chunk_count: int
    embedding_model: str | None
    last_indexed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    latest_job: ProcessingJobSummary | None


class DocumentListResponse(BaseModel):
    """Paginated list of document summaries."""

    items: list[DocumentSummary]
    page: int
    page_size: int
    total: int
