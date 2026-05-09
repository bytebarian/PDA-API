"""Pydantic schemas/DTOs for ProcessingJob resources."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProcessingJobStatus = Literal["awaiting", "processing", "ready", "failed"]


class ProcessingJobBase(BaseModel):
    """Fields shared across create/read/update operations."""

    document_id: uuid.UUID
    status: ProcessingJobStatus = "awaiting"
    stage: str = "queued"
    attempt_count: int = 0
    max_attempts: int = 3
    error_message: str | None = None
    error_details_jsonb: dict[str, Any] | None = None
    stage_history_jsonb: list[Any] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ProcessingJobCreate(ProcessingJobBase):
    """Schema for creating a new processing job."""


class ProcessingJobUpdate(BaseModel):
    """Schema for partially updating an existing processing job."""

    status: ProcessingJobStatus | None = None
    stage: str | None = None
    attempt_count: int | None = None
    max_attempts: int | None = None
    error_message: str | None = None
    error_details_jsonb: dict[str, Any] | None = None
    stage_history_jsonb: list[Any] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ProcessingJobRead(ProcessingJobBase):
    """Schema for reading processing job data, including DB-generated fields."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
