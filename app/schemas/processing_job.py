"""Pydantic schemas/DTOs for ProcessingJob resources."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.status import ProcessingJobStage, ProcessingJobStatus


class ProcessingJobBase(BaseModel):
    """Fields shared across create/read/update operations."""

    document_id: uuid.UUID
    status: ProcessingJobStatus = ProcessingJobStatus.awaiting
    stage: ProcessingJobStage = ProcessingJobStage.queued
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
    stage: ProcessingJobStage | None = None
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
