"""Schemas for document reprocess requests and responses."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from app.domain.status import DocumentStatus, ProcessingJobStage, ProcessingJobStatus


class ReprocessRequest(BaseModel):
    """Optional request payload for reprocess endpoint."""

    force: bool = False
    reason: str | None = None


class ReprocessResponse(BaseModel):
    """Stable response returned after requesting document reprocessing."""

    document_id: uuid.UUID
    job_id: uuid.UUID
    document_status: DocumentStatus
    job_status: ProcessingJobStatus
    job_stage: ProcessingJobStage
