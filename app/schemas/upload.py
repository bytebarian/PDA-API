"""Pydantic schemas for the file upload endpoint."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from app.domain.status import DocumentStatus, ProcessingJobStage, ProcessingJobStatus


class UploadResponse(BaseModel):
    """Response body returned after a successful document upload."""

    document_id: uuid.UUID
    job_id: uuid.UUID
    filename: str
    status: DocumentStatus
    job_status: ProcessingJobStatus
    job_stage: ProcessingJobStage
