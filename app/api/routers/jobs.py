from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.processing_job import ProcessingJob
from app.schemas.processing_job import ProcessingJobRead

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id:uuid}",
    response_model=ProcessingJobRead,
    summary="Get processing job status",
)
async def get_processing_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ProcessingJobRead:
    job = await db.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processing job not found",
        )
    return ProcessingJobRead.model_validate(job)
