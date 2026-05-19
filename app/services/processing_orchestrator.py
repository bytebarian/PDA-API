"""Processing orchestration skeleton for document jobs."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.status import DocumentStatus, ProcessingJobStage, ProcessingJobStatus
from app.models.document import Document
from app.models.processing_job import ProcessingJob


class ProcessingJobNotFoundError(LookupError):
    """Raised when a processing job cannot be found."""


class ProcessingOrchestratorStateError(RuntimeError):
    """Raised when a job/document is not in a processable state."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _append_stage_history(
    job: ProcessingJob,
    *,
    stage: ProcessingJobStage,
    status: str,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "stage": stage.value,
        "status": status,
        "timestamp": _utcnow().isoformat(),
    }
    if message is not None:
        entry["message"] = message
    if details is not None:
        entry["details"] = details

    if job.stage_history_jsonb is None:
        job.stage_history_jsonb = []
    job.stage_history_jsonb.append(entry)


def _validate_processable(document: Document, job: ProcessingJob) -> None:
    if job.status != ProcessingJobStatus.awaiting.value:
        raise ProcessingOrchestratorStateError("Processing job is not awaiting")

    allowed_initial_stages = {
        ProcessingJobStage.queued.value,
        ProcessingJobStage.upload_received.value,
    }
    if job.stage not in allowed_initial_stages:
        raise ProcessingOrchestratorStateError("Processing job is not in a queueable stage")

    if document.status not in {
        DocumentStatus.awaiting.value,
        DocumentStatus.failed.value,
    }:
        raise ProcessingOrchestratorStateError("Document is not awaiting processing")


def _mark_processing(document: Document, job: ProcessingJob) -> None:
    job.status = ProcessingJobStatus.processing.value
    document.status = DocumentStatus.processing.value
    job.started_at = _utcnow()
    job.completed_at = None
    job.error_message = None
    job.error_details_jsonb = None
    job.stage_history_jsonb = []
    job.attempt_count += 1


def _mark_ready(document: Document, job: ProcessingJob) -> None:
    document.status = DocumentStatus.ready.value
    job.status = ProcessingJobStatus.ready.value
    job.stage = ProcessingJobStage.completed.value
    job.completed_at = _utcnow()
    _append_stage_history(job, stage=ProcessingJobStage.completed, status="completed")


def _mark_failed(
    document: Document,
    job: ProcessingJob,
    *,
    failed_stage: ProcessingJobStage,
    error: Exception,
) -> None:
    document.status = DocumentStatus.failed.value
    job.status = ProcessingJobStatus.failed.value
    job.stage = ProcessingJobStage.failed.value
    job.completed_at = _utcnow()

    message = str(error) or error.__class__.__name__
    job.error_message = message
    details = {
        "stage": failed_stage.value,
        "error_type": error.__class__.__name__,
        "message": message,
    }
    job.error_details_jsonb = details
    _append_stage_history(
        job,
        stage=failed_stage,
        status="failed",
        message="Stage failed",
        details=details,
    )


async def _run_queued_stage(job: ProcessingJob) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.queued, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.queued, status="completed")


async def _run_upload_received_stage(job: ProcessingJob) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.upload_received, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.upload_received, status="completed")


async def _run_ocr_stage(job: ProcessingJob) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.ocr, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.ocr, status="completed")


async def _run_text_extraction_stage(job: ProcessingJob) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.text_extraction, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.text_extraction, status="completed")


async def _run_chunking_stage(job: ProcessingJob) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.chunking, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.chunking, status="completed")


async def _run_embedding_stage(job: ProcessingJob) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.embedding, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.embedding, status="completed")


async def _run_indexing_stage(job: ProcessingJob) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.indexing, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.indexing, status="completed")


def _stage_flow(start_stage: ProcessingJobStage) -> tuple[tuple[ProcessingJobStage, Any], ...]:
    flow: tuple[tuple[ProcessingJobStage, Any], ...] = (
        (ProcessingJobStage.upload_received, _run_upload_received_stage),
        (ProcessingJobStage.queued, _run_queued_stage),
        (ProcessingJobStage.ocr, _run_ocr_stage),
        (ProcessingJobStage.text_extraction, _run_text_extraction_stage),
        (ProcessingJobStage.chunking, _run_chunking_stage),
        (ProcessingJobStage.embedding, _run_embedding_stage),
        (ProcessingJobStage.indexing, _run_indexing_stage),
    )
    for index, (stage, _) in enumerate(flow):
        if stage == start_stage:
            return flow[index:]
    raise ProcessingOrchestratorStateError(f"Unsupported initial stage: {start_stage.value}")


async def process_job(db: AsyncSession, job_id: uuid.UUID) -> ProcessingJob:
    """Run a processing job through the placeholder orchestration stages."""
    job = await db.get(ProcessingJob, job_id)
    if job is None:
        raise ProcessingJobNotFoundError(f"Processing job not found: {job_id}")

    document = await db.get(Document, job.document_id)
    if document is None:
        raise ProcessingOrchestratorStateError("Document linked to processing job was not found")

    _validate_processable(document, job)
    start_stage = ProcessingJobStage(job.stage)
    _mark_processing(document, job)
    await db.commit()

    for stage, stage_runner in _stage_flow(start_stage):
        job.stage = stage.value
        try:
            await stage_runner(job)
            await db.commit()
        except Exception as error:
            _mark_failed(document, job, failed_stage=stage, error=error)
            await db.commit()
            await db.refresh(job)
            raise

    _mark_ready(document, job)
    await db.commit()
    await db.refresh(job)
    return job
