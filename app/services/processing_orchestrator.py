"""Processing orchestration for document jobs."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.status import DocumentStatus, ProcessingJobStage, ProcessingJobStatus
from app.models.document import Document
from app.models.processing_job import ProcessingJob

# Type alias for stage runner callables.
_StageRunner = Callable[[AsyncSession, Document, ProcessingJob], Awaitable[None]]


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

    if job.attempt_count >= job.max_attempts:
        raise ProcessingOrchestratorStateError(f"Processing job has exhausted retry attempts: {job.attempt_count}/{job.max_attempts}")


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
    last_entry = job.stage_history_jsonb[-1] if job.stage_history_jsonb else None
    if not (
        isinstance(last_entry, dict)
        and last_entry.get("stage") == failed_stage.value
        and last_entry.get("status") == "failed"
    ):
        _append_stage_history(
            job,
            stage=failed_stage,
            status="failed",
            message="Stage failed",
            details=details,
        )


async def _run_queued_stage(
    db: AsyncSession, document: Document, job: ProcessingJob
) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.queued, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.queued, status="completed")


async def _run_upload_received_stage(
    db: AsyncSession, document: Document, job: ProcessingJob
) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.upload_received, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.upload_received, status="completed")


async def _run_ocr_stage(
    db: AsyncSession, document: Document, job: ProcessingJob
) -> None:
    from app.services.ocr_service import OCRService, document_requires_ocr

    if not document_requires_ocr(document):
        _append_stage_history(
            job,
            stage=ProcessingJobStage.ocr,
            status="processing",
        )
        _append_stage_history(
            job,
            stage=ProcessingJobStage.ocr,
            status="completed",
            details={"skipped": True},
        )
        return

    await OCRService(db).extract_text_for_document(document.id, job_id=job.id)


async def _run_text_extraction_stage(
    db: AsyncSession, document: Document, job: ProcessingJob
) -> None:
    from app.core.config import get_settings
    from app.services.ocr_service import document_requires_ocr
    from app.services.file_storage import resolve_stored_file_path
    from app.services.text_extraction import extract_text_from_file

    _append_stage_history(job, stage=ProcessingJobStage.text_extraction, status="processing")

    needs_extraction = (
        not document_requires_ocr(document)
        and (document.extracted_text is None or not document.extracted_text.strip())
    )
    if needs_extraction:
        stored_path = document.path or ""
        resolved_path = resolve_stored_file_path(get_settings().storage_path, stored_path)
        if resolved_path is None:
            raise FileNotFoundError(
                f"Document file path is missing or outside storage root for document {document.id}"
            )
        result = await extract_text_from_file(
            resolved_path,
            mime_type=document.mime_type,
            filename=document.filename,
            document=document,
        )
        document.extracted_text = result.text

    _append_stage_history(
        job,
        stage=ProcessingJobStage.text_extraction,
        status="completed",
        details={"char_count": len((document.extracted_text or "").strip())},
    )


async def _run_normalize_text_stage(
    db: AsyncSession, document: Document, job: ProcessingJob
) -> None:
    from app.core.config import get_settings
    from app.services.text_normalization import (
        TextNormalizationEmptyInputError,
        TextNormalizationEmptyOutputError,
        TextNormalizationOptions,
        normalize_text,
    )

    settings = get_settings()
    _append_stage_history(job, stage=ProcessingJobStage.normalize_text, status="processing")

    if not settings.text_normalization_enabled:
        _append_stage_history(
            job,
            stage=ProcessingJobStage.normalize_text,
            status="completed",
            details={"skipped": True},
        )
        return

    raw_text = document.extracted_text

    # Handle absent/empty input according to pipeline policy.
    if raw_text is None:
        if settings.text_normalization_fail_on_empty_output:
            raise TextNormalizationEmptyInputError(
                "No extracted text available for normalization"
            )
        _append_stage_history(
            job,
            stage=ProcessingJobStage.normalize_text,
            status="completed",
            details={"skipped": True, "reason": "no_input"},
        )
        return

    if raw_text == "":
        if settings.text_normalization_fail_on_empty_output:
            raise TextNormalizationEmptyInputError(
                "Extracted text is empty and cannot be normalized"
            )
        _append_stage_history(
            job,
            stage=ProcessingJobStage.normalize_text,
            status="completed",
            details={"skipped": True, "reason": "empty_input"},
        )
        return

    options = TextNormalizationOptions(
        unicode_form=settings.text_normalization_unicode_form,
        max_blank_lines=settings.text_normalization_max_blank_lines,
        dehyphenate_line_breaks=settings.text_normalization_dehyphenate_line_breaks,
        remove_control_characters=settings.text_normalization_remove_control_chars,
    )
    result = normalize_text(
        raw_text,
        options=options,
        warn_removal_ratio=settings.text_normalization_warn_removal_ratio,
    )
    if settings.text_normalization_fail_on_empty_output and result.output_character_count == 0:
        raise TextNormalizationEmptyOutputError(
            "Normalization produced empty output from non-empty input"
        )

    # Persist the normalized text as the canonical downstream representation.
    # Raw extraction provenance is recorded in metadata_jsonb["normalization"]
    # so it remains inspectable without duplicating large text blobs.
    document.extracted_text = result.normalized_text

    # Keys use snake_case to match the existing pipeline provenance schema
    # (e.g. OCR metadata in metadata_jsonb["ocr"]).
    normalization_meta: dict[str, Any] = {
        "provider": "pda-local-normalizer",
        "rule_set_version": result.rule_set_version,
        "input_character_count": result.input_character_count,
        "output_character_count": result.output_character_count,
        "input_line_count": result.input_line_count,
        "output_line_count": result.output_line_count,
        "changed": result.changed,
        "warnings": [
            {"code": w.code, "message": w.message} for w in result.warnings
        ],
    }
    metadata = dict(document.metadata_jsonb or {})
    metadata["normalization"] = normalization_meta
    document.metadata_jsonb = metadata

    _append_stage_history(
        job,
        stage=ProcessingJobStage.normalize_text,
        status="completed",
        details={
            "input_char_count": result.input_character_count,
            "output_char_count": result.output_character_count,
            "changed": result.changed,
            "rule_set_version": result.rule_set_version,
            "warning_count": len(result.warnings),
        },
    )


async def _run_chunking_stage(
    db: AsyncSession, document: Document, job: ProcessingJob
) -> None:
    from app.services.chunking_service import chunk_document

    _append_stage_history(job, stage=ProcessingJobStage.chunking, status="processing")

    chunks = await chunk_document(db, document)

    _append_stage_history(
        job,
        stage=ProcessingJobStage.chunking,
        status="completed",
        details={"chunk_count": len(chunks)},
    )


async def _run_embedding_stage(
    db: AsyncSession, document: Document, job: ProcessingJob
) -> None:
    from app.services.embedding_service import EmbeddingService

    _append_stage_history(job, stage=ProcessingJobStage.embedding, status="processing")
    result = await EmbeddingService(db).generate_embeddings_for_document(
        document.id,
        job_id=job.id,
    )
    _append_stage_history(
        job,
        stage=ProcessingJobStage.embedding,
        status="completed",
        details={
            "embedded_chunk_count": result.embedded_chunk_count,
            "provider": result.provider,
            "model": result.model,
            "dimensions": result.dimensions,
        },
    )


async def _run_indexing_stage(
    db: AsyncSession, document: Document, job: ProcessingJob
) -> None:
    _append_stage_history(job, stage=ProcessingJobStage.indexing, status="processing")
    _append_stage_history(job, stage=ProcessingJobStage.indexing, status="completed")


def _stage_flow(start_stage: ProcessingJobStage) -> tuple[tuple[ProcessingJobStage, _StageRunner], ...]:
    flow: tuple[tuple[ProcessingJobStage, _StageRunner], ...] = (
        (ProcessingJobStage.upload_received, _run_upload_received_stage),
        (ProcessingJobStage.queued, _run_queued_stage),
        (ProcessingJobStage.ocr, _run_ocr_stage),
        (ProcessingJobStage.text_extraction, _run_text_extraction_stage),
        (ProcessingJobStage.normalize_text, _run_normalize_text_stage),
        (ProcessingJobStage.chunking, _run_chunking_stage),
        (ProcessingJobStage.embedding, _run_embedding_stage),
        (ProcessingJobStage.indexing, _run_indexing_stage),
    )
    for index, (stage, _) in enumerate(flow):
        if stage == start_stage:
            return flow[index:]
    raise ProcessingOrchestratorStateError(f"Processing job is not in a queueable stage: {start_stage.value}")


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
            await stage_runner(db, document, job)
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
