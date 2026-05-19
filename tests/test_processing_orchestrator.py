"""Tests for processing orchestrator stage/state transitions."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.processing_job import ProcessingJob
from app.services import processing_orchestrator
from app.services.processing_orchestrator import ProcessingJobNotFoundError, process_job


async def test_process_job_success_updates_statuses_and_stage_history(db_session: AsyncSession) -> None:
    document = Document(filename="success.pdf", status="awaiting")
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    processed = await process_job(db_session, job.id)

    refreshed_document = await db_session.get(Document, document.id)
    assert refreshed_document is not None

    assert processed.status == "ready"
    assert processed.stage == "completed"
    assert processed.attempt_count == 1
    assert processed.started_at is not None
    assert processed.completed_at is not None
    assert refreshed_document.status == "ready"

    expected_flow = [
        ("queued", "processing"),
        ("queued", "completed"),
        ("ocr", "processing"),
        ("ocr", "completed"),
        ("text_extraction", "processing"),
        ("text_extraction", "completed"),
        ("chunking", "processing"),
        ("chunking", "completed"),
        ("embedding", "processing"),
        ("embedding", "completed"),
        ("indexing", "processing"),
        ("indexing", "completed"),
        ("completed", "completed"),
    ]
    assert [(entry["stage"], entry["status"]) for entry in processed.stage_history_jsonb] == expected_flow
    assert all("timestamp" in entry for entry in processed.stage_history_jsonb)


async def test_process_job_success_replaces_prepopulated_legacy_stage_history(
    db_session: AsyncSession,
) -> None:
    document = Document(filename="reprocess.pdf", status="awaiting")
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(
        document_id=document.id,
        status="awaiting",
        stage="queued",
        stage_history_jsonb=[{"stage": "queued", "force": True, "reason": "reprocess"}],
    )
    db_session.add(job)
    await db_session.commit()

    processed = await process_job(db_session, job.id)

    refreshed_document = await db_session.get(Document, document.id)
    assert refreshed_document is not None

    assert processed.status == "ready"
    assert processed.stage == "completed"
    assert processed.attempt_count == 1
    assert processed.started_at is not None
    assert processed.completed_at is not None
    assert refreshed_document.status == "ready"

    expected_flow = [
        ("queued", "processing"),
        ("queued", "completed"),
        ("ocr", "processing"),
        ("ocr", "completed"),
        ("text_extraction", "processing"),
        ("text_extraction", "completed"),
        ("chunking", "processing"),
        ("chunking", "completed"),
        ("embedding", "processing"),
        ("embedding", "completed"),
        ("indexing", "processing"),
        ("indexing", "completed"),
        ("completed", "completed"),
    ]
    assert [(entry["stage"], entry["status"]) for entry in processed.stage_history_jsonb] == expected_flow
    assert all("timestamp" in entry for entry in processed.stage_history_jsonb)
    assert all("force" not in entry for entry in processed.stage_history_jsonb)
    assert all("reason" not in entry for entry in processed.stage_history_jsonb)


async def test_process_job_missing_job_raises_not_found(db_session: AsyncSession) -> None:
    with pytest.raises(ProcessingJobNotFoundError):
        await process_job(db_session, uuid.uuid4())


async def test_process_job_failed_stage_marks_job_and_document_failed(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = Document(filename="fail.pdf", status="awaiting")
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    async def fail_chunking(_: ProcessingJob) -> None:
        raise RuntimeError("chunking exploded")

    monkeypatch.setattr(processing_orchestrator, "_run_chunking_stage", fail_chunking)

    with pytest.raises(RuntimeError, match="chunking exploded"):
        await process_job(db_session, job.id)

    refreshed_job = await db_session.get(ProcessingJob, job.id)
    refreshed_document = await db_session.get(Document, document.id)

    assert refreshed_job is not None
    assert refreshed_document is not None

    assert refreshed_job.status == "failed"
    assert refreshed_job.stage == "failed"
    assert refreshed_document.status == "failed"

    assert refreshed_job.attempt_count == 1
    assert refreshed_job.started_at is not None
    assert refreshed_job.completed_at is not None
    assert refreshed_job.error_message == "chunking exploded"
    assert refreshed_job.error_details_jsonb == {
        "stage": "chunking",
        "error_type": "RuntimeError",
        "message": "chunking exploded",
    }

    assert refreshed_job.stage_history_jsonb[-1]["stage"] == "chunking"
    assert refreshed_job.stage_history_jsonb[-1]["status"] == "failed"
