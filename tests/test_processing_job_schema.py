"""Tests for ProcessingJob Pydantic schemas."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.processing_job import (
    ProcessingJobBase,
    ProcessingJobCreate,
    ProcessingJobRead,
    ProcessingJobUpdate,
)


def test_processing_job_base_requires_document_id() -> None:
    """ProcessingJobBase must require document_id."""
    with pytest.raises(ValidationError):
        ProcessingJobBase()  # type: ignore[call-arg]


def test_processing_job_base_defaults() -> None:
    """ProcessingJobBase must apply correct default values."""
    job = ProcessingJobBase(document_id=uuid.uuid4())
    assert job.status == "awaiting"
    assert job.stage == "queued"
    assert job.attempt_count == 0
    assert job.max_attempts == 3
    assert job.error_message is None
    assert job.error_details_jsonb is None
    assert job.stage_history_jsonb == []
    assert job.started_at is None
    assert job.completed_at is None


def test_processing_job_create_inherits_base_defaults() -> None:
    """ProcessingJobCreate must inherit ProcessingJobBase defaults."""
    job = ProcessingJobCreate(document_id=uuid.uuid4())
    assert job.status == "awaiting"
    assert job.stage_history_jsonb == []


def test_processing_job_update_all_optional() -> None:
    """ProcessingJobUpdate must allow instantiation with no fields."""
    update = ProcessingJobUpdate()
    assert update.status is None
    assert update.stage is None
    assert update.attempt_count is None


def test_processing_job_update_partial_payload() -> None:
    """ProcessingJobUpdate accepts a subset of fields."""
    update = ProcessingJobUpdate(status="processing", stage="chunking", attempt_count=1)  # type: ignore[arg-type]
    assert update.status == "processing"
    assert update.stage == "chunking"
    assert update.attempt_count == 1
    assert update.error_message is None


def test_processing_job_invalid_status_rejected() -> None:
    """Only known status values should be accepted by the schemas."""
    with pytest.raises(ValidationError):
        ProcessingJobBase(  # type: ignore[call-arg]
            document_id=uuid.uuid4(),
            status="unknown",  # type: ignore[arg-type]
        )


def test_processing_job_read_requires_id_and_timestamps() -> None:
    """ProcessingJobRead must require id, created_at, and updated_at."""
    now = datetime.now(tz=timezone.utc)
    job_id = uuid.uuid4()
    document_id = uuid.uuid4()

    job = ProcessingJobRead(
        id=job_id,
        document_id=document_id,
        created_at=now,
        updated_at=now,
    )
    assert job.id == job_id
    assert job.document_id == document_id
    assert job.created_at == now
    assert job.updated_at == now
    assert job.stage_history_jsonb == []


def test_processing_job_read_from_attributes() -> None:
    """ProcessingJobRead must be constructible from ORM model attributes."""
    from app.models.processing_job import ProcessingJob

    now = datetime.now(tz=timezone.utc)
    orm_obj = ProcessingJob(
        document_id=uuid.uuid4(),
        status="awaiting",
        stage="queued",
        attempt_count=0,
        max_attempts=3,
        stage_history_jsonb=[{"stage": "queued"}],
    )
    orm_obj.id = uuid.uuid4()
    orm_obj.created_at = now
    orm_obj.updated_at = now

    schema = ProcessingJobRead.model_validate(orm_obj)
    assert schema.id == orm_obj.id
    assert schema.status == "awaiting"
    assert schema.stage_history_jsonb == [{"stage": "queued"}]
