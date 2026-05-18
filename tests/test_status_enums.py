"""Tests for shared status/stage enums and their integration with schemas."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.domain.status import DocumentStatus, ProcessingJobStage, ProcessingJobStatus
from app.schemas.document import DocumentBase, DocumentUpdate
from app.schemas.processing_job import ProcessingJobBase, ProcessingJobUpdate


# ---------------------------------------------------------------------------
# Enum member completeness
# ---------------------------------------------------------------------------


def test_document_status_members() -> None:
    """DocumentStatus must define the required status vocabulary."""
    members = {s.value for s in DocumentStatus}
    assert members == {"awaiting", "processing", "ready", "failed"}


def test_processing_job_status_members() -> None:
    """ProcessingJobStatus must define the required status vocabulary."""
    members = {s.value for s in ProcessingJobStatus}
    assert members == {"awaiting", "processing", "ready", "failed"}


def test_processing_job_stage_members() -> None:
    """ProcessingJobStage must define the recommended stage vocabulary."""
    members = {s.value for s in ProcessingJobStage}
    assert {
        "queued",
        "upload_received",
        "ocr",
        "text_extraction",
        "chunking",
        "embedding",
        "indexing",
        "completed",
        "failed",
    }.issubset(members)


def test_enums_are_str_subclass() -> None:
    """All status/stage enums must be str subclasses for JSON serialisation."""
    assert isinstance(DocumentStatus.awaiting, str)
    assert isinstance(ProcessingJobStatus.processing, str)
    assert isinstance(ProcessingJobStage.chunking, str)


# ---------------------------------------------------------------------------
# Default stability
# ---------------------------------------------------------------------------


def test_document_default_status() -> None:
    """DocumentBase must default to DocumentStatus.awaiting."""
    doc = DocumentBase(filename="test.pdf")
    assert doc.status == DocumentStatus.awaiting
    assert doc.status == "awaiting"


def test_processing_job_default_status() -> None:
    """ProcessingJobBase must default to ProcessingJobStatus.awaiting."""
    job = ProcessingJobBase(document_id=uuid.uuid4())
    assert job.status == ProcessingJobStatus.awaiting
    assert job.status == "awaiting"


def test_processing_job_default_stage() -> None:
    """ProcessingJobBase must default to ProcessingJobStage.queued."""
    job = ProcessingJobBase(document_id=uuid.uuid4())
    assert job.stage == ProcessingJobStage.queued
    assert job.stage == "queued"


# ---------------------------------------------------------------------------
# Valid values accepted
# ---------------------------------------------------------------------------


def test_document_base_accepts_all_valid_statuses() -> None:
    """DocumentBase must accept every value in DocumentStatus."""
    for status in DocumentStatus:
        doc = DocumentBase(filename="f.pdf", status=status)
        assert doc.status == status


def test_processing_job_base_accepts_all_valid_statuses() -> None:
    """ProcessingJobBase must accept every value in ProcessingJobStatus."""
    for status in ProcessingJobStatus:
        job = ProcessingJobBase(document_id=uuid.uuid4(), status=status)
        assert job.status == status


def test_processing_job_base_accepts_all_valid_stages() -> None:
    """ProcessingJobBase must accept every value in ProcessingJobStage."""
    for stage in ProcessingJobStage:
        job = ProcessingJobBase(document_id=uuid.uuid4(), stage=stage)
        assert job.stage == stage


# ---------------------------------------------------------------------------
# Invalid values rejected
# ---------------------------------------------------------------------------


def test_document_base_rejects_invalid_status() -> None:
    """DocumentBase must reject status values not in DocumentStatus."""
    with pytest.raises(ValidationError):
        DocumentBase(filename="f.pdf", status="unknown")  # type: ignore[arg-type]


def test_document_update_rejects_status_field() -> None:
    """DocumentUpdate must reject status because it is not updatable metadata."""
    with pytest.raises(ValidationError):
        DocumentUpdate(status="pending")  # type: ignore[call-arg]


def test_processing_job_base_rejects_invalid_status() -> None:
    """ProcessingJobBase must reject status values not in ProcessingJobStatus."""
    with pytest.raises(ValidationError):
        ProcessingJobBase(
            document_id=uuid.uuid4(),
            status="bad_status",  # type: ignore[arg-type]
        )


def test_processing_job_base_rejects_invalid_stage() -> None:
    """ProcessingJobBase must reject stage values not in ProcessingJobStage."""
    with pytest.raises(ValidationError):
        ProcessingJobBase(
            document_id=uuid.uuid4(),
            stage="bad_stage",  # type: ignore[arg-type]
        )


def test_processing_job_update_rejects_invalid_status() -> None:
    """ProcessingJobUpdate must reject status values not in ProcessingJobStatus."""
    with pytest.raises(ValidationError):
        ProcessingJobUpdate(status="unknown")  # type: ignore[arg-type]


def test_processing_job_update_rejects_invalid_stage() -> None:
    """ProcessingJobUpdate must reject stage values not in ProcessingJobStage."""
    with pytest.raises(ValidationError):
        ProcessingJobUpdate(stage="bad_stage")  # type: ignore[arg-type]
