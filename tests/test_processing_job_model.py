"""Tests for the ProcessingJob SQLAlchemy ORM model."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Table
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.processing_job import ProcessingJob


def test_processing_job_table_name() -> None:
    """ProcessingJob model must use the 'processing_jobs' table name."""
    assert ProcessingJob.__tablename__ == "processing_jobs"


def test_processing_job_inherits_base() -> None:
    """ProcessingJob must inherit from the shared declarative Base."""
    from app.db.base import Base

    assert issubclass(ProcessingJob, Base)


def test_processing_job_has_expected_columns() -> None:
    """All required processing job columns must be present in the table mapping."""
    expected = {
        "id",
        "document_id",
        "status",
        "stage",
        "attempt_count",
        "max_attempts",
        "error_message",
        "error_details_jsonb",
        "stage_history_jsonb",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
    }
    actual = {col.name for col in ProcessingJob.__table__.columns}
    assert expected == actual


def test_processing_job_primary_key_is_id() -> None:
    """The 'id' column must be the primary key."""
    pk_cols = {col.name for col in ProcessingJob.__table__.primary_key}
    assert pk_cols == {"id"}


def test_processing_job_has_document_fk() -> None:
    """document_id must have a foreign key to documents.id."""
    document_id_column = ProcessingJob.__table__.c.document_id
    fk_targets = {fk.target_fullname for fk in document_id_column.foreign_keys}
    assert fk_targets == {"documents.id"}


def test_processing_job_indexes_defined() -> None:
    """Processing jobs should define single and composite indexes for retrieval."""
    table: Table = ProcessingJob.__table__  # type: ignore[assignment]
    index_names = {idx.name for idx in table.indexes}
    assert "ix_processing_jobs_document_id" in index_names
    assert "ix_processing_jobs_document_id_status" in index_names


def test_processing_job_defaults() -> None:
    """Default values must match the persistence contract."""
    assert ProcessingJob.__table__.c.status.default is not None
    assert ProcessingJob.__table__.c.status.default.arg == "awaiting"
    assert ProcessingJob.__table__.c.stage.default is not None
    assert ProcessingJob.__table__.c.stage.default.arg == "queued"
    assert ProcessingJob.__table__.c.attempt_count.default is not None
    assert ProcessingJob.__table__.c.attempt_count.default.arg == 0
    assert ProcessingJob.__table__.c.max_attempts.default is not None
    assert ProcessingJob.__table__.c.max_attempts.default.arg == 3
    assert ProcessingJob.__table__.c.stage_history_jsonb.default is not None
    assert callable(ProcessingJob.__table__.c.stage_history_jsonb.default.arg)


def test_document_and_processing_job_relationships_exist() -> None:
    """Document and ProcessingJob should expose reciprocal relationships."""
    assert ProcessingJob.document.property.mapper.class_ is Document
    assert Document.jobs.property.mapper.class_ is ProcessingJob


async def test_processing_job_insert_and_read_defaults(db_session: AsyncSession) -> None:
    """A processing job row can be inserted and retrieved with ORM defaults."""
    document = Document(filename="report.pdf")
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id)
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)

    assert job.id is not None
    assert job.document_id == document.id
    assert job.status == "awaiting"
    assert job.stage == "queued"
    assert job.attempt_count == 0
    assert job.max_attempts == 3
    assert job.stage_history_jsonb == []


async def test_processing_job_json_and_timestamps_roundtrip(
    db_session: AsyncSession,
) -> None:
    """JSON payloads and optional timestamps should persist correctly."""
    document = Document(filename="processing.pdf")
    db_session.add(document)
    await db_session.flush()

    started_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    completed_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    job = ProcessingJob(
        document_id=document.id,
        status="failed",
        stage="ocr",
        attempt_count=1,
        max_attempts=5,
        error_message="OCR failed",
        error_details_jsonb={"provider": "tesseract", "code": "timeout"},
        stage_history_jsonb=[{"stage": "queued"}, {"stage": "ocr"}],
        started_at=started_at,
        completed_at=completed_at,
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)

    assert job.error_details_jsonb == {"provider": "tesseract", "code": "timeout"}
    assert job.stage_history_jsonb == [{"stage": "queued"}, {"stage": "ocr"}]
    assert job.started_at == started_at
    assert job.completed_at == completed_at
