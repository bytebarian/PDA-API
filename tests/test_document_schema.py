"""Tests for Document Pydantic schemas."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.document import DocumentBase, DocumentCreate, DocumentRead, DocumentUpdate


# ---------------------------------------------------------------------------
# DocumentBase
# ---------------------------------------------------------------------------


def test_document_base_requires_filename() -> None:
    """DocumentBase must require a filename."""
    with pytest.raises(ValidationError):
        DocumentBase()  # type: ignore[call-arg]


def test_document_base_defaults() -> None:
    """DocumentBase must apply correct default values."""
    doc = DocumentBase(filename="test.pdf")
    assert doc.filename == "test.pdf"
    assert doc.status == "awaiting"
    assert doc.size == 0
    assert doc.chunk_count == 0
    assert doc.category is None
    assert doc.file_type is None
    assert doc.mime_type is None
    assert doc.path is None
    assert doc.checksum_sha256 is None
    assert doc.metadata_jsonb is None
    assert doc.extracted_text is None
    assert doc.summary is None
    assert doc.embedding_model is None
    assert doc.last_indexed_at is None


def test_document_base_full_payload() -> None:
    """DocumentBase accepts all optional fields."""
    doc = DocumentBase(
        filename="report.pdf",
        category="finance",
        file_type="pdf",
        mime_type="application/pdf",
        status="ready",
        path="/uploads/report.pdf",
        size=4096,
        checksum_sha256="deadbeef",
        metadata_jsonb={"pages": 10},
        extracted_text="text content",
        summary="A short summary",
        chunk_count=5,
        embedding_model="text-embedding-3-small",
    )
    assert doc.category == "finance"
    assert doc.size == 4096
    assert doc.chunk_count == 5
    assert doc.metadata_jsonb == {"pages": 10}


# ---------------------------------------------------------------------------
# DocumentCreate
# ---------------------------------------------------------------------------


def test_document_create_inherits_base_defaults() -> None:
    """DocumentCreate must inherit DocumentBase defaults."""
    doc = DocumentCreate(filename="upload.pdf")
    assert doc.status == "awaiting"
    assert doc.size == 0


def test_document_create_accepts_full_payload() -> None:
    """DocumentCreate accepts all fields from DocumentBase."""
    doc = DocumentCreate(
        filename="upload.pdf",
        category="legal",
        status="awaiting",
    )
    assert doc.filename == "upload.pdf"
    assert doc.category == "legal"


# ---------------------------------------------------------------------------
# DocumentUpdate
# ---------------------------------------------------------------------------


def test_document_update_all_optional() -> None:
    """DocumentUpdate must allow instantiation with no fields."""
    update = DocumentUpdate()
    assert update.filename is None
    assert update.status is None
    assert update.size is None


def test_document_update_partial_payload() -> None:
    """DocumentUpdate accepts a subset of fields."""
    update = DocumentUpdate(status="processing", chunk_count=3)
    assert update.status == "processing"
    assert update.chunk_count == 3
    assert update.filename is None


# ---------------------------------------------------------------------------
# DocumentRead
# ---------------------------------------------------------------------------


def test_document_read_requires_id_and_timestamps() -> None:
    """DocumentRead must require id, created_at, and updated_at."""
    now = datetime.now(tz=timezone.utc)
    doc_id = uuid.uuid4()

    doc = DocumentRead(
        id=doc_id,
        filename="read.pdf",
        created_at=now,
        updated_at=now,
    )
    assert doc.id == doc_id
    assert doc.created_at == now
    assert doc.updated_at == now


def test_document_read_from_attributes() -> None:
    """DocumentRead must be constructible from ORM model attributes."""
    from app.models.document import Document

    now = datetime.now(tz=timezone.utc)
    orm_obj = Document(
        filename="orm.pdf",
        status="awaiting",
        size=0,
        chunk_count=0,
    )
    orm_obj.id = uuid.uuid4()
    orm_obj.created_at = now
    orm_obj.updated_at = now

    schema = DocumentRead.model_validate(orm_obj)
    assert schema.filename == "orm.pdf"
    assert schema.status == "awaiting"
    assert schema.id == orm_obj.id


def test_document_read_missing_id_raises() -> None:
    """DocumentRead must raise when id is missing."""
    now = datetime.now(tz=timezone.utc)
    with pytest.raises(ValidationError):
        DocumentRead(  # type: ignore[call-arg]
            filename="read.pdf",
            created_at=now,
            updated_at=now,
        )
