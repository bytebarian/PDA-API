"""Tests for DocumentChunk Pydantic schemas."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.document_chunk import (
    DocumentChunkBase,
    DocumentChunkCreate,
    DocumentChunkRead,
    DocumentChunkUpdate,
)


def test_document_chunk_base_requires_fields() -> None:
    """DocumentChunkBase must require document_id, chunk_index, and content."""
    with pytest.raises(ValidationError):
        DocumentChunkBase()  # type: ignore[call-arg]


def test_document_chunk_base_defaults() -> None:
    """DocumentChunkBase must apply correct default values."""
    payload = DocumentChunkBase(
        document_id=uuid.uuid4(),
        chunk_index=0,
        content="chunk text",
    )
    assert payload.token_count == 0
    assert payload.page_number is None
    assert payload.source_start_offset is None
    assert payload.source_end_offset is None
    assert payload.metadata_jsonb is None
    assert payload.embedding is None
    assert payload.embedding_model is None
    assert payload.embedding_provider is None
    assert payload.embedding_dimension is None
    assert payload.embedding_created_at is None


def test_document_chunk_create_inherits_base_defaults() -> None:
    """DocumentChunkCreate must inherit defaults from DocumentChunkBase."""
    payload = DocumentChunkCreate(
        document_id=uuid.uuid4(),
        chunk_index=1,
        content="chunk",
    )
    assert payload.token_count == 0


def test_document_chunk_update_all_optional() -> None:
    """DocumentChunkUpdate must allow instantiation with no fields."""
    update = DocumentChunkUpdate()
    assert update.content is None
    assert update.embedding is None
    assert update.token_count is None


def test_document_chunk_update_partial_payload() -> None:
    """DocumentChunkUpdate accepts a subset of fields."""
    update = DocumentChunkUpdate(token_count=42, embedding_model="text-embedding-3-small")
    assert update.token_count == 42
    assert update.embedding_model == "text-embedding-3-small"
    assert update.content is None


def test_document_chunk_read_requires_id_and_timestamps() -> None:
    """DocumentChunkRead must require id, created_at, and updated_at."""
    now = datetime.now(tz=timezone.utc)
    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()

    chunk = DocumentChunkRead(
        id=chunk_id,
        document_id=document_id,
        chunk_index=2,
        content="read text",
        created_at=now,
        updated_at=now,
    )
    assert chunk.id == chunk_id
    assert chunk.document_id == document_id
    assert chunk.created_at == now
    assert chunk.updated_at == now


def test_document_chunk_read_from_attributes() -> None:
    """DocumentChunkRead must be constructible from ORM model attributes."""
    from app.models.document_chunk import DocumentChunk

    now = datetime.now(tz=timezone.utc)
    orm_obj = DocumentChunk(
        document_id=uuid.uuid4(),
        chunk_index=0,
        content="orm chunk",
    )
    orm_obj.token_count = 0
    orm_obj.id = uuid.uuid4()
    orm_obj.created_at = now
    orm_obj.updated_at = now

    schema = DocumentChunkRead.model_validate(orm_obj)
    assert schema.id == orm_obj.id
    assert schema.content == "orm chunk"
