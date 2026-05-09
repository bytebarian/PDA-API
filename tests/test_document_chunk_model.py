"""Tests for the DocumentChunk SQLAlchemy ORM model."""

from __future__ import annotations

from sqlalchemy import Table
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_chunk import DocumentChunk


def test_document_chunk_table_name() -> None:
    """DocumentChunk model must use the 'document_chunks' table name."""
    assert DocumentChunk.__tablename__ == "document_chunks"


def test_document_chunk_inherits_base() -> None:
    """DocumentChunk must inherit from the shared declarative Base."""
    from app.db.base import Base

    assert issubclass(DocumentChunk, Base)


def test_document_chunk_has_expected_columns() -> None:
    """All required chunk columns must be present in the table mapping."""
    expected = {
        "id",
        "document_id",
        "chunk_index",
        "content",
        "token_count",
        "page_number",
        "source_start_offset",
        "source_end_offset",
        "metadata_jsonb",
        "embedding",
        "embedding_model",
        "created_at",
        "updated_at",
    }
    actual = {col.name for col in DocumentChunk.__table__.columns}
    assert expected == actual


def test_document_chunk_primary_key_is_id() -> None:
    """The 'id' column must be the primary key."""
    pk_cols = {col.name for col in DocumentChunk.__table__.primary_key}
    assert pk_cols == {"id"}


def test_document_chunk_has_document_fk() -> None:
    """document_id must have a foreign key to documents.id."""
    document_id_column = DocumentChunk.__table__.c.document_id
    fk_targets = {fk.target_fullname for fk in document_id_column.foreign_keys}
    assert fk_targets == {"documents.id"}


def test_document_chunk_indexes_defined() -> None:
    """Chunk table should define single and composite indexes for retrieval."""
    table: Table = DocumentChunk.__table__  # type: ignore[assignment]
    index_names = {idx.name for idx in table.indexes}
    assert "ix_document_chunks_document_id" in index_names
    assert "ix_document_chunks_document_id_chunk_index" in index_names


def test_document_chunk_unique_constraint_defined() -> None:
    """Chunks should be unique within a document by chunk_index."""
    table: Table = DocumentChunk.__table__  # type: ignore[assignment]
    unique_names = {constraint.name for constraint in table.constraints}
    assert "uq_document_chunks_document_id_chunk_index" in unique_names


def test_document_chunk_token_count_default() -> None:
    """The token_count column must declare a default of 0."""
    col = DocumentChunk.__table__.c.token_count
    assert col.default is not None
    assert col.default.arg == 0


def test_document_and_chunk_relationships_exist() -> None:
    """Document and DocumentChunk should expose reciprocal relationships."""
    assert DocumentChunk.document.property.mapper.class_ is Document
    assert Document.chunks.property.mapper.class_ is DocumentChunk


async def test_document_chunk_insert_and_read(db_session: AsyncSession) -> None:
    """A chunk row can be inserted and retrieved from the DB."""
    document = Document(filename="report.pdf")
    db_session.add(document)
    await db_session.flush()

    chunk = DocumentChunk(document_id=document.id, chunk_index=0, content="hello world")
    db_session.add(chunk)
    await db_session.commit()
    await db_session.refresh(chunk)

    assert chunk.id is not None
    assert chunk.document_id == document.id
    assert chunk.content == "hello world"
    assert chunk.token_count == 0


async def test_document_chunk_embedding_roundtrip(db_session: AsyncSession) -> None:
    """Embedding data should persist in the SQLite fallback path."""
    document = Document(filename="vectors.pdf")
    db_session.add(document)
    await db_session.flush()

    chunk = DocumentChunk(
        document_id=document.id,
        chunk_index=0,
        content="vectorized text",
        embedding=[0.1, 0.2, 0.3],
    )
    db_session.add(chunk)
    await db_session.commit()
    await db_session.refresh(chunk)

    assert chunk.embedding == [0.1, 0.2, 0.3]


async def test_document_chunk_unique_per_document(db_session: AsyncSession) -> None:
    """Duplicate chunk_index values are rejected within the same document."""
    document = Document(filename="unique.pdf")
    db_session.add(document)
    await db_session.flush()

    first = DocumentChunk(document_id=document.id, chunk_index=1, content="A")
    duplicate = DocumentChunk(document_id=document.id, chunk_index=1, content="B")
    db_session.add_all([first, duplicate])

    try:
        await db_session.commit()
    except IntegrityError:
        await db_session.rollback()
    else:  # pragma: no cover
        raise AssertionError("Expected unique constraint violation for duplicate chunk_index")
