"""Tests for the Document SQLAlchemy ORM model."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document


# ---------------------------------------------------------------------------
# Metadata / structural tests (no DB required)
# ---------------------------------------------------------------------------


def test_document_table_name() -> None:
    """Document model must use the 'documents' table name."""
    assert Document.__tablename__ == "documents"


def test_document_inherits_base() -> None:
    """Document must inherit from the shared declarative Base."""
    from app.db.base import Base

    assert issubclass(Document, Base)


def test_document_has_expected_columns() -> None:
    """All required columns must be present in the table mapping."""
    expected = {
        "id",
        "filename",
        "category",
        "file_type",
        "mime_type",
        "status",
        "path",
        "size",
        "checksum_sha256",
        "metadata_jsonb",
        "extracted_text",
        "summary",
        "chunk_count",
        "embedding_model",
        "last_indexed_at",
        "created_at",
        "updated_at",
    }
    actual = {col.name for col in Document.__table__.columns}
    assert expected == actual


def test_document_primary_key_is_id() -> None:
    """The 'id' column must be the primary key."""
    pk_cols = {col.name for col in Document.__table__.primary_key}
    assert pk_cols == {"id"}


def test_document_checksum_index_defined() -> None:
    """An index on checksum_sha256 must be declared."""
    from sqlalchemy import Table

    table: Table = Document.__table__  # type: ignore[assignment]
    index_names = {idx.name for idx in table.indexes}
    assert "ix_documents_checksum_sha256" in index_names


def test_document_status_column_default() -> None:
    """The status column must declare a default of 'awaiting'."""
    col = Document.__table__.c.status
    assert col.default is not None
    assert col.default.arg == "awaiting"


def test_document_size_column_default() -> None:
    """The size column must declare a default of 0."""
    col = Document.__table__.c.size
    assert col.default is not None
    assert col.default.arg == 0


def test_document_chunk_count_column_default() -> None:
    """The chunk_count column must declare a default of 0."""
    col = Document.__table__.c.chunk_count
    assert col.default is not None
    assert col.default.arg == 0


def test_document_optional_fields_default_none() -> None:
    """Nullable fields must default to None when not supplied."""
    doc = Document(filename="test.pdf")
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


def test_document_id_callable_default() -> None:
    """The id column must declare a callable UUID default."""
    col = Document.__table__.c.id
    assert col.default is not None
    assert callable(col.default.arg)


# ---------------------------------------------------------------------------
# Persistence tests (SQLite in-memory via conftest db_session fixture)
# ---------------------------------------------------------------------------


async def test_document_insert_and_read(db_session: AsyncSession) -> None:
    """A Document row can be inserted and retrieved from the DB."""
    doc = Document(filename="report.pdf", status="awaiting", size=1024)
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    assert doc.id is not None
    assert doc.filename == "report.pdf"
    assert doc.status == "awaiting"
    assert doc.size == 1024


async def test_document_created_at_set_on_insert(db_session: AsyncSession) -> None:
    """created_at must be populated after commit/refresh."""
    doc = Document(filename="invoice.pdf")
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    assert doc.created_at is not None


async def test_document_metadata_jsonb_roundtrip(db_session: AsyncSession) -> None:
    """JSON metadata survives an insert/read cycle."""
    payload = {"pages": 5, "author": "Alice"}
    doc = Document(filename="doc.pdf", metadata_jsonb=payload)
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    assert doc.metadata_jsonb == payload


async def test_document_all_optional_fields(db_session: AsyncSession) -> None:
    """A Document with all fields populated persists correctly."""
    doc = Document(
        filename="full.pdf",
        category="invoice",
        file_type="pdf",
        mime_type="application/pdf",
        status="ready",
        path="/uploads/full.pdf",
        size=2048,
        checksum_sha256="abc123",
        metadata_jsonb={"key": "value"},
        extracted_text="Hello world",
        summary="A summary",
        chunk_count=3,
        embedding_model="text-embedding-3-small",
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    assert doc.category == "invoice"
    assert doc.file_type == "pdf"
    assert doc.mime_type == "application/pdf"
    assert doc.status == "ready"
    assert doc.path == "/uploads/full.pdf"
    assert doc.checksum_sha256 == "abc123"
    assert doc.chunk_count == 3
    assert doc.embedding_model == "text-embedding-3-small"


async def test_document_unique_ids_persisted(db_session: AsyncSession) -> None:
    """Two documents inserted in the same session have distinct UUIDs."""
    doc1 = Document(filename="a.pdf")
    doc2 = Document(filename="b.pdf")
    db_session.add_all([doc1, doc2])
    await db_session.commit()
    await db_session.refresh(doc1)
    await db_session.refresh(doc2)

    assert doc1.id != doc2.id
