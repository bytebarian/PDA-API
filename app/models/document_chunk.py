"""SQLAlchemy ORM model for the document_chunks table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.document import Document


class DocumentChunk(Base):
    """Persistence model for a chunk extracted from a document."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_start_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_end_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_jsonb: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(postgresql.JSONB(), "postgresql"),
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector().with_variant(JSON(), "sqlite"),
        nullable=True,
    )
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    @validates("metadata_jsonb")
    def _coerce_metadata_jsonb(
        self, _key: str, value: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Normalize ``None`` to an empty object so NOT NULL inserts succeed.

        Call-sites may pass ``metadata_jsonb=None`` explicitly, which bypasses
        the column ``default`` and would violate the NOT NULL constraint
        (notably on PostgreSQL).
        """
        return {} if value is None else value

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_id_chunk_index",
        ),
        Index(
            "ix_document_chunks_document_id_chunk_index",
            "document_id",
            "chunk_index",
        ),
        Index("ix_document_chunks_embedding_model", "embedding_model"),
    )
