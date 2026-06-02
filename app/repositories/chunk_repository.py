"""Repository for semantic chunk search with document-level filters."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.repositories.vector_search_repository import (
    bounded_search_limit,
    cosine_distance,
)
from app.schemas.search_filters import SearchFilters
from app.services.vector_validation import similarity_from_cosine_distance


@dataclass
class ChunkSearchRow:
    """Raw row returned by the semantic search query."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    document_path: str | None
    category: str | None
    file_type: str | None
    page_number: int | None
    chunk_index: int
    start_offset: int | None
    end_offset: int | None
    content: str
    metadata: dict[str, Any]
    embedding_model: str | None
    distance: float


@dataclass
class ChunkSearchDiagnostics:
    """Counts describing search scope before and after vector ranking."""

    candidate_document_count: int
    candidate_chunk_count: int


@dataclass
class ChunkSearchResult:
    """Semantic search rows plus candidate diagnostics."""

    rows: list[ChunkSearchRow]
    diagnostics: ChunkSearchDiagnostics


class ChunkRepository:
    """Repository for semantic similarity search over document chunks."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def semantic_search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        search_filters: SearchFilters | None = None,
        document_ids: list[uuid.UUID] | None = None,
        min_score: float | None = None,
        categories: list[str] | None = None,
        file_types: list[str] | None = None,
        embedding_model: str | None = None,
    ) -> ChunkSearchResult:
        """Search chunks by vector similarity, returning at most *limit* rows.

        Only chunks belonging to documents with status ``ready`` are
        considered.  Chunks without an embedding are excluded.  Results are
        ordered by ascending cosine distance (most similar first), with
        ``chunk_index`` as a stable tie-breaker.
        """
        bounded_limit = bounded_search_limit(limit)
        effective_filters = search_filters or SearchFilters(
            document_ids=document_ids,
            categories=categories,
            file_types=file_types,
        )
        if self._db.bind is None:
            raise RuntimeError("Database session is not bound to an engine")
        dialect = self._db.bind.dialect.name
        base_statement = self._base_statement(
            search_filters=effective_filters,
            embedding_model=embedding_model,
            embedding_dimensions=len(query_embedding),
        )
        diagnostics = await self._candidate_diagnostics(base_statement)

        if diagnostics.candidate_chunk_count == 0:
            return ChunkSearchResult(rows=[], diagnostics=diagnostics)

        if dialect == "postgresql":
            rows = await self._search_postgres(
                query_embedding=query_embedding,
                base_statement=base_statement,
                limit=bounded_limit,
                min_score=min_score,
            )
        else:
            rows = await self._search_generic(
                query_embedding=query_embedding,
                base_statement=base_statement,
                limit=bounded_limit,
                min_score=min_score,
            )
        return ChunkSearchResult(rows=rows, diagnostics=diagnostics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_statement(
        self,
        *,
        search_filters: SearchFilters,
        embedding_model: str | None,
        embedding_dimensions: int,
    ) -> Select[tuple[DocumentChunk, Document]]:
        statement: Select[tuple[DocumentChunk, Document]] = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.embedding.is_not(None))
            .where(DocumentChunk.embedding_dimension == embedding_dimensions)
            .where(Document.status.in_(search_filters.resolved_statuses()))
        )
        if search_filters.document_ids:
            statement = statement.where(DocumentChunk.document_id.in_(search_filters.document_ids))
        if search_filters.categories:
            statement = statement.where(Document.category.in_(search_filters.categories))
        if search_filters.file_types:
            statement = statement.where(Document.file_type.in_(search_filters.file_types))
        if search_filters.filename_contains:
            statement = statement.where(
                Document.filename.ilike(
                    f"%{self._escape_like(search_filters.filename_contains)}%",
                    escape="\\",
                )
            )
        if search_filters.created_from:
            statement = statement.where(Document.created_at >= search_filters.created_from)
        if search_filters.created_to:
            statement = statement.where(Document.created_at < search_filters.created_to)
        if search_filters.modified_from:
            statement = statement.where(Document.updated_at >= search_filters.modified_from)
        if search_filters.modified_to:
            statement = statement.where(Document.updated_at < search_filters.modified_to)
        if search_filters.metadata:
            if self._db.bind is not None and self._db.bind.dialect.name == "postgresql":
                statement = statement.where(Document.metadata_jsonb.contains(search_filters.metadata))
                for key, value in search_filters.metadata.items():
                    statement = statement.where(
                        func.json_extract(Document.metadata_jsonb, f'$."{key}"') == value
                    )
        if embedding_model:
            statement = statement.where(DocumentChunk.embedding_model == embedding_model)
        return statement

    @staticmethod
    def _escape_like(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    async def _candidate_diagnostics(
        self,
        base_statement: Select[tuple[DocumentChunk, Document]],
    ) -> ChunkSearchDiagnostics:
        scope = base_statement.subquery()
        chunk_count_stmt = select(func.count()).select_from(scope)
        doc_count_stmt = select(func.count(distinct(scope.c.document_id)))
        candidate_chunk_count = int((await self._db.execute(chunk_count_stmt)).scalar_one())
        candidate_document_count = int((await self._db.execute(doc_count_stmt)).scalar_one())
        return ChunkSearchDiagnostics(
            candidate_document_count=candidate_document_count,
            candidate_chunk_count=candidate_chunk_count,
        )

    async def _search_postgres(
        self,
        *,
        query_embedding: list[float],
        base_statement: Select[tuple[DocumentChunk, Document]],
        limit: int,
        min_score: float | None,
    ) -> list[ChunkSearchRow]:
        distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)
        statement = (
            base_statement
            .add_columns(distance_expr.label("distance"))
        )
        if min_score is not None and min_score > 0.0:
            # score = max(0, 1 - distance / 2) >= min_score
            # max(0, ...) does not affect the bound for positive scores.
            # => distance <= 2 * (1 - min_score)
            statement = statement.where(distance_expr <= (2.0 * (1.0 - min_score)))
        statement = (
            statement.order_by(distance_expr.asc(), DocumentChunk.chunk_index.asc())
            .limit(limit)
        )

        rows = (await self._db.execute(statement)).all()
        return [
            ChunkSearchRow(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_name=doc.filename,
                document_path=doc.path,
                category=doc.category,
                file_type=doc.file_type,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                start_offset=chunk.source_start_offset,
                end_offset=chunk.source_end_offset,
                content=chunk.content,
                metadata=chunk.metadata_jsonb or {},
                embedding_model=chunk.embedding_model,
                distance=float(distance),
            )
            for chunk, doc, distance in rows
        ]

    async def _search_generic(
        self,
        *,
        query_embedding: list[float],
        base_statement: Select[tuple[DocumentChunk, Document]],
        limit: int,
        min_score: float | None,
    ) -> list[ChunkSearchRow]:
        rows = (await self._db.execute(base_statement)).all()

        scored: list[ChunkSearchRow] = []
        for chunk, doc in rows:
            embedding = chunk.embedding
            if embedding is None:
                continue
            if len(embedding) != len(query_embedding):
                continue
            distance = cosine_distance(list(embedding), query_embedding)
            score = similarity_from_cosine_distance(distance)
            if min_score is not None and score < min_score:
                continue
            scored.append(
                ChunkSearchRow(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_name=doc.filename,
                    document_path=doc.path,
                    category=doc.category,
                    file_type=doc.file_type,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    start_offset=chunk.source_start_offset,
                    end_offset=chunk.source_end_offset,
                    content=chunk.content,
                    metadata=chunk.metadata_jsonb or {},
                    embedding_model=chunk.embedding_model,
                    distance=distance,
                )
            )

        scored.sort(key=lambda r: (r.distance, r.chunk_index))
        return scored[:limit]
