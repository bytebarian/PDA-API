"""Repository for semantic chunk search with document-level filters."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, distinct, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.repositories.vector_search_repository import (
    bounded_search_limit,
    cosine_distance,
)
from app.schemas.search_filters import SearchFilters
from app.services.vector_validation import similarity_from_cosine_distance

_MAX_FULL_TEXT_LIMIT = 200


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


@dataclass
class FullTextSearchRow:
    """Raw row returned by a full-text search query."""

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
    ft_score: float


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
        if search_filters is None:
            effective_filters = SearchFilters(
                document_ids=document_ids,
                categories=categories,
                file_types=file_types,
            )
        else:
            effective_filters = search_filters.model_copy(deep=True)
            if effective_filters.document_ids is None and document_ids is not None:
                effective_filters.document_ids = document_ids
            if effective_filters.categories is None and categories is not None:
                effective_filters.categories = categories
            if effective_filters.file_types is None and file_types is not None:
                effective_filters.file_types = file_types
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
            dialect = self._db.bind.dialect.name if self._db.bind is not None else None
            if dialect == "postgresql":
                statement = statement.where(
                    Document.metadata_jsonb.contains(search_filters.metadata)
                )
            else:
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
        counts_stmt = (
            select(
                func.count().label("chunk_count"),
                func.count(distinct(scope.c.document_id)).label("document_count"),
            )
            .select_from(scope)
        )
        counts = (await self._db.execute(counts_stmt)).one()
        candidate_chunk_count = int(counts.chunk_count)
        candidate_document_count = int(counts.document_count)
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

    # ------------------------------------------------------------------
    # Full-text search helpers
    # ------------------------------------------------------------------

    async def full_text_candidates(
        self,
        query: str,
        *,
        limit: int = 30,
        search_filters: SearchFilters | None = None,
    ) -> list[FullTextSearchRow]:
        """Return chunks matching *query* via full-text search.

        On PostgreSQL, ``to_tsvector('simple', content)`` is matched against
        ``websearch_to_tsquery('simple', query)`` and scored with
        ``ts_rank_cd``.  Results are ordered by descending rank, then by
        ascending ``chunk_index`` as a stable tie-breaker.

        On SQLite (used in tests), a case-insensitive ``LIKE`` match is used as
        a fallback so unit tests remain runnable without PostgreSQL.
        """
        query = query.strip()
        if not query:
            return []
        effective_filters = (
            search_filters if search_filters is not None else SearchFilters()
        )
        bounded_limit = min(max(1, limit), _MAX_FULL_TEXT_LIMIT)
        if self._db.bind is None:
            raise RuntimeError("Database session is not bound to an engine")
        dialect = self._db.bind.dialect.name

        if dialect == "postgresql":
            return await self._full_text_postgres(
                query=query,
                filters=effective_filters,
                limit=bounded_limit,
            )
        return await self._full_text_generic(
            query=query,
            filters=effective_filters,
            limit=bounded_limit,
        )

    async def _full_text_postgres(
        self,
        *,
        query: str,
        filters: SearchFilters,
        limit: int,
    ) -> list[FullTextSearchRow]:
        """PostgreSQL full-text search using to_tsvector / websearch_to_tsquery."""
        # Build base filtered subquery (documents only – no embedding constraint)
        doc_stmt = (
            select(Document.id)
            .where(Document.status.in_(filters.resolved_statuses()))
        )
        if filters.document_ids:
            doc_stmt = doc_stmt.where(Document.id.in_(filters.document_ids))
        if filters.categories:
            doc_stmt = doc_stmt.where(Document.category.in_(filters.categories))
        if filters.file_types:
            doc_stmt = doc_stmt.where(Document.file_type.in_(filters.file_types))
        if filters.filename_contains:
            doc_stmt = doc_stmt.where(
                Document.filename.ilike(
                    f"%{self._escape_like(filters.filename_contains)}%",
                    escape="\\",
                )
            )
        if filters.created_from:
            doc_stmt = doc_stmt.where(Document.created_at >= filters.created_from)
        if filters.created_to:
            doc_stmt = doc_stmt.where(Document.created_at < filters.created_to)
        if filters.modified_from:
            doc_stmt = doc_stmt.where(Document.updated_at >= filters.modified_from)
        if filters.modified_to:
            doc_stmt = doc_stmt.where(Document.updated_at < filters.modified_to)
        if filters.metadata:
            doc_stmt = doc_stmt.where(Document.metadata_jsonb.contains(filters.metadata))

        # Full-text search over chunks from filtered documents
        ft_stmt = text(
            """
            SELECT
                c.id            AS chunk_id,
                c.document_id,
                d.filename      AS document_name,
                d.path          AS document_path,
                d.category,
                d.file_type,
                c.page_number,
                c.chunk_index,
                c.source_start_offset AS start_offset,
                c.source_end_offset   AS end_offset,
                c.content,
                c.metadata_jsonb      AS metadata,
                ts_rank_cd(
                    to_tsvector('simple', c.content),
                    websearch_to_tsquery('simple', :query)
                ) AS ft_score
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE
                c.document_id = ANY(:doc_ids::uuid[])
                AND to_tsvector('simple', c.content)
                    @@ websearch_to_tsquery('simple', :query)
            ORDER BY ft_score DESC, c.chunk_index ASC
            LIMIT :lim
            """
        )

        # Materialize the candidate doc IDs so we can pass them as an array
        raw_doc_ids = (await self._db.execute(doc_stmt)).scalars().all()
        if not raw_doc_ids:
            return []

        result = await self._db.execute(
            ft_stmt,
            {"query": query, "doc_ids": list(raw_doc_ids), "lim": limit},
        )
        rows = result.mappings().all()
        return [
            FullTextSearchRow(
                chunk_id=uuid.UUID(str(row["chunk_id"])),
                document_id=uuid.UUID(str(row["document_id"])),
                document_name=str(row["document_name"]),
                document_path=row["document_path"],
                category=row["category"],
                file_type=row["file_type"],
                page_number=row["page_number"],
                chunk_index=int(row["chunk_index"]),
                start_offset=row["start_offset"],
                end_offset=row["end_offset"],
                content=str(row["content"]),
                metadata=dict(row["metadata"]) if row["metadata"] else {},
                ft_score=float(row["ft_score"]),
            )
            for row in rows
        ]

    async def _full_text_generic(
        self,
        *,
        query: str,
        filters: SearchFilters,
        limit: int,
    ) -> list[FullTextSearchRow]:
        """SQLite fallback: case-insensitive LIKE match for testing."""
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status.in_(filters.resolved_statuses()))
        )
        if filters.document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(filters.document_ids))
        if filters.categories:
            stmt = stmt.where(Document.category.in_(filters.categories))
        if filters.file_types:
            stmt = stmt.where(Document.file_type.in_(filters.file_types))
        if filters.filename_contains:
            stmt = stmt.where(
                Document.filename.ilike(
                    f"%{self._escape_like(filters.filename_contains)}%",
                    escape="\\",
                )
            )
        if filters.created_from:
            stmt = stmt.where(Document.created_at >= filters.created_from)
        if filters.created_to:
            stmt = stmt.where(Document.created_at < filters.created_to)
        if filters.modified_from:
            stmt = stmt.where(Document.updated_at >= filters.modified_from)
        if filters.modified_to:
            stmt = stmt.where(Document.updated_at < filters.modified_to)
        if filters.metadata:
            for key, value in filters.metadata.items():
                stmt = stmt.where(
                    func.json_extract(Document.metadata_jsonb, f'$."{key}"') == value
                )

        # Apply LIKE filter for each whitespace-delimited term
        terms = [t.strip() for t in query.split() if t.strip()]
        for term in terms:
            escaped = self._escape_like(term)
            stmt = stmt.where(
                DocumentChunk.content.ilike(f"%{escaped}%", escape="\\")
            )

        rows = (await self._db.execute(stmt)).all()
        scored: list[FullTextSearchRow] = []
        for chunk, doc in rows:
            # Simple term-frequency proxy: count of matched terms
            content_lower = chunk.content.lower()
            term_hits = sum(
                1 for t in terms if t.lower() in content_lower
            )
            ft_score = term_hits / max(len(terms), 1)
            scored.append(
                FullTextSearchRow(
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
                    ft_score=ft_score,
                )
            )

        scored.sort(key=lambda r: (-r.ft_score, r.chunk_index))
        return scored[:limit]
