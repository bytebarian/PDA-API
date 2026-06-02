"""Repository for semantic chunk search with document-level filters."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.status import DocumentStatus
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

MAX_SEARCH_LIMIT = 50


def _cosine_distance(left: list[float], right: list[float]) -> float:
    """Compute cosine distance in [0, 2] between two equal-length vectors."""
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    cosine_similarity = dot_product / (left_norm * right_norm)
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    return 1.0 - cosine_similarity


def score_from_distance(distance: float) -> float:
    """Map cosine distance [0, 2] to a stable score in [0, 1].

    Uses the formula ``max(0, 1 - distance / 2)`` so the returned value is
    always non-negative and reaches 1.0 for identical vectors.
    """
    return max(0.0, 1.0 - distance / 2.0)


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


class ChunkRepository:
    """Repository for semantic similarity search over document chunks."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def semantic_search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        document_ids: list[uuid.UUID] | None = None,
        min_score: float | None = None,
        categories: list[str] | None = None,
        file_types: list[str] | None = None,
        embedding_model: str | None = None,
    ) -> list[ChunkSearchRow]:
        """Search chunks by vector similarity, returning at most *limit* rows.

        Only chunks belonging to documents with status ``ready`` are
        considered.  Chunks without an embedding are excluded.  Results are
        ordered by ascending cosine distance (most similar first), with
        ``chunk_index`` as a stable tie-breaker.
        """
        bounded_limit = min(max(limit, 1), MAX_SEARCH_LIMIT)
        if self._db.bind is None:
            raise RuntimeError("Database session is not bound to an engine")
        dialect = self._db.bind.dialect.name
        if dialect == "postgresql":
            return await self._search_postgres(
                query_embedding=query_embedding,
                limit=bounded_limit,
                document_ids=document_ids,
                min_score=min_score,
                categories=categories,
                file_types=file_types,
                embedding_model=embedding_model,
            )
        return await self._search_generic(
            query_embedding=query_embedding,
            limit=bounded_limit,
            document_ids=document_ids,
            min_score=min_score,
            categories=categories,
            file_types=file_types,
            embedding_model=embedding_model,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_statement(
        self,
        *,
        document_ids: list[uuid.UUID] | None,
        categories: list[str] | None,
        file_types: list[str] | None,
        embedding_model: str | None,
    ) -> Select[tuple[DocumentChunk, Document]]:
        statement = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == DocumentStatus.ready.value)
            .where(DocumentChunk.embedding.is_not(None))
        )
        if document_ids:
            statement = statement.where(DocumentChunk.document_id.in_(document_ids))
        if categories:
            statement = statement.where(Document.category.in_(categories))
        if file_types:
            statement = statement.where(Document.file_type.in_(file_types))
        if embedding_model:
            statement = statement.where(DocumentChunk.embedding_model == embedding_model)
        return statement

    async def _search_postgres(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        document_ids: list[uuid.UUID] | None,
        min_score: float | None,
        categories: list[str] | None,
        file_types: list[str] | None,
        embedding_model: str | None,
    ) -> list[ChunkSearchRow]:
        distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)
        statement = (
            self._base_statement(
                document_ids=document_ids,
                categories=categories,
                file_types=file_types,
                embedding_model=embedding_model,
            )
            .where(DocumentChunk.embedding_dimension == len(query_embedding))
            .add_columns(distance_expr.label("distance"))
        )
        if min_score is not None:
            # score = max(0, 1 - distance / 2) >= min_score
            # ⟹ distance <= 2 * (1 - min_score)
            statement = statement.where(distance_expr <= 2.0 * (1.0 - min_score))
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
        limit: int,
        document_ids: list[uuid.UUID] | None,
        min_score: float | None,
        categories: list[str] | None,
        file_types: list[str] | None,
        embedding_model: str | None,
    ) -> list[ChunkSearchRow]:
        rows = (
            await self._db.execute(
                self._base_statement(
                    document_ids=document_ids,
                    categories=categories,
                    file_types=file_types,
                    embedding_model=embedding_model,
                )
            )
        ).all()

        scored: list[ChunkSearchRow] = []
        for chunk, doc in rows:
            embedding = chunk.embedding
            if embedding is None or len(embedding) != len(query_embedding):
                continue
            distance = _cosine_distance(list(embedding), query_embedding)
            score = score_from_distance(distance)
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
