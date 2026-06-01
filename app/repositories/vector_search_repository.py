"""Repository for chunk-level vector similarity search."""

from __future__ import annotations

import math
import uuid
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.schemas.vector_search import SimilarityResult
from app.services.vector_validation import similarity_from_cosine_distance

MAX_SEARCH_LIMIT = 50


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    cosine_similarity = dot_product / (left_norm * right_norm)
    return 1.0 - cosine_similarity


class VectorSearchRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        document_ids: list[uuid.UUID] | None = None,
        min_similarity: float | None = None,
        embedding_model: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SimilarityResult]:
        bounded_limit = min(max(limit, 1), MAX_SEARCH_LIMIT)
        if self._db.bind is None:
            raise RuntimeError("Database session is not bound to an engine")
        dialect = self._db.bind.dialect.name
        if dialect == "postgresql":
            return await self._search_postgres(
                query_embedding=query_embedding,
                limit=bounded_limit,
                document_ids=document_ids,
                min_similarity=min_similarity,
                embedding_model=embedding_model,
                metadata_filter=metadata_filter,
            )
        return await self._search_generic(
            query_embedding=query_embedding,
            limit=bounded_limit,
            document_ids=document_ids,
            min_similarity=min_similarity,
            embedding_model=embedding_model,
            metadata_filter=metadata_filter,
        )

    def _base_statement(
        self,
        *,
        document_ids: list[uuid.UUID] | None,
        embedding_model: str | None,
        metadata_filter: dict[str, Any] | None,
    ) -> Select[tuple[DocumentChunk, str | None]]:
        statement = (
            select(DocumentChunk, Document.filename)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.embedding.is_not(None))
        )
        if document_ids:
            statement = statement.where(DocumentChunk.document_id.in_(document_ids))
        if embedding_model:
            statement = statement.where(DocumentChunk.embedding_model == embedding_model)
        if metadata_filter:
            statement = statement.where(DocumentChunk.metadata_jsonb.contains(metadata_filter))
        return statement

    async def _search_postgres(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        document_ids: list[uuid.UUID] | None,
        min_similarity: float | None,
        embedding_model: str | None,
        metadata_filter: dict[str, Any] | None,
    ) -> list[SimilarityResult]:
        distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)
        statement = self._base_statement(
            document_ids=document_ids,
            embedding_model=embedding_model,
            metadata_filter=metadata_filter,
        ).add_columns(distance_expr.label("distance"))
        if min_similarity is not None:
            statement = statement.where(distance_expr <= (1.0 - min_similarity))
        statement = statement.order_by(distance_expr.asc()).limit(limit)

        rows = (await self._db.execute(statement)).all()
        return [
            SimilarityResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_name=document_name,
                chunk_index=chunk.chunk_index,
                excerpt=chunk.content,
                page_number=chunk.page_number,
                distance=float(distance),
                similarity=similarity_from_cosine_distance(float(distance)),
                embedding_model=chunk.embedding_model,
                metadata=chunk.metadata_jsonb,
            )
            for chunk, document_name, distance in rows
        ]

    async def _search_generic(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        document_ids: list[uuid.UUID] | None,
        min_similarity: float | None,
        embedding_model: str | None,
        metadata_filter: dict[str, Any] | None,
    ) -> list[SimilarityResult]:
        rows = (
            await self._db.execute(
                self._base_statement(
                    document_ids=document_ids,
                    embedding_model=embedding_model,
                    metadata_filter=metadata_filter,
                )
            )
        ).all()

        scored: list[SimilarityResult] = []
        for chunk, document_name in rows:
            embedding = chunk.embedding
            if embedding is None or len(embedding) != len(query_embedding):
                continue
            distance = _cosine_distance(list(embedding), query_embedding)
            similarity = similarity_from_cosine_distance(distance)
            if min_similarity is not None and similarity < min_similarity:
                continue
            scored.append(
                SimilarityResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_name=document_name,
                    chunk_index=chunk.chunk_index,
                    excerpt=chunk.content,
                    page_number=chunk.page_number,
                    distance=distance,
                    similarity=similarity,
                    embedding_model=chunk.embedding_model,
                    metadata=chunk.metadata_jsonb,
                )
            )

        scored.sort(key=lambda result: result.distance)
        return scored[:limit]
