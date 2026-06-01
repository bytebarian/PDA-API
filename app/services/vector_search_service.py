"""Business service for vector similarity search."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.repositories.vector_search_repository import VectorSearchRepository
from app.schemas.vector_search import SimilarityResult
from app.services.vector_validation import validate_embedding_vector

logger = logging.getLogger(__name__)


class VectorSearchService:
    def __init__(self, db: AsyncSession, *, settings: Settings | None = None) -> None:
        self._db = db
        self._settings = settings or get_settings()
        self._repository = VectorSearchRepository(db)

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
        validated_query = validate_embedding_vector(
            query_embedding,
            expected_dimensions=self._settings.embedding_dimensions,
            field_name="query_embedding",
        )
        started = time.perf_counter()
        results = await self._repository.search_similar_chunks(
            validated_query,
            limit=limit,
            document_ids=document_ids,
            min_similarity=min_similarity,
            embedding_model=embedding_model,
            metadata_filter=metadata_filter,
        )
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Vector similarity search completed",
            extra={
                "result_count": len(results),
                "document_ids_count": len(document_ids or []),
                "embedding_model": embedding_model,
                "embedding_dimension": self._settings.embedding_dimensions,
                "duration_ms": duration_ms,
            },
        )
        return results
