"""Semantic search service orchestrating query embedding and vector retrieval."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderUnavailableError,
)
from app.core.config import Settings, get_settings
from app.models.app_settings import AppSettings
from app.repositories.chunk_repository import ChunkRepository
from app.schemas.search import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from app.services.vector_validation import similarity_from_cosine_distance

logger = logging.getLogger(__name__)

EXCERPT_MAX_CHARS = 300


class SearchServiceError(RuntimeError):
    """Base class for search service failures."""


class EmbeddingProviderNotAvailableError(SearchServiceError):
    """Raised when the configured embedding provider is unreachable."""


class SearchConfigurationError(SearchServiceError):
    """Raised when the search configuration is missing or invalid."""


def _make_excerpt(text: str, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """Return a deterministic excerpt suitable for UI citation cards.

    If *text* fits within *max_chars* it is returned unchanged.  Otherwise
    the text is truncated at the last word boundary before *max_chars* and an
    ellipsis character (U+2026) is appended.  The slice never breaks in the
    middle of a UTF-8 code-point because Python strings are already sequences
    of Unicode characters.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "\u2026"


@dataclass(frozen=True)
class _EmbeddingRuntime:
    provider: str
    model: str
    dimensions: int


def _build_providers(settings: Settings) -> dict[str, EmbeddingProvider]:
    from app.services.embedding_service import build_embedding_providers

    return build_embedding_providers(settings)


class SearchService:
    """Orchestrate query embedding generation and semantic chunk retrieval."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        providers: dict[str, EmbeddingProvider] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._db = db
        self._settings = settings or get_settings()
        self._owns_providers = providers is None
        self._providers = (
            providers if providers is not None else _build_providers(self._settings)
        )
        self._repository = ChunkRepository(db)

    async def semantic_search(
        self,
        request: SemanticSearchRequest,
    ) -> SemanticSearchResponse:
        """Run a full semantic search pipeline for *request*.

        Steps:
        1. Resolve embedding runtime configuration (provider / model / dimensions).
        2. Generate a query embedding using the configured provider.
        3. Run vector similarity search via :class:`ChunkRepository`.
        4. Normalise scores and generate excerpts.
        5. Return a :class:`SemanticSearchResponse`.

        Raises :exc:`EmbeddingProviderNotAvailableError` when the provider is
        unreachable so the router can return a ``503``.
        """
        runtime = await self._resolve_runtime()
        provider = self._providers.get(runtime.provider)
        if provider is None:
            raise SearchConfigurationError(
                f"Unknown embedding provider '{runtime.provider}'"
            )

        started = time.perf_counter()
        try:
            embed_results = await provider.embed_texts(
                [request.query],
                model=runtime.model,
                dimensions=runtime.dimensions,
                truncate=self._settings.embedding_truncate,
            )
        except EmbeddingProviderUnavailableError as exc:
            raise EmbeddingProviderNotAvailableError(
                f"Embedding provider '{runtime.provider}' is unavailable: {exc}"
            ) from exc
        except EmbeddingProviderError as exc:
            raise SearchServiceError(
                f"Embedding provider returned an error: {exc}"
            ) from exc

        from app.services.vector_validation import validate_embedding_vector

        if len(embed_results) != 1:
            raise SearchServiceError(
                f"Embedding provider returned {len(embed_results)} embeddings for 1 query"
            )
        result = embed_results[0]
        if result.text_index != 0:
            raise SearchServiceError(
                f"Embedding provider returned embedding for unexpected index {result.text_index}"
            )

        query_vector = validate_embedding_vector(
            result.vector,
            expected_dimensions=runtime.dimensions,
            field_name="query_embedding",
        )
        actual_model: str = result.model or runtime.model
        rows = await self._repository.semantic_search(
            query_vector,
            limit=request.top_k,
            document_ids=request.document_ids,
            min_score=request.min_score,
            categories=request.categories,
            file_types=request.file_types,
            embedding_model=actual_model,
        )

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Semantic search completed",
            extra={
                "query_length": len(request.query),
                "top_k": request.top_k,
                "result_count": len(rows),
                "embedding_model": actual_model,
                "duration_ms": duration_ms,
            },
        )

        results: list[SemanticSearchResult] = []
        for row in rows:
            score = similarity_from_cosine_distance(row.distance)
            text_value = row.content if request.include_content else None
            excerpt = _make_excerpt(row.content)
            results.append(
                SemanticSearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    document_name=row.document_name,
                    document_path=row.document_path,
                    category=row.category,
                    file_type=row.file_type,
                    page_number=row.page_number,
                    chunk_index=row.chunk_index,
                    start_offset=row.start_offset,
                    end_offset=row.end_offset,
                    text=text_value,
                    excerpt=excerpt,
                    distance=row.distance,
                    score=score,
                    metadata={
                        **(row.metadata or {}),
                        "source": "document_chunks",
                        "embedding_model": actual_model,
                    },
                )
            )

        return SemanticSearchResponse(
            query=request.query,
            embedding_model=actual_model,
            top_k=request.top_k,
            result_count=len(results),
            results=results,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_runtime(self) -> _EmbeddingRuntime:
        """Return provider/model/dimensions resolved from DB → env fallback."""
        persisted = await self._load_app_settings()
        provider = (
            (persisted.embedding_provider if persisted else None)
            or self._settings.embedding_provider
        )
        model = (
            (persisted.embedding_model if persisted else None)
            or self._settings.embedding_model
        )
        raw_dimensions: int | None = (
            persisted.embedding_dimensions if persisted else None
        )
        dimensions: int = (
            raw_dimensions
            if raw_dimensions is not None
            else self._settings.embedding_dimensions
        )
        if not provider or not model or dimensions <= 0:
            raise SearchConfigurationError(
                "Embedding provider, model, and dimensions must be configured"
            )
        return _EmbeddingRuntime(provider=provider, model=model, dimensions=dimensions)

    async def _load_app_settings(self) -> AppSettings | None:
        result = await self._db.execute(
            select(AppSettings).order_by(AppSettings.updated_at.desc()).limit(1)
        )
        return result.scalar_one_or_none()
