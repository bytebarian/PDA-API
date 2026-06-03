"""Hybrid search service combining vector similarity and full-text retrieval."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderUnavailableError,
)
from app.core.config import Settings, get_settings
from app.models.app_settings import AppSettings
from app.repositories.chunk_repository import (
    ChunkRepository,
    ChunkSearchRow,
    FullTextSearchRow,
)
from app.schemas.hybrid_search import (
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
)
from app.schemas.search_filters import SearchFilters
from app.services.search_service import (
    EmbeddingProviderNotAvailableError,
    SearchConfigurationError,
    SearchServiceError,
    _make_excerpt,
)
from app.services.vector_validation import (
    similarity_from_cosine_distance,
    validate_embedding_vector,
)

logger = logging.getLogger(__name__)

# RRF smoothing constant – standard value recommended in the literature.
_RRF_K = 60


# ---------------------------------------------------------------------------
# Internal fusion data structures
# ---------------------------------------------------------------------------


@dataclass
class _FusionEntry:
    """Accumulated per-chunk data during result fusion."""

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
    metadata: dict  # type: ignore[type-arg]

    # Per-source diagnostics
    vector_score: float | None = None
    full_text_score: float | None = None
    vector_rank: int | None = None
    full_text_rank: int | None = None
    matched_by: list[Literal["vector", "full_text"]] = field(default_factory=list)

    # Final fused score (populated after fusion)
    score: float = 0.0


# ---------------------------------------------------------------------------
# RRF + weighted fusion helpers
# ---------------------------------------------------------------------------


def _rrf_score(
    vector_rank: int | None,
    full_text_rank: int | None,
    vector_weight: float,
    full_text_weight: float,
    k: int = _RRF_K,
) -> float:
    """Compute the weighted Reciprocal Rank Fusion score for a chunk.

    ``rrf_score = Σ weight(source) * (1 / (k + rank_in_source))``
    """
    score = 0.0
    if vector_rank is not None:
        score += vector_weight * (1.0 / (k + vector_rank))
    if full_text_rank is not None:
        score += full_text_weight * (1.0 / (k + full_text_rank))
    return score


def _weighted_score(
    vector_score: float | None,
    full_text_score: float | None,
    vector_weight: float,
    full_text_weight: float,
) -> float:
    """Compute a normalised weighted combination of component scores.

    Missing scores are ignored (their weights are excluded from normalisation).
    The result is normalised by the sum of weights present so a chunk matched
    by only one source is not artificially penalised relative to the max achievable score.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    if vector_score is not None:
        weighted_sum += vector_weight * vector_score
        total_weight += vector_weight
    if full_text_score is not None:
        weighted_sum += full_text_weight * full_text_score
        total_weight += full_text_weight
    if total_weight == 0.0:
        return 0.0
    return weighted_sum / total_weight


def _fuse_results(
    vector_rows: list[ChunkSearchRow],
    ft_rows: list[FullTextSearchRow],
    *,
    strategy: Literal["rrf", "weighted"],
    vector_weight: float,
    full_text_weight: float,
) -> list[_FusionEntry]:
    """Merge and rank vector + full-text candidates into one ordered list."""
    entries: dict[uuid.UUID, _FusionEntry] = {}

    # --- Incorporate vector candidates ---
    for rank_0, row in enumerate(vector_rows, start=1):
        vector_score = similarity_from_cosine_distance(row.distance)
        entry = _FusionEntry(
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
            content=row.content,
            metadata=row.metadata or {},
            vector_score=vector_score,
            vector_rank=rank_0,
            matched_by=["vector"],
        )
        entries[row.chunk_id] = entry

    # --- Incorporate full-text candidates ---
    ft_score_denom = (
        max((r.ft_score for r in ft_rows), default=0.0) if strategy == "weighted" else 1.0
    )
    for rank_0, ft_row in enumerate(ft_rows, start=1):
        ft_score = (
            (ft_row.ft_score / ft_score_denom)
            if strategy == "weighted" and ft_score_denom > 0.0
            else ft_row.ft_score
        )
        if ft_row.chunk_id in entries:
            existing = entries[ft_row.chunk_id]
            existing.full_text_score = ft_score
            existing.full_text_rank = rank_0
            if "full_text" not in existing.matched_by:
                existing.matched_by.append("full_text")
        else:
            entry = _FusionEntry(
                chunk_id=ft_row.chunk_id,
                document_id=ft_row.document_id,
                document_name=ft_row.document_name,
                document_path=ft_row.document_path,
                category=ft_row.category,
                file_type=ft_row.file_type,
                page_number=ft_row.page_number,
                chunk_index=ft_row.chunk_index,
                start_offset=ft_row.start_offset,
                end_offset=ft_row.end_offset,
                content=ft_row.content,
                metadata=ft_row.metadata or {},
                full_text_score=ft_score,
                full_text_rank=rank_0,
                matched_by=["full_text"],
            )
            entries[ft_row.chunk_id] = entry

    # --- Score each entry ---
    for entry in entries.values():
        if strategy == "rrf":
            entry.score = _rrf_score(
                entry.vector_rank,
                entry.full_text_rank,
                vector_weight=vector_weight,
                full_text_weight=full_text_weight,
            )
        else:
            entry.score = _weighted_score(
                entry.vector_score,
                entry.full_text_score,
                vector_weight=vector_weight,
                full_text_weight=full_text_weight,
            )

    # Stable sort: score desc → best source rank asc → document_id asc → chunk_index asc
    def _sort_key(e: _FusionEntry) -> tuple:  # type: ignore[type-arg]
        best_rank = min(
            r for r in (e.vector_rank, e.full_text_rank) if r is not None
        )
        return (-e.score, best_rank, str(e.document_id), e.chunk_index)

    return sorted(entries.values(), key=_sort_key)


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EmbeddingRuntime:
    provider: str
    model: str
    dimensions: int


def _build_providers(settings: Settings) -> dict[str, EmbeddingProvider]:
    from app.services.embedding_service import build_embedding_providers

    return build_embedding_providers(settings)


class HybridSearchService:
    """Orchestrate hybrid retrieval (vector + full-text) over document chunks."""

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
        self._providers: dict[str, EmbeddingProvider] = (
            providers if providers is not None else _build_providers(self._settings)
        )
        self._repository = ChunkRepository(db)

    async def hybrid_search(
        self,
        request: HybridSearchRequest,
    ) -> HybridSearchResponse:
        """Run the full hybrid retrieval pipeline.

        Steps:
        1. Resolve embedding runtime configuration.
        2. Generate a query embedding.
        3. Fetch vector candidates (filtered).
        4. Fetch full-text candidates (filtered).
        5. Fuse candidates using RRF or weighted strategy.
        6. Apply minScore filter on final fused score.
        7. Take topK.
        8. Return citation-ready response with diagnostics.
        """
        # Normalize weights
        weight_sum = request.vector_weight + request.full_text_weight
        v_weight = request.vector_weight / weight_sum
        ft_weight = request.full_text_weight / weight_sum

        effective_filters: SearchFilters = request.resolved_filters()

        started = time.perf_counter()
        actual_model = self._settings.embedding_model

        # --- Vector candidates ---
        vector_rows: list[ChunkSearchRow] = []
        if request.vector_weight > 0.0:
            runtime = await self._resolve_runtime()
            provider = self._providers.get(runtime.provider)
            if provider is None:
                raise SearchConfigurationError(
                    f"Unknown embedding provider '{runtime.provider}'"
                )

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
                from app.adapters.embeddings import EmbeddingDimensionMismatchError

                if isinstance(exc, EmbeddingDimensionMismatchError):
                    raise SearchConfigurationError(
                        f"Embedding dimensions mismatch for provider '{runtime.provider}'"
                        f" / model '{runtime.model}': {exc}"
                    ) from exc
                raise SearchServiceError(
                    f"Embedding provider returned an error: {exc}"
                ) from exc
            finally:
                if self._owns_providers:
                    for owned_provider in self._providers.values():
                        close = getattr(owned_provider, "aclose", None)
                        if callable(close):
                            try:
                                await close()
                            except Exception:
                                logger.debug(
                                    "Failed to close embedding provider cleanly",
                                    exc_info=True,
                                )

            if len(embed_results) != 1:
                raise SearchServiceError(
                    f"Embedding provider returned {len(embed_results)} embeddings for 1 query"
                )
            result = embed_results[0]
            if result.text_index != 0:
                raise SearchServiceError(
                    "Embedding provider returned embedding for unexpected index "
                    f"{result.text_index}"
                )

            query_vector = validate_embedding_vector(
                result.vector,
                expected_dimensions=runtime.dimensions,
                field_name="query_embedding",
            )
            actual_model = result.model or runtime.model

            vector_search_result = await self._repository.semantic_search(
                query_vector,
                limit=request.effective_vector_top_k(),
                min_score=None,  # minScore applied after fusion
                search_filters=effective_filters,
                embedding_model=actual_model,
            )
            vector_rows = vector_search_result.rows

        # --- Full-text candidates ---
        ft_rows: list[FullTextSearchRow] = []
        if request.full_text_weight > 0.0:
            ft_rows = await self._repository.full_text_candidates(
                request.query,
                limit=request.effective_full_text_top_k(),
                search_filters=effective_filters,
            )

        # Candidate diagnostics for the fused result set (unique chunks/documents
        # returned by either retrieval path).
        vector_candidate_chunk_ids = {row.chunk_id for row in vector_rows}
        ft_candidate_chunk_ids = {row.chunk_id for row in ft_rows}
        vector_candidate_doc_ids = {row.document_id for row in vector_rows}
        ft_candidate_doc_ids = {row.document_id for row in ft_rows}

        candidate_chunk_count = len(vector_candidate_chunk_ids | ft_candidate_chunk_ids)
        candidate_document_count = len(vector_candidate_doc_ids | ft_candidate_doc_ids)

        # --- Fuse ---
        fused = _fuse_results(
            vector_rows,
            ft_rows,
            strategy=request.fusion_strategy,
            vector_weight=v_weight,
            full_text_weight=ft_weight,
        )

        # --- Apply minScore ---
        if request.min_score is not None:
            fused = [e for e in fused if e.score >= request.min_score]

        # --- Take topK ---
        fused = fused[: request.top_k]

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Hybrid search completed",
            extra={
                "query_length": len(request.query),
                "top_k": request.top_k,
                "fusion_strategy": request.fusion_strategy,
                "candidate_document_count": candidate_document_count,
                "candidate_chunk_count": candidate_chunk_count,
                "vector_candidate_count": len(vector_rows),
                "full_text_candidate_count": len(ft_rows),
                "result_count": len(fused),
                "embedding_model": actual_model,
                "duration_ms": duration_ms,
            },
        )

        # --- Build response ---
        results: list[HybridSearchResult] = []
        for entry in fused:
            text_value = entry.content if request.include_content else None
            excerpt = _make_excerpt(entry.content)
            results.append(
                HybridSearchResult(
                    chunk_id=entry.chunk_id,
                    document_id=entry.document_id,
                    document_name=entry.document_name,
                    document_path=entry.document_path,
                    category=entry.category,
                    file_type=entry.file_type,
                    page_number=entry.page_number,
                    chunk_index=entry.chunk_index,
                    start_offset=entry.start_offset,
                    end_offset=entry.end_offset,
                    text=text_value,
                    excerpt=excerpt,
                    score=entry.score,
                    vector_score=entry.vector_score,
                    full_text_score=entry.full_text_score,
                    vector_rank=entry.vector_rank,
                    full_text_rank=entry.full_text_rank,
                    matched_by=entry.matched_by,
                    metadata={
                        **(entry.metadata or {}),
                        "source": "hybrid_retrieval",
                        "embedding_model": actual_model,
                    },
                )
            )

        return HybridSearchResponse(
            query=request.query,
            top_k=request.top_k,
            fusion_strategy=request.fusion_strategy,
            embedding_model=actual_model,
            filters_applied=effective_filters.filters_applied(),
            candidate_document_count=candidate_document_count,
            candidate_chunk_count=candidate_chunk_count,
            vector_candidate_count=len(vector_rows),
            full_text_candidate_count=len(ft_rows),
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
