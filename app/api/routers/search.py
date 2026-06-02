"""HTTP endpoints for semantic search.

Exposes:
  POST /search/semantic  – natural-language semantic search over document chunks
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embeddings import EmbeddingProvider
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.search import SemanticSearchRequest, SemanticSearchResponse
from app.services.embedding_service import build_embedding_providers
from app.services.search_service import (
    EmbeddingProviderNotAvailableError,
    SearchConfigurationError,
    SearchService,
    SearchServiceError,
)

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)
_PROVIDER_CACHE: dict[tuple[Any, ...], dict[str, EmbeddingProvider]] = {}


def _provider_cache_key(settings: Settings) -> tuple[Any, ...]:
    return (
        settings.ollama_base_url,
        settings.embedding_timeout_seconds,
    )


def _get_shared_providers(settings: Settings) -> dict[str, EmbeddingProvider]:
    cache_key = _provider_cache_key(settings)
    providers = _PROVIDER_CACHE.get(cache_key)
    if providers is None:
        providers = build_embedding_providers(settings)
        _PROVIDER_CACHE[cache_key] = providers
    return providers


async def close_search_providers() -> None:
    for providers in _PROVIDER_CACHE.values():
        for provider in providers.values():
            close = getattr(provider, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    logger.debug(
                        "Failed to close embedding provider cleanly",
                        exc_info=True,
                    )
    _PROVIDER_CACHE.clear()


def get_search_service(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> SearchService:
    return SearchService(
        db,
        providers=_get_shared_providers(settings),
        settings=settings,
    )


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    summary="Semantic search over document chunks",
    description=(
        "Embed the query with the configured embedding provider and return the "
        "most relevant document chunks ordered by vector similarity. "
        "Document-level metadata filters are applied before vector ordering. "
        "By default, only documents with status ``ready`` are searched."
    ),
)
async def semantic_search(
    request: SemanticSearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SemanticSearchResponse:
    """Run semantic search and return ranked chunk results."""
    try:
        return await service.semantic_search(request)
    except EmbeddingProviderNotAvailableError as exc:
        logger.warning("Embedding provider unavailable during search: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding provider is currently unavailable. Please try again later.",
        ) from exc
    except SearchConfigurationError as exc:
        logger.error("Search configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search is not configured on this server.",
        ) from exc
    except SearchServiceError as exc:
        logger.error("Search service error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during search.",
        ) from exc
