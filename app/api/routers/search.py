"""HTTP endpoints for semantic search.

Exposes:
  POST /search/semantic  – natural-language semantic search over document chunks
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.search import SemanticSearchRequest, SemanticSearchResponse
from app.services.search_service import (
    EmbeddingProviderNotAvailableError,
    SearchConfigurationError,
    SearchService,
    SearchServiceError,
)

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    summary="Semantic search over document chunks",
    description=(
        "Embed the query with the configured embedding provider and return the "
        "most relevant document chunks ordered by vector similarity.  "
        "Only documents with status ``ready`` are searched."
    ),
)
async def semantic_search(
    request: SemanticSearchRequest,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> SemanticSearchResponse:
    """Run semantic search and return ranked chunk results."""
    service = SearchService(db, settings=settings)
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
