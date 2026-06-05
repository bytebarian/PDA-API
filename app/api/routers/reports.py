"""HTTP endpoints for grounded report generation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routers.search import _get_shared_providers
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.reports import ReportGenerateRequest, ReportGenerateResponse
from app.services.report_service import (
    ReportConfigurationError,
    ReportProviderNotAvailableError,
    ReportService,
    ReportServiceError,
)

router = APIRouter(prefix="/reports", tags=["reports"])
logger = logging.getLogger(__name__)


def get_report_service(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ReportService:
    return ReportService(
        db,
        embedding_providers=_get_shared_providers(settings),
        settings=settings,
    )


@router.post(
    "/generate",
    response_model=ReportGenerateResponse,
    summary="Generate a markdown report grounded in indexed document chunks",
    description=(
        "Retrieve relevant document chunks, assemble report-oriented prompt context, "
        "call the configured local model, and return markdown with source citations."
    ),
)
async def generate_report(
    request: ReportGenerateRequest,
    service: ReportService = Depends(get_report_service),
) -> ReportGenerateResponse:
    """Generate a markdown report using retrieved document context."""
    try:
        return await service.generate_report(request)
    except ReportProviderNotAvailableError as exc:
        logger.warning("Report provider unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report model provider is currently unavailable. Please try again later.",
        ) from exc
    except ReportConfigurationError as exc:
        logger.error("Report configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report generation is not configured on this server.",
        ) from exc
    except ReportServiceError as exc:
        logger.error("Report service error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during report generation.",
        ) from exc
