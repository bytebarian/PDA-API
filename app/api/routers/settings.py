"""HTTP endpoints for application settings."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.app_settings import (
    SUPPORTED_LLM_MODELS,
    AppSettingsRead,
    AppSettingsUpdate,
)
from app.services.settings_service import SettingsService

router = APIRouter(prefix="/settings", tags=["settings"])
logger = logging.getLogger(__name__)


def get_settings_service(db: AsyncSession = Depends(get_db)) -> SettingsService:
    return SettingsService(db)


@router.get(
    "",
    response_model=AppSettingsRead,
    summary="Return current application settings",
    description=(
        "Return the singleton application settings row.  "
        "A row is created from canonical defaults when none exists yet."
    ),
)
async def get_settings(
    service: SettingsService = Depends(get_settings_service),
) -> AppSettingsRead:
    row = await service.get_settings()
    return AppSettingsRead.model_validate(row)


@router.patch(
    "",
    response_model=AppSettingsRead,
    summary="Partially update application settings",
    description=(
        "Apply a partial update to the singleton application settings row.  "
        f"The ``llm_model`` field must be one of the supported values: "
        f"{sorted(SUPPORTED_LLM_MODELS)}."
    ),
)
async def patch_settings(
    update: AppSettingsUpdate,
    service: SettingsService = Depends(get_settings_service),
) -> AppSettingsRead:
    try:
        row = await service.update_settings(update)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc
    return AppSettingsRead.model_validate(row)
