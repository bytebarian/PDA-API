"""Service for managing the singleton application settings."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.app_settings import AppSettings
from app.repositories.settings_repository import SettingsRepository
from app.schemas.app_settings import AppSettingsUpdate


class SettingsService:
    """Business logic for reading and updating application settings."""

    def __init__(self, db: AsyncSession) -> None:
        self._repo = SettingsRepository(db)

    async def get_settings(self) -> AppSettings:
        """Return the singleton settings row, creating it from defaults when absent."""
        return await self._repo.get_or_create()

    async def update_settings(self, update: AppSettingsUpdate) -> AppSettings:
        """Apply a partial update and return the updated settings row.

        Only fields explicitly set in the update payload (non-None values) are
        written.  Validation of ``llm_model`` against the allow-list is
        performed by the ``AppSettingsUpdate`` schema before this method is
        called.
        """
        row = await self._repo.get_or_create()
        changes = {
            field: value
            for field, value in update.model_dump(exclude_unset=True).items()
            if value is not None
        }
        if not changes:
            return row
        return await self._repo.update(row, changes)
