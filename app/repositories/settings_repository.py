"""Repository for the singleton AppSettings row."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.app_settings import AppSettings


class SettingsRepository:
    """Data-access layer for the singleton application settings row."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get(self) -> AppSettings | None:
        """Return the singleton settings row, or None when not yet created."""
        result = await self._db.execute(select(AppSettings).limit(1))
        return result.scalars().first()

    async def get_or_create(self) -> AppSettings:
        """Return the existing row, creating one from ORM defaults when absent."""
        existing = await self.get()
        if existing is not None:
            return existing

        row = AppSettings()
        self._db.add(row)
        await self._db.flush()
        await self._db.refresh(row)
        return row

    async def update(self, row: AppSettings, updates: dict[str, object]) -> AppSettings:
        """Apply a dict of field updates to the given row and flush."""
        for field, value in updates.items():
            setattr(row, field, value)
        self._db.add(row)
        await self._db.flush()
        await self._db.refresh(row)
        return row
