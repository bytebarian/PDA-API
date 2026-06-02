"""Mock categorization provider for tests and CI."""

from __future__ import annotations

from app.adapters.categorization.base import (
    CategorizationError,
    CategorizationResult,
    CategorizationSource,
    DocumentCategory,
)


class MockCategorizationProvider:
    """Deterministic in-memory categorization provider for use in tests.

    By default returns ``other`` with full confidence.  Pass ``category`` to
    control the returned category, or ``error`` to simulate a provider failure
    without requiring any external service.
    """

    name: str = CategorizationSource.mock.value

    def __init__(
        self,
        *,
        category: str = DocumentCategory.other.value,
        confidence: float = 1.0,
        reason: str = "mock categorization result",
        error: CategorizationError | None = None,
    ) -> None:
        self._category = category
        self._confidence = confidence
        self._reason = reason
        self._error = error

    async def categorize(
        self,
        text: str,
        *,
        filename: str | None = None,
        summary: str | None = None,
        metadata: dict | None = None,
    ) -> CategorizationResult:
        if self._error is not None:
            raise self._error

        return CategorizationResult(
            category=self._category,
            confidence=self._confidence,
            reason=self._reason,
            source=CategorizationSource.mock.value,
        )

    async def healthcheck(self) -> bool:
        return self._error is None
