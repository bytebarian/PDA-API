"""Categorization provider contracts, domain exceptions, and shared vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


# ---------------------------------------------------------------------------
# Allowed category vocabulary
# ---------------------------------------------------------------------------


class DocumentCategory(str, Enum):
    """Stable machine-readable document categories.

    Values are stored in the database as-is and must remain stable across
    migrations.  Display labels may be added in API response schemas.
    """

    contract = "contract"
    agreement = "agreement"
    offer = "offer"
    invoice = "invoice"
    official_document = "official_document"
    documentation = "documentation"
    note = "note"
    other = "other"


ALLOWED_CATEGORIES: frozenset[str] = frozenset(c.value for c in DocumentCategory)


# ---------------------------------------------------------------------------
# Allowed source vocabulary
# ---------------------------------------------------------------------------


class CategorizationSource(str, Enum):
    """Identifies which provider/strategy assigned the category."""

    rules = "rules"
    local_model = "local_model"
    manual = "manual"
    fallback = "fallback"
    mock = "mock"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CategorizationError(RuntimeError):
    """Base class for categorization provider failures."""


class CategorizationProviderUnavailableError(CategorizationError):
    """Raised when the provider is unreachable or times out."""


class CategorizationProviderResponseError(CategorizationError):
    """Raised when the provider returns an invalid or unparseable response."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategorizationResult:
    """Output from a categorization provider call."""

    category: str
    confidence: float
    reason: str
    source: str
    model: str | None = None


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class CategorizationProvider(Protocol):
    """Provider contract for document categorization."""

    name: str

    async def categorize(
        self,
        text: str,
        *,
        filename: str | None = None,
        summary: str | None = None,
        metadata: dict | None = None,
    ) -> CategorizationResult:
        """Assign a category to the document described by the given inputs."""

    async def healthcheck(self) -> bool:
        """Return True when the provider is ready to serve requests."""
