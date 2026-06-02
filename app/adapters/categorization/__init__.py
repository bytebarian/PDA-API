"""Categorization adapters."""

from app.adapters.categorization.base import (
    ALLOWED_CATEGORIES,
    CategorizationError,
    CategorizationProvider,
    CategorizationProviderResponseError,
    CategorizationProviderUnavailableError,
    CategorizationResult,
    CategorizationSource,
    DocumentCategory,
)
from app.adapters.categorization.mock import MockCategorizationProvider
from app.adapters.categorization.ollama import OllamaCategorizationProvider
from app.adapters.categorization.rules import RulesCategorizationProvider

__all__ = [
    "ALLOWED_CATEGORIES",
    "CategorizationError",
    "CategorizationProvider",
    "CategorizationProviderResponseError",
    "CategorizationProviderUnavailableError",
    "CategorizationResult",
    "CategorizationSource",
    "DocumentCategory",
    "MockCategorizationProvider",
    "OllamaCategorizationProvider",
    "RulesCategorizationProvider",
]
