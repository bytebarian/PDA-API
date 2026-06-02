"""Summarization adapters."""

from app.adapters.summarization.base import (
    SummarizationError,
    SummarizationProvider,
    SummarizationProviderResponseError,
    SummarizationProviderUnavailableError,
    SummarizationResult,
)
from app.adapters.summarization.mock import MockSummarizationProvider
from app.adapters.summarization.ollama import OllamaSummarizationProvider

__all__ = [
    "SummarizationError",
    "SummarizationProvider",
    "SummarizationProviderResponseError",
    "SummarizationProviderUnavailableError",
    "SummarizationResult",
    "MockSummarizationProvider",
    "OllamaSummarizationProvider",
]
