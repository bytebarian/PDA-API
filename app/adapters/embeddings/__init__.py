"""Embeddings provider implementations."""

from app.adapters.embeddings.base import (
    EmbeddingDimensionMismatchError,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderResponseError,
    EmbeddingProviderUnavailableError,
    EmbeddingResult,
)
from app.adapters.embeddings.fake import FakeEmbeddingProvider
from app.adapters.embeddings.ollama import OllamaEmbeddingProvider

__all__ = [
    "EmbeddingDimensionMismatchError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingProviderResponseError",
    "EmbeddingProviderUnavailableError",
    "EmbeddingResult",
    "FakeEmbeddingProvider",
    "OllamaEmbeddingProvider",
]
