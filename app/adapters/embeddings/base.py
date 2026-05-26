"""Embedding provider contracts and domain exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EmbeddingProviderError(RuntimeError):
    """Base class for embedding provider failures."""


class EmbeddingProviderUnavailableError(EmbeddingProviderError):
    """Raised when the provider is unreachable or times out."""


class EmbeddingProviderResponseError(EmbeddingProviderError):
    """Raised when the provider returns an invalid response."""


class EmbeddingDimensionMismatchError(EmbeddingProviderError):
    """Raised when returned vectors do not match expected dimensions."""


@dataclass(frozen=True)
class EmbeddingResult:
    """One embedding output tied to the input index."""

    text_index: int
    vector: list[float]
    model: str
    dimensions: int


class EmbeddingProvider(Protocol):
    """Provider contract for batched embedding generation."""

    name: str

    async def embed_texts(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None = None,
        truncate: bool = True,
    ) -> list[EmbeddingResult]:
        """Generate vectors for *texts* in deterministic input order."""

    async def healthcheck(self) -> bool:
        """Return provider health status."""
