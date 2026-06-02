"""Summarization provider contracts and domain exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SummarizationError(RuntimeError):
    """Base class for summarization provider failures."""


class SummarizationProviderUnavailableError(SummarizationError):
    """Raised when the provider is unreachable or times out."""


class SummarizationProviderResponseError(SummarizationError):
    """Raised when the provider returns an invalid or unparseable response."""


@dataclass(frozen=True)
class SummarizationResult:
    """Output from a summarization provider call."""

    summary: str
    model: str
    provider: str
    input_char_count: int
    output_char_count: int


class SummarizationProvider(Protocol):
    """Provider contract for document summarization."""

    name: str

    async def summarize(
        self,
        text: str,
        *,
        model: str,
        max_output_chars: int = 600,
    ) -> SummarizationResult:
        """Generate a short factual summary of *text*."""

    async def healthcheck(self) -> bool:
        """Return True when the provider is ready to serve requests."""
