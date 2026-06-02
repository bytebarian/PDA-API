"""Mock summarization provider for tests and CI."""

from __future__ import annotations

from app.adapters.summarization.base import (
    SummarizationError,
    SummarizationResult,
)


class MockSummarizationProvider:
    """Deterministic in-memory summarization provider for use in tests.

    By default returns a fixed summary string. Pass ``error`` to simulate
    a provider failure without a live Ollama instance.
    """

    name = "mock"

    def __init__(
        self,
        *,
        summary: str = "Mock summary of the document.",
        error: SummarizationError | None = None,
    ) -> None:
        self._summary = summary
        self._error = error

    async def summarize(
        self,
        text: str,
        *,
        model: str,
        max_output_chars: int = 600,
    ) -> SummarizationResult:
        if self._error is not None:
            raise self._error

        result_text = self._summary
        # Respect max_output_chars the same way the real providers do.
        if len(result_text) > max_output_chars:
            result_text = result_text[:max_output_chars]

        return SummarizationResult(
            summary=result_text,
            model=model,
            provider=self.name,
            input_char_count=len(text),
            output_char_count=len(result_text),
        )

    async def healthcheck(self) -> bool:
        return self._error is None
