"""Mock chat model provider for tests and CI."""

from __future__ import annotations

from app.adapters.llm.base import (
    ChatMessage,
    ChatModelError,
    ChatModelResult,
)


class MockChatModelProvider:
    """Deterministic in-memory chat model provider."""

    name = "mock"

    def __init__(
        self,
        *,
        answer: str = "Mock answer. [S1]",
        error: ChatModelError | None = None,
    ) -> None:
        self._answer = answer
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> ChatModelResult:
        if self._error is not None:
            raise self._error

        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return ChatModelResult(
            text=self._answer,
            model=model,
            provider=self.name,
            input_message_count=len(messages),
            output_char_count=len(self._answer),
        )

    async def healthcheck(self) -> bool:
        return self._error is None
