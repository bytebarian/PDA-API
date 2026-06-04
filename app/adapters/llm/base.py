"""Chat model provider contracts and domain exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


class ChatModelError(RuntimeError):
    """Base class for chat model provider failures."""


class ChatModelUnavailableError(ChatModelError):
    """Raised when the provider is unreachable or times out."""


class ChatModelResponseError(ChatModelError):
    """Raised when the provider returns an invalid response."""


@dataclass(frozen=True)
class ChatMessage:
    """One chat message sent to a provider."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatModelResult:
    """Normalized provider response."""

    text: str
    model: str
    provider: str
    input_message_count: int
    output_char_count: int


class ChatModelProvider(Protocol):
    """Provider contract for grounded chat generation."""

    name: str

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> ChatModelResult:
        """Generate a grounded answer from chat messages."""

    async def healthcheck(self) -> bool:
        """Return True when the provider is ready to serve requests."""
