"""Ollama-backed chat model provider."""

from __future__ import annotations

import asyncio

import httpx

from app.adapters.llm.base import (
    ChatMessage,
    ChatModelResponseError,
    ChatModelResult,
    ChatModelUnavailableError,
)


class OllamaChatModelProvider:
    """Generate chat answers via the Ollama `/api/chat` endpoint."""

    name = "local"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock: asyncio.Lock | None = None

    def _get_client_lock(self) -> asyncio.Lock:
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        return self._client_lock

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._get_client_lock():
            if self._client is None or self._client.is_closed:
                if self._transport is not None:
                    self._client = httpx.AsyncClient(
                        base_url=self._base_url,
                        timeout=float(self._timeout_seconds),
                        transport=self._transport,
                    )
                else:
                    self._client = httpx.AsyncClient(
                        base_url=self._base_url,
                        timeout=float(self._timeout_seconds),
                    )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> ChatModelResult:
        payload = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        client = await self._get_client()
        try:
            response = await client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise ChatModelUnavailableError(
                f"Ollama chat timed out after {self._timeout_seconds}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise ChatModelUnavailableError(
                f"Ollama is not reachable at {self._base_url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ChatModelUnavailableError(
                f"Ollama HTTP error during chat generation: {exc}"
            ) from exc

        if response.status_code != 200:
            raise ChatModelResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise ChatModelResponseError(
                f"Ollama returned non-JSON response: {response.text[:200]}"
            ) from exc

        message = data.get("message")
        text = message.get("content", "").strip() if isinstance(message, dict) else ""
        if not text:
            raise ChatModelResponseError("Ollama returned an empty chat response")

        return ChatModelResult(
            text=text,
            model=data.get("model", model),
            provider=self.name,
            input_message_count=len(messages),
            output_char_count=len(text),
        )

    async def healthcheck(self) -> bool:
        client = await self._get_client()
        try:
            response = await client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
