"""Ollama-backed summarization provider."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.adapters.summarization.base import (
    SummarizationProviderResponseError,
    SummarizationProviderUnavailableError,
    SummarizationResult,
)

logger = logging.getLogger(__name__)

_SUMMARIZATION_PROMPT_TEMPLATE = """\
You are summarizing a private personal document for a local document assistant.
Create a concise factual summary in English.
Use only the provided document text.
Do not infer facts that are not present.
Do not mention that you are an AI model.
Keep the summary under {max_chars} characters.

Document text:
{text}"""


class OllamaSummarizationProvider:
    """Generate document summaries via the Ollama `/api/generate` endpoint."""

    name = "ollama"

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
                kwargs: dict = {"base_url": self._base_url, "timeout": float(self._timeout_seconds)}
                if self._transport is not None:
                    kwargs["transport"] = self._transport
                self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def summarize(
        self,
        text: str,
        *,
        model: str,
        max_output_chars: int = 600,
    ) -> SummarizationResult:
        """Generate a short factual summary of *text* via Ollama."""
        prompt = _SUMMARIZATION_PROMPT_TEMPLATE.format(
            max_chars=max_output_chars,
            text=text,
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                # Limit output tokens as a guard; the prompt already sets the character limit.
                "num_predict": max(64, max_output_chars // 3),
            },
        }

        client = await self._get_client()
        try:
            response = await client.post("/api/generate", json=payload)
        except httpx.TimeoutException as exc:
            raise SummarizationProviderUnavailableError(
                f"Ollama summarization timed out after {self._timeout_seconds}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise SummarizationProviderUnavailableError(
                f"Ollama is not reachable at {self._base_url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SummarizationProviderUnavailableError(
                f"Ollama HTTP error during summarization: {exc}"
            ) from exc

        if response.status_code != 200:
            raise SummarizationProviderResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise SummarizationProviderResponseError(
                f"Ollama returned non-JSON response: {response.text[:200]}"
            ) from exc

        summary_text = data.get("response", "").strip()
        if not summary_text:
            raise SummarizationProviderResponseError(
                "Ollama returned an empty summarization response"
            )

        # Truncate to max_output_chars if the model exceeded the requested limit.
        if len(summary_text) > max_output_chars:
            summary_text = summary_text[:max_output_chars].rsplit(" ", 1)[0]

        return SummarizationResult(
            summary=summary_text,
            model=data.get("model", model),
            provider=self.name,
            input_char_count=len(text),
            output_char_count=len(summary_text),
        )

    async def healthcheck(self) -> bool:
        """Return True when Ollama responds to a GET /api/tags request."""
        client = await self._get_client()
        try:
            response = await client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
