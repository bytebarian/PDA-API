"""Ollama-compatible embeddings provider."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.adapters.embeddings.base import (
    EmbeddingDimensionMismatchError,
    EmbeddingProviderResponseError,
    EmbeddingProviderUnavailableError,
    EmbeddingResult,
)


class OllamaEmbeddingProvider:
    """Generate embeddings via Ollama `/api/embed`."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        base_url=self._base_url,
                        timeout=self._timeout_seconds,
                        transport=self._transport,
                    )
        return self._client

    async def aclose(self) -> None:
        async with self._client_lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def embed_texts(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None = None,
        truncate: bool = True,
    ) -> list[EmbeddingResult]:
        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
            "truncate": truncate,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions

        try:
            client = await self._get_client()
            response = await client.post("/api/embed", json=payload)
        except httpx.TimeoutException as error:
            raise EmbeddingProviderUnavailableError(
                f"Ollama request timed out after {self._timeout_seconds}s"
            ) from error
        except httpx.HTTPError as error:
            raise EmbeddingProviderUnavailableError(f"Ollama request failed: {error}") from error

        if response.status_code >= 400:
            raise EmbeddingProviderResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except ValueError as error:
            raise EmbeddingProviderResponseError(
                "Ollama response is not valid JSON"
            ) from error
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingProviderResponseError("Ollama response missing embeddings list")
        if len(embeddings) != len(texts):
            raise EmbeddingProviderResponseError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
            )

        response_model = data.get("model")
        if not isinstance(response_model, str) or not response_model.strip():
            response_model = model

        results: list[EmbeddingResult] = []
        for index, embedding in enumerate(embeddings):
            if not isinstance(embedding, list) or not embedding:
                raise EmbeddingProviderResponseError(
                    f"Ollama embedding at index {index} is not a non-empty list"
                )
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError) as error:
                raise EmbeddingProviderResponseError(
                    f"Ollama embedding at index {index} contains non-numeric values"
                ) from error
            if dimensions is not None and len(vector) != dimensions:
                raise EmbeddingDimensionMismatchError(
                    f"Ollama embedding at index {index} has dimensions {len(vector)}, expected {dimensions}"
                )
            results.append(
                EmbeddingResult(
                    text_index=index,
                    vector=vector,
                    model=response_model,
                    dimensions=len(vector),
                )
            )
        return results

    async def healthcheck(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(
                "/api/tags",
                timeout=min(self._timeout_seconds, 5),
            )
            return response.status_code < 400
        except httpx.HTTPError:
            return False
