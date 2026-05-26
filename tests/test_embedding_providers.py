"""Unit tests for embedding provider adapters."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.adapters.embeddings import (
    EmbeddingDimensionMismatchError,
    EmbeddingProviderResponseError,
    FakeEmbeddingProvider,
    OllamaEmbeddingProvider,
)


async def test_fake_provider_returns_deterministic_vectors() -> None:
    provider = FakeEmbeddingProvider()

    first = await provider.embed_texts(["hello"], model="fake-model", dimensions=8)
    second = await provider.embed_texts(["hello"], model="fake-model", dimensions=8)

    assert first[0].vector == second[0].vector


async def test_fake_provider_respects_requested_dimensions() -> None:
    provider = FakeEmbeddingProvider()

    result = await provider.embed_texts(["hello"], model="fake-model", dimensions=12)

    assert len(result[0].vector) == 12


async def test_fake_provider_raises_domain_error_for_invalid_dimensions() -> None:
    provider = FakeEmbeddingProvider()

    with pytest.raises(EmbeddingDimensionMismatchError, match="greater than 0"):
        await provider.embed_texts(["hello"], model="fake-model", dimensions=0)


async def test_ollama_provider_builds_api_embed_request_payload() -> None:
    import httpx

    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"model": "all-minilm", "embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        )

    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(handler),
    )
    try:
        await provider.embed_texts(
            ["chunk text 1", "chunk text 2"],
            model="all-minilm",
            dimensions=2,
            truncate=True,
        )
    finally:
        await provider.aclose()

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/embed"
    assert captured["payload"] == {
        "model": "all-minilm",
        "input": ["chunk text 1", "chunk text 2"],
        "truncate": True,
        "dimensions": 2,
    }


async def test_ollama_provider_parses_valid_response() -> None:
    import httpx

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "all-minilm", "embeddings": [[0.1, 0.2]]})

    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(handler),
    )

    try:
        result = await provider.embed_texts(["chunk"], model="all-minilm", dimensions=2)
    finally:
        await provider.aclose()

    assert result[0].text_index == 0
    assert result[0].model == "all-minilm"
    assert result[0].dimensions == 2
    assert result[0].vector == [0.1, 0.2]


async def test_ollama_provider_fails_on_invalid_response_count() -> None:
    import httpx

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "all-minilm", "embeddings": [[0.1, 0.2]]},
        )

    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(handler),
    )

    try:
        with pytest.raises(EmbeddingProviderResponseError, match="returned 1 embeddings for 2 texts"):
            await provider.embed_texts(["chunk 1", "chunk 2"], model="all-minilm", dimensions=2)
    finally:
        await provider.aclose()


async def test_ollama_provider_fails_on_dimension_mismatch() -> None:
    import httpx

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "all-minilm", "embeddings": [[0.1, 0.2]]},
        )

    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(handler),
    )

    try:
        with pytest.raises(EmbeddingDimensionMismatchError, match="expected 3"):
            await provider.embed_texts(["chunk"], model="all-minilm", dimensions=3)
    finally:
        await provider.aclose()


async def test_ollama_provider_reuses_and_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import httpx

    clients: list[httpx.AsyncClient] = []

    class CountingAsyncClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            clients.append(self)
            super().__init__(*args, **kwargs)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "all-minilm", "embeddings": [[0.1, 0.2]]})

    monkeypatch.setattr(httpx, "AsyncClient", CountingAsyncClient)
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(handler),
    )

    try:
        await asyncio.gather(
            provider.embed_texts(["first"], model="all-minilm", dimensions=2),
            provider.embed_texts(["second"], model="all-minilm", dimensions=2),
        )
    finally:
        await provider.aclose()

    assert len(clients) == 1
    assert clients[0].is_closed
