"""Deterministic local fake embeddings provider for tests and CI."""

from __future__ import annotations

import hashlib

from app.adapters.embeddings.base import EmbeddingDimensionMismatchError
from app.adapters.embeddings.base import EmbeddingResult

_BYTE_TO_UNIT_SCALE = 127.5


class FakeEmbeddingProvider:
    """Network-free deterministic embedding provider."""

    name = "fake"

    async def embed_texts(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None = None,
        truncate: bool = True,
    ) -> list[EmbeddingResult]:
        del truncate
        vector_dimensions = dimensions if dimensions is not None else 8
        if vector_dimensions <= 0:
            raise EmbeddingDimensionMismatchError("dimensions must be greater than 0")

        return [
            EmbeddingResult(
                text_index=index,
                vector=_deterministic_vector(text, model, vector_dimensions),
                model=model,
                dimensions=vector_dimensions,
            )
            for index, text in enumerate(texts)
        ]

    async def healthcheck(self) -> bool:
        return True


def _deterministic_vector(text: str, model: str, dimensions: int) -> list[float]:
    seed = f"{model}\0{text}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    output: list[float] = []
    cursor = 0
    while len(output) < dimensions:
        if cursor >= len(digest):
            digest = hashlib.sha256(digest).digest()
            cursor = 0
        byte_value = digest[cursor]
        output.append((byte_value / _BYTE_TO_UNIT_SCALE) - 1.0)
        cursor += 1
    return output
