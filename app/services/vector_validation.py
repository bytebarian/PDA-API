"""Validation and scoring helpers for embedding vectors."""

from __future__ import annotations

import math
from collections.abc import Sequence


class InvalidEmbeddingVectorError(ValueError):
    """Raised when a vector is malformed or incompatible with configuration."""


def validate_embedding_vector(
    vector: Sequence[float] | None,
    *,
    expected_dimensions: int,
    field_name: str = "embedding",
) -> list[float]:
    if vector is None:
        raise InvalidEmbeddingVectorError(f"{field_name} cannot be null")
    if expected_dimensions <= 0:
        raise InvalidEmbeddingVectorError("Embedding dimensions must be greater than 0")

    values = list(vector)
    if len(values) != expected_dimensions:
        raise InvalidEmbeddingVectorError(
            f"{field_name} must contain exactly {expected_dimensions} values"
        )

    normalized: list[float] = []
    for index, value in enumerate(values):
        number = float(value)
        if not math.isfinite(number):
            raise InvalidEmbeddingVectorError(
                f"{field_name}[{index}] must be a finite number"
            )
        normalized.append(number)

    return normalized


def similarity_from_cosine_distance(distance: float) -> float:
    return 1.0 - float(distance)
