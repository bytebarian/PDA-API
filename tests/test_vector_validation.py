from __future__ import annotations

import math

import pytest

from app.services.vector_validation import (
    InvalidEmbeddingVectorError,
    similarity_from_cosine_distance,
    validate_embedding_vector,
)


def test_validate_embedding_vector_rejects_dimension_mismatch() -> None:
    with pytest.raises(InvalidEmbeddingVectorError, match="exactly 3"):
        validate_embedding_vector([1.0, 2.0], expected_dimensions=3)


def test_validate_embedding_vector_rejects_nan_and_infinity() -> None:
    with pytest.raises(InvalidEmbeddingVectorError, match="finite number"):
        validate_embedding_vector([math.nan, 1.0, 2.0], expected_dimensions=3)
    with pytest.raises(InvalidEmbeddingVectorError, match="finite number"):
        validate_embedding_vector([math.inf, 1.0, 2.0], expected_dimensions=3)


def test_similarity_from_cosine_distance() -> None:
    assert similarity_from_cosine_distance(0.2) == pytest.approx(0.8)
