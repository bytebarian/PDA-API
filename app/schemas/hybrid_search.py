"""Pydantic schemas for the hybrid search endpoint."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from app.schemas.search_filters import SearchFilters

_DEFAULT_VECTOR_WEIGHT = 0.65
_DEFAULT_FULL_TEXT_WEIGHT = 0.35
_DEFAULT_TOP_K = 10
_DEFAULT_CANDIDATE_MULTIPLIER = 3
_MAX_TOP_K = 50
_MAX_CANDIDATE_K = 200


class HybridSearchRequest(BaseModel):
    """Request body for POST /search/hybrid."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(
        default=_DEFAULT_TOP_K,
        ge=1,
        le=_MAX_TOP_K,
        validation_alias=AliasChoices("top_k", "topK"),
    )
    vector_top_k: int | None = Field(
        default=None,
        ge=1,
        le=_MAX_CANDIDATE_K,
        validation_alias=AliasChoices("vector_top_k", "vectorTopK"),
    )
    full_text_top_k: int | None = Field(
        default=None,
        ge=1,
        le=_MAX_CANDIDATE_K,
        validation_alias=AliasChoices("full_text_top_k", "fullTextTopK"),
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("min_score", "minScore"),
    )
    fusion_strategy: Literal["rrf", "weighted"] = Field(
        default="rrf",
        validation_alias=AliasChoices("fusion_strategy", "fusionStrategy"),
    )
    vector_weight: float = Field(
        default=_DEFAULT_VECTOR_WEIGHT,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("vector_weight", "vectorWeight"),
    )
    full_text_weight: float = Field(
        default=_DEFAULT_FULL_TEXT_WEIGHT,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("full_text_weight", "fullTextWeight"),
    )
    filters: SearchFilters | None = None
    include_content: bool = Field(
        default=True,
        validation_alias=AliasChoices("include_content", "includeContent"),
    )

    @model_validator(mode="after")
    def _validate_weights(self) -> HybridSearchRequest:
        if self.vector_weight + self.full_text_weight <= 0.0:
            raise ValueError(
                "vectorWeight + fullTextWeight must be greater than zero"
            )
        return self

    def resolved_filters(self) -> SearchFilters:
        """Return explicit filters or a default empty filter set."""
        return self.filters or SearchFilters()

    def effective_vector_top_k(self) -> int:
        """Return the candidate limit for vector retrieval."""
        return self.vector_top_k or max(
            self.top_k * _DEFAULT_CANDIDATE_MULTIPLIER, 30
        )

    def effective_full_text_top_k(self) -> int:
        """Return the candidate limit for full-text retrieval."""
        return self.full_text_top_k or max(
            self.top_k * _DEFAULT_CANDIDATE_MULTIPLIER, 30
        )


class HybridSearchResult(BaseModel):
    """One result entry returned from a hybrid search."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    document_path: str | None = None
    category: str | None = None
    file_type: str | None = None
    page_number: int | None = None
    chunk_index: int
    start_offset: int | None = None
    end_offset: int | None = None
    text: str | None = None
    excerpt: str
    score: float
    vector_score: float | None = None
    full_text_score: float | None = None
    vector_rank: int | None = None
    full_text_rank: int | None = None
    matched_by: list[Literal["vector", "full_text"]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class HybridSearchResponse(BaseModel):
    """Response body for POST /search/hybrid."""

    query: str
    top_k: int
    fusion_strategy: str
    embedding_model: str
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    candidate_document_count: int
    candidate_chunk_count: int
    vector_candidate_count: int
    full_text_candidate_count: int
    result_count: int
    results: list[HybridSearchResult]
