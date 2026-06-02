"""Pydantic schemas for the semantic search endpoint."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from app.schemas.search_filters import SearchFilters


class SemanticSearchRequest(BaseModel):
    """Request body for POST /search/semantic."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=50, validation_alias=AliasChoices("top_k", "topK"))
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("min_score", "minScore"),
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        validation_alias=AliasChoices("document_ids", "documentIds"),
    )
    categories: list[str] | None = None
    file_types: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("file_types", "fileTypes"),
    )
    include_content: bool = Field(
        default=True,
        validation_alias=AliasChoices("include_content", "includeContent"),
    )
    filters: SearchFilters | None = None

    @model_validator(mode="after")
    def _normalize_legacy_top_level_filters(self) -> SemanticSearchRequest:
        merged = self.filters.model_copy(deep=True) if self.filters is not None else SearchFilters()
        if merged.document_ids is None and self.document_ids is not None:
            merged.document_ids = self.document_ids
        if merged.categories is None and self.categories is not None:
            merged.categories = self.categories
        if merged.file_types is None and self.file_types is not None:
            merged.file_types = self.file_types
        self.filters = merged
        return self

    def resolved_filters(self) -> SearchFilters:
        return self.filters or SearchFilters()


class SemanticSearchResult(BaseModel):
    """One result entry returned from a semantic search."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    category: str | None = None
    file_type: str | None = None
    page_number: int | None = None
    chunk_index: int
    start_offset: int | None = None
    end_offset: int | None = None
    text: str | None = None
    excerpt: str
    distance: float
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticSearchResponse(BaseModel):
    """Response body for POST /search/semantic."""

    query: str
    embedding_model: str
    top_k: int
    candidate_document_count: int
    candidate_chunk_count: int
    result_count: int
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    results: list[SemanticSearchResult]
