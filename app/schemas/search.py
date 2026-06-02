"""Pydantic schemas for the semantic search endpoint."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class SemanticSearchRequest(BaseModel):
    """Request body for POST /search/semantic."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=50)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    document_ids: list[uuid.UUID] | None = None
    categories: list[str] | None = None
    file_types: list[str] | None = None
    include_content: bool = True


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
    result_count: int
    results: list[SemanticSearchResult]
