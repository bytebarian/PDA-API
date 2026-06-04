"""Pydantic schemas for the citation builder service."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """One frontend-ready citation entry produced by the citation builder."""

    model_config = ConfigDict(populate_by_name=True)

    source_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_id", "sourceId"),
    )
    citation_index: int = Field(
        validation_alias=AliasChoices("citation_index", "citationIndex"),
    )
    document_id: uuid.UUID = Field(
        validation_alias=AliasChoices("document_id", "documentId"),
    )
    document_name: str = Field(
        validation_alias=AliasChoices("document_name", "documentName"),
    )
    document_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("document_path", "documentPath"),
    )
    document_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("document_url", "documentUrl"),
    )
    chunk_id: uuid.UUID = Field(
        validation_alias=AliasChoices("chunk_id", "chunkId"),
    )
    page_number: int | None = Field(
        default=None,
        validation_alias=AliasChoices("page_number", "pageNumber"),
    )
    chunk_index: int = Field(
        validation_alias=AliasChoices("chunk_index", "chunkIndex"),
    )
    start_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("start_offset", "startOffset"),
    )
    end_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("end_offset", "endOffset"),
    )
    excerpt: str
    score: float | None = None
    relevance_source: Literal["vector", "full_text", "hybrid", "context"] | None = Field(
        default=None,
        validation_alias=AliasChoices("relevance_source", "relevanceSource"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationDiagnostics(BaseModel):
    """Diagnostics produced by the citation builder."""

    model_config = ConfigDict(populate_by_name=True)

    source_markers_found: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("source_markers_found", "sourceMarkersFound"),
    )
    unknown_source_markers: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("unknown_source_markers", "unknownSourceMarkers"),
    )
    duplicate_markers_ignored: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("duplicate_markers_ignored", "duplicateMarkersIgnored"),
    )
    sources_available: int = Field(
        default=0,
        validation_alias=AliasChoices("sources_available", "sourcesAvailable"),
    )
    citation_count: int = Field(
        default=0,
        validation_alias=AliasChoices("citation_count", "citationCount"),
    )
    excerpts_truncated: int = Field(
        default=0,
        validation_alias=AliasChoices("excerpts_truncated", "excerptsTruncated"),
    )
    missing_required_metadata: int = Field(
        default=0,
        validation_alias=AliasChoices("missing_required_metadata", "missingRequiredMetadata"),
    )


class CitationBuildRequest(BaseModel):
    """Request body for POST /citations/build."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    answer_text: str | None = Field(
        default=None,
        validation_alias=AliasChoices("answer_text", "answerText"),
    )
    sources: list[CitationSourceInput]
    max_excerpt_characters: int = Field(
        default=500,
        ge=50,
        le=4000,
        validation_alias=AliasChoices("max_excerpt_characters", "maxExcerptCharacters"),
    )
    include_uncited_sources: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_uncited_sources", "includeUncitedSources"),
    )


class CitationSourceInput(BaseModel):
    """One source entry accepted by POST /citations/build."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    source_id: str = Field(
        validation_alias=AliasChoices("source_id", "sourceId"),
    )
    document_id: uuid.UUID = Field(
        validation_alias=AliasChoices("document_id", "documentId"),
    )
    document_name: str = Field(
        validation_alias=AliasChoices("document_name", "documentName"),
    )
    document_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("document_path", "documentPath"),
    )
    chunk_id: uuid.UUID = Field(
        validation_alias=AliasChoices("chunk_id", "chunkId"),
    )
    page_number: int | None = Field(
        default=None,
        validation_alias=AliasChoices("page_number", "pageNumber"),
    )
    chunk_index: int = Field(
        validation_alias=AliasChoices("chunk_index", "chunkIndex"),
    )
    start_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("start_offset", "startOffset"),
    )
    end_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("end_offset", "endOffset"),
    )
    excerpt: str | None = None
    text: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationBuildResponse(BaseModel):
    """Response body for POST /citations/build."""

    model_config = ConfigDict(populate_by_name=True)

    citations: list[Citation]
    diagnostics: CitationDiagnostics


# Forward-reference resolution
CitationBuildRequest.model_rebuild()
