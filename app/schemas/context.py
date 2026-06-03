"""Schemas for assembling retrieval results into model-ready prompt context."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

DEFAULT_CONTEXT_MAX_CHUNKS = 12
DEFAULT_CONTEXT_MAX_CHARACTERS = 12000


class RetrievalResultForContext(BaseModel):
    """Normalized retrieval chunk input accepted by the context builder."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    chunk_id: uuid.UUID = Field(
        validation_alias=AliasChoices("chunk_id", "chunkId")
    )
    document_id: uuid.UUID = Field(
        validation_alias=AliasChoices("document_id", "documentId")
    )
    document_name: str = Field(
        min_length=1, validation_alias=AliasChoices("document_name", "documentName")
    )
    document_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("document_path", "documentPath"),
    )
    category: str | None = None
    file_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("file_type", "fileType"),
    )
    page_number: int | None = Field(
        default=None,
        validation_alias=AliasChoices("page_number", "pageNumber"),
    )
    chunk_index: int = Field(
        ge=0, validation_alias=AliasChoices("chunk_index", "chunkIndex")
    )
    start_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("start_offset", "startOffset"),
    )
    end_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("end_offset", "endOffset"),
    )
    text: str | None = None
    excerpt: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IncludedTextRange(BaseModel):
    """Character range included from the original chunk content."""

    start: int
    end: int


class ContextSource(BaseModel):
    """Citation/source metadata for one included chunk."""

    source_id: str = Field(validation_alias=AliasChoices("source_id", "sourceId"))
    chunk_id: uuid.UUID = Field(
        validation_alias=AliasChoices("chunk_id", "chunkId")
    )
    document_id: uuid.UUID = Field(
        validation_alias=AliasChoices("document_id", "documentId")
    )
    document_name: str = Field(
        validation_alias=AliasChoices("document_name", "documentName")
    )
    document_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("document_path", "documentPath"),
    )
    page_number: int | None = Field(
        default=None,
        validation_alias=AliasChoices("page_number", "pageNumber"),
    )
    chunk_index: int = Field(
        validation_alias=AliasChoices("chunk_index", "chunkIndex")
    )
    start_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("start_offset", "startOffset"),
    )
    end_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("end_offset", "endOffset"),
    )
    score: float | None = None
    included_text_range: IncludedTextRange | None = Field(
        default=None,
        validation_alias=AliasChoices("included_text_range", "includedTextRange"),
    )
    excerpt_only: bool = Field(
        default=False,
        validation_alias=AliasChoices("excerpt_only", "excerptOnly"),
    )


class ExcludedContextChunk(BaseModel):
    """Reason a candidate chunk was excluded from the built context."""

    chunk_id: uuid.UUID = Field(
        validation_alias=AliasChoices("chunk_id", "chunkId")
    )
    reason: Literal[
        "budget_exceeded",
        "max_chunks_exceeded",
        "missing_content",
        "duplicate_chunk_id",
        "duplicate_document_chunk",
        "duplicate_text",
    ]


class BuiltContext(BaseModel):
    """Structured context output produced by the context builder."""

    context_text: str = Field(
        validation_alias=AliasChoices("context_text", "contextText")
    )
    sources: list[ContextSource]
    included_chunk_count: int = Field(
        validation_alias=AliasChoices("included_chunk_count", "includedChunkCount")
    )
    excluded_chunk_count: int = Field(
        validation_alias=AliasChoices("excluded_chunk_count", "excludedChunkCount")
    )
    estimated_tokens: int = Field(
        validation_alias=AliasChoices("estimated_tokens", "estimatedTokens")
    )
    character_count: int = Field(
        validation_alias=AliasChoices("character_count", "characterCount")
    )
    truncated: bool
    excluded: list[ExcludedContextChunk] = Field(default_factory=list)


class ModelContextPayload(BaseModel):
    """Handoff payload that can be passed directly to model orchestration."""

    system_instruction: str
    context_text: str
    source_map: list[ContextSource]
    estimated_tokens: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBuildRequest(BaseModel):
    """Internal request DTO for context assembly operations."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query: str | None = None
    retrieval_results: list[RetrievalResultForContext] = Field(
        validation_alias=AliasChoices("retrieval_results", "retrievalResults")
    )
    max_chunks: int = Field(default=DEFAULT_CONTEXT_MAX_CHUNKS, ge=1, le=50)
    max_characters: int = Field(
        default=DEFAULT_CONTEXT_MAX_CHARACTERS,
        ge=1,
        le=100000,
        validation_alias=AliasChoices("max_characters", "maxCharacters"),
    )
    max_tokens: int | None = Field(
        default=None,
        ge=256,
        le=128000,
        validation_alias=AliasChoices("max_tokens", "maxTokens"),
    )
    include_metadata: bool = Field(
        default=True,
        validation_alias=AliasChoices("include_metadata", "includeMetadata"),
    )
    include_scores: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_scores", "includeScores"),
    )
    group_by_document: bool = Field(
        default=True,
        validation_alias=AliasChoices("group_by_document", "groupByDocument"),
    )
    context_style: Literal["chat", "report", "raw"] = Field(
        default="chat",
        validation_alias=AliasChoices("context_style", "contextStyle"),
    )
