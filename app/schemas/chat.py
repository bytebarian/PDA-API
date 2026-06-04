"""Schemas for grounded chat requests and citation-ready responses."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from app.schemas.citations import Citation
from app.schemas.search_filters import SearchFilters

_DEFAULT_CHAT_TOP_K = 8
_MAX_CHAT_TOP_K = 30
_DEFAULT_MAX_CONTEXT_TOKENS = 6000
_DEFAULT_MAX_ANSWER_TOKENS = 800


class ChatAskRequest(BaseModel):
    """Request body for POST /chat/ask."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    question: str = Field(min_length=1, max_length=4000)
    filters: SearchFilters | None = None
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        validation_alias=AliasChoices("document_ids", "documentIds"),
    )
    top_k: int = Field(
        default=_DEFAULT_CHAT_TOP_K,
        ge=1,
        le=_MAX_CHAT_TOP_K,
        validation_alias=AliasChoices("top_k", "topK"),
    )
    retrieval_strategy: Literal["hybrid", "semantic"] = Field(
        default="hybrid",
        validation_alias=AliasChoices("retrieval_strategy", "retrievalStrategy"),
    )
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_context_tokens: int = Field(
        default=_DEFAULT_MAX_CONTEXT_TOKENS,
        ge=512,
        le=128000,
        validation_alias=AliasChoices("max_context_tokens", "maxContextTokens"),
    )
    max_answer_tokens: int = Field(
        default=_DEFAULT_MAX_ANSWER_TOKENS,
        ge=64,
        le=8192,
        validation_alias=AliasChoices("max_answer_tokens", "maxAnswerTokens"),
    )
    include_diagnostics: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_diagnostics", "includeDiagnostics"),
    )

    @model_validator(mode="after")
    def _normalize_document_ids(self) -> ChatAskRequest:
        if not self.document_ids:
            return self

        filters = self.filters.model_copy(deep=True) if self.filters is not None else SearchFilters()
        if filters.document_ids:
            deduped = {str(value): value for value in filters.document_ids}
            for value in self.document_ids:
                deduped.setdefault(str(value), value)
            filters = filters.model_copy(update={"document_ids": list(deduped.values())})
        else:
            filters = filters.model_copy(update={"document_ids": self.document_ids})

        self.filters = filters
        self.document_ids = None
        return self

    def resolved_filters(self) -> SearchFilters:
        """Return explicit filters or the default ready-document filter set."""
        return self.filters or SearchFilters()


class ChatCitation(BaseModel):
    """One frontend-ready citation entry (legacy alias; use Citation from citations.py)."""

    source_id: str = Field(validation_alias=AliasChoices("source_id", "sourceId"))
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
    chunk_id: uuid.UUID = Field(validation_alias=AliasChoices("chunk_id", "chunkId"))
    page_number: int | None = Field(
        default=None,
        validation_alias=AliasChoices("page_number", "pageNumber"),
    )
    chunk_index: int = Field(
        validation_alias=AliasChoices("chunk_index", "chunkIndex")
    )
    excerpt: str
    start_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("start_offset", "startOffset"),
    )
    end_offset: int | None = Field(
        default=None,
        validation_alias=AliasChoices("end_offset", "endOffset"),
    )
    score: float | None = None


class RetrievalDiagnostics(BaseModel):
    """Safe retrieval and context diagnostics."""

    strategy: Literal["hybrid", "semantic"]
    result_count: int = Field(validation_alias=AliasChoices("result_count", "resultCount"))
    candidate_document_count: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "candidate_document_count", "candidateDocumentCount"
        ),
    )
    candidate_chunk_count: int | None = Field(
        default=None,
        validation_alias=AliasChoices("candidate_chunk_count", "candidateChunkCount"),
    )
    included_context_chunk_count: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "included_context_chunk_count", "includedContextChunkCount"
        ),
    )
    excluded_context_chunk_count: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "excluded_context_chunk_count", "excludedContextChunkCount"
        ),
    )
    filters_applied: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("filters_applied", "filtersApplied"),
    )
    warning: str | None = None


class ModelDiagnostics(BaseModel):
    """Safe model selection diagnostics."""

    provider: str
    name: str
    citation_marker_count: int | None = Field(
        default=None,
        validation_alias=AliasChoices("citation_marker_count", "citationMarkerCount"),
    )


class UsageDiagnostics(BaseModel):
    """Safe token-usage diagnostics."""

    estimated_context_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "estimated_context_tokens", "estimatedContextTokens"
        ),
    )
    estimated_answer_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "estimated_answer_tokens", "estimatedAnswerTokens"
        ),
    )


class ChatAskResponse(BaseModel):
    """Response body for POST /chat/ask."""

    answer: str
    citations: list[Citation]
    retrieval: RetrievalDiagnostics | None = None
    model: ModelDiagnostics | None = None
    usage: UsageDiagnostics | None = None
