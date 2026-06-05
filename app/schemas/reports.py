"""Pydantic schemas for grounded report generation."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from app.schemas.chat import ModelDiagnostics, RetrievalDiagnostics, UsageDiagnostics
from app.schemas.citations import Citation
from app.schemas.search_filters import SearchFilters

_DEFAULT_REPORT_TOP_K = 12
_MAX_REPORT_TOP_K = 50
_DEFAULT_MAX_CONTEXT_TOKENS = 12000
_DEFAULT_MAX_REPORT_TOKENS = 2000


class ReportGenerateRequest(BaseModel):
    """Request body for POST /reports/generate."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    topic: str = Field(min_length=1, max_length=4000)
    instructions: str | None = Field(default=None, max_length=4000)
    title: str | None = Field(default=None, max_length=200)
    filters: SearchFilters | None = None
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        validation_alias=AliasChoices("document_ids", "documentIds"),
    )
    top_k: int = Field(
        default=_DEFAULT_REPORT_TOP_K,
        ge=1,
        le=_MAX_REPORT_TOP_K,
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
    max_report_tokens: int = Field(
        default=_DEFAULT_MAX_REPORT_TOKENS,
        ge=128,
        le=16384,
        validation_alias=AliasChoices("max_report_tokens", "maxReportTokens"),
    )
    include_diagnostics: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_diagnostics", "includeDiagnostics"),
    )

    @model_validator(mode="after")
    def _normalize_document_ids(self) -> ReportGenerateRequest:
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


class ReportGenerateResponse(BaseModel):
    """Response body for POST /reports/generate."""

    markdown: str
    citations: list[Citation]
    retrieval: RetrievalDiagnostics | None = None
    model: ModelDiagnostics | None = None
    usage: UsageDiagnostics | None = None
