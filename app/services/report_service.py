"""Service orchestration for grounded markdown report generation."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embeddings import EmbeddingProvider
from app.adapters.llm import (
    ChatMessage,
    ChatModelError,
    ChatModelProvider,
    ChatModelUnavailableError,
)
from app.core.config import Settings, get_settings
from app.schemas.chat import ModelDiagnostics, RetrievalDiagnostics, UsageDiagnostics
from app.schemas.hybrid_search import HybridSearchRequest, HybridSearchResult
from app.schemas.reports import ReportGenerateRequest, ReportGenerateResponse
from app.schemas.search import SemanticSearchRequest, SemanticSearchResult
from app.services.chat_service import get_chat_model_provider
from app.services.citation_builder import CitationBuilder
from app.services.context_builder import ContextBuilderService
from app.services.hybrid_search_service import HybridSearchService
from app.services.search_service import (
    EmbeddingProviderNotAvailableError,
    SearchConfigurationError,
    SearchService,
    SearchServiceError,
)

logger = logging.getLogger(__name__)

_INSUFFICIENT_CONTEXT_MARKDOWN = (
    "# Report\n\n"
    "I could not find enough relevant information in the indexed documents to generate "
    "a grounded report for this topic."
)
_MISSING_CITATIONS_WARNING = (
    "Model report did not include matching citation markers; returning top included sources."
)


class ReportServiceError(RuntimeError):
    """Base class for report generation failures."""


class ReportProviderNotAvailableError(ReportServiceError):
    """Raised when a configured model or embedding provider is unavailable."""


class ReportConfigurationError(ReportServiceError):
    """Raised when report generation configuration is missing or invalid."""


@dataclass(frozen=True)
class _SearchOutcome:
    strategy: str
    results: Sequence[HybridSearchResult | SemanticSearchResult]
    candidate_document_count: int | None
    candidate_chunk_count: int | None
    filters_applied: dict[str, object]


class ReportService:
    """Orchestrate retrieval, report context assembly, model generation, and citations."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        hybrid_search_service: HybridSearchService | None = None,
        search_service: SearchService | None = None,
        embedding_providers: dict[str, EmbeddingProvider] | None = None,
        context_builder: ContextBuilderService | None = None,
        model_provider: ChatModelProvider | None = None,
        citation_builder: CitationBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._db = db
        self._settings = settings or get_settings()
        self._hybrid_search_service_instance = hybrid_search_service
        self._search_service_instance = search_service
        self._embedding_providers = embedding_providers
        self._context_builder = context_builder or ContextBuilderService()
        self._model_provider = model_provider or get_chat_model_provider(
            settings=self._settings
        )
        self._citation_builder = citation_builder or CitationBuilder()
        self._owns_model_provider = model_provider is None

    @property
    def _hybrid_search_service(self) -> HybridSearchService:
        if self._hybrid_search_service_instance is None:
            self._hybrid_search_service_instance = HybridSearchService(
                self._db,
                providers=self._embedding_providers,
                settings=self._settings,
            )
        return self._hybrid_search_service_instance

    @property
    def _search_service(self) -> SearchService:
        if self._search_service_instance is None:
            self._search_service_instance = SearchService(
                self._db,
                providers=self._embedding_providers,
                settings=self._settings,
            )
        return self._search_service_instance

    async def generate_report(
        self, request: ReportGenerateRequest
    ) -> ReportGenerateResponse:
        """Generate a markdown report grounded in retrieved document context."""
        started = time.perf_counter()
        search_outcome = await self._retrieve(request)
        built_context = await self._context_builder.build_context(
            list(search_outcome.results),
            query=request.topic,
            max_tokens=request.max_context_tokens,
            context_style="report",
            include_scores=False,
            group_by_document=True,
        )

        if not search_outcome.results or built_context.included_chunk_count == 0:
            return ReportGenerateResponse(
                markdown=_INSUFFICIENT_CONTEXT_MARKDOWN,
                citations=[],
                retrieval=self._build_retrieval_diagnostics(
                    request,
                    search_outcome,
                    built_context.included_chunk_count,
                    built_context.excluded_chunk_count,
                    warning=None,
                ),
            )

        model_name = request.model or self._settings.model_name
        messages = self._build_messages(request, built_context.context_text)

        try:
            model_result = await self._model_provider.generate(
                messages,
                model=model_name,
                temperature=request.temperature,
                max_tokens=request.max_report_tokens,
            )
        except ChatModelUnavailableError as exc:
            raise ReportProviderNotAvailableError(str(exc)) from exc
        except ChatModelError as exc:
            raise ReportServiceError("Report model provider returned an error") from exc
        finally:
            if self._owns_model_provider:
                close = getattr(self._model_provider, "aclose", None)
                if callable(close):
                    await close()

        extracted_source_ids = self._citation_builder.extract_source_markers(
            model_result.text
        )
        citations, _diag = self._citation_builder.build_from_sources(
            built_context.sources,
            answer_text=model_result.text,
            retrieval_results=search_outcome.results,
        )
        warning: str | None = None
        if not citations and built_context.sources:
            citations, _diag = self._citation_builder.build_from_sources(
                built_context.sources[: min(5, len(built_context.sources))],
                retrieval_results=search_outcome.results,
            )
            warning = _MISSING_CITATIONS_WARNING

        logger.info(
            "report generated",
            extra={
                "topic_length": len(request.topic),
                "retrieval_strategy": request.retrieval_strategy,
                "result_count": len(search_outcome.results),
                "included_context_chunk_count": built_context.included_chunk_count,
                "excluded_context_chunk_count": built_context.excluded_chunk_count,
                "estimated_context_tokens": built_context.estimated_tokens,
                "estimated_report_tokens": math.ceil(len(model_result.text) / 4),
                "citation_count": len(citations),
                "model_provider": model_result.provider,
                "model_name": model_result.model,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )

        return ReportGenerateResponse(
            markdown=model_result.text,
            citations=citations,
            retrieval=self._build_retrieval_diagnostics(
                request,
                search_outcome,
                built_context.included_chunk_count,
                built_context.excluded_chunk_count,
                warning=warning,
            ),
            model=self._build_model_diagnostics(
                provider=model_result.provider,
                name=model_result.model,
                citation_marker_count=len(extracted_source_ids),
                include_diagnostics=request.include_diagnostics,
            ),
            usage=UsageDiagnostics(
                estimated_context_tokens=built_context.estimated_tokens,
                estimated_answer_tokens=math.ceil(len(model_result.text) / 4),
            ),
        )

    async def _retrieve(self, request: ReportGenerateRequest) -> _SearchOutcome:
        try:
            if request.retrieval_strategy == "semantic":
                semantic_response = await self._search_service.semantic_search(
                    SemanticSearchRequest(
                        query=request.topic,
                        top_k=request.top_k,
                        filters=request.resolved_filters(),
                    )
                )
                return _SearchOutcome(
                    strategy="semantic",
                    results=semantic_response.results,
                    candidate_document_count=semantic_response.candidate_document_count,
                    candidate_chunk_count=semantic_response.candidate_chunk_count,
                    filters_applied=semantic_response.filters_applied,
                )

            hybrid_response = await self._hybrid_search_service.hybrid_search(
                HybridSearchRequest(
                    query=request.topic,
                    top_k=request.top_k,
                    filters=request.resolved_filters(),
                    include_content=True,
                )
            )
            return _SearchOutcome(
                strategy="hybrid",
                results=hybrid_response.results,
                candidate_document_count=hybrid_response.candidate_document_count,
                candidate_chunk_count=hybrid_response.candidate_chunk_count,
                filters_applied=hybrid_response.filters_applied,
            )
        except EmbeddingProviderNotAvailableError as exc:
            raise ReportProviderNotAvailableError(str(exc)) from exc
        except SearchConfigurationError as exc:
            raise ReportConfigurationError(str(exc)) from exc
        except SearchServiceError as exc:
            raise ReportServiceError("Retrieval failed") from exc

    def _build_messages(
        self, request: ReportGenerateRequest, context_text: str
    ) -> list[ChatMessage]:
        title_line = f"Preferred report title: {request.title}\n" if request.title else ""
        instructions = (
            f"Additional user instructions:\n{request.instructions}\n\n"
            if request.instructions
            else ""
        )
        return [
            ChatMessage(
                role="system",
                content=(
                    "You are PDA, a privacy-first personal document assistant. "
                    "Write a markdown report using only the provided document context. "
                    "Ground every document-specific claim in the sources and cite source IDs "
                    "such as [S1], [S2]. Do not invent facts, sources, dates, or totals. "
                    "If evidence is limited, state the limitation in the report. Use clear "
                    "headings and concise bullet points where helpful."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Document context:\n{context_text}\n\n"
                    f"Report topic:\n{request.topic}\n"
                    f"{title_line}"
                    f"{instructions}"
                    "Return only the markdown report."
                ),
            ),
        ]

    def _build_retrieval_diagnostics(
        self,
        request: ReportGenerateRequest,
        outcome: _SearchOutcome,
        included_context_chunk_count: int,
        excluded_context_chunk_count: int,
        *,
        warning: str | None,
    ) -> RetrievalDiagnostics:
        return RetrievalDiagnostics(
            strategy=request.retrieval_strategy,
            result_count=len(outcome.results),
            candidate_document_count=outcome.candidate_document_count,
            candidate_chunk_count=outcome.candidate_chunk_count,
            included_context_chunk_count=(
                included_context_chunk_count if request.include_diagnostics else None
            ),
            excluded_context_chunk_count=(
                excluded_context_chunk_count if request.include_diagnostics else None
            ),
            filters_applied=outcome.filters_applied,
            warning=warning if request.include_diagnostics else None,
        )

    def _build_model_diagnostics(
        self,
        *,
        provider: str,
        name: str,
        citation_marker_count: int,
        include_diagnostics: bool,
    ) -> ModelDiagnostics:
        return ModelDiagnostics(
            provider=provider,
            name=name,
            citation_marker_count=(
                citation_marker_count if include_diagnostics else None
            ),
        )
