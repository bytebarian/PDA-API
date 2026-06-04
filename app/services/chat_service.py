"""Service orchestration for one-shot grounded chat answers with citations."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.llm import (
    ChatMessage,
    ChatModelError,
    ChatModelProvider,
    ChatModelUnavailableError,
    MockChatModelProvider,
    OllamaChatModelProvider,
)
from app.core.config import Settings, get_settings
from app.schemas.chat import (
    ChatAskRequest,
    ChatAskResponse,
    ModelDiagnostics,
    RetrievalDiagnostics,
    UsageDiagnostics,
)
from app.schemas.hybrid_search import HybridSearchRequest, HybridSearchResult
from app.schemas.search import (
    SemanticSearchRequest,
    SemanticSearchResult,
)
from app.services.citation_mapper import CitationMapper
from app.services.context_builder import ContextBuilderService
from app.services.hybrid_search_service import HybridSearchService
from app.services.search_service import (
    EmbeddingProviderNotAvailableError,
    SearchConfigurationError,
    SearchService,
    SearchServiceError,
)

logger = logging.getLogger(__name__)

_INSUFFICIENT_CONTEXT_ANSWER = (
    "I could not find enough relevant information in the indexed documents to answer this question."
)
_MISSING_MARKERS_WARNING = (
    "Model answer did not include citation markers; returning top included sources."
)


class ChatServiceError(RuntimeError):
    """Base class for chat orchestration failures."""


class ChatProviderNotAvailableError(ChatServiceError):
    """Raised when the configured chat provider is unreachable."""


class ChatConfigurationError(ChatServiceError):
    """Raised when chat configuration is missing or invalid."""


@dataclass(frozen=True)
class _SearchOutcome:
    strategy: str
    results: Sequence[HybridSearchResult | SemanticSearchResult]
    candidate_document_count: int | None
    candidate_chunk_count: int | None
    filters_applied: dict[str, object]


def get_chat_model_provider(
    *,
    provider_name: str | None = None,
    settings: Settings | None = None,
) -> ChatModelProvider:
    """Resolve and return a chat model provider instance."""
    resolved_settings = settings or get_settings()
    name = (provider_name or resolved_settings.model_provider).strip().lower()

    if name == "mock":
        return MockChatModelProvider()
    if name in {"local", "ollama"}:
        return OllamaChatModelProvider(
            base_url=resolved_settings.ollama_base_url,
            timeout_seconds=resolved_settings.ollama_timeout_seconds,
        )

    raise ChatConfigurationError(f"Unknown chat model provider '{name}'")


class ChatService:
    """Orchestrate retrieval, context assembly, generation, and citation mapping."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        hybrid_search_service: HybridSearchService | None = None,
        search_service: SearchService | None = None,
        context_builder: ContextBuilderService | None = None,
        model_provider: ChatModelProvider | None = None,
        citation_mapper: CitationMapper | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._db = db
        self._settings = settings or get_settings()
        self._hybrid_search_service = hybrid_search_service or HybridSearchService(
            db,
            settings=self._settings,
        )
        self._search_service = search_service or SearchService(
            db,
            settings=self._settings,
        )
        self._context_builder = context_builder or ContextBuilderService()
        self._model_provider = model_provider or get_chat_model_provider(
            settings=self._settings
        )
        self._citation_mapper = citation_mapper or CitationMapper()
        self._owns_model_provider = model_provider is None

    async def ask_question(self, request: ChatAskRequest) -> ChatAskResponse:
        """Answer one question using retrieved document context."""
        started = time.perf_counter()
        search_outcome = await self._retrieve(request)
        built_context = await self._context_builder.build_context(
            list(search_outcome.results),
            query=request.question,
            max_tokens=request.max_context_tokens,
            context_style="chat",
        )

        if not search_outcome.results or built_context.included_chunk_count == 0:
            return ChatAskResponse(
                answer=_INSUFFICIENT_CONTEXT_ANSWER,
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
        messages = self._build_messages(request.question, built_context.context_text)

        try:
            model_result = await self._model_provider.generate(
                messages,
                model=model_name,
                temperature=request.temperature,
                max_tokens=request.max_answer_tokens,
            )
        except ChatModelUnavailableError as exc:
            raise ChatProviderNotAvailableError(str(exc)) from exc
        except ChatModelError as exc:
            raise ChatServiceError("Chat model provider returned an error") from exc
        finally:
            if self._owns_model_provider:
                close = getattr(self._model_provider, "aclose", None)
                if callable(close):
                    await close()

        extracted_source_ids = self._citation_mapper.extract_source_ids(model_result.text)
        citations = self._citation_mapper.map_to_citations(
            extracted_source_ids,
            built_context.sources,
            search_outcome.results,
        )
        warning: str | None = None
        if not extracted_source_ids and built_context.sources:
            fallback_source_ids = [
                source.source_id for source in built_context.sources[: min(3, len(built_context.sources))]
            ]
            citations = self._citation_mapper.map_to_citations(
                fallback_source_ids,
                built_context.sources,
                search_outcome.results,
            )
            warning = _MISSING_MARKERS_WARNING

        logger.info(
            "chat answer generated",
            extra={
                "question_length": len(request.question),
                "retrieval_strategy": request.retrieval_strategy,
                "result_count": len(search_outcome.results),
                "included_context_chunk_count": built_context.included_chunk_count,
                "excluded_context_chunk_count": built_context.excluded_chunk_count,
                "estimated_context_tokens": built_context.estimated_tokens,
                "estimated_answer_tokens": math.ceil(len(model_result.text) / 4),
                "citation_count": len(citations),
                "model_provider": model_result.provider,
                "model_name": model_result.model,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )

        return ChatAskResponse(
            answer=model_result.text,
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

    async def _retrieve(self, request: ChatAskRequest) -> _SearchOutcome:
        try:
            if request.retrieval_strategy == "semantic":
                semantic_response = await self._search_service.semantic_search(
                    SemanticSearchRequest(
                        query=request.question,
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
                    query=request.question,
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
            raise ChatProviderNotAvailableError(str(exc)) from exc
        except SearchConfigurationError as exc:
            raise ChatConfigurationError(str(exc)) from exc
        except SearchServiceError as exc:
            raise ChatServiceError("Retrieval failed") from exc

    def _build_messages(self, question: str, context_text: str) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=(
                    "You are PDA, a privacy-first personal document assistant. "
                    "Answer the user's question using only the provided document context. "
                    "If the context does not contain enough information, say that the available "
                    "documents do not contain enough evidence. Cite supporting statements with "
                    "source IDs such as [S1], [S2]. Do not invent document facts or cite sources "
                    "that are not present in the context. Keep the answer concise and directly useful."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Document context:\n{context_text}\n\nQuestion:\n{question}",
            ),
        ]

    def _build_retrieval_diagnostics(
        self,
        request: ChatAskRequest,
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
