"""Tests for grounded chat orchestration, citation mapping, and API contract."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.adapters.embeddings import FakeEmbeddingProvider
from app.adapters.llm.base import ChatModelUnavailableError
from app.adapters.llm.mock import MockChatModelProvider
from app.api.routers.chat import get_chat_service
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.domain.status import DocumentStatus
from app.main import app as fastapi_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.schemas.chat import ChatAskRequest
from app.schemas.context import ContextSource, IncludedTextRange
from app.schemas.hybrid_search import HybridSearchResponse, HybridSearchResult
from app.services.citation_builder import CitationBuilder
from app.services.chat_service import ChatProviderNotAvailableError, ChatService
from app.services.citation_mapper import CitationMapper
from app.services.context_builder import ContextBuilderService
from app.services.hybrid_search_service import HybridSearchService

import app.models  # noqa: F401


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(
        storage_path=tmp_path,  # type: ignore[arg-type]
        embedding_provider="fake",
        embedding_model="test-chat-embedding",
        embedding_dimensions=8,
        model_provider="mock",
        model_name="llama3.1",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def api_client(
    tmp_path: Path, db_session: AsyncSession, api_settings: Settings
) -> Generator[TestClient, None, None]:
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    fastapi_app.dependency_overrides[get_db] = override_get_db
    fastapi_app.dependency_overrides[get_settings] = lambda: api_settings

    with TestClient(fastapi_app) as client:
        yield client

    fastapi_app.dependency_overrides.clear()


async def _seed_doc(
    db: AsyncSession,
    *,
    filename: str,
    status: str = DocumentStatus.ready.value,
) -> Document:
    doc = Document(
        filename=filename,
        status=status,
        category="contracts",
        file_type="pdf",
        path=f"/documents/{filename}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata_jsonb={},
    )
    db.add(doc)
    await db.flush()
    return doc


async def _seed_chunk(
    db: AsyncSession,
    provider: FakeEmbeddingProvider,
    settings: Settings,
    document_id: uuid.UUID,
    *,
    chunk_index: int,
    content: str,
    page_number: int = 3,
) -> DocumentChunk:
    embedding_result = (
        await provider.embed_texts(
            [content],
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    )[0]
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        embedding=embedding_result.vector,
        embedding_model=settings.embedding_model,
        embedding_provider=settings.embedding_provider,
        embedding_dimension=settings.embedding_dimensions,
        page_number=page_number,
        source_start_offset=chunk_index * 100,
        source_end_offset=(chunk_index * 100) + len(content),
        metadata_jsonb={},
    )
    db.add(chunk)
    await db.flush()
    return chunk


def test_chat_request_normalizes_top_level_document_ids() -> None:
    document_id = uuid.uuid4()
    request = ChatAskRequest.model_validate(
        {
            "question": "What is the notice period?",
            "documentIds": [str(document_id)],
        }
    )

    assert request.document_ids is None
    assert request.filters is not None
    assert request.filters.document_ids == [document_id]


def test_citation_mapper_extracts_unique_source_ids_in_order() -> None:
    mapper = CitationMapper()

    assert mapper.extract_source_ids("Answer [S2] then [S1] and again [S2].") == [
        "S2",
        "S1",
    ]


def test_citation_mapper_ignores_unknown_source_ids() -> None:
    mapper = CitationMapper()
    source = ContextSource(
        source_id="S1",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_name="employment-contract.pdf",
        document_path="/documents/employment-contract.pdf",
        page_number=3,
        chunk_index=7,
        start_offset=1200,
        end_offset=1840,
        score=0.91,
        included_text_range=IncludedTextRange(start=0, end=31),
        excerpt_only=False,
    )
    result = HybridSearchResult(
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        document_name=source.document_name,
        document_path=source.document_path,
        category="contracts",
        file_type="pdf",
        page_number=3,
        chunk_index=7,
        start_offset=1200,
        end_offset=1840,
        text="The notice period is three months.",
        excerpt="The notice period is three months.",
        score=0.91,
        vector_score=0.91,
        full_text_score=0.91,
        vector_rank=1,
        full_text_rank=1,
        matched_by=["vector", "full_text"],
        metadata={},
    )

    citations = mapper.map_to_citations(["S99", "S1"], [source], [result])

    assert len(citations) == 1
    assert citations[0].source_id == "S1"
    assert citations[0].document_name == "employment-contract.pdf"


def test_chat_service_ignores_legacy_citation_mapper_instance(
    db_session: AsyncSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="app.services.chat_service")

    service = ChatService(
        db_session,
        model_provider=MockChatModelProvider(),
        citation_mapper=CitationMapper(),
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )

    assert isinstance(service._citation_builder, CitationBuilder)
    assert "ignoring legacy citation_mapper" in caplog.text
    assert any(
        getattr(record, "citation_mapper_type", None) == "CitationMapper"
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_chat_service_builds_prompt_and_maps_citation(
    db_session: AsyncSession,
) -> None:
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=HybridSearchResponse(
            query="What is the notice period?",
            top_k=8,
            fusion_strategy="rrf",
            embedding_model="test-chat-embedding",
            filters_applied={},
            candidate_document_count=1,
            candidate_chunk_count=1,
            vector_candidate_count=1,
            full_text_candidate_count=1,
            result_count=1,
            results=[
                HybridSearchResult(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    document_name="employment-contract.pdf",
                    document_path="/documents/employment-contract.pdf",
                    category="contracts",
                    file_type="pdf",
                    page_number=3,
                    chunk_index=7,
                    start_offset=1200,
                    end_offset=1840,
                    text="The notice period is three months.",
                    excerpt="The notice period is three months.",
                    score=0.91,
                    vector_score=0.91,
                    full_text_score=0.91,
                    vector_rank=1,
                    full_text_rank=1,
                    matched_by=["vector", "full_text"],
                    metadata={},
                )
            ],
        )
    )
    provider = MockChatModelProvider(
        answer="The termination notice period is three months. [S1]"
    )
    service = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        context_builder=ContextBuilderService(),
        model_provider=provider,
        settings=Settings(model_provider="mock", model_name="llama3.1", _env_file=None),  # type: ignore[call-arg]
    )

    response = await service.ask_question(
        ChatAskRequest(question="What is the notice period?")
    )

    assert response.answer.endswith("[S1]")
    assert len(response.citations) == 1
    assert response.citations[0].source_id == "S1"
    assert response.citations[0].document_name == "employment-contract.pdf"
    assert provider.calls
    messages = provider.calls[0]["messages"]
    assert isinstance(messages, list)
    assert "Answer the user's question using only the provided document context" in messages[0].content
    assert "Document context:" in messages[1].content
    assert "Question:\nWhat is the notice period?" in messages[1].content


@pytest.mark.asyncio
async def test_chat_service_skips_model_call_when_no_context(
    db_session: AsyncSession,
) -> None:
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=HybridSearchResponse(
            query="What is the notice period?",
            top_k=8,
            fusion_strategy="rrf",
            embedding_model="test-chat-embedding",
            filters_applied={},
            candidate_document_count=0,
            candidate_chunk_count=0,
            vector_candidate_count=0,
            full_text_candidate_count=0,
            result_count=0,
            results=[],
        )
    )
    provider = MockChatModelProvider(answer="This should not be used.")
    service = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        context_builder=ContextBuilderService(),
        model_provider=provider,
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )

    response = await service.ask_question(
        ChatAskRequest(question="What is the notice period?")
    )

    assert "could not find enough relevant information" in response.answer.lower()
    assert response.citations == []
    assert provider.calls == []


def test_chat_service_defers_search_provider_construction(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fail_if_called(_settings: Settings) -> dict[str, FakeEmbeddingProvider]:
        calls.append("called")
        raise AssertionError("embedding providers should be constructed lazily")

    monkeypatch.setattr("app.services.search_service._build_providers", fail_if_called)
    monkeypatch.setattr(
        "app.services.hybrid_search_service._build_providers", fail_if_called
    )

    ChatService(
        db_session,
        model_provider=MockChatModelProvider(),
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )

    assert calls == []


@pytest.mark.parametrize("strategy", ["hybrid", "semantic"])
@pytest.mark.asyncio
async def test_chat_service_lazily_constructs_requested_search_service(
    db_session: AsyncSession,
    api_settings: Settings,
    strategy: Literal["hybrid", "semantic"],
) -> None:
    provider = FakeEmbeddingProvider()
    doc = await _seed_doc(db_session, filename="employment-contract.pdf")
    await _seed_chunk(
        db_session,
        provider,
        api_settings,
        doc.id,
        chunk_index=7,
        content=(
            "According to the employment contract, the termination notice period is "
            "three months."
        ),
    )
    await db_session.commit()
    service = ChatService(
        db_session,
        embedding_providers={"fake": provider},
        model_provider=MockChatModelProvider(
            answer="The termination notice period is three months. [S1]"
        ),
        settings=api_settings,
    )

    response = await service.ask_question(
        ChatAskRequest(
            question="What is the termination notice period?",
            retrieval_strategy=strategy,
        )
    )

    assert response.retrieval is not None
    assert response.retrieval.strategy == strategy
    assert len(response.citations) == 1
    assert response.citations[0].document_id == doc.id
    if strategy == "semantic":
        assert service._search_service_instance is not None
        assert service._hybrid_search_service_instance is None
    else:
        assert service._hybrid_search_service_instance is not None
        assert service._search_service_instance is None


@pytest.mark.asyncio
async def test_chat_service_adds_fallback_citations_when_model_omits_markers(
    db_session: AsyncSession,
) -> None:
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=HybridSearchResponse(
            query="What is the notice period?",
            top_k=8,
            fusion_strategy="rrf",
            embedding_model="test-chat-embedding",
            filters_applied={},
            candidate_document_count=1,
            candidate_chunk_count=1,
            vector_candidate_count=1,
            full_text_candidate_count=1,
            result_count=1,
            results=[
                HybridSearchResult(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    document_name="employment-contract.pdf",
                    document_path="/documents/employment-contract.pdf",
                    category="contracts",
                    file_type="pdf",
                    page_number=3,
                    chunk_index=7,
                    start_offset=1200,
                    end_offset=1840,
                    text="The notice period is three months.",
                    excerpt="The notice period is three months.",
                    score=0.91,
                    vector_score=0.91,
                    full_text_score=0.91,
                    vector_rank=1,
                    full_text_rank=1,
                    matched_by=["vector", "full_text"],
                    metadata={},
                )
            ],
        )
    )
    provider = MockChatModelProvider(answer="The termination notice period is three months.")
    service = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        context_builder=ContextBuilderService(),
        model_provider=provider,
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )

    response = await service.ask_question(
        ChatAskRequest(question="What is the notice period?", include_diagnostics=True)
    )

    assert len(response.citations) == 1
    assert response.citations[0].source_id == "S1"
    assert response.retrieval is not None
    assert response.retrieval.warning is not None


@pytest.mark.asyncio
async def test_chat_service_provider_unavailable_maps_to_safe_error(
    db_session: AsyncSession,
) -> None:
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=HybridSearchResponse(
            query="What is the notice period?",
            top_k=8,
            fusion_strategy="rrf",
            embedding_model="test-chat-embedding",
            filters_applied={},
            candidate_document_count=1,
            candidate_chunk_count=1,
            vector_candidate_count=1,
            full_text_candidate_count=1,
            result_count=1,
            results=[
                HybridSearchResult(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    document_name="employment-contract.pdf",
                    document_path="/documents/employment-contract.pdf",
                    category="contracts",
                    file_type="pdf",
                    page_number=3,
                    chunk_index=7,
                    start_offset=1200,
                    end_offset=1840,
                    text="The notice period is three months.",
                    excerpt="The notice period is three months.",
                    score=0.91,
                    vector_score=0.91,
                    full_text_score=0.91,
                    vector_rank=1,
                    full_text_rank=1,
                    matched_by=["vector", "full_text"],
                    metadata={},
                )
            ],
        )
    )
    provider = MockChatModelProvider(
        error=ChatModelUnavailableError("Ollama is not reachable")
    )
    service = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        model_provider=provider,
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )

    with pytest.raises(ChatProviderNotAvailableError):
        await service.ask_question(ChatAskRequest(question="What is the notice period?"))


def test_chat_endpoint_exists_in_openapi(api_client: TestClient) -> None:
    schema = api_client.get("/openapi.json").json()

    assert "/chat/ask" in schema["paths"]


def test_chat_endpoint_rejects_empty_question(api_client: TestClient) -> None:
    response = api_client.post("/chat/ask", json={"question": ""})

    assert response.status_code == 422


def test_chat_endpoint_rejects_invalid_top_k(api_client: TestClient) -> None:
    response = api_client.post("/chat/ask", json={"question": "hi", "topK": 0})

    assert response.status_code == 422


def test_chat_endpoint_provider_unavailable_returns_503(
    api_client: TestClient, db_session: AsyncSession
) -> None:
    service = ChatService(
        db_session,
        hybrid_search_service=MagicMock(
            hybrid_search=AsyncMock(
                return_value=HybridSearchResponse(
                    query="What is the notice period?",
                    top_k=8,
                    fusion_strategy="rrf",
                    embedding_model="test-chat-embedding",
                    filters_applied={},
                    candidate_document_count=1,
                    candidate_chunk_count=1,
                    vector_candidate_count=1,
                    full_text_candidate_count=1,
                    result_count=1,
                    results=[
                        HybridSearchResult(
                            chunk_id=uuid.uuid4(),
                            document_id=uuid.uuid4(),
                            document_name="employment-contract.pdf",
                            document_path="/documents/employment-contract.pdf",
                            category="contracts",
                            file_type="pdf",
                            page_number=3,
                            chunk_index=7,
                            start_offset=1200,
                            end_offset=1840,
                            text="The notice period is three months.",
                            excerpt="The notice period is three months.",
                            score=0.91,
                            vector_score=0.91,
                            full_text_score=0.91,
                            vector_rank=1,
                            full_text_rank=1,
                            matched_by=["vector", "full_text"],
                            metadata={},
                        )
                    ],
                )
            )
        ),
        search_service=MagicMock(),
        model_provider=MockChatModelProvider(
            error=ChatModelUnavailableError("Ollama is not reachable")
        ),
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )
    fastapi_app.dependency_overrides[get_chat_service] = lambda: service

    response = api_client.post(
        "/chat/ask", json={"question": "What is the termination notice period?"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Chat model provider is currently unavailable. Please try again later."
    )


@pytest.mark.asyncio
async def test_chat_endpoint_returns_answer_and_real_citation(
    api_client: TestClient,
    db_session: AsyncSession,
    api_settings: Settings,
) -> None:
    provider = FakeEmbeddingProvider()
    doc = await _seed_doc(db_session, filename="employment-contract.pdf")
    chunk = await _seed_chunk(
        db_session,
        provider,
        api_settings,
        doc.id,
        chunk_index=7,
        content=(
            "According to the employment contract, the termination notice period is "
            "three months."
        ),
    )
    await db_session.commit()

    service = ChatService(
        db_session,
        hybrid_search_service=HybridSearchService(
            db_session,
            providers={"fake": provider},
            settings=api_settings,
        ),
        model_provider=MockChatModelProvider(
            answer="The termination notice period is three months. [S1]"
        ),
        settings=api_settings,
    )
    fastapi_app.dependency_overrides[get_chat_service] = lambda: service

    response = api_client.post(
        "/chat/ask",
        json={
            "question": "What is the termination notice period?",
            "topK": 8,
            "filters": {"categories": ["contracts"], "filenameContains": "employment"},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "The termination notice period is three months. [S1]"
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["source_id"] == "S1"
    assert citation["document_id"] == str(doc.id)
    assert citation["document_name"] == "employment-contract.pdf"
    assert citation["chunk_id"] == str(chunk.id)
    assert "three months" in citation["excerpt"]
    assert body["retrieval"]["strategy"] == "hybrid"
    assert body["retrieval"]["result_count"] >= 1


@pytest.mark.asyncio
async def test_chat_endpoint_no_results_skips_model_call(
    api_client: TestClient,
    db_session: AsyncSession,
    api_settings: Settings,
) -> None:
    provider = MockChatModelProvider(answer="Should not be called.")
    service = ChatService(
        db_session,
        hybrid_search_service=MagicMock(
            hybrid_search=AsyncMock(
                return_value=HybridSearchResponse(
                    query="What is the notice period?",
                    top_k=8,
                    fusion_strategy="rrf",
                    embedding_model="test-chat-embedding",
                    filters_applied={},
                    candidate_document_count=0,
                    candidate_chunk_count=0,
                    vector_candidate_count=0,
                    full_text_candidate_count=0,
                    result_count=0,
                    results=[],
                )
            )
        ),
        search_service=MagicMock(),
        model_provider=provider,
        settings=api_settings,
    )
    fastapi_app.dependency_overrides[get_chat_service] = lambda: service

    response = api_client.post(
        "/chat/ask",
        json={"question": "What is the termination notice period?"},
    )

    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert provider.calls == []


def test_chat_endpoint_includes_diagnostics_only_when_requested(
    api_client: TestClient, db_session: AsyncSession
) -> None:
    result = HybridSearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_name="employment-contract.pdf",
        document_path="/documents/employment-contract.pdf",
        category="contracts",
        file_type="pdf",
        page_number=3,
        chunk_index=7,
        start_offset=1200,
        end_offset=1840,
        text="The notice period is three months.",
        excerpt="The notice period is three months.",
        score=0.91,
        vector_score=0.91,
        full_text_score=0.91,
        vector_rank=1,
        full_text_rank=1,
        matched_by=["vector", "full_text"],
        metadata={},
    )
    base_response = HybridSearchResponse(
        query="What is the notice period?",
        top_k=8,
        fusion_strategy="rrf",
        embedding_model="test-chat-embedding",
        filters_applied={},
        candidate_document_count=1,
        candidate_chunk_count=1,
        vector_candidate_count=1,
        full_text_candidate_count=1,
        result_count=1,
        results=[result],
    )
    service = ChatService(
        db_session,
        hybrid_search_service=MagicMock(hybrid_search=AsyncMock(return_value=base_response)),
        search_service=MagicMock(),
        model_provider=MockChatModelProvider(answer="The notice period is three months. [S1]"),
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )
    fastapi_app.dependency_overrides[get_chat_service] = lambda: service

    without_diag = api_client.post("/chat/ask", json={"question": "What is the notice period?"})
    with_diag = api_client.post(
        "/chat/ask",
        json={
            "question": "What is the notice period?",
            "includeDiagnostics": True,
        },
    )

    assert without_diag.status_code == 200
    assert with_diag.status_code == 200
    assert without_diag.json()["retrieval"]["included_context_chunk_count"] is None
    assert with_diag.json()["retrieval"]["included_context_chunk_count"] == 1
    assert without_diag.json()["model"]["citation_marker_count"] is None
    assert with_diag.json()["model"]["citation_marker_count"] == 1


def test_chat_endpoint_document_ids_filter_contract(
    api_client: TestClient, db_session: AsyncSession
) -> None:
    service = ChatService(
        db_session,
        hybrid_search_service=MagicMock(
            hybrid_search=AsyncMock(
                return_value=HybridSearchResponse(
                    query="What is the notice period?",
                    top_k=8,
                    fusion_strategy="rrf",
                    embedding_model="test-chat-embedding",
                    filters_applied={
                        "document_ids": [str(uuid.uuid4())],
                    },
                    candidate_document_count=0,
                    candidate_chunk_count=0,
                    vector_candidate_count=0,
                    full_text_candidate_count=0,
                    result_count=0,
                    results=[],
                )
            )
        ),
        search_service=MagicMock(),
        model_provider=MockChatModelProvider(),
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )
    fastapi_app.dependency_overrides[get_chat_service] = lambda: service

    document_id = uuid.uuid4()
    response = api_client.post(
        "/chat/ask",
        json={
            "question": "What is the notice period?",
            "documentIds": [str(document_id)],
        },
    )

    assert response.status_code == 200
    called_request = service._hybrid_search_service.hybrid_search.await_args.args[0]  # type: ignore[attr-defined]
    assert called_request.filters is not None
    assert called_request.filters.document_ids == [document_id]
    assert called_request.filters.resolved_statuses() == [DocumentStatus.ready.value]
