"""Tests for grounded report generation orchestration and API contract."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.adapters.embeddings import FakeEmbeddingProvider
from app.adapters.llm.base import ChatModelUnavailableError
from app.adapters.llm.mock import MockChatModelProvider
from app.api.routers.reports import get_report_service
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.domain.status import DocumentStatus
from app.main import app as fastapi_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.schemas.hybrid_search import HybridSearchResponse, HybridSearchResult
from app.schemas.reports import ReportGenerateRequest
from app.services.context_builder import ContextBuilderService
from app.services.hybrid_search_service import HybridSearchService
from app.services.report_service import ReportProviderNotAvailableError, ReportService

import app.models  # noqa: F401


@pytest.fixture
def db_session() -> Generator[AsyncSession, None, None]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async def setup() -> AsyncSession:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        return factory()

    session = asyncio.run(setup())
    try:
        yield session
    finally:
        asyncio.run(session.close())
        asyncio.run(engine.dispose())


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(
        storage_path=tmp_path,  # type: ignore[arg-type]
        embedding_provider="fake",
        embedding_model="test-report-embedding",
        embedding_dimensions=8,
        model_provider="mock",
        model_name="llama3.1",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def api_client(
    db_session: AsyncSession, api_settings: Settings
) -> Generator[TestClient, None, None]:
    async def override_get_db() -> AsyncSession:
        return db_session

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
        category="finance",
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
    page_number: int = 1,
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


def _hybrid_response(result: HybridSearchResult | None) -> HybridSearchResponse:
    results = [] if result is None else [result]
    return HybridSearchResponse(
        query="Summarize tax documents",
        top_k=12,
        fusion_strategy="rrf",
        embedding_model="test-report-embedding",
        filters_applied={},
        candidate_document_count=len(results),
        candidate_chunk_count=len(results),
        vector_candidate_count=len(results),
        full_text_candidate_count=0,
        result_count=len(results),
        results=results,
    )


def test_report_request_normalizes_top_level_document_ids() -> None:
    document_id = uuid.uuid4()
    request = ReportGenerateRequest.model_validate(
        {
            "topic": "Summarize tax documents",
            "documentIds": [str(document_id)],
        }
    )

    assert request.document_ids is None
    assert request.filters is not None
    assert request.filters.document_ids == [document_id]


def test_report_service_builds_markdown_prompt_and_maps_citations(
    db_session: AsyncSession,
) -> None:
    document_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=_hybrid_response(
            HybridSearchResult(
                chunk_id=chunk_id,
                document_id=document_id,
                document_name="tax-return.pdf",
                document_path="/documents/tax-return.pdf",
                category="finance",
                file_type="pdf",
                page_number=2,
                chunk_index=4,
                start_offset=400,
                end_offset=460,
                text="The 2025 tax return lists total income of $75,000.",
                excerpt="The 2025 tax return lists total income of $75,000.",
                score=0.92,
                vector_score=0.92,
                full_text_score=None,
                vector_rank=1,
                full_text_rank=None,
                matched_by=["vector"],
                metadata={},
            )
        )
    )
    provider = MockChatModelProvider(
        answer="# Tax Summary\n\n- Total income was $75,000. [S1]"
    )
    service = ReportService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        context_builder=ContextBuilderService(),
        model_provider=provider,
        settings=Settings(model_provider="mock", model_name="llama3.1", _env_file=None),  # type: ignore[call-arg]
    )

    response = asyncio.run(
        service.generate_report(
            ReportGenerateRequest(
                topic="Summarize tax documents",
                instructions="Highlight income figures.",
                title="Tax Summary",
                include_diagnostics=True,
            )
        )
    )

    assert response.markdown.startswith("# Tax Summary")
    assert len(response.citations) == 1
    assert response.citations[0].source_id == "S1"
    assert response.citations[0].document_id == document_id
    assert response.retrieval is not None
    assert response.retrieval.included_context_chunk_count == 1
    assert response.model is not None
    assert response.model.citation_marker_count == 1
    assert provider.calls
    messages = provider.calls[0]["messages"]
    assert isinstance(messages, list)
    assert "Write a markdown report using only the provided document context" in messages[0].content
    assert "<report_document_context>" in messages[1].content
    assert "Report topic:\nSummarize tax documents" in messages[1].content
    assert "Additional user instructions:\nHighlight income figures." in messages[1].content
    assert provider.calls[0]["max_tokens"] == 2000


def test_report_service_skips_model_call_when_no_context(
    db_session: AsyncSession,
) -> None:
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(return_value=_hybrid_response(None))
    provider = MockChatModelProvider(answer="This should not be called.")
    service = ReportService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        model_provider=provider,
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )

    response = asyncio.run(
        service.generate_report(ReportGenerateRequest(topic="Summarize tax documents"))
    )

    assert response.markdown.startswith("# Report")
    assert response.citations == []
    assert provider.calls == []


def test_report_service_provider_unavailable_maps_to_safe_error(
    db_session: AsyncSession,
) -> None:
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=_hybrid_response(
            HybridSearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                document_name="tax-return.pdf",
                document_path="/documents/tax-return.pdf",
                category="finance",
                file_type="pdf",
                page_number=2,
                chunk_index=4,
                start_offset=400,
                end_offset=460,
                text="The 2025 tax return lists total income of $75,000.",
                excerpt="The 2025 tax return lists total income of $75,000.",
                score=0.92,
                vector_score=0.92,
                full_text_score=None,
                vector_rank=1,
                full_text_rank=None,
                matched_by=["vector"],
                metadata={},
            )
        )
    )
    provider = MockChatModelProvider(
        error=ChatModelUnavailableError("Ollama is not reachable")
    )
    service = ReportService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        model_provider=provider,
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )

    with pytest.raises(ReportProviderNotAvailableError):
        asyncio.run(
            service.generate_report(ReportGenerateRequest(topic="Summarize tax documents"))
        )


def test_report_endpoint_exists_in_openapi(api_client: TestClient) -> None:
    schema = api_client.get("/openapi.json").json()

    assert "/reports/generate" in schema["paths"]


def test_report_endpoint_rejects_empty_topic(api_client: TestClient) -> None:
    response = api_client.post("/reports/generate", json={"topic": ""})

    assert response.status_code == 422


def test_report_endpoint_provider_unavailable_returns_503(
    api_client: TestClient, db_session: AsyncSession
) -> None:
    result = HybridSearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_name="tax-return.pdf",
        document_path="/documents/tax-return.pdf",
        category="finance",
        file_type="pdf",
        page_number=2,
        chunk_index=4,
        start_offset=400,
        end_offset=460,
        text="The 2025 tax return lists total income of $75,000.",
        excerpt="The 2025 tax return lists total income of $75,000.",
        score=0.92,
        vector_score=0.92,
        full_text_score=None,
        vector_rank=1,
        full_text_rank=None,
        matched_by=["vector"],
        metadata={},
    )
    service = ReportService(
        db_session,
        hybrid_search_service=MagicMock(
            hybrid_search=AsyncMock(return_value=_hybrid_response(result))
        ),
        search_service=MagicMock(),
        model_provider=MockChatModelProvider(
            error=ChatModelUnavailableError("Ollama is not reachable")
        ),
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )
    fastapi_app.dependency_overrides[get_report_service] = lambda: service

    response = api_client.post(
        "/reports/generate", json={"topic": "Summarize tax documents"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Report model provider is currently unavailable. Please try again later."
    )


def test_report_endpoint_returns_markdown_and_real_citation(
    api_client: TestClient,
    db_session: AsyncSession,
    api_settings: Settings,
) -> None:
    provider = FakeEmbeddingProvider()
    doc = asyncio.run(_seed_doc(db_session, filename="tax-return.pdf"))
    chunk = asyncio.run(
        _seed_chunk(
            db_session,
            provider,
            api_settings,
            doc.id,
            chunk_index=4,
            content="The 2025 tax return lists total income of $75,000.",
            page_number=2,
        )
    )
    asyncio.run(db_session.commit())

    service = ReportService(
        db_session,
        hybrid_search_service=HybridSearchService(
            db_session,
            providers={"fake": provider},
            settings=api_settings,
        ),
        model_provider=MockChatModelProvider(
            answer="# Tax Summary\n\n- Total income was $75,000. [S1]"
        ),
        settings=api_settings,
    )
    fastapi_app.dependency_overrides[get_report_service] = lambda: service

    response = api_client.post(
        "/reports/generate",
        json={
            "topic": "Summarize tax documents",
            "topK": 12,
            "filters": {"categories": ["finance"], "filenameContains": "tax"},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["markdown"] == "# Tax Summary\n\n- Total income was $75,000. [S1]"
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["source_id"] == "S1"
    assert citation["document_id"] == str(doc.id)
    assert citation["document_name"] == "tax-return.pdf"
    assert citation["chunk_id"] == str(chunk.id)
    assert "$75,000" in citation["excerpt"]
    assert body["retrieval"]["strategy"] == "hybrid"
    assert body["retrieval"]["result_count"] >= 1


def test_report_endpoint_document_ids_filter_contract(
    api_client: TestClient, db_session: AsyncSession
) -> None:
    service = ReportService(
        db_session,
        hybrid_search_service=MagicMock(
            hybrid_search=AsyncMock(return_value=_hybrid_response(None))
        ),
        search_service=MagicMock(),
        model_provider=MockChatModelProvider(),
        settings=Settings(model_provider="mock", _env_file=None),  # type: ignore[call-arg]
    )
    fastapi_app.dependency_overrides[get_report_service] = lambda: service

    document_id = uuid.uuid4()
    response = api_client.post(
        "/reports/generate",
        json={
            "topic": "Summarize tax documents",
            "documentIds": [str(document_id)],
        },
    )

    assert response.status_code == 200
    called_request = service._hybrid_search_service.hybrid_search.await_args.args[0]  # type: ignore[attr-defined]
    assert called_request.filters is not None
    assert called_request.filters.document_ids == [document_id]
    assert called_request.filters.resolved_statuses() == [DocumentStatus.ready.value]
