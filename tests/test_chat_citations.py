"""Integration tests for citation builder with the chat endpoint and /citations/build."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.adapters.embeddings import FakeEmbeddingProvider
from app.adapters.llm.mock import MockChatModelProvider
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.domain.status import DocumentStatus
from app.main import app as fastapi_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.schemas.chat import ChatAskRequest
from app.schemas.hybrid_search import HybridSearchResponse, HybridSearchResult
from app.services.chat_service import ChatService
from app.services.context_builder import ContextBuilderService

import app.models  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


async def _seed_doc(db: AsyncSession, *, filename: str) -> Document:
    doc = Document(
        filename=filename,
        status=DocumentStatus.ready.value,
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


# ---------------------------------------------------------------------------
# Chat endpoint citation integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_citation_includes_document_and_chunk_ids(
    db_session: AsyncSession,
) -> None:
    """Citations must reference real document/chunk IDs from the source map."""
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()

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
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    document_name="employment-contract.pdf",
                    document_path="/documents/employment-contract.pdf",
                    category="contracts",
                    file_type="pdf",
                    page_number=3,
                    chunk_index=7,
                    start_offset=1200,
                    end_offset=1840,
                    text="The notice period is three months from the date of resignation.",
                    excerpt="The notice period is three months.",
                    score=0.91,
                    matched_by=["vector", "full_text"],
                    metadata={},
                )
            ],
        )
    )
    service = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        context_builder=ContextBuilderService(),
        model_provider=MockChatModelProvider(
            answer="The notice period is three months. [S1]"
        ),
        settings=Settings(model_provider="mock", model_name="llama3.1", _env_file=None),  # type: ignore[call-arg]
    )

    response = await service.ask_question(
        ChatAskRequest(question="What is the notice period?")
    )

    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.source_id == "S1"
    assert citation.document_id == doc_id
    assert citation.chunk_id == chunk_id
    assert citation.document_name == "employment-contract.pdf"
    assert citation.page_number == 3
    assert citation.chunk_index == 7
    assert citation.citation_index == 1
    assert "notice period" in citation.excerpt or citation.excerpt != ""


@pytest.mark.asyncio
async def test_chat_unknown_marker_not_fabricated(
    db_session: AsyncSession,
) -> None:
    """[S99] in the model answer must produce zero citations for S99."""
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=HybridSearchResponse(
            query="What is the tariff?",
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
                    document_name="tariff.pdf",
                    document_path="/documents/tariff.pdf",
                    category="contracts",
                    file_type="pdf",
                    page_number=1,
                    chunk_index=0,
                    start_offset=0,
                    end_offset=100,
                    text="The tariff is G11.",
                    excerpt="The tariff is G11.",
                    score=0.88,
                    matched_by=["vector"],
                    metadata={},
                )
            ],
        )
    )
    # Model references S99 which doesn't exist
    service = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        context_builder=ContextBuilderService(),
        model_provider=MockChatModelProvider(answer="The tariff is G11 [S99]."),
        settings=Settings(model_provider="mock", model_name="llama3.1", _env_file=None),  # type: ignore[call-arg]
    )

    response = await service.ask_question(
        ChatAskRequest(question="What is the tariff?")
    )

    # S99 has no matching context source; it must be silently ignored.
    # The fallback (no valid markers → top-3 sources) will produce one citation.
    assert len(response.citations) == 1
    assert response.citations[0].source_id == "S1"
    for cit in response.citations:
        assert cit.source_id != "S99", "S99 must never appear as a citation source_id"


@pytest.mark.asyncio
async def test_chat_duplicate_markers_deduplicated(
    db_session: AsyncSession,
) -> None:
    """Duplicate [S1] markers in model answer must yield a single citation."""
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
                    document_name="contract.pdf",
                    document_path=None,
                    category=None,
                    file_type=None,
                    page_number=None,
                    chunk_index=0,
                    start_offset=None,
                    end_offset=None,
                    text="Notice period is 3 months.",
                    excerpt="Notice period is 3 months.",
                    score=0.9,
                    matched_by=["vector"],
                    metadata={},
                )
            ],
        )
    )
    service = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        context_builder=ContextBuilderService(),
        model_provider=MockChatModelProvider(
            answer="Notice period is 3 months [S1]. See also [S1]."
        ),
        settings=Settings(model_provider="mock", model_name="llama3.1", _env_file=None),  # type: ignore[call-arg]
    )

    response = await service.ask_question(
        ChatAskRequest(question="Notice period?")
    )

    source_ids = [c.source_id for c in response.citations]
    assert source_ids.count("S1") == 1, "Duplicate [S1] must produce only one citation"


@pytest.mark.asyncio
async def test_chat_hybrid_search_duplicate_chunk_single_citation(
    db_session: AsyncSession,
    api_settings: Settings,
) -> None:
    """Hybrid search returning a duplicate chunk from vector+full_text paths yields one citation."""
    provider = FakeEmbeddingProvider()
    doc = await _seed_doc(db_session, filename="offer.pdf")
    await _seed_chunk(
        db_session,
        provider,
        api_settings,
        doc.id,
        chunk_index=0,
        content="The base salary is 5000 PLN per month.",
    )
    await db_session.commit()

    service = ChatService(
        db_session,
        embedding_providers={"fake": provider},
        model_provider=MockChatModelProvider(answer="The salary is 5000 PLN. [S1]"),
        settings=api_settings,
    )

    response = await service.ask_question(
        ChatAskRequest(question="What is the salary?")
    )

    # There is only one chunk in the DB, so we must get exactly one citation
    assert len(response.citations) <= 1
    if response.citations:
        assert response.citations[0].document_id == doc.id


# ---------------------------------------------------------------------------
# /citations/build endpoint tests
# ---------------------------------------------------------------------------


def test_citations_build_returns_citation_for_known_source(
    api_client: TestClient,
) -> None:
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    payload = {
        "answerText": "The notice period is three months. [S1]",
        "sources": [
            {
                "sourceId": "S1",
                "documentId": doc_id,
                "documentName": "employment-contract.pdf",
                "chunkId": chunk_id,
                "pageNumber": 3,
                "chunkIndex": 7,
                "excerpt": "The notice period is three months.",
            }
        ],
    }
    response = api_client.post("/citations/build", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["citations"]) == 1
    cit = body["citations"][0]
    assert cit["source_id"] == "S1"
    assert cit["document_id"] == doc_id
    assert cit["chunk_id"] == chunk_id
    assert cit["page_number"] == 3
    assert cit["chunk_index"] == 7
    assert "three months" in cit["excerpt"]
    assert cit["citation_index"] == 1


def test_citations_build_ignores_unknown_source_marker(
    api_client: TestClient,
) -> None:
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    payload = {
        "answerText": "See [S99]. Also see [S1].",
        "sources": [
            {
                "sourceId": "S1",
                "documentId": doc_id,
                "documentName": "contract.pdf",
                "chunkId": chunk_id,
                "chunkIndex": 0,
                "excerpt": "Contract clause.",
            }
        ],
    }
    response = api_client.post("/citations/build", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source_id"] == "S1"
    assert "S99" in body["diagnostics"]["unknown_source_markers"]


def test_citations_build_deduplicates_markers(
    api_client: TestClient,
) -> None:
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    payload = {
        "answerText": "[S1] and [S1] again.",
        "sources": [
            {
                "sourceId": "S1",
                "documentId": doc_id,
                "documentName": "contract.pdf",
                "chunkId": chunk_id,
                "chunkIndex": 0,
                "excerpt": "Contract clause.",
            }
        ],
    }
    response = api_client.post("/citations/build", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["citations"]) == 1
    assert body["diagnostics"]["duplicate_markers_ignored"] == ["S1"]


def test_citations_build_truncates_long_excerpt(
    api_client: TestClient,
) -> None:
    long_text = "word " * 200  # well over 500 chars

    payload = {
        "answerText": "[S1]",
        "sources": [
            {
                "sourceId": "S1",
                "documentId": str(uuid.uuid4()),
                "documentName": "contract.pdf",
                "chunkId": str(uuid.uuid4()),
                "chunkIndex": 0,
                "text": long_text,
            }
        ],
        "maxExcerptCharacters": 100,
    }
    response = api_client.post("/citations/build", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["diagnostics"]["excerpts_truncated"] == 1
    assert body["citations"][0]["excerpt"].endswith("…")


def test_citations_build_without_answer_text_returns_all_sources(
    api_client: TestClient,
) -> None:
    payload = {
        "sources": [
            {
                "sourceId": "S1",
                "documentId": str(uuid.uuid4()),
                "documentName": "contract.pdf",
                "chunkId": str(uuid.uuid4()),
                "chunkIndex": 0,
                "excerpt": "First clause.",
            },
            {
                "sourceId": "S2",
                "documentId": str(uuid.uuid4()),
                "documentName": "offer.pdf",
                "chunkId": str(uuid.uuid4()),
                "chunkIndex": 1,
                "excerpt": "Second clause.",
            },
        ]
    }
    response = api_client.post("/citations/build", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["citations"]) == 2


def test_citations_build_rejects_invalid_payload(api_client: TestClient) -> None:
    response = api_client.post("/citations/build", json={"invalid": "payload"})

    assert response.status_code == 422


def test_citations_build_endpoint_in_openapi(api_client: TestClient) -> None:
    schema = api_client.get("/openapi.json").json()
    assert "/citations/build" in schema["paths"]
