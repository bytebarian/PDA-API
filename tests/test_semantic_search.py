"""Tests for semantic search: schemas, service, repository, and API contract."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.adapters.embeddings import (
    EmbeddingProviderUnavailableError,
    FakeEmbeddingProvider,
)
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.domain.status import DocumentStatus
from app.main import app as fastapi_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.repositories.chunk_repository import ChunkRepository
from app.schemas.search import SemanticSearchRequest
from app.services.search_service import (
    EmbeddingProviderNotAvailableError,
    SearchService,
    _make_excerpt,
)
from app.services.vector_validation import (
    similarity_from_cosine_distance as score_from_distance,
)

import app.models  # noqa: F401 – register all ORM models

# ---------------------------------------------------------------------------
# Shared fixture: in-memory SQLite database
# ---------------------------------------------------------------------------

_DIM = 4  # small dimensionality for tests


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


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

async def _seed_ready_doc(
    db: AsyncSession,
    *,
    filename: str = "contract.pdf",
    category: str | None = "contract",
    file_type: str | None = "pdf",
    path: str | None = "/docs/contract.pdf",
) -> Document:
    doc = Document(
        filename=filename,
        status=DocumentStatus.ready.value,
        category=category,
        file_type=file_type,
        path=path,
    )
    db.add(doc)
    await db.flush()
    return doc


async def _seed_chunk(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    chunk_index: int = 0,
    content: str = "sample text",
    embedding: list[float] | None = None,
    embedding_model: str = "test-model",
    embedding_dimension: int = _DIM,
    page_number: int | None = 1,
    source_start_offset: int | None = 0,
    source_end_offset: int | None = 100,
    metadata_jsonb: dict[str, Any] | None = None,
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        embedding=embedding,
        embedding_model=embedding_model if embedding is not None else None,
        embedding_provider="fake" if embedding is not None else None,
        embedding_dimension=embedding_dimension if embedding is not None else None,
        page_number=page_number,
        source_start_offset=source_start_offset,
        source_end_offset=source_end_offset,
        metadata_jsonb=metadata_jsonb or {},
    )
    db.add(chunk)
    await db.flush()
    return chunk


# ---------------------------------------------------------------------------
# Unit tests – schema validation
# ---------------------------------------------------------------------------


def test_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="")


def test_request_rejects_query_over_4000_chars() -> None:
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="x" * 4001)


def test_request_rejects_top_k_zero() -> None:
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="hello", top_k=0)


def test_request_rejects_top_k_over_50() -> None:
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="hello", top_k=51)


def test_request_rejects_negative_min_score() -> None:
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="hello", min_score=-0.1)


def test_request_rejects_min_score_over_1() -> None:
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="hello", min_score=1.01)


def test_request_accepts_valid_defaults() -> None:
    req = SemanticSearchRequest(query="hello world")
    assert req.top_k == 10
    assert req.min_score is None
    assert req.include_content is True
    assert req.document_ids is None
    assert req.categories is None
    assert req.file_types is None


def test_request_accepts_boundary_top_k() -> None:
    assert SemanticSearchRequest(query="q", top_k=1).top_k == 1
    assert SemanticSearchRequest(query="q", top_k=50).top_k == 50


# ---------------------------------------------------------------------------
# Unit tests – score normalisation
# ---------------------------------------------------------------------------


def test_score_from_distance_zero_is_one() -> None:
    assert score_from_distance(0.0) == 1.0


def test_score_from_distance_two_is_zero() -> None:
    assert score_from_distance(2.0) == 0.0


def test_score_from_distance_one_is_half() -> None:
    assert score_from_distance(1.0) == 0.5


def test_score_from_distance_never_negative() -> None:
    assert score_from_distance(3.0) == 0.0


# ---------------------------------------------------------------------------
# Unit tests – excerpt generation
# ---------------------------------------------------------------------------


def test_excerpt_short_text_unchanged() -> None:
    text = "Short text."
    assert _make_excerpt(text, max_chars=300) == text


def test_excerpt_exact_limit_unchanged() -> None:
    text = "a" * 300
    assert _make_excerpt(text, max_chars=300) == text


def test_excerpt_long_text_truncated_at_word_boundary() -> None:
    text = "word " * 200  # well over 300 chars
    result = _make_excerpt(text, max_chars=300)
    assert len(result) <= 301  # 300 chars + ellipsis char
    assert result.endswith("…")
    # The character immediately before the ellipsis must not be a space
    # (no trailing space left after trimming at the word boundary).
    assert result[-2] != " "


def test_excerpt_deterministic() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 20
    assert _make_excerpt(text) == _make_excerpt(text)


def test_excerpt_does_not_break_unicode() -> None:
    # Multi-byte chars must not be split
    text = "é " * 200
    result = _make_excerpt(text, max_chars=300)
    result.encode("utf-8")  # must not raise


# ---------------------------------------------------------------------------
# Integration tests – ChunkRepository
# ---------------------------------------------------------------------------


async def test_chunk_repository_returns_ready_documents_only(
    db_session: AsyncSession,
) -> None:
    ready_doc = await _seed_ready_doc(db_session, filename="ready.pdf")
    awaiting_doc = Document(
        filename="awaiting.pdf",
        status=DocumentStatus.awaiting.value,
        category="contract",
    )
    db_session.add(awaiting_doc)
    await db_session.flush()

    emb = [1.0, 0.0, 0.0, 0.0]
    await _seed_chunk(db_session, ready_doc.id, content="ready chunk", embedding=emb)
    await _seed_chunk(
        db_session, awaiting_doc.id, content="awaiting chunk", embedding=emb
    )
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.semantic_search(emb)

    assert all(r.document_id == ready_doc.id for r in results)


async def test_chunk_repository_excludes_null_embeddings(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_ready_doc(db_session)
    await _seed_chunk(db_session, doc.id, content="has embedding", embedding=[1.0, 0.0, 0.0, 0.0])
    await _seed_chunk(db_session, doc.id, chunk_index=1, content="no embedding", embedding=None)
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.semantic_search([1.0, 0.0, 0.0, 0.0])

    assert len(results) == 1
    assert results[0].content == "has embedding"


async def test_chunk_repository_orders_by_similarity(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_ready_doc(db_session)
    # chunk_0 is closest to query [1,0,0,0]
    await _seed_chunk(
        db_session, doc.id, chunk_index=0, content="nearest", embedding=[1.0, 0.0, 0.0, 0.0]
    )
    await _seed_chunk(
        db_session, doc.id, chunk_index=1, content="farther", embedding=[0.0, 1.0, 0.0, 0.0]
    )
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.semantic_search([0.9, 0.1, 0.0, 0.0])

    assert results[0].content == "nearest"
    assert results[0].distance < results[1].distance


async def test_chunk_repository_respects_limit(db_session: AsyncSession) -> None:
    doc = await _seed_ready_doc(db_session)
    for i in range(5):
        emb = [float(j == i) for j in range(_DIM)]
        await _seed_chunk(db_session, doc.id, chunk_index=i, content=f"chunk {i}", embedding=emb)
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.semantic_search([1.0, 0.0, 0.0, 0.0], limit=2)

    assert len(results) == 2


async def test_chunk_repository_filters_by_document_ids(
    db_session: AsyncSession,
) -> None:
    doc_a = await _seed_ready_doc(db_session, filename="a.pdf")
    doc_b = await _seed_ready_doc(db_session, filename="b.pdf")
    emb = [1.0, 0.0, 0.0, 0.0]
    await _seed_chunk(db_session, doc_a.id, content="doc a chunk", embedding=emb)
    await _seed_chunk(db_session, doc_b.id, content="doc b chunk", embedding=emb)
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.semantic_search(emb, document_ids=[doc_a.id])

    assert all(r.document_id == doc_a.id for r in results)


async def test_chunk_repository_filters_by_category(db_session: AsyncSession) -> None:
    doc_contract = await _seed_ready_doc(db_session, filename="c.pdf", category="contract")
    doc_note = await _seed_ready_doc(db_session, filename="n.pdf", category="note")
    emb = [1.0, 0.0, 0.0, 0.0]
    await _seed_chunk(db_session, doc_contract.id, content="contract chunk", embedding=emb)
    await _seed_chunk(db_session, doc_note.id, content="note chunk", embedding=emb)
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.semantic_search(emb, categories=["contract"])

    assert all(r.category == "contract" for r in results)


async def test_chunk_repository_filters_by_file_type(db_session: AsyncSession) -> None:
    doc_pdf = await _seed_ready_doc(db_session, filename="d.pdf", file_type="pdf")
    doc_txt = await _seed_ready_doc(db_session, filename="d.txt", file_type="txt")
    emb = [1.0, 0.0, 0.0, 0.0]
    await _seed_chunk(db_session, doc_pdf.id, content="pdf chunk", embedding=emb)
    await _seed_chunk(db_session, doc_txt.id, content="txt chunk", embedding=emb)
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.semantic_search(emb, file_types=["pdf"])

    assert all(r.file_type == "pdf" for r in results)


async def test_chunk_repository_min_score_filter(db_session: AsyncSession) -> None:
    doc = await _seed_ready_doc(db_session)
    # Perfect match
    await _seed_chunk(
        db_session, doc.id, chunk_index=0, content="exact match", embedding=[1.0, 0.0, 0.0, 0.0]
    )
    # Orthogonal – score 0.0
    await _seed_chunk(
        db_session, doc.id, chunk_index=1, content="orthogonal", embedding=[0.0, 1.0, 0.0, 0.0]
    )
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.semantic_search(
        [1.0, 0.0, 0.0, 0.0], min_score=0.9
    )

    assert len(results) == 1
    assert results[0].content == "exact match"


async def test_chunk_repository_empty_db_returns_empty(db_session: AsyncSession) -> None:
    repo = ChunkRepository(db_session)
    results = await repo.semantic_search([1.0, 0.0, 0.0, 0.0])
    assert results == []


async def test_chunk_repository_result_includes_document_metadata(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_ready_doc(
        db_session,
        filename="employment-contract.pdf",
        category="contract",
        file_type="pdf",
        path="/docs/employment-contract.pdf",
    )
    emb = [1.0, 0.0, 0.0, 0.0]
    await _seed_chunk(
        db_session,
        doc.id,
        content="The notice period is three months.",
        embedding=emb,
        page_number=3,
        source_start_offset=1200,
        source_end_offset=1840,
    )
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.semantic_search(emb)

    assert len(results) == 1
    r = results[0]
    assert r.document_name == "employment-contract.pdf"
    assert r.document_path == "/docs/employment-contract.pdf"
    assert r.category == "contract"
    assert r.file_type == "pdf"
    assert r.page_number == 3
    assert r.start_offset == 1200
    assert r.end_offset == 1840


# ---------------------------------------------------------------------------
# Integration tests – SearchService
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


async def _make_service(
    db: AsyncSession,
    *,
    provider: Any = None,
    dimensions: int = _DIM,
) -> SearchService:
    p = provider or FakeEmbeddingProvider()
    return SearchService(
        db,
        providers={"fake": p, "ollama": p},
        settings=Settings(
            embedding_provider="fake",
            embedding_model="test-model",
            embedding_dimensions=dimensions,
            _env_file=None,  # type: ignore[call-arg]
        ),
    )


async def test_service_returns_correct_chunk_for_semantic_query(
    db_session: AsyncSession,
) -> None:
    """Semantic query about termination returns employment-contract chunk first."""
    doc_employment = await _seed_ready_doc(
        db_session, filename="employment-contract.pdf", category="contract"
    )
    doc_energy = await _seed_ready_doc(
        db_session, filename="energy-tariff.pdf", category="contract"
    )
    doc_note = await _seed_ready_doc(db_session, filename="personal-note.txt", category="note")

    provider = FakeEmbeddingProvider()
    model = "test-model"

    async def _embed(text: str) -> list[float]:
        results = await provider.embed_texts([text], model=model, dimensions=_DIM)
        return results[0].vector

    emb_termination = await _embed("termination notice period contract")
    emb_tariff = await _embed("energy tariff cost supplier")
    emb_note = await _embed("personal shopping list reminder")

    await _seed_chunk(
        db_session,
        doc_employment.id,
        content="The notice period for contract termination is three months.",
        embedding=emb_termination,
        embedding_model=model,
        embedding_dimension=_DIM,
    )
    await _seed_chunk(
        db_session,
        doc_energy.id,
        content="Your energy tariff rate is 28p per kWh.",
        embedding=emb_tariff,
        embedding_model=model,
        embedding_dimension=_DIM,
    )
    await _seed_chunk(
        db_session,
        doc_note.id,
        content="Buy milk and bread tomorrow.",
        embedding=emb_note,
        embedding_model=model,
        embedding_dimension=_DIM,
    )
    await db_session.commit()

    service = await _make_service(db_session, provider=provider)
    request = SemanticSearchRequest(query="termination notice period contract", top_k=5)
    response = await service.semantic_search(request)

    assert response.result_count >= 1
    assert response.results[0].document_id == doc_employment.id


async def test_service_empty_corpus_returns_empty_response(
    db_session: AsyncSession,
) -> None:
    service = await _make_service(db_session)
    request = SemanticSearchRequest(query="any query", top_k=10)
    response = await service.semantic_search(request)

    assert response.result_count == 0
    assert response.results == []


async def test_service_top_k_limits_results(db_session: AsyncSession) -> None:
    doc = await _seed_ready_doc(db_session)
    provider = FakeEmbeddingProvider()
    for i in range(10):
        results = await provider.embed_texts([f"chunk {i}"], model="test-model", dimensions=_DIM)
        emb = results[0].vector
        await _seed_chunk(
            db_session, doc.id, chunk_index=i, content=f"chunk {i}",
            embedding=emb, embedding_model="test-model", embedding_dimension=_DIM,
        )
    await db_session.commit()

    service = await _make_service(db_session, provider=provider)
    response = await service.semantic_search(
        SemanticSearchRequest(query="hello world", top_k=3)
    )

    assert len(response.results) <= 3
    assert response.top_k == 3


async def test_service_min_score_filters_low_quality_results(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_ready_doc(db_session)
    provider = FakeEmbeddingProvider()
    model = "test-model"

    # Use hand-crafted vectors to get known distances
    query_emb = [1.0, 0.0, 0.0, 0.0]
    high_emb = [1.0, 0.0, 0.0, 0.0]   # distance ≈ 0  → score ≈ 1.0
    low_emb = [0.0, 1.0, 0.0, 0.0]    # distance ≈ 1  → score ≈ 0.5

    await _seed_chunk(
        db_session, doc.id, chunk_index=0, content="high score",
        embedding=high_emb, embedding_model=model, embedding_dimension=_DIM,
    )
    await _seed_chunk(
        db_session, doc.id, chunk_index=1, content="low score",
        embedding=low_emb, embedding_model=model, embedding_dimension=_DIM,
    )
    await db_session.commit()

    # Intercept embed_texts so the service uses our hand-crafted query vector

    async def _patched_embed(texts: list[str], *, model: str, dimensions: int | None = None, truncate: bool = True):  # type: ignore[override]
        from app.adapters.embeddings.base import EmbeddingResult
        return [
            EmbeddingResult(text_index=0, vector=query_emb, model=model, dimensions=_DIM)
        ]

    provider.embed_texts = _patched_embed  # type: ignore[method-assign]

    service = await _make_service(db_session, provider=provider)
    response = await service.semantic_search(
        SemanticSearchRequest(query="query", top_k=10, min_score=0.9)
    )

    assert all(r.score >= 0.9 for r in response.results)


async def test_service_document_ids_filter(db_session: AsyncSession) -> None:
    doc_a = await _seed_ready_doc(db_session, filename="a.pdf")
    doc_b = await _seed_ready_doc(db_session, filename="b.pdf")
    provider = FakeEmbeddingProvider()
    model = "test-model"

    for doc in [doc_a, doc_b]:
        results = await provider.embed_texts(["content"], model=model, dimensions=_DIM)
        emb = results[0].vector
        await _seed_chunk(
            db_session, doc.id, content="content",
            embedding=emb, embedding_model=model, embedding_dimension=_DIM,
        )
    await db_session.commit()

    service = await _make_service(db_session, provider=provider)
    response = await service.semantic_search(
        SemanticSearchRequest(query="content", top_k=10, document_ids=[doc_a.id])
    )

    assert all(r.document_id == doc_a.id for r in response.results)


async def test_service_skips_non_ready_documents(db_session: AsyncSession) -> None:
    ready = await _seed_ready_doc(db_session, filename="ready.pdf")
    failed = Document(filename="failed.pdf", status=DocumentStatus.failed.value)
    db_session.add(failed)
    await db_session.flush()

    provider = FakeEmbeddingProvider()
    model = "test-model"
    emb = [1.0, 0.0, 0.0, 0.0]
    await _seed_chunk(
        db_session, ready.id, content="ready chunk",
        embedding=emb, embedding_model=model, embedding_dimension=_DIM,
    )
    await _seed_chunk(
        db_session, failed.id, content="failed chunk",
        embedding=emb, embedding_model=model, embedding_dimension=_DIM,
    )
    await db_session.commit()

    service = await _make_service(db_session, provider=provider)
    response = await service.semantic_search(SemanticSearchRequest(query="chunk", top_k=10))

    assert all(r.document_id == ready.id for r in response.results)


async def test_service_include_content_false_omits_text(db_session: AsyncSession) -> None:
    doc = await _seed_ready_doc(db_session)
    provider = FakeEmbeddingProvider()
    model = "test-model"
    results = await provider.embed_texts(["content"], model=model, dimensions=_DIM)
    emb = results[0].vector
    await _seed_chunk(
        db_session, doc.id, content="sensitive content",
        embedding=emb, embedding_model=model, embedding_dimension=_DIM,
    )
    await db_session.commit()

    service = await _make_service(db_session, provider=provider)
    response = await service.semantic_search(
        SemanticSearchRequest(query="content", top_k=10, include_content=False)
    )

    assert response.result_count > 0
    assert all(r.text is None for r in response.results)
    # excerpt is always present
    assert all(r.excerpt for r in response.results)


async def test_service_raises_on_provider_unavailable(db_session: AsyncSession) -> None:
    unavailable = MagicMock()
    unavailable.name = "fake"
    unavailable.embed_texts = AsyncMock(
        side_effect=EmbeddingProviderUnavailableError("connection refused")
    )

    service = SearchService(
        db_session,
        providers={"fake": unavailable},
        settings=Settings(
            embedding_provider="fake",
            embedding_model="test-model",
            embedding_dimensions=_DIM,
            _env_file=None,  # type: ignore[call-arg]
        ),
    )

    with pytest.raises(EmbeddingProviderNotAvailableError):
        await service.semantic_search(SemanticSearchRequest(query="hello"))


async def test_service_response_includes_citation_metadata(db_session: AsyncSession) -> None:
    doc = await _seed_ready_doc(
        db_session,
        filename="contract.pdf",
        category="contract",
        file_type="pdf",
        path="/docs/contract.pdf",
    )
    provider = FakeEmbeddingProvider()
    model = "test-model"
    results = await provider.embed_texts(["content"], model=model, dimensions=_DIM)
    emb = results[0].vector
    await _seed_chunk(
        db_session, doc.id, content="The notice period is three months.",
        embedding=emb, embedding_model=model, embedding_dimension=_DIM,
        page_number=3, source_start_offset=100, source_end_offset=200,
    )
    await db_session.commit()

    service = await _make_service(db_session, provider=provider)
    response = await service.semantic_search(SemanticSearchRequest(query="notice period"))

    assert response.result_count == 1
    r = response.results[0]
    assert r.document_name == "contract.pdf"
    assert "document_path" not in r.model_dump()
    assert r.category == "contract"
    assert r.file_type == "pdf"
    assert r.page_number == 3
    assert r.start_offset == 100
    assert r.end_offset == 200
    assert r.chunk_id is not None
    assert r.document_id == doc.id
    assert 0.0 <= r.score <= 1.0
    assert r.distance >= 0.0
    assert "embedding_model" in r.metadata


# ---------------------------------------------------------------------------
# API contract tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def api_db() -> AsyncGenerator[AsyncSession, None]:
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
def api_client(
    tmp_path: Path, api_db: AsyncSession
) -> Generator[TestClient, None, None]:
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield api_db

    fastapi_app.dependency_overrides[get_db] = override_get_db
    fastapi_app.dependency_overrides[get_settings] = lambda: Settings(
        storage_path=tmp_path,  # type: ignore[arg-type]
        embedding_provider="fake",
        embedding_model="test-model",
        embedding_dimensions=_DIM,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


def test_api_semantic_search_empty_corpus_returns_200(api_client: TestClient) -> None:
    response = api_client.post(
        "/search/semantic",
        json={"query": "termination notice period"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 0
    assert body["results"] == []


def test_api_semantic_search_invalid_empty_query_returns_422(
    api_client: TestClient,
) -> None:
    response = api_client.post("/search/semantic", json={"query": ""})
    assert response.status_code == 422


def test_api_semantic_search_invalid_top_k_over_50_returns_422(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/search/semantic", json={"query": "hello", "top_k": 51}
    )
    assert response.status_code == 422


def test_api_semantic_search_valid_filters_accepted(api_client: TestClient) -> None:
    doc_id = str(uuid.uuid4())
    response = api_client.post(
        "/search/semantic",
        json={
            "query": "test query",
            "top_k": 5,
            "min_score": 0.5,
            "document_ids": [doc_id],
            "categories": ["contract"],
            "file_types": ["pdf"],
            "include_content": False,
        },
    )
    # Empty corpus → 200 with empty results
    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 0


def test_api_semantic_search_provider_unavailable_returns_503(
    tmp_path: Path, api_db: AsyncSession
) -> None:
    """Verify the router maps EmbeddingProviderNotAvailableError → 503."""
    from unittest.mock import AsyncMock, MagicMock

    unavailable = MagicMock()
    unavailable.name = "fake"
    unavailable.embed_texts = AsyncMock(
        side_effect=EmbeddingProviderUnavailableError("down")
    )

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield api_db

    # Patch SearchService so it uses the unavailable provider
    original_init = SearchService.__init__

    def patched_init(
        self: SearchService, db: AsyncSession, **kwargs: Any
    ) -> None:
        original_init(
            self,
            db,
            providers={"fake": unavailable},
            settings=Settings(
                embedding_provider="fake",
                embedding_model="test-model",
                embedding_dimensions=_DIM,
                _env_file=None,  # type: ignore[call-arg]
            ),
        )

    fastapi_app.dependency_overrides[get_db] = override_get_db
    fastapi_app.dependency_overrides[get_settings] = lambda: Settings(
        storage_path=tmp_path,  # type: ignore[arg-type]
        embedding_provider="fake",
        embedding_model="test-model",
        embedding_dimensions=_DIM,
        _env_file=None,  # type: ignore[call-arg]
    )

    import app.api.routers.search  # noqa: F401 – ensure module import is exercised

    original_service_init = SearchService.__init__
    SearchService.__init__ = patched_init  # type: ignore[method-assign]

    try:
        with TestClient(fastapi_app) as c:
            resp = c.post("/search/semantic", json={"query": "hello"})
        assert resp.status_code == 503
    finally:
        SearchService.__init__ = original_service_init  # type: ignore[method-assign]
        fastapi_app.dependency_overrides.clear()


def test_api_semantic_search_openapi_schema_includes_endpoint(
    api_client: TestClient,
) -> None:
    """Verify the new endpoint is reflected in the OpenAPI schema."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    assert "/search/semantic" in paths
    assert "post" in paths["/search/semantic"]
