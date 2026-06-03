"""Tests for hybrid search: schemas, service (unit), repository, and API contract."""

from __future__ import annotations

import math
import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.adapters.embeddings import FakeEmbeddingProvider
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.domain.status import DocumentStatus
from app.main import app as fastapi_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.repositories.chunk_repository import ChunkRepository
from app.schemas.hybrid_search import HybridSearchRequest, HybridSearchResponse
from app.schemas.search_filters import SearchFilters
from app.services.hybrid_search_service import (
    HybridSearchService,
    _fuse_results,
    _rrf_score,
    _weighted_score,
)

import app.models  # noqa: F401

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DIM = 4  # small embedding dimensionality for tests


# ---------------------------------------------------------------------------
# Shared fixture: in-memory SQLite database
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


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


async def _seed_doc(
    db: AsyncSession,
    *,
    filename: str = "doc.pdf",
    category: str | None = "contract",
    file_type: str | None = "pdf",
    status: str = DocumentStatus.ready.value,
    metadata_jsonb: dict[str, Any] | None = None,
) -> Document:
    doc = Document(
        filename=filename,
        status=status,
        category=category,
        file_type=file_type,
        path=f"/docs/{filename}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata_jsonb=metadata_jsonb or {},
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
    embedding_dimension: int = _DIM,
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        embedding=embedding,
        embedding_model="test-model" if embedding is not None else None,
        embedding_provider="fake" if embedding is not None else None,
        embedding_dimension=embedding_dimension if embedding is not None else None,
        page_number=1,
        source_start_offset=0,
        source_end_offset=len(content),
        metadata_jsonb={},
    )
    db.add(chunk)
    await db.flush()
    return chunk


# ---------------------------------------------------------------------------
# Unit tests – HybridSearchRequest schema
# ---------------------------------------------------------------------------


def test_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        HybridSearchRequest(query="")


def test_request_rejects_query_over_4000_chars() -> None:
    with pytest.raises(ValidationError):
        HybridSearchRequest(query="x" * 4001)


def test_request_rejects_top_k_zero() -> None:
    with pytest.raises(ValidationError):
        HybridSearchRequest(query="hello", top_k=0)


def test_request_rejects_top_k_over_50() -> None:
    with pytest.raises(ValidationError):
        HybridSearchRequest(query="hello", top_k=51)


def test_request_rejects_negative_min_score() -> None:
    with pytest.raises(ValidationError):
        HybridSearchRequest(query="hello", min_score=-0.1)


def test_request_rejects_min_score_over_1() -> None:
    with pytest.raises(ValidationError):
        HybridSearchRequest(query="hello", min_score=1.01)


def test_request_rejects_both_weights_zero() -> None:
    with pytest.raises(ValidationError):
        HybridSearchRequest(query="hello", vector_weight=0.0, full_text_weight=0.0)


def test_request_accepts_valid_defaults() -> None:
    req = HybridSearchRequest(query="hello world")
    assert req.top_k == 10
    assert req.min_score is None
    assert req.include_content is True
    assert req.fusion_strategy == "rrf"
    assert req.vector_weight == 0.65
    assert req.full_text_weight == 0.35
    assert req.vector_top_k is None
    assert req.full_text_top_k is None
    assert req.filters is None


def test_request_resolved_filters_defaults_to_ready_status() -> None:
    req = HybridSearchRequest(query="hello")
    assert req.resolved_filters().resolved_statuses() == [DocumentStatus.ready.value]


def test_request_effective_top_k_defaults() -> None:
    req = HybridSearchRequest(query="hello", top_k=10)
    # Default: max(top_k * 3, 30)
    assert req.effective_vector_top_k() == 30
    assert req.effective_full_text_top_k() == 30


def test_request_effective_top_k_uses_explicit_values() -> None:
    req = HybridSearchRequest(
        query="hello",
        top_k=10,
        vector_top_k=50,
        full_text_top_k=40,
    )
    assert req.effective_vector_top_k() == 50
    assert req.effective_full_text_top_k() == 40


def test_request_accepts_camel_case_fields() -> None:
    req = HybridSearchRequest.model_validate(
        {
            "query": "termination notice",
            "topK": 5,
            "vectorTopK": 20,
            "fullTextTopK": 20,
            "fusionStrategy": "weighted",
            "vectorWeight": 0.7,
            "fullTextWeight": 0.3,
            "minScore": 0.1,
            "includeContent": False,
        }
    )
    assert req.top_k == 5
    assert req.vector_top_k == 20
    assert req.full_text_top_k == 20
    assert req.fusion_strategy == "weighted"
    assert req.vector_weight == 0.7
    assert req.full_text_weight == 0.3
    assert req.min_score == 0.1
    assert req.include_content is False


def test_request_rejects_unknown_fusion_strategy() -> None:
    with pytest.raises(ValidationError):
        HybridSearchRequest.model_validate(
            {"query": "hello", "fusionStrategy": "bm25"}
        )


# ---------------------------------------------------------------------------
# Unit tests – RRF and weighted scoring
# ---------------------------------------------------------------------------


def test_rrf_score_both_sources() -> None:
    score = _rrf_score(1, 1, vector_weight=1.0, full_text_weight=1.0, k=60)
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_rrf_score_vector_only() -> None:
    score = _rrf_score(2, None, vector_weight=0.65, full_text_weight=0.35, k=60)
    expected = 0.65 / (60 + 2)
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_rrf_score_full_text_only() -> None:
    score = _rrf_score(None, 3, vector_weight=0.65, full_text_weight=0.35, k=60)
    expected = 0.35 / (60 + 3)
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_rrf_score_higher_rank_gives_lower_score() -> None:
    score_rank1 = _rrf_score(1, None, vector_weight=1.0, full_text_weight=0.0)
    score_rank5 = _rrf_score(5, None, vector_weight=1.0, full_text_weight=0.0)
    assert score_rank1 > score_rank5


def test_weighted_score_both_sources() -> None:
    score = _weighted_score(0.8, 0.6, vector_weight=0.65, full_text_weight=0.35)
    expected = (0.65 * 0.8 + 0.35 * 0.6) / (0.65 + 0.35)
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_weighted_score_vector_only() -> None:
    score = _weighted_score(0.9, None, vector_weight=0.65, full_text_weight=0.35)
    expected = (0.65 * 0.9) / 0.65
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_weighted_score_full_text_only() -> None:
    score = _weighted_score(None, 0.7, vector_weight=0.65, full_text_weight=0.35)
    expected = (0.35 * 0.7) / 0.35
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_weighted_score_both_zero_weights() -> None:
    assert _weighted_score(0.9, 0.7, vector_weight=0.0, full_text_weight=0.0) == 0.0


# ---------------------------------------------------------------------------
# Unit tests – _fuse_results
# ---------------------------------------------------------------------------


def _make_chunk_row(
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    chunk_index: int = 0,
    distance: float = 0.5,
    content: str = "sample text",
) -> Any:
    """Create a minimal ChunkSearchRow-like object for fusion tests."""
    from app.repositories.chunk_repository import ChunkSearchRow

    return ChunkSearchRow(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        document_name="doc.pdf",
        document_path="/docs/doc.pdf",
        category="contract",
        file_type="pdf",
        page_number=1,
        chunk_index=chunk_index,
        start_offset=0,
        end_offset=100,
        content=content,
        metadata={},
        embedding_model="test-model",
        distance=distance,
    )


def _make_ft_row(
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    chunk_index: int = 0,
    ft_score: float = 0.5,
    content: str = "sample text",
) -> Any:
    """Create a minimal FullTextSearchRow-like object for fusion tests."""
    from app.repositories.chunk_repository import FullTextSearchRow

    return FullTextSearchRow(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        document_name="doc.pdf",
        document_path="/docs/doc.pdf",
        category="contract",
        file_type="pdf",
        page_number=1,
        chunk_index=chunk_index,
        start_offset=0,
        end_offset=100,
        content=content,
        metadata={},
        ft_score=ft_score,
    )


def test_fusion_deduplicates_shared_chunk() -> None:
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    v_row = _make_chunk_row(chunk_id=chunk_id, document_id=doc_id)
    ft_row = _make_ft_row(chunk_id=chunk_id, document_id=doc_id)

    fused = _fuse_results(
        [v_row],
        [ft_row],
        strategy="rrf",
        vector_weight=0.65,
        full_text_weight=0.35,
    )
    assert len(fused) == 1
    entry = fused[0]
    assert entry.chunk_id == chunk_id
    assert "vector" in entry.matched_by
    assert "full_text" in entry.matched_by
    assert entry.vector_rank == 1
    assert entry.full_text_rank == 1


def test_fusion_vector_only_chunk_included() -> None:
    v_row = _make_chunk_row()
    fused = _fuse_results(
        [v_row],
        [],
        strategy="rrf",
        vector_weight=0.65,
        full_text_weight=0.35,
    )
    assert len(fused) == 1
    assert fused[0].matched_by == ["vector"]
    assert fused[0].full_text_rank is None


def test_fusion_full_text_only_chunk_included() -> None:
    ft_row = _make_ft_row()
    fused = _fuse_results(
        [],
        [ft_row],
        strategy="rrf",
        vector_weight=0.65,
        full_text_weight=0.35,
    )
    assert len(fused) == 1
    assert fused[0].matched_by == ["full_text"]
    assert fused[0].vector_rank is None


def test_fusion_empty_both_sources() -> None:
    fused = _fuse_results(
        [],
        [],
        strategy="rrf",
        vector_weight=0.65,
        full_text_weight=0.35,
    )
    assert fused == []


def test_fusion_rrf_shared_chunk_scores_higher_than_single_source() -> None:
    """A chunk matched by both sources should outscore a chunk matched by only one."""
    shared_id = uuid.uuid4()
    shared_doc = uuid.uuid4()
    other_id = uuid.uuid4()
    other_doc = uuid.uuid4()

    # shared chunk appears at rank 1 in both sources
    v_shared = _make_chunk_row(chunk_id=shared_id, document_id=shared_doc, distance=0.1)
    ft_shared = _make_ft_row(chunk_id=shared_id, document_id=shared_doc, ft_score=0.9)

    # another chunk appears only in vector at rank 2
    v_other = _make_chunk_row(chunk_id=other_id, document_id=other_doc, distance=0.2)

    fused = _fuse_results(
        [v_shared, v_other],
        [ft_shared],
        strategy="rrf",
        vector_weight=0.65,
        full_text_weight=0.35,
    )
    shared_entry = next(e for e in fused if e.chunk_id == shared_id)
    other_entry = next(e for e in fused if e.chunk_id == other_id)
    assert shared_entry.score > other_entry.score


def test_fusion_min_score_filters_low_scoring_results() -> None:
    fused = _fuse_results(
        [_make_chunk_row()],
        [],
        strategy="rrf",
        vector_weight=0.65,
        full_text_weight=0.35,
    )
    # All scores from RRF are << 1.0 – any high threshold should filter everything
    high_min = 0.99
    filtered = [e for e in fused if e.score >= high_min]
    assert filtered == []


def test_fusion_stable_tie_breaking() -> None:
    """Identical-score entries sort by best_rank then doc_id then chunk_index."""
    from app.repositories.chunk_repository import FullTextSearchRow

    doc_a = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000000")
    doc_b = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000000")
    chunk_a = uuid.uuid4()
    chunk_b = uuid.uuid4()

    # Both chunks only in full-text at the same rank; doc_a < doc_b
    ft_a = FullTextSearchRow(
        chunk_id=chunk_a,
        document_id=doc_a,
        document_name="a.pdf",
        document_path=None,
        category=None,
        file_type=None,
        page_number=None,
        chunk_index=0,
        start_offset=None,
        end_offset=None,
        content="text",
        metadata={},
        ft_score=0.8,
    )
    ft_b = FullTextSearchRow(
        chunk_id=chunk_b,
        document_id=doc_b,
        document_name="b.pdf",
        document_path=None,
        category=None,
        file_type=None,
        page_number=None,
        chunk_index=0,
        start_offset=None,
        end_offset=None,
        content="text",
        metadata={},
        ft_score=0.8,
    )

    fused = _fuse_results(
        [],
        [ft_a, ft_b],
        strategy="rrf",
        vector_weight=0.65,
        full_text_weight=0.35,
    )
    # Both at rank 1 and rank 2 respectively
    # ft_a gets rank 1, ft_b gets rank 2 → ft_a has higher score
    assert fused[0].chunk_id == chunk_a


def test_fusion_weighted_strategy() -> None:
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    v_row = _make_chunk_row(chunk_id=chunk_id, document_id=doc_id, distance=0.2)
    ft_row = _make_ft_row(chunk_id=chunk_id, document_id=doc_id, ft_score=0.7)

    fused = _fuse_results(
        [v_row],
        [ft_row],
        strategy="weighted",
        vector_weight=0.65,
        full_text_weight=0.35,
    )
    assert len(fused) == 1
    entry = fused[0]
    from app.services.vector_validation import similarity_from_cosine_distance

    v_score = similarity_from_cosine_distance(0.2)
    expected = _weighted_score(v_score, 0.7, vector_weight=0.65, full_text_weight=0.35)
    assert math.isclose(entry.score, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Repository integration tests (SQLite)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_text_candidates_returns_matching_chunks(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_doc(db_session, filename="employment-contract.pdf")
    await _seed_chunk(
        db_session,
        doc.id,
        chunk_index=0,
        content="The termination notice period is three months as per contract.",
    )
    await _seed_chunk(
        db_session,
        doc.id,
        chunk_index=1,
        content="Vacation entitlement is 26 days per year.",
    )
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.full_text_candidates(
        "termination notice",
        limit=10,
        search_filters=SearchFilters(),
    )
    assert len(results) >= 1
    matched_contents = [r.content for r in results]
    assert any("termination" in c.lower() for c in matched_contents)


@pytest.mark.asyncio
async def test_full_text_candidates_excludes_processing_documents(
    db_session: AsyncSession,
) -> None:
    # processing document – should be excluded by default
    doc_proc = await _seed_doc(
        db_session,
        filename="draft.md",
        status=DocumentStatus.processing.value,
    )
    await _seed_chunk(
        db_session,
        doc_proc.id,
        content="termination notice period draft document",
    )
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.full_text_candidates(
        "termination notice",
        limit=10,
        search_filters=SearchFilters(),
    )
    # Default status is ready; processing document must be excluded
    for r in results:
        assert r.document_id != doc_proc.id


@pytest.mark.asyncio
async def test_full_text_candidates_respects_category_filter(
    db_session: AsyncSession,
) -> None:
    doc_contract = await _seed_doc(
        db_session, filename="contract.pdf", category="contract"
    )
    doc_note = await _seed_doc(
        db_session, filename="notes.md", category="note"
    )
    await _seed_chunk(
        db_session,
        doc_contract.id,
        content="non-compete clause applies to all employees",
    )
    await _seed_chunk(
        db_session,
        doc_note.id,
        content="non-compete reminder from HR meeting",
    )
    await db_session.commit()

    repo = ChunkRepository(db_session)
    results = await repo.full_text_candidates(
        "non-compete",
        limit=10,
        search_filters=SearchFilters(categories=["contract"]),
    )
    assert all(r.document_id == doc_contract.id for r in results)


@pytest.mark.asyncio
async def test_full_text_candidates_empty_query_no_results(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_doc(db_session)
    await _seed_chunk(db_session, doc.id, content="some generic text")
    await db_session.commit()

    repo = ChunkRepository(db_session)
    # A query with no matching terms should return no results
    results = await repo.full_text_candidates(
        "xyzzynonexistentterm12345",
        limit=10,
        search_filters=SearchFilters(),
    )
    assert results == []


# ---------------------------------------------------------------------------
# Service integration tests (SQLite with FakeEmbeddingProvider)
# ---------------------------------------------------------------------------


def _make_fake_providers(dim: int = _DIM) -> dict[str, Any]:
    provider = FakeEmbeddingProvider()
    return {"fake": provider}


def _make_settings(dim: int = _DIM) -> Settings:
    return Settings(
        embedding_provider="fake",
        embedding_model="test-model",
        embedding_dimensions=dim,
    )


@pytest.mark.asyncio
async def test_hybrid_search_returns_response(db_session: AsyncSession) -> None:
    doc = await _seed_doc(db_session, filename="employment-contract.pdf")
    embedding = [0.1, 0.2, 0.3, 0.4]
    await _seed_chunk(
        db_session,
        doc.id,
        content="The termination notice period is three months.",
        embedding=embedding,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    request = HybridSearchRequest(
        query="termination notice",
        top_k=10,
    )
    response = await service.hybrid_search(request)

    assert isinstance(response, HybridSearchResponse)
    assert response.query == "termination notice"
    assert response.fusion_strategy == "rrf"
    assert response.embedding_model == "test-model"
    assert response.vector_candidate_count >= 0
    assert response.full_text_candidate_count >= 0
    assert response.result_count == len(response.results)


@pytest.mark.asyncio
async def test_hybrid_search_includes_both_retrieval_diagnostics(
    db_session: AsyncSession,
) -> None:
    """Response must always expose both vector and full-text candidate counts."""
    doc = await _seed_doc(db_session)
    embedding = [0.1, 0.2, 0.3, 0.4]
    await _seed_chunk(
        db_session,
        doc.id,
        content="energy tariff G11 consumer agreement",
        embedding=embedding,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    response = await service.hybrid_search(HybridSearchRequest(query="energy tariff G11"))
    assert hasattr(response, "vector_candidate_count")
    assert hasattr(response, "full_text_candidate_count")


@pytest.mark.asyncio
async def test_hybrid_search_candidate_counts_include_full_text_only_matches(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_doc(db_session, filename="notes.md")
    await _seed_chunk(
        db_session,
        doc.id,
        content="energy tariff G11 consumer agreement",
        embedding=None,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    response = await service.hybrid_search(HybridSearchRequest(query="energy tariff G11"))

    assert response.vector_candidate_count == 0
    assert response.full_text_candidate_count > 0
    assert response.candidate_chunk_count >= response.full_text_candidate_count
    assert response.candidate_document_count >= 1


@pytest.mark.asyncio
async def test_hybrid_search_honors_vector_top_k_above_50(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_doc(db_session, filename="vector-heavy.pdf")
    embedding = [0.1, 0.2, 0.3, 0.4]
    for index in range(90):
        await _seed_chunk(
            db_session,
            doc.id,
            chunk_index=index,
            content=f"content-{index}",
            embedding=embedding,
        )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    response = await service.hybrid_search(
        HybridSearchRequest(
            query="unmatched-terms",
            top_k=10,
            vector_top_k=80,
            full_text_top_k=10,
        )
    )

    assert response.vector_candidate_count == 80
    assert response.full_text_candidate_count == 0


@pytest.mark.asyncio
async def test_hybrid_search_result_has_scoring_diagnostics(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_doc(db_session, filename="energy-contract.pdf")
    embedding = [0.1, 0.2, 0.3, 0.4]
    await _seed_chunk(
        db_session,
        doc.id,
        content="PGE tariff G11 termination notice period applicable.",
        embedding=embedding,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    response = await service.hybrid_search(
        HybridSearchRequest(query="PGE tariff G11 termination notice")
    )
    if response.results:
        result = response.results[0]
        assert result.score >= 0.0
        # matched_by is always present
        assert isinstance(result.matched_by, list)
        assert len(result.matched_by) >= 1


@pytest.mark.asyncio
async def test_hybrid_search_metadata_filter_reduces_candidates(
    db_session: AsyncSession,
) -> None:
    """Metadata filter must narrow the corpus before both retrieval paths."""
    embedding = [0.1, 0.2, 0.3, 0.4]

    doc_contract = await _seed_doc(
        db_session, filename="contract.pdf", category="contract"
    )
    await _seed_chunk(
        db_session,
        doc_contract.id,
        content="non-compete clause in employment contract",
        embedding=embedding,
    )

    doc_note = await _seed_doc(
        db_session, filename="notes.md", category="note"
    )
    await _seed_chunk(
        db_session,
        doc_note.id,
        content="non-compete reminder notes for staff",
        embedding=embedding,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )

    # Filter to contracts only
    filtered_response = await service.hybrid_search(
        HybridSearchRequest(
            query="non-compete",
            filters=SearchFilters(categories=["contract"]),
        )
    )
    # Unfiltered
    unfiltered_response = await service.hybrid_search(
        HybridSearchRequest(query="non-compete")
    )

    # Filtered candidate pool must be smaller (at most equal)
    assert filtered_response.candidate_chunk_count <= unfiltered_response.candidate_chunk_count
    # All results must belong to contracts
    for result in filtered_response.results:
        assert result.category == "contract"


@pytest.mark.asyncio
async def test_hybrid_search_excludes_processing_documents_by_default(
    db_session: AsyncSession,
) -> None:
    embedding = [0.1, 0.2, 0.3, 0.4]
    doc_processing = await _seed_doc(
        db_session,
        filename="draft.md",
        status=DocumentStatus.processing.value,
    )
    await _seed_chunk(
        db_session,
        doc_processing.id,
        content="termination notice found in draft",
        embedding=embedding,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    response = await service.hybrid_search(
        HybridSearchRequest(query="termination notice")
    )
    doc_ids = {r.document_id for r in response.results}
    assert doc_processing.id not in doc_ids


@pytest.mark.asyncio
async def test_hybrid_search_deduplicates_chunks_from_both_paths(
    db_session: AsyncSession,
) -> None:
    """A chunk matched by both sources must appear only once in the response."""
    doc = await _seed_doc(db_session, filename="contract.pdf")
    embedding = [0.1, 0.2, 0.3, 0.4]
    await _seed_chunk(
        db_session,
        doc.id,
        content="termination notice period is three months in this contract",
        embedding=embedding,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    response = await service.hybrid_search(
        HybridSearchRequest(query="termination notice", top_k=50)
    )

    chunk_ids = [r.chunk_id for r in response.results]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs in response"


@pytest.mark.asyncio
async def test_hybrid_search_empty_results_when_no_matching_docs(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_doc(db_session, filename="unrelated.pdf", category="note")
    embedding = [0.1, 0.2, 0.3, 0.4]
    await _seed_chunk(
        db_session,
        doc.id,
        content="generic text with no relevant keywords",
        embedding=embedding,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    response = await service.hybrid_search(
        HybridSearchRequest(
            query="invoices",
            filters=SearchFilters(categories=["invoice"]),
        )
    )
    assert response.result_count == 0
    assert response.results == []


@pytest.mark.asyncio
async def test_hybrid_search_include_content_false_omits_text(
    db_session: AsyncSession,
) -> None:
    doc = await _seed_doc(db_session)
    embedding = [0.1, 0.2, 0.3, 0.4]
    await _seed_chunk(
        db_session, doc.id, content="sensitive content text here", embedding=embedding
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    response = await service.hybrid_search(
        HybridSearchRequest(query="sensitive content", include_content=False)
    )
    for result in response.results:
        assert result.text is None


@pytest.mark.asyncio
async def test_hybrid_search_improved_relevance_over_vector_only(
    db_session: AsyncSession,
) -> None:
    """Exact-term full-text match must rank above generic semantic-only chunk.

    Fixture:
    - Chunk A: "The termination notice period is three months in this employment contract."
      → Contains exact query terms; should rank #1 in hybrid.
    - Chunk B: "Employee benefits include vacation, health insurance, and retirement plans."
      → Semantically related to employment contracts but has no exact query terms.

    FakeEmbeddingProvider generates deterministic embeddings from the text hash, so
    vector scores will differ.  Hybrid fusion should push chunk A above chunk B.
    """
    doc = await _seed_doc(db_session, filename="employment-contract.pdf", category="contract")

    # Use embeddings that are different to make vector search non-trivial
    # but full-text will clearly match chunk A for "termination notice"
    embedding_a = [0.9, 0.1, 0.1, 0.1]  # distinct direction
    embedding_b = [0.1, 0.9, 0.1, 0.1]  # distinct direction

    chunk_a = await _seed_chunk(
        db_session,
        doc.id,
        chunk_index=0,
        content="The termination notice period is three months in this employment contract.",
        embedding=embedding_a,
    )
    chunk_b = await _seed_chunk(
        db_session,
        doc.id,
        chunk_index=1,
        content="Employee benefits include vacation, health insurance, and retirement plans.",
        embedding=embedding_b,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )

    # Query uses exact terms from chunk A
    response = await service.hybrid_search(
        HybridSearchRequest(query="termination notice", top_k=10)
    )

    chunk_ids = [r.chunk_id for r in response.results]
    assert chunk_a.id in chunk_ids, "Exact-term chunk must appear in hybrid results"

    # Chunk A must be ranked above chunk B (or chunk B not present at all)
    if chunk_b.id in chunk_ids:
        rank_a = chunk_ids.index(chunk_a.id)
        rank_b = chunk_ids.index(chunk_b.id)
        assert rank_a < rank_b, (
            "Hybrid search must rank exact-term chunk above generic semantic chunk"
        )

    # Chunk A must have been matched via full-text at minimum
    result_a = next(r for r in response.results if r.chunk_id == chunk_a.id)
    assert "full_text" in result_a.matched_by, (
        "Exact-term chunk must be matched via full-text path"
    )


@pytest.mark.asyncio
async def test_hybrid_search_full_text_heavy_weights(
    db_session: AsyncSession,
) -> None:
    """With full_text_weight=1.0 and vector_weight=0.0 full-text drives ranking."""
    doc = await _seed_doc(db_session, filename="energy-contract.pdf")
    embedding = [0.1, 0.2, 0.3, 0.4]
    chunk = await _seed_chunk(
        db_session,
        doc.id,
        content="PGE tariff G11 non-compete clause",
        embedding=embedding,
    )
    await db_session.commit()

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    response = await service.hybrid_search(
        HybridSearchRequest(
            query="PGE tariff G11",
            vector_weight=0.0,
            full_text_weight=1.0,
        )
    )
    chunk_ids = [r.chunk_id for r in response.results]
    assert chunk.id in chunk_ids
    result = next(r for r in response.results if r.chunk_id == chunk.id)
    assert "full_text" in result.matched_by


@pytest.mark.asyncio
async def test_hybrid_search_full_text_only_skips_vector_path(
    db_session: AsyncSession,
) -> None:
    unused_provider = MagicMock()
    unused_provider.embed_texts = AsyncMock(
        side_effect=AssertionError("unexpected embed")
    )

    service = HybridSearchService(
        db_session,
        providers={"fake": unused_provider},
        settings=_make_settings(),
    )
    ft_row = _make_ft_row(content="PGE tariff G11 non-compete clause")
    service._repository.semantic_search = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("unexpected vector search")
    )
    service._repository.full_text_candidates = AsyncMock(  # type: ignore[method-assign]
        return_value=[ft_row]
    )

    response = await service.hybrid_search(
        HybridSearchRequest(
            query="PGE tariff G11",
            vector_weight=0.0,
            full_text_weight=1.0,
        )
    )

    unused_provider.embed_texts.assert_not_called()
    service._repository.semantic_search.assert_not_awaited()
    service._repository.full_text_candidates.assert_awaited_once()
    assert response.vector_candidate_count == 0
    assert response.full_text_candidate_count == 1
    assert response.results[0].matched_by == ["full_text"]


@pytest.mark.asyncio
async def test_hybrid_search_vector_only_skips_full_text_path(
    db_session: AsyncSession,
) -> None:
    from app.repositories.chunk_repository import (
        ChunkSearchDiagnostics,
        ChunkSearchResult,
    )

    service = HybridSearchService(
        db_session,
        providers=_make_fake_providers(),
        settings=_make_settings(),
    )
    vector_row = _make_chunk_row(content="PGE tariff G11 non-compete clause")
    service._repository.semantic_search = AsyncMock(  # type: ignore[method-assign]
        return_value=ChunkSearchResult(
            rows=[vector_row],
            diagnostics=ChunkSearchDiagnostics(
                candidate_document_count=1,
                candidate_chunk_count=1,
            ),
        )
    )
    service._repository.full_text_candidates = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("unexpected full-text search")
    )

    response = await service.hybrid_search(
        HybridSearchRequest(
            query="PGE tariff G11",
            vector_weight=1.0,
            full_text_weight=0.0,
        )
    )

    service._repository.semantic_search.assert_awaited_once()
    service._repository.full_text_candidates.assert_not_awaited()
    assert response.vector_candidate_count == 1
    assert response.full_text_candidate_count == 0
    assert response.results[0].matched_by == ["vector"]


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
    tmp_path: Any, api_db: AsyncSession
) -> Generator[TestClient, None, None]:
    from pathlib import Path

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield api_db

    fastapi_app.dependency_overrides[get_db] = override_get_db
    fastapi_app.dependency_overrides[get_settings] = lambda: Settings(
        storage_path=Path(str(tmp_path)),  # type: ignore[arg-type]
        embedding_provider="fake",
        embedding_model="test-model",
        embedding_dimensions=_DIM,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


def test_hybrid_endpoint_exists_in_openapi(api_client: TestClient) -> None:
    response = api_client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/search/hybrid" in paths


def test_hybrid_endpoint_rejects_empty_query(api_client: TestClient) -> None:
    response = api_client.post("/search/hybrid", json={"query": ""})
    assert response.status_code == 422


def test_hybrid_endpoint_rejects_invalid_weights(api_client: TestClient) -> None:
    response = api_client.post(
        "/search/hybrid",
        json={"query": "hello", "vectorWeight": 0.0, "fullTextWeight": 0.0},
    )
    assert response.status_code == 422


def test_hybrid_endpoint_rejects_unknown_fusion_strategy(api_client: TestClient) -> None:
    response = api_client.post(
        "/search/hybrid",
        json={"query": "hello", "fusionStrategy": "bm25"},
    )
    assert response.status_code == 422


def test_hybrid_endpoint_returns_503_when_provider_unavailable(
    tmp_path: Any, api_db: AsyncSession
) -> None:
    """Verify the router maps EmbeddingProviderNotAvailableError → 503."""
    from pathlib import Path

    from app.adapters.embeddings import EmbeddingProviderUnavailableError
    from app.services.hybrid_search_service import HybridSearchService

    unavailable = MagicMock()
    unavailable.name = "fake"
    unavailable.embed_texts = AsyncMock(
        side_effect=EmbeddingProviderUnavailableError("down")
    )

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield api_db

    original_init = HybridSearchService.__init__

    def patched_init(
        self: HybridSearchService, db: AsyncSession, **kwargs: Any
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
        storage_path=Path(str(tmp_path)),  # type: ignore[arg-type]
        embedding_provider="fake",
        embedding_model="test-model",
        embedding_dimensions=_DIM,
        _env_file=None,  # type: ignore[call-arg]
    )
    HybridSearchService.__init__ = patched_init  # type: ignore[method-assign]

    try:
        with TestClient(fastapi_app) as c:
            response = c.post("/search/hybrid", json={"query": "hello"})
        assert response.status_code == 503
    finally:
        HybridSearchService.__init__ = original_init  # type: ignore[method-assign]
        fastapi_app.dependency_overrides.clear()


def test_semantic_endpoint_still_works_after_hybrid_added(api_client: TestClient) -> None:
    """Existing semantic endpoint must remain compatible."""
    response = api_client.post("/search/semantic", json={"query": "hello world"})
    # 200 with empty corpus; never 404
    assert response.status_code == 200


def test_hybrid_endpoint_empty_corpus_returns_200(api_client: TestClient) -> None:
    """POST /search/hybrid on empty DB must return 200 with empty results."""
    response = api_client.post(
        "/search/hybrid",
        json={"query": "termination notice period"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "termination notice period"
    assert "results" in data
    assert data["result_count"] == 0
    assert data["results"] == []
    assert "vector_candidate_count" in data
    assert "full_text_candidate_count" in data
    assert "fusion_strategy" in data
    assert "embedding_model" in data


def test_hybrid_endpoint_with_filters_returns_valid_response(api_client: TestClient) -> None:
    response = api_client.post(
        "/search/hybrid",
        json={
            "query": "termination notice period in employment contract",
            "topK": 10,
            "fusionStrategy": "rrf",
            "vectorWeight": 0.65,
            "fullTextWeight": 0.35,
            "filters": {
                "categories": ["contract"],
                "fileTypes": ["pdf"],
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filters_applied"]["categories"] == ["contract"]
    assert data["filters_applied"]["file_types"] == ["pdf"]


def test_hybrid_endpoint_vector_only_weights(api_client: TestClient) -> None:
    response = api_client.post(
        "/search/hybrid",
        json={
            "query": "non-compete clause",
            "vectorWeight": 1.0,
            "fullTextWeight": 0.0,
        },
    )
    assert response.status_code == 200


def test_hybrid_endpoint_full_text_only_weights(api_client: TestClient) -> None:
    response = api_client.post(
        "/search/hybrid",
        json={
            "query": "non-compete clause",
            "vectorWeight": 0.0,
            "fullTextWeight": 1.0,
        },
    )
    assert response.status_code == 200
