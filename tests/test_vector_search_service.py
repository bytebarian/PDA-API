from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.vector_search_service import VectorSearchService
from app.services.vector_validation import InvalidEmbeddingVectorError


async def _seed_chunks(db_session: AsyncSession) -> tuple[uuid.UUID, uuid.UUID]:
    doc_a = Document(filename="energy-contract.pdf", status="awaiting")
    doc_b = Document(filename="insurance-policy.pdf", status="awaiting")
    db_session.add_all([doc_a, doc_b])
    await db_session.flush()

    db_session.add_all(
        [
            DocumentChunk(
                document_id=doc_a.id,
                chunk_index=0,
                content="energy contract termination period",
                embedding=[1.0, 0.0, 0.0],
                embedding_model="nomic-embed-text",
                embedding_provider="fake",
                embedding_dimension=3,
                metadata_jsonb={"section": "termination"},
            ),
            DocumentChunk(
                document_id=doc_a.id,
                chunk_index=1,
                content="payment schedule and invoice address",
                embedding=[0.0, 1.0, 0.0],
                embedding_model="nomic-embed-text",
                embedding_provider="fake",
                embedding_dimension=3,
                metadata_jsonb={"section": "billing"},
            ),
            DocumentChunk(
                document_id=doc_b.id,
                chunk_index=0,
                content="insurance policy coverage",
                embedding=[0.0, 0.0, 1.0],
                embedding_model="nomic-embed-text",
                embedding_provider="fake",
                embedding_dimension=3,
                metadata_jsonb={"section": "coverage"},
            ),
            DocumentChunk(
                document_id=doc_b.id,
                chunk_index=1,
                content="no embedding",
                embedding=None,
            ),
        ]
    )
    await db_session.commit()
    return doc_a.id, doc_b.id


async def test_vector_search_returns_expected_nearest_chunk(db_session: AsyncSession) -> None:
    doc_a_id, _ = await _seed_chunks(db_session)
    service = VectorSearchService(db_session, settings=Settings(embedding_dimensions=3))

    results = await service.search_similar_chunks([0.9, 0.1, 0.0], limit=5)

    assert results
    assert results[0].document_id == doc_a_id
    assert results[0].chunk_index == 0
    assert results[0].similarity > results[1].similarity


async def test_vector_search_filters_by_document_ids(db_session: AsyncSession) -> None:
    doc_a_id, _ = await _seed_chunks(db_session)
    service = VectorSearchService(db_session, settings=Settings(embedding_dimensions=3))

    results = await service.search_similar_chunks(
        [0.9, 0.1, 0.0],
        limit=5,
        document_ids=[doc_a_id],
    )

    assert results
    assert {item.document_id for item in results} == {doc_a_id}


async def test_vector_search_ignores_null_embeddings_and_supports_empty_results(
    db_session: AsyncSession,
) -> None:
    await _seed_chunks(db_session)
    service = VectorSearchService(db_session, settings=Settings(embedding_dimensions=3))

    no_results = await service.search_similar_chunks(
        [0.0, 0.0, 1.0],
        min_similarity=1.01,
    )

    assert no_results == []


async def test_vector_search_supports_metadata_filter(db_session: AsyncSession) -> None:
    await _seed_chunks(db_session)
    service = VectorSearchService(db_session, settings=Settings(embedding_dimensions=3))

    results = await service.search_similar_chunks(
        [0.9, 0.1, 0.0],
        metadata_filter={"section": "billing"},
    )

    assert len(results) == 1
    assert results[0].metadata == {"section": "billing"}


async def test_vector_search_rejects_query_dimension_mismatch(db_session: AsyncSession) -> None:
    await _seed_chunks(db_session)
    service = VectorSearchService(db_session, settings=Settings(embedding_dimensions=3))

    with pytest.raises(InvalidEmbeddingVectorError, match="exactly 3"):
        await service.search_similar_chunks([1.0, 0.0])
