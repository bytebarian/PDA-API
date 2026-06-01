"""Unit tests for embedding service."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.adapters.embeddings.base import EmbeddingResult
from app.core.config import Settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.embedding_service import EmbeddingService, NoChunksToEmbedError
from app.services.vector_validation import InvalidEmbeddingVectorError


class _OutOfOrderProvider:
    name = "fake"

    async def embed_texts(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None = None,
        truncate: bool = True,
    ) -> list[EmbeddingResult]:
        del model, dimensions, truncate
        return [
            EmbeddingResult(text_index=1, vector=[2.0] * 1536, model="fake-model", dimensions=1536),
            EmbeddingResult(text_index=0, vector=[1.0] * 1536, model="fake-model", dimensions=1536),
        ][: len(texts)]

    async def healthcheck(self) -> bool:
        return True


class _NanProvider:
    name = "fake"

    async def embed_texts(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None = None,
        truncate: bool = True,
    ) -> list[EmbeddingResult]:
        del model, dimensions, truncate
        return [
            EmbeddingResult(text_index=index, vector=[float("nan")] * 1536, model="fake-model", dimensions=1536)
            for index, _ in enumerate(texts)
        ]

    async def healthcheck(self) -> bool:
        return True


async def test_embedding_service_rejects_documents_with_no_chunks(db_session) -> None:
    document = Document(filename="no-chunks.txt", status="awaiting", extracted_text="hello")
    db_session.add(document)
    await db_session.commit()

    service = EmbeddingService(
        db_session,
        providers={"fake": _OutOfOrderProvider()},
        settings=Settings(embedding_provider="fake", embedding_model="fake-model"),
    )

    with pytest.raises(NoChunksToEmbedError):
        await service.generate_embeddings_for_document(document.id)


async def test_embedding_service_maps_vectors_to_ordered_chunks(db_session) -> None:
    document = Document(filename="ordered.txt", status="awaiting", extracted_text="hello world")
    db_session.add(document)
    await db_session.flush()
    db_session.add(
        DocumentChunk(document_id=document.id, chunk_index=1, content="second")
    )
    db_session.add(
        DocumentChunk(document_id=document.id, chunk_index=0, content="first")
    )
    await db_session.commit()

    service = EmbeddingService(
        db_session,
        providers={"fake": _OutOfOrderProvider()},
        settings=Settings(embedding_provider="fake", embedding_model="fake-model"),
    )
    result = await service.generate_embeddings_for_document(document.id)
    await db_session.commit()

    refreshed = list(
        (
            await db_session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index)
            )
        )
        .scalars()
        .all()
    )

    assert result.embedded_chunk_count == 2
    assert refreshed[0].embedding[0] == 1.0
    assert refreshed[1].embedding[0] == 2.0


async def test_embedding_service_rejects_nan_vectors(db_session) -> None:
    document = Document(filename="nan.txt", status="awaiting", extracted_text="hello world")
    db_session.add(document)
    await db_session.flush()
    db_session.add(DocumentChunk(document_id=document.id, chunk_index=0, content="first"))
    await db_session.commit()

    service = EmbeddingService(
        db_session,
        providers={"fake": _NanProvider()},
        settings=Settings(embedding_provider="fake", embedding_model="fake-model"),
    )

    with pytest.raises(InvalidEmbeddingVectorError, match="finite number"):
        await service.generate_embeddings_for_document(document.id)
