"""Integration tests for embeddings stage in processing orchestration."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.processing_job import ProcessingJob
from app.services import embedding_service
from app.services.processing_orchestrator import process_job


def _configure_fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDA_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("PDA_EMBEDDING_MODEL", "fake-embed-model")
    monkeypatch.setenv("PDA_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("PDA_EMBEDDING_BATCH_SIZE", "2")
    get_settings.cache_clear()


async def _load_chunks(db: AsyncSession, document_id) -> list[DocumentChunk]:
    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    return list(result.scalars().all())


async def test_embeddings_stage_persists_vectors_and_updates_metadata(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_fake_provider(monkeypatch)
    document = Document(
        filename="embed.txt",
        status="awaiting",
        extracted_text="Hello world. This text should produce chunks for embeddings.",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    processed = await process_job(db_session, job.id)

    chunks = await _load_chunks(db_session, document.id)
    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    assert chunks
    assert all(chunk.embedding is not None for chunk in chunks)
    assert all(len(chunk.embedding or []) == 1536 for chunk in chunks)
    assert refreshed.embedding_model == "fake-embed-model"
    assert refreshed.chunk_count == len(chunks)
    assert refreshed.last_indexed_at is not None

    embedding_completed = next(
        entry
        for entry in processed.stage_history_jsonb
        if entry["stage"] == "embedding" and entry["status"] == "completed"
    )
    assert embedding_completed["details"]["embedded_chunk_count"] == len(chunks)
    assert embedding_completed["details"]["provider"] == "fake"
    assert embedding_completed["details"]["model"] == "fake-embed-model"
    assert embedding_completed["details"]["dimensions"] == 1536
    get_settings.cache_clear()


async def test_reembedding_overwrites_existing_vectors(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_fake_provider(monkeypatch)
    document = Document(
        filename="reembed.txt",
        status="awaiting",
        extracted_text="One paragraph. Two paragraph. Three paragraph.",
    )
    db_session.add(document)
    await db_session.flush()
    chunks = [
        DocumentChunk(document_id=document.id, chunk_index=0, content="first"),
        DocumentChunk(document_id=document.id, chunk_index=1, content="second"),
    ]
    db_session.add_all(chunks)
    await db_session.commit()

    service = embedding_service.EmbeddingService(db_session)
    await service.generate_embeddings_for_document(document.id, model="model-a")
    await db_session.commit()
    first_pass = [chunk.embedding for chunk in await _load_chunks(db_session, document.id)]

    await service.generate_embeddings_for_document(document.id, model="model-b")
    await db_session.commit()
    second_pass = [chunk.embedding for chunk in await _load_chunks(db_session, document.id)]

    assert first_pass != second_pass
    get_settings.cache_clear()


class _FailingProvider:
    name = "fake"

    async def embed_texts(self, texts, *, model, dimensions=None, truncate=True):
        del texts, model, dimensions, truncate
        raise RuntimeError("embedding provider down")

    async def healthcheck(self) -> bool:
        return False


async def test_embeddings_failure_marks_job_and_document_failed(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_fake_provider(monkeypatch)
    document = Document(
        filename="failed-embed.txt",
        status="awaiting",
        extracted_text="Failure case with enough text to chunk.",
    )
    db_session.add(document)
    await db_session.flush()
    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    monkeypatch.setattr(
        embedding_service,
        "build_embedding_providers",
        lambda settings: {"fake": _FailingProvider()},
    )

    with pytest.raises(RuntimeError, match="embedding provider down"):
        await process_job(db_session, job.id)

    refreshed_job = await db_session.get(ProcessingJob, job.id)
    refreshed_document = await db_session.get(Document, document.id)
    assert refreshed_job is not None
    assert refreshed_document is not None
    assert refreshed_job.status == "failed"
    assert refreshed_document.status == "failed"
    assert refreshed_job.error_details_jsonb == {
        "stage": "embedding",
        "error_type": "RuntimeError",
        "message": "embedding provider down",
    }
    get_settings.cache_clear()
