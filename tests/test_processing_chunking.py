"""Integration tests for the chunking stage within the processing orchestrator."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.app_settings import AppSettings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.processing_job import ProcessingJob
from app.services.processing_orchestrator import process_job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _load_chunks(db: AsyncSession, document: Document) -> list[DocumentChunk]:
    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index)
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Core persistence tests
# ---------------------------------------------------------------------------


async def test_chunking_persists_chunks_into_db(db_session: AsyncSession) -> None:
    """process_job should insert DocumentChunk rows for the document."""
    document = Document(
        filename="report.txt",
        status="awaiting",
        extracted_text="Hello world. This is a short document.",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    chunks = await _load_chunks(db_session, document)
    assert len(chunks) >= 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content != ""
    assert chunks[0].source_start_offset is not None
    assert chunks[0].source_end_offset is not None


async def test_chunking_updates_document_chunk_count(db_session: AsyncSession) -> None:
    """document.chunk_count must equal the number of persisted chunks."""
    document = Document(
        filename="count.txt",
        status="awaiting",
        extracted_text="Sentence one. Sentence two. Sentence three.",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    chunks = await _load_chunks(db_session, document)
    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    assert refreshed.chunk_count == len(chunks)


async def test_chunking_replaces_stale_chunks_on_reprocess(
    db_session: AsyncSession,
) -> None:
    """Reprocessing must delete old chunks and replace with fresh ones."""
    document = Document(
        filename="reprocess.txt",
        status="awaiting",
        extracted_text="Initial content for chunking.",
    )
    db_session.add(document)
    await db_session.flush()

    # Pre-populate a stale chunk.
    stale = DocumentChunk(
        document_id=document.id,
        chunk_index=0,
        content="stale chunk content",
    )
    db_session.add(stale)

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    chunks = await _load_chunks(db_session, document)
    contents = [c.content for c in chunks]
    assert "stale chunk content" not in contents
    assert any("Initial content" in c for c in contents)


# ---------------------------------------------------------------------------
# Stage history tests
# ---------------------------------------------------------------------------


async def test_chunking_stage_history_has_processing_and_completed(
    db_session: AsyncSession,
) -> None:
    """Stage history must record chunking processing → completed entries."""
    document = Document(
        filename="history.txt",
        status="awaiting",
        extracted_text="Some text to chunk.",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    processed = await process_job(db_session, job.id)

    history = processed.stage_history_jsonb
    chunking_entries = [e for e in history if e["stage"] == "chunking"]
    statuses = [e["status"] for e in chunking_entries]
    assert "processing" in statuses
    assert "completed" in statuses


async def test_chunking_stage_history_completed_entry_has_chunk_count(
    db_session: AsyncSession,
) -> None:
    """The chunking completed entry must carry the chunk_count in details."""
    document = Document(
        filename="count_history.txt",
        status="awaiting",
        extracted_text="A sentence. Another sentence. Third sentence.",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    processed = await process_job(db_session, job.id)

    completed_entry = next(
        (
            e
            for e in processed.stage_history_jsonb
            if e["stage"] == "chunking" and e["status"] == "completed"
        ),
        None,
    )
    assert completed_entry is not None
    assert "details" in completed_entry
    assert "chunk_count" in completed_entry["details"]
    assert isinstance(completed_entry["details"]["chunk_count"], int)
    assert completed_entry["details"]["chunk_count"] >= 1


# ---------------------------------------------------------------------------
# Empty / whitespace text → chunking stage failure
# ---------------------------------------------------------------------------


async def test_chunking_fails_when_extracted_text_is_none(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document with no extracted text must fail the chunking stage."""

    # The processing pipeline runs text extraction before chunking; for this test we
    # intentionally keep extracted_text empty so chunking fails.
    from app.domain.status import ProcessingJobStage
    from app.services import processing_orchestrator

    async def noop_text_extraction(
        _db: AsyncSession, _doc: Document, job: ProcessingJob
    ) -> None:
        processing_orchestrator._append_stage_history(
            job, stage=ProcessingJobStage.text_extraction, status="processing"
        )
        processing_orchestrator._append_stage_history(
            job, stage=ProcessingJobStage.text_extraction, status="completed"
        )

    monkeypatch.setattr(processing_orchestrator, "_run_text_extraction_stage", noop_text_extraction)

    document = Document(
        filename="empty.txt",
        status="awaiting",
        extracted_text=None,
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    from app.services.chunking_service import ChunkingEmptyTextError

    with pytest.raises(ChunkingEmptyTextError, match="No extractable text"):
        await process_job(db_session, job.id)

    refreshed_job = await db_session.get(ProcessingJob, job.id)
    refreshed_doc = await db_session.get(Document, document.id)
    assert refreshed_job is not None
    assert refreshed_doc is not None
    assert refreshed_job.status == "failed"
    assert refreshed_doc.status == "failed"


async def test_chunking_fails_when_extracted_text_is_whitespace(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document with whitespace-only extracted text must fail the chunking stage."""

    # The processing pipeline runs text extraction before chunking; for this test we
    # intentionally keep extracted_text whitespace-only so chunking fails.
    from app.domain.status import ProcessingJobStage
    from app.services import processing_orchestrator

    async def noop_text_extraction(
        _db: AsyncSession, _doc: Document, job: ProcessingJob
    ) -> None:
        processing_orchestrator._append_stage_history(
            job, stage=ProcessingJobStage.text_extraction, status="processing"
        )
        processing_orchestrator._append_stage_history(
            job, stage=ProcessingJobStage.text_extraction, status="completed"
        )

    monkeypatch.setattr(processing_orchestrator, "_run_text_extraction_stage", noop_text_extraction)

    document = Document(
        filename="whitespace.txt",
        status="awaiting",
        extracted_text="   \n\t  ",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    from app.services.chunking_service import ChunkingEmptyTextError

    with pytest.raises(ChunkingEmptyTextError, match="No extractable text"):
        await process_job(db_session, job.id)

    refreshed_job = await db_session.get(ProcessingJob, job.id)
    assert refreshed_job is not None
    assert refreshed_job.status == "failed"
    assert refreshed_job.error_message is not None
    assert "No extractable text" in refreshed_job.error_message


# ---------------------------------------------------------------------------
# Settings-driven chunking
# ---------------------------------------------------------------------------


async def test_chunking_uses_persisted_settings(db_session: AsyncSession) -> None:
    """ChunkingService should load chunk_size and chunk_overlap from AppSettings."""
    # Insert settings with a very small chunk_size to force multiple chunks.
    settings = AppSettings(chunk_size=20, chunk_overlap=0)
    db_session.add(settings)

    text = "word " * 20  # 100 chars – will need several 20-char chunks
    document = Document(
        filename="settings.txt",
        status="awaiting",
        extracted_text=text,
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    chunks = await _load_chunks(db_session, document)
    # With chunk_size=20 on 100-char text we expect multiple chunks.
    assert len(chunks) > 1


async def test_chunking_defaults_when_no_app_settings_row(
    db_session: AsyncSession,
) -> None:
    """Chunking must use defaults when no AppSettings row exists."""
    short_text = "Short text."
    document = Document(
        filename="defaults.txt",
        status="awaiting",
        extracted_text=short_text,
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    chunks = await _load_chunks(db_session, document)
    # Short text < default chunk_size (1000) → exactly one chunk.
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Offset validity
# ---------------------------------------------------------------------------


async def test_chunk_offsets_reconstruct_content(db_session: AsyncSession) -> None:
    """source_start_offset/source_end_offset must reproduce each chunk's content."""
    raw_text = (
        "First paragraph of content.\n\n"
        "Second paragraph with more words.\n\n"
        "Third paragraph concludes the document."
    )
    document = Document(
        filename="offsets.txt",
        status="awaiting",
        extracted_text=raw_text,
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    # Normalise the way the service does.
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    chunks = await _load_chunks(db_session, document)
    for chunk in chunks:
        assert chunk.source_start_offset is not None
        assert chunk.source_end_offset is not None
        sliced = normalized[chunk.source_start_offset : chunk.source_end_offset].strip()
        assert sliced == chunk.content
