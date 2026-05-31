"""Integration tests for the normalize_text stage within the processing orchestrator.

These tests verify that:
- Normalization runs after text extraction and before chunking.
- Normalized text is persisted in Document.extracted_text.
- Normalization metadata is stored in Document.metadata_jsonb["normalization"].
- Stage history records normalize_text processing → completed entries.
- Normalization failure marks the job/document as failed.
- Chunking uses the normalized text, not the raw extraction output.
- Reprocessing reruns normalization and replaces stale artifacts.
- OCR output is normalized before chunking.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.processing_job import ProcessingJob
from app.services import processing_orchestrator
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


def _patch_noop_text_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.domain.status import ProcessingJobStage

    async def noop(
        _db: AsyncSession, _doc: Document, job: ProcessingJob
    ) -> None:
        processing_orchestrator._append_stage_history(
            job, stage=ProcessingJobStage.text_extraction, status="processing"
        )
        processing_orchestrator._append_stage_history(
            job, stage=ProcessingJobStage.text_extraction, status="completed"
        )

    monkeypatch.setattr(processing_orchestrator, "_run_text_extraction_stage", noop)


# ---------------------------------------------------------------------------
# Normalization persists normalized text
# ---------------------------------------------------------------------------


async def test_normalization_persists_normalized_text(db_session: AsyncSession) -> None:
    """Extracted text must be normalized and stored before chunking."""
    raw = "  Hello\r\n\r\nWorld.   "
    document = Document(filename="norm.txt", status="awaiting", extracted_text=raw)
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    # Leading/trailing whitespace trimmed, CRLF converted to LF.
    assert refreshed.extracted_text == "Hello\n\nWorld."


async def test_normalization_stores_metadata_in_document(db_session: AsyncSession) -> None:
    """Document.metadata_jsonb must contain a 'normalization' entry after processing."""
    document = Document(
        filename="meta.txt",
        status="awaiting",
        extracted_text="Some text\r\nwith CRLF.",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    assert refreshed.metadata_jsonb is not None
    norm_meta = refreshed.metadata_jsonb.get("normalization")
    assert norm_meta is not None
    assert norm_meta["provider"] == "pda-local-normalizer"
    assert norm_meta["ruleSetVersion"] == "pda-normalization-v1"
    assert isinstance(norm_meta["inputCharacterCount"], int)
    assert isinstance(norm_meta["outputCharacterCount"], int)
    assert isinstance(norm_meta["changed"], bool)
    assert isinstance(norm_meta["warnings"], list)


async def test_normalization_metadata_changed_flag_true_when_text_changed(
    db_session: AsyncSession,
) -> None:
    """changed flag must be True when normalization modifies the text."""
    document = Document(
        filename="changed.txt",
        status="awaiting",
        extracted_text="messy   \r\n  text",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    assert refreshed.metadata_jsonb is not None
    assert refreshed.metadata_jsonb["normalization"]["changed"] is True


async def test_normalization_metadata_changed_flag_false_for_clean_text(
    db_session: AsyncSession,
) -> None:
    """changed flag must be False when the text is already clean."""
    clean = "Already clean text.\n\nSecond paragraph."
    document = Document(filename="clean.txt", status="awaiting", extracted_text=clean)
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    assert refreshed.metadata_jsonb is not None
    assert refreshed.metadata_jsonb["normalization"]["changed"] is False


# ---------------------------------------------------------------------------
# Chunking uses normalized text
# ---------------------------------------------------------------------------


async def test_chunking_uses_normalized_text(db_session: AsyncSession) -> None:
    """Chunk content must come from normalized text, not raw extracted text."""
    # Raw text has control chars and BOM which normalization will strip.
    raw = "\ufeffHello\x00World. Second sentence."
    document = Document(filename="chunk_norm.txt", status="awaiting", extracted_text=raw)
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    chunks = await _load_chunks(db_session, document)
    assert chunks
    combined = " ".join(c.content for c in chunks)
    # BOM and null byte must not appear in any chunk.
    assert "\ufeff" not in combined
    assert "\x00" not in combined
    # Meaningful content preserved.
    assert "Hello" in combined
    assert "World" in combined


async def test_chunks_are_from_normalized_not_raw_crlf_text(
    db_session: AsyncSession,
) -> None:
    """Chunks must not contain CRLF sequences after normalization."""
    raw = "Sentence one.\r\nSentence two.\r\nSentence three."
    document = Document(filename="crlf_chunks.txt", status="awaiting", extracted_text=raw)
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    chunks = await _load_chunks(db_session, document)
    for chunk in chunks:
        assert "\r" not in chunk.content


# ---------------------------------------------------------------------------
# Stage history
# ---------------------------------------------------------------------------


async def test_normalization_stage_history_has_processing_and_completed(
    db_session: AsyncSession,
) -> None:
    """Stage history must record normalize_text processing → completed entries."""
    document = Document(
        filename="stage_history.txt",
        status="awaiting",
        extracted_text="Stage history test text.",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    processed = await process_job(db_session, job.id)

    history = processed.stage_history_jsonb
    norm_entries = [e for e in history if e["stage"] == "normalize_text"]
    statuses = [e["status"] for e in norm_entries]
    assert "processing" in statuses
    assert "completed" in statuses


async def test_normalization_stage_history_completed_entry_has_details(
    db_session: AsyncSession,
) -> None:
    """The normalize_text completed entry must carry provenance details."""
    document = Document(
        filename="details.txt",
        status="awaiting",
        extracted_text="Details test document.",
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
            if e["stage"] == "normalize_text" and e["status"] == "completed"
        ),
        None,
    )
    assert completed_entry is not None
    details = completed_entry.get("details", {})
    assert "input_char_count" in details
    assert "output_char_count" in details
    assert "changed" in details
    assert "rule_set_version" in details
    assert "warning_count" in details


async def test_normalize_text_stage_runs_between_extraction_and_chunking(
    db_session: AsyncSession,
) -> None:
    """normalize_text must appear after text_extraction and before chunking in history."""
    document = Document(
        filename="order.txt",
        status="awaiting",
        extracted_text="Pipeline order test.",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    processed = await process_job(db_session, job.id)

    stages = [e["stage"] for e in processed.stage_history_jsonb]
    norm_idx = next(i for i, s in enumerate(stages) if s == "normalize_text")
    ext_idx = next(i for i, s in enumerate(stages) if s == "text_extraction")
    chunk_idx = next(i for i, s in enumerate(stages) if s == "chunking")
    assert ext_idx < norm_idx < chunk_idx


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


async def test_normalization_failure_marks_job_and_document_failed(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normalization stage exception must fail the job and document."""
    _patch_noop_text_extraction(monkeypatch)

    document = Document(
        filename="fail_norm.txt",
        status="awaiting",
        extracted_text="Some text.",
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    async def fail_normalization(
        _db: AsyncSession, _doc: Document, _job: ProcessingJob
    ) -> None:
        raise RuntimeError("normalization exploded")

    monkeypatch.setattr(
        processing_orchestrator, "_run_normalize_text_stage", fail_normalization
    )

    with pytest.raises(RuntimeError, match="normalization exploded"):
        await process_job(db_session, job.id)

    refreshed_job = await db_session.get(ProcessingJob, job.id)
    refreshed_doc = await db_session.get(Document, document.id)
    assert refreshed_job is not None
    assert refreshed_doc is not None
    assert refreshed_job.status == "failed"
    assert refreshed_doc.status == "failed"
    assert refreshed_job.error_details_jsonb == {
        "stage": "normalize_text",
        "error_type": "RuntimeError",
        "message": "normalization exploded",
    }


async def test_normalization_failure_on_none_extracted_text(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent extracted text must fail the normalize_text stage (fail_on_empty=True)."""
    _patch_noop_text_extraction(monkeypatch)

    document = Document(
        filename="none_text.txt",
        status="awaiting",
        extracted_text=None,
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    from app.services.text_normalization import TextNormalizationEmptyInputError

    with pytest.raises(TextNormalizationEmptyInputError):
        await process_job(db_session, job.id)

    refreshed_job = await db_session.get(ProcessingJob, job.id)
    assert refreshed_job is not None
    assert refreshed_job.status == "failed"
    assert any(
        e["stage"] == "normalize_text" and e["status"] == "failed"
        for e in refreshed_job.stage_history_jsonb
    )


async def test_normalization_failure_on_whitespace_only_text(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only text that normalizes to empty must fail the normalize_text stage."""
    _patch_noop_text_extraction(monkeypatch)

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

    from app.services.text_normalization import TextNormalizationEmptyOutputError

    with pytest.raises(TextNormalizationEmptyOutputError):
        await process_job(db_session, job.id)

    refreshed_job = await db_session.get(ProcessingJob, job.id)
    assert refreshed_job is not None
    assert refreshed_job.status == "failed"


# ---------------------------------------------------------------------------
# Reprocessing
# ---------------------------------------------------------------------------


async def test_reprocessing_replaces_normalized_text(db_session: AsyncSession) -> None:
    """Reprocessing must run normalization again and replace the previous output."""
    first_raw = "First   content.\r\n"
    document = Document(
        filename="reprocess_norm.txt",
        status="awaiting",
        extracted_text=first_raw,
    )
    db_session.add(document)
    await db_session.flush()

    job1 = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job1)
    await db_session.commit()

    await process_job(db_session, job1.id)

    # Refresh and check first normalization result.
    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    first_normalized = refreshed.extracted_text
    assert first_normalized == "First content."

    # Simulate reprocess: update extracted_text and create a new awaiting job.
    refreshed.extracted_text = "Second   content.\r\n"
    refreshed.status = "awaiting"
    job2 = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job2)
    await db_session.commit()

    await process_job(db_session, job2.id)

    refreshed2 = await db_session.get(Document, document.id)
    assert refreshed2 is not None
    assert refreshed2.extracted_text == "Second content."
    # Metadata updated to reflect second run.
    assert refreshed2.metadata_jsonb is not None
    assert "normalization" in refreshed2.metadata_jsonb


async def test_reprocessing_replaces_stale_chunks_based_on_normalized_text(
    db_session: AsyncSession,
) -> None:
    """Reprocessing must replace stale chunks derived from old normalization output."""
    document = Document(
        filename="reprocess_chunks.txt",
        status="awaiting",
        extracted_text="Initial content for normalization and chunking.",
    )
    db_session.add(document)
    await db_session.flush()

    stale = DocumentChunk(
        document_id=document.id,
        chunk_index=0,
        content="stale chunk from previous run",
    )
    db_session.add(stale)

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    await process_job(db_session, job.id)

    chunks = await _load_chunks(db_session, document)
    contents = [c.content for c in chunks]
    assert "stale chunk from previous run" not in contents
    assert any("Initial content" in c for c in contents)


# ---------------------------------------------------------------------------
# OCR text normalization
# ---------------------------------------------------------------------------


async def test_ocr_output_is_normalized_before_chunking(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OCR output stored in extracted_text must be normalized before chunking."""
    # Simulate OCR result: BOM + CRLF + null byte + meaningful content.
    ocr_output = "\ufeffScanned\r\nDocument\x00 text. Second line."

    # Pre-populate extracted_text as if the OCR stage already ran.
    document = Document(
        filename="scan.png",
        mime_type="image/png",
        status="awaiting",
        extracted_text=ocr_output,
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    # Patch OCR and text_extraction to noops so only normalization + chunking run.
    from app.domain.status import ProcessingJobStage

    async def noop_ocr(
        _db: AsyncSession, _doc: Document, _job: ProcessingJob
    ) -> None:
        pass

    async def noop_text_extraction(
        _db: AsyncSession, _doc: Document, _job: ProcessingJob
    ) -> None:
        processing_orchestrator._append_stage_history(
            _job, stage=ProcessingJobStage.text_extraction, status="processing"
        )
        processing_orchestrator._append_stage_history(
            _job, stage=ProcessingJobStage.text_extraction, status="completed"
        )

    monkeypatch.setattr(processing_orchestrator, "_run_ocr_stage", noop_ocr)
    monkeypatch.setattr(
        processing_orchestrator, "_run_text_extraction_stage", noop_text_extraction
    )

    await process_job(db_session, job.id)

    refreshed = await db_session.get(Document, document.id)
    assert refreshed is not None
    normalized = refreshed.extracted_text or ""

    # Normalization artifacts cleaned.
    assert "\ufeff" not in normalized
    assert "\r" not in normalized
    assert "\x00" not in normalized
    # Meaningful content preserved.
    assert "Scanned" in normalized
    assert "Document" in normalized

    chunks = await _load_chunks(db_session, document)
    assert chunks
    for chunk in chunks:
        assert "\ufeff" not in chunk.content
        assert "\r" not in chunk.content
        assert "\x00" not in chunk.content
