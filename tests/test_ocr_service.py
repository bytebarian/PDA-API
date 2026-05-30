"""Tests for OCR persistence and orchestrator integration."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.ocr import (
    FakeOCRProvider,
    OCREmptyResultError,
    OCRUnreadableImageError,
)
from app.core.config import get_settings
from app.models.document import Document
from app.models.processing_job import ProcessingJob
from app.services.ocr_service import OCRService
from app.services.processing_orchestrator import process_job


@pytest.fixture
def ocr_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    storage_path = tmp_path / "storage"
    storage_path.mkdir()
    monkeypatch.setenv("PDA_STORAGE_PATH", str(storage_path))
    get_settings.cache_clear()
    return storage_path


async def test_ocr_service_persists_extracted_text_metadata_and_stage_history(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    ocr_storage: Path,
) -> None:
    image_path = ocr_storage / "receipt.png"
    image_path.write_bytes(b"fake image bytes")

    document = Document(
        filename="receipt.png",
        mime_type="image/png",
        status="awaiting",
        path=str(image_path),
        metadata_jsonb={"existing": {"keep": True}},
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="ocr")
    db_session.add(job)
    await db_session.flush()

    monkeypatch.setattr(
        "app.services.ocr_service.get_ocr_provider",
        lambda **_: FakeOCRProvider(
            text="Detected receipt text",
            confidence=91.5,
            warnings=["low contrast"],
        ),
    )

    result = await OCRService(db_session).extract_text_for_document(document.id, job_id=job.id)

    assert result.extracted_text == "Detected receipt text"
    assert result.char_count == len("Detected receipt text")
    assert result.confidence == pytest.approx(91.5)

    assert document.extracted_text == "Detected receipt text"
    assert document.metadata_jsonb is not None
    assert document.metadata_jsonb["existing"] == {"keep": True}
    assert document.metadata_jsonb["ocr"]["provider"] == "fake"
    assert document.metadata_jsonb["ocr"]["languages"] == ["eng"]
    assert document.metadata_jsonb["ocr"]["char_count"] == len("Detected receipt text")
    assert document.metadata_jsonb["ocr"]["confidence"] == pytest.approx(91.5)

    assert [(entry["stage"], entry["status"]) for entry in job.stage_history_jsonb] == [
        ("ocr", "processing"),
        ("ocr", "completed"),
    ]


async def test_ocr_service_empty_output_fails_cleanly(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    ocr_storage: Path,
) -> None:
    image_path = ocr_storage / "empty.png"
    image_path.write_bytes(b"fake image bytes")

    document = Document(
        filename="empty.png",
        mime_type="image/png",
        status="awaiting",
        path=str(image_path),
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="ocr")
    db_session.add(job)
    await db_session.flush()

    monkeypatch.setattr(
        "app.services.ocr_service.get_ocr_provider",
        lambda **_: FakeOCRProvider(text="   "),
    )

    with pytest.raises(OCREmptyResultError, match="empty text"):
        await OCRService(db_session).extract_text_for_document(document.id, job_id=job.id)

    assert document.extracted_text is None
    assert job.stage_history_jsonb[-1]["stage"] == "ocr"
    assert job.stage_history_jsonb[-1]["status"] == "failed"


async def test_process_job_runs_ocr_for_images_and_overwrites_stale_text(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    ocr_storage: Path,
) -> None:
    image_path = ocr_storage / "invoice.png"
    image_path.write_bytes(b"fake image bytes")

    document = Document(
        filename="invoice.png",
        mime_type="image/png",
        status="awaiting",
        path=str(image_path),
        extracted_text="stale text",
        metadata_jsonb={"ocr": {"provider": "old"}},
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    monkeypatch.setattr(
        "app.services.ocr_service.get_ocr_provider",
        lambda **_: FakeOCRProvider(text="Fresh OCR text", confidence=88.0),
    )

    processed = await process_job(db_session, job.id)

    refreshed_document = await db_session.get(Document, document.id)
    assert refreshed_document is not None
    assert refreshed_document.status == "ready"
    assert refreshed_document.extracted_text == "Fresh OCR text"
    assert refreshed_document.metadata_jsonb is not None
    assert refreshed_document.metadata_jsonb["ocr"]["provider"] == "fake"
    assert refreshed_document.metadata_jsonb["ocr"]["char_count"] == len("Fresh OCR text")

    ocr_completed = next(
        entry
        for entry in processed.stage_history_jsonb
        if entry["stage"] == "ocr" and entry["status"] == "completed"
    )
    assert ocr_completed["details"]["provider"] == "fake"
    assert ocr_completed["details"]["confidence"] == pytest.approx(88.0)


async def test_process_job_ocr_failure_marks_document_and_job_failed(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    ocr_storage: Path,
) -> None:
    image_path = ocr_storage / "bad.png"
    image_path.write_bytes(b"fake image bytes")

    document = Document(
        filename="bad.png",
        mime_type="image/png",
        status="awaiting",
        path=str(image_path),
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    monkeypatch.setattr(
        "app.services.ocr_service.get_ocr_provider",
        lambda **_: FakeOCRProvider(
            error=OCRUnreadableImageError("corrupt image fixture")
        ),
    )

    with pytest.raises(OCRUnreadableImageError, match="corrupt image fixture"):
        await process_job(db_session, job.id)

    refreshed_job = await db_session.get(ProcessingJob, job.id)
    refreshed_document = await db_session.get(Document, document.id)
    assert refreshed_job is not None
    assert refreshed_document is not None

    assert refreshed_job.status == "failed"
    assert refreshed_document.status == "failed"
    assert refreshed_job.error_details_jsonb == {
        "stage": "ocr",
        "error_type": "OCRUnreadableImageError",
        "message": "corrupt image fixture",
    }
    failed_entries = [
        entry
        for entry in refreshed_job.stage_history_jsonb
        if entry["stage"] == "ocr" and entry["status"] == "failed"
    ]
    assert len(failed_entries) == 1
