"""Tests for OCR persistence and orchestrator integration."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.ocr import (
    FakeOCRProvider,
    OCREmptyResultError,
    OCRPageResult,
    OCRResult,
    OCRUnreadableImageError,
)
from app.core.config import get_settings
from app.models.document import Document
from app.models.processing_job import ProcessingJob
from app.services.ocr_service import OCRService
from app.services.processing_orchestrator import process_job
from app.services.text_extraction import ExtractedTextResult


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


async def test_process_job_falls_back_to_ocr_when_pdf_text_is_empty(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    ocr_storage: Path,
) -> None:
    monkeypatch.setenv("PDA_SUMMARIZATION_PROVIDER", "mock")
    get_settings.cache_clear()
    pdf_path = ocr_storage / "scanned.pdf"
    pdf_path.write_bytes(b"fake scanned pdf")

    document = Document(
        filename="scanned.pdf",
        mime_type="application/pdf",
        status="awaiting",
        path=str(pdf_path),
    )
    db_session.add(document)
    await db_session.flush()

    job = ProcessingJob(document_id=document.id, status="awaiting", stage="queued")
    db_session.add(job)
    await db_session.commit()

    async def empty_pdf_extraction(*args: object, **kwargs: object) -> ExtractedTextResult:
        del args, kwargs
        return ExtractedTextResult(text="", metadata={"extractor": "PdfAdapter"})

    rendered_directories: list[Path] = []

    class FakePDFRenderer:
        async def render_pages(
            self,
            pdf_path: Path,
            output_directory: Path,
            *,
            dpi: int,
        ) -> list[Path]:
            assert pdf_path.name == "scanned.pdf"
            assert dpi == 300
            rendered_directories.append(output_directory)
            page_paths = [
                output_directory / "page-000001.png",
                output_directory / "page-000002.png",
            ]
            for page_path in page_paths:
                page_path.write_bytes(b"fake page")
            return page_paths

    class PageAwareOCRProvider:
        name = "fake"

        async def extract_text(
            self,
            file_path: Path,
            *,
            mime_type: str,
            languages: list[str] | None = None,
            timeout_seconds: int | None = None,
        ) -> OCRResult:
            del timeout_seconds
            page_number = int(file_path.stem.rsplit("-", 1)[1])
            return OCRResult(
                provider=self.name,
                engine_version="fake-1",
                languages=list(languages or ["eng"]),
                pages=[
                    OCRPageResult(
                        page_number=1,
                        text=f"Scanned page {page_number}",
                        confidence=90.0 + page_number,
                    )
                ],
                metadata={"mime_type": mime_type},
            )

        async def healthcheck(self) -> bool:
            return True

    monkeypatch.setattr(
        "app.services.text_extraction.extract_text_from_file",
        empty_pdf_extraction,
    )
    monkeypatch.setattr(
        "app.services.ocr_service.get_pdf_renderer",
        lambda: FakePDFRenderer(),
    )
    monkeypatch.setattr(
        "app.services.ocr_service.get_ocr_provider",
        lambda **_: PageAwareOCRProvider(),
    )

    processed = await process_job(db_session, job.id)

    refreshed_document = await db_session.get(Document, document.id)
    assert refreshed_document is not None
    assert refreshed_document.status == "ready"
    assert refreshed_document.extracted_text == "Scanned page 1\n\nScanned page 2"
    assert refreshed_document.metadata_jsonb is not None
    assert refreshed_document.metadata_jsonb["ocr"]["extraction_method"] == "pdf_ocr"
    assert refreshed_document.metadata_jsonb["ocr"]["page_count"] == 2
    assert [
        page["page_number"] for page in refreshed_document.metadata_jsonb["ocr"]["pages"]
    ] == [1, 2]

    extraction_entry = next(
        entry
        for entry in processed.stage_history_jsonb
        if entry["stage"] == "text_extraction" and entry["status"] == "completed"
    )
    assert extraction_entry["details"]["extraction_method"] == "pdf_ocr"
    assert len(rendered_directories) == 1
    assert not rendered_directories[0].exists()


async def test_pdf_ocr_fallback_rejects_empty_ocr_output(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    ocr_storage: Path,
) -> None:
    pdf_path = ocr_storage / "blank-scan.pdf"
    pdf_path.write_bytes(b"fake blank pdf")
    document = Document(
        filename="blank-scan.pdf",
        mime_type="application/pdf",
        status="awaiting",
        path=str(pdf_path),
    )
    db_session.add(document)
    await db_session.flush()

    class FakePDFRenderer:
        async def render_pages(
            self,
            pdf_path: Path,
            output_directory: Path,
            *,
            dpi: int,
        ) -> list[Path]:
            del pdf_path, dpi
            page_path = output_directory / "page-000001.png"
            page_path.write_bytes(b"fake page")
            return [page_path]

    monkeypatch.setattr(
        "app.services.ocr_service.get_pdf_renderer",
        lambda: FakePDFRenderer(),
    )
    monkeypatch.setattr(
        "app.services.ocr_service.get_ocr_provider",
        lambda **_: FakeOCRProvider(text="   "),
    )

    with pytest.raises(OCREmptyResultError, match="empty text for PDF"):
        await OCRService(db_session).extract_text_from_pdf_document(document.id)

    assert document.extracted_text is None
    assert document.metadata_jsonb is None
