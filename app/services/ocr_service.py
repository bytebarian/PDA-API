"""OCR service for image-based documents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.ocr import (
    OCREmptyResultError,
    OCRPageResult,
    OCRProvider,
    OCRProviderUnavailableError,
    OCRResult,
    OCRUnsupportedMimeTypeError,
    FakeOCRProvider,
    TesseractOCRProvider,
    mime_type_requires_ocr,
    normalize_mime_type,
)
from app.core.config import Settings, get_settings
from app.domain.status import ProcessingJobStage
from app.models.document import Document
from app.models.processing_job import ProcessingJob
from app.services.file_storage import resolve_stored_file_path


@dataclass(frozen=True)
class OCRExtractionResult:
    """Persisted OCR result summary for callers."""

    document_id: uuid.UUID
    provider: str
    extracted_text: str
    char_count: int
    confidence: float | None
    languages: list[str]
    warnings: list[str]
    duration_seconds: float
    metadata: dict[str, Any]


def _append_stage_history(
    job: ProcessingJob,
    *,
    status: str,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    from app.services.processing_orchestrator import _append_stage_history as append_job_stage_history

    append_job_stage_history(
        job,
        stage=ProcessingJobStage.ocr,
        status=status,
        message=message,
        details=details,
    )


def _default_languages(settings: Settings) -> list[str]:
    if settings.tesseract_languages:
        return list(settings.tesseract_languages)
    if settings.ocr_language:
        return [settings.ocr_language]
    return ["eng"]


def _average_confidence(pages: list[OCRPageResult]) -> float | None:
    confidences = [page.confidence for page in pages if page.confidence is not None]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _combine_pages(result: OCRResult) -> str:
    ordered_pages = sorted(result.pages, key=lambda page: page.page_number)
    page_texts = [page.text.strip() for page in ordered_pages if page.text.strip()]
    return "\n\n".join(page_texts).strip()


def get_ocr_provider(
    *,
    provider_name: str | None = None,
    settings: Settings | None = None,
) -> OCRProvider:
    resolved_settings = settings or get_settings()
    selected_provider = (provider_name or resolved_settings.ocr_provider).strip().lower()

    if selected_provider == "tesseract":
        return TesseractOCRProvider(
            command=resolved_settings.tesseract_cmd,
            default_languages=_default_languages(resolved_settings),
            default_timeout_seconds=resolved_settings.tesseract_timeout_seconds,
            psm=resolved_settings.tesseract_psm,
            oem=resolved_settings.tesseract_oem,
        )
    if selected_provider == "fake":
        return FakeOCRProvider(text="fake ocr text")

    raise OCRProviderUnavailableError(f"Unsupported OCR provider: {selected_provider}")


class OCRService:
    """Loads image documents, runs OCR, and persists extracted text/metadata."""

    def __init__(self, db: AsyncSession, *, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()

    async def extract_text_for_document(
        self,
        document_id: uuid.UUID,
        *,
        job_id: uuid.UUID | None = None,
        provider_name: str | None = None,
        languages: list[str] | None = None,
    ) -> OCRExtractionResult:
        document = await self.db.get(Document, document_id)
        if document is None:
            raise LookupError(f"Document not found: {document_id}")

        job = await self.db.get(ProcessingJob, job_id) if job_id is not None else None
        normalized_mime = normalize_mime_type(document.mime_type)
        if not mime_type_requires_ocr(normalized_mime):
            raise OCRUnsupportedMimeTypeError(
                f"Unsupported OCR MIME type: {normalized_mime or '<unknown>'}"
            )

        stored_path = document.path or ""
        resolved_path = resolve_stored_file_path(self.settings.storage_path, stored_path)
        if resolved_path is None:
            raise FileNotFoundError(
                f"Document file path is missing or outside storage root for document {document.id}"
            )

        provider = get_ocr_provider(provider_name=provider_name, settings=self.settings)
        selected_languages = list(languages or _default_languages(self.settings))
        started = perf_counter()
        if job is not None:
            _append_stage_history(
                job,
                status="processing",
                details={
                    "provider": provider.name,
                    "languages": selected_languages,
                    "mime_type": normalized_mime,
                },
            )

        try:
            result = await provider.extract_text(
                resolved_path,
                mime_type=normalized_mime,
                languages=selected_languages,
                timeout_seconds=self.settings.tesseract_timeout_seconds,
            )
            extracted_text = _combine_pages(result)
            if not extracted_text:
                raise OCREmptyResultError(
                    f"OCR produced empty text for document {document.id}"
                )

            confidence = _average_confidence(result.pages)
            duration_seconds = round(perf_counter() - started, 6)
            metadata = {
                "provider": result.provider,
                "engine_version": result.engine_version,
                "languages": list(result.languages),
                "char_count": len(extracted_text),
                "confidence": confidence,
                "duration_seconds": duration_seconds,
                "warnings": list(result.warnings),
                "page_count": len(result.pages),
                "mime_type": normalized_mime,
                "source_path": str(resolved_path),
            }
            merged_metadata = dict(document.metadata_jsonb or {})
            merged_metadata["ocr"] = metadata

            document.extracted_text = extracted_text
            document.metadata_jsonb = merged_metadata
            await self.db.flush()

            if job is not None:
                _append_stage_history(job, status="completed", details=metadata)

            return OCRExtractionResult(
                document_id=document.id,
                provider=result.provider,
                extracted_text=extracted_text,
                char_count=len(extracted_text),
                confidence=confidence,
                languages=list(result.languages),
                warnings=list(result.warnings),
                duration_seconds=duration_seconds,
                metadata=metadata,
            )
        except Exception as error:
            if job is not None:
                duration_seconds = round(perf_counter() - started, 6)
                _append_stage_history(
                    job,
                    status="failed",
                    message="Stage failed",
                    details={
                        "provider": provider.name,
                        "languages": selected_languages,
                        "mime_type": normalized_mime,
                        "duration_seconds": duration_seconds,
                        "error_type": error.__class__.__name__,
                        "message": str(error) or error.__class__.__name__,
                    },
                )
            raise


def document_requires_ocr(document: Document) -> bool:
    """Return True when *document* should go through OCR."""

    return mime_type_requires_ocr(document.mime_type)


def resolve_document_path(document: Document, settings: Settings) -> Path | None:
    """Resolve a stored document file path relative to the configured storage root."""

    return resolve_stored_file_path(settings.storage_path, document.path or "")
