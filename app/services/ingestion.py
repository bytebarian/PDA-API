"""File ingestion service.

Validates an incoming file upload, persists the file locally, and creates
the corresponding ``Document`` and ``ProcessingJob`` database rows.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.domain.status import DocumentStatus, ProcessingJobStage, ProcessingJobStatus
from app.models.document import Document
from app.models.processing_job import ProcessingJob
from app.services.file_storage import save_file


async def ingest_upload(
    *,
    db: AsyncSession,
    filename: str,
    content_type: str,
    data: bytes,
    settings: Settings,
) -> tuple[Document, ProcessingJob]:
    """Validate, store, and record a file upload.

    Raises:
        HTTPException 400 – when the file is empty.
        HTTPException 413 – when the file exceeds the configured size limit.
        HTTPException 415 – when the MIME type is not in the allow-list.
    """
    # --- Size validation -------------------------------------------------------
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(data) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"File size {len(data)} bytes exceeds the maximum allowed "
                f"{settings.max_file_size_bytes} bytes."
            ),
        )

    # --- MIME type validation ---------------------------------------------------
    normalised_ct = content_type.split(";")[0].strip().lower()
    if normalised_ct not in settings.allowed_file_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type '{normalised_ct}' is not allowed. "
                f"Accepted types: {', '.join(settings.allowed_file_types)}."
            ),
        )

    # --- Persist file ----------------------------------------------------------
    storage_path = Path(settings.storage_path)
    file_path, checksum = save_file(storage_path, filename, data)

    # Derive the sanitized name from the stored path so there is a single
    # source of truth (save_file's own sanitization).
    safe_name = file_path.name

    # Determine a simple file-type label from the extension or MIME type.
    suffix = Path(safe_name).suffix.lstrip(".").lower() or normalised_ct.split("/")[-1]

    # --- Create Document row ---------------------------------------------------
    document = Document(
        filename=safe_name,
        file_type=suffix or None,
        mime_type=normalised_ct,
        status=DocumentStatus.awaiting.value,
        path=str(file_path),
        size=len(data),
        checksum_sha256=checksum,
    )
    db.add(document)
    await db.flush()  # Populate document.id before creating the job.

    # --- Create ProcessingJob row ----------------------------------------------
    job = ProcessingJob(
        document_id=document.id,
        status=ProcessingJobStatus.awaiting.value,
        stage=ProcessingJobStage.upload_received.value,
    )
    db.add(job)
    await db.commit()
    await db.refresh(document)
    await db.refresh(job)

    return document, job
