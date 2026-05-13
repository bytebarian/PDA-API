"""HTTP endpoints for document management.

Currently exposes:
  POST /documents/upload – multipart file upload that persists the file
  locally and creates Document + ProcessingJob database rows.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.upload import UploadResponse
from app.services.ingestion import ingest_upload

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_READ_CHUNK_SIZE = 1024 * 1024

async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_SIZE)
        if not chunk:
            break

        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"File size exceeds the maximum allowed {max_bytes} bytes."
                ),
            )

        chunks.append(chunk)

    return b"".join(chunks)

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
    summary="Upload a document file",
)
async def upload_document(
    file: UploadFile = File(..., description="The document file to upload."),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Accept a multipart file upload and start the ingestion pipeline.

    - Validates that a file is present.
    - Validates file size against the configured maximum.
    - Validates the MIME type against the configured allow-list.
    - Persists the file to local storage.
    - Creates a **Document** row and a linked **ProcessingJob** row.
    - Returns document and job identifiers plus initial status values.
    """
    data = await read_upload_limited(file, settings.max_file_size_bytes)
    content_type = file.content_type or ""

    document, job = await ingest_upload(
        db=db,
        filename=file.filename or "",
        content_type=content_type,
        data=data,
        settings=settings,
    )

    return UploadResponse(
        document_id=document.id,
        job_id=job.id,
        filename=document.filename,
        status=document.status,  # type: ignore[arg-type]
        job_status=job.status,  # type: ignore[arg-type]
        job_stage=job.stage,  # type: ignore[arg-type]
    )
