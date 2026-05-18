"""HTTP endpoints for document management.

Exposes:
  POST /documents/upload      – multipart file upload
  GET  /documents             – paginated document list
  GET  /documents/{id}        – document detail
  POST /documents/{id}/reprocess – create a new processing job stub
  PATCH /documents/{id}       – safe metadata updates
  DELETE /documents/{id}      – delete document and related state
  GET  /documents/{id}/download – download original file
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.domain.status import DocumentStatus, ProcessingJobStage, ProcessingJobStatus
from app.models.document_chunk import DocumentChunk
from app.models.document import Document
from app.models.processing_job import ProcessingJob
from app.schemas.reprocess import ReprocessRequest, ReprocessResponse
from app.schemas.document import (
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    DocumentUpdate,
    ProcessingJobSummary,
)
from app.schemas.upload import UploadResponse
from app.services.ingestion import ingest_upload
from app.services.file_storage import resolve_stored_file_path, sanitize_filename

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_READ_CHUNK_SIZE = 1024 * 1024
logger = logging.getLogger(__name__)


def _escape_like_pattern(value: str) -> str:
    """Escape SQL LIKE wildcard characters in *value* for literal substring matching.

    Escapes backslash first (the chosen escape character), then ``%`` and
    ``_``, so the caller can safely pass the result to
    ``ColumnElement.ilike(f"%{result}%", escape="\\")``.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sanitize_updated_filename(filename: str) -> str:
    """Sanitize a user-provided filename and reject empty values."""
    safe_filename = sanitize_filename(filename)
    base_name = Path(filename).name.lstrip(".")
    if not base_name.strip() or not safe_filename.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Filename must not be empty.",
        )
    return safe_filename


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


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    document_status: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    category: Annotated[str | None, Query()] = None,
    file_type: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query()] = None,
    sort: Annotated[Literal["newest", "oldest"], Query()] = "newest",
) -> DocumentListResponse:
    """Return a paginated list of documents with optional filters.

    Query parameters:
    - **page**: 1-based page number (default 1)
    - **page_size**: items per page, max 100 (default 20)
    - **status**: filter by DocumentStatus value
    - **category**: exact category match
    - **file_type**: exact file_type match
    - **q**: case-insensitive filename substring search
    - **sort**: ``newest`` (default) or ``oldest``
    """
    filters = []
    if document_status is not None:
        filters.append(Document.status == document_status.value)
    if category is not None:
        filters.append(Document.category == category)
    if file_type is not None:
        filters.append(Document.file_type == file_type)
    if q is not None:
        filters.append(Document.filename.ilike(f"%{_escape_like_pattern(q)}%", escape="\\"))

    count_stmt = select(func.count()).select_from(Document)
    if filters:
        count_stmt = count_stmt.where(*filters)
    total: int = (await db.execute(count_stmt)).scalar_one()

    order_by_clauses = (
        (Document.created_at.asc(), Document.id.asc())
        if sort == "oldest"
        else (Document.created_at.desc(), Document.id.desc())
    )
    offset = (page - 1) * page_size
    list_stmt = (
        select(Document)
        .where(*filters)
        .order_by(*order_by_clauses)
        .offset(offset)
        .limit(page_size)
    )
    rows = (await db.execute(list_stmt)).scalars().all()

    items = [DocumentSummary.model_validate(row) for row in rows]
    return DocumentListResponse(items=items, page=page, page_size=page_size, total=total)


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="Get document detail",
)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetail:
    """Return full detail for a single document by UUID.

    Includes a summary of the most recently created processing job when one
    exists.  Returns **404** if the document is not found.
    """
    doc_row = await db.get(Document, document_id)
    if doc_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    job_stmt = (
        select(ProcessingJob)
        .where(ProcessingJob.document_id == document_id)
        .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
        .limit(1)
    )
    job_row = (await db.execute(job_stmt)).scalars().first()

    latest_job: ProcessingJobSummary | None = None
    if job_row is not None:
        latest_job = ProcessingJobSummary.model_validate(job_row)

    return DocumentDetail(
        id=doc_row.id,
        filename=doc_row.filename,
        category=doc_row.category,
        file_type=doc_row.file_type,
        mime_type=doc_row.mime_type,
        status=doc_row.status,  # type: ignore[arg-type]
        size=doc_row.size,
        checksum_sha256=doc_row.checksum_sha256,
        summary=doc_row.summary,
        chunk_count=doc_row.chunk_count,
        embedding_model=doc_row.embedding_model,
        last_indexed_at=doc_row.last_indexed_at,
        created_at=doc_row.created_at,
        updated_at=doc_row.updated_at,
        latest_job=latest_job,
    )


@router.patch(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="Update document metadata",
)
async def update_document(
    document_id: uuid.UUID,
    payload: DocumentUpdate,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetail:
    """Apply a safe partial metadata update for a document."""
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    update_data = payload.model_dump(exclude_unset=True)
    if "filename" in update_data:
        filename = update_data["filename"]
        if filename is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Filename must not be empty.",
            )
        update_data["filename"] = _sanitize_updated_filename(filename)

    for field, value in update_data.items():
        setattr(document, field, value)

    await db.commit()
    return await get_document(document.id, db)


@router.post(
    "/{document_id}/reprocess",
    response_model=ReprocessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Request document reprocessing",
)
async def reprocess_document(
    document_id: uuid.UUID,
    payload: ReprocessRequest | None = Body(default=None),
    db: AsyncSession = Depends(get_db),
) -> ReprocessResponse:
    """Create a new processing job and reset document status to awaiting."""
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    preferred_stage = ProcessingJobStage.queued
    job = ProcessingJob(
        document_id=document.id,
        status=ProcessingJobStatus.awaiting.value,
        stage=preferred_stage.value,
    )

    if payload is not None:
        history_event: dict[str, str | bool | None] = {
            "stage": preferred_stage.value,
            "force": payload.force,
            "reason": payload.reason,
        }
        job.stage_history_jsonb = [history_event]

    document.status = DocumentStatus.awaiting.value
    db.add(job)
    await db.commit()
    await db.refresh(job)
    await db.refresh(document)

    return ReprocessResponse(
        document_id=document.id,
        job_id=job.id,
        document_status=DocumentStatus(document.status),
        job_status=ProcessingJobStatus(job.status),
        job_stage=ProcessingJobStage(job.stage),
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Delete a document, related rows, and its stored file when safely resolvable."""
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    resolved_path = resolve_stored_file_path(Path(settings.storage_path), document.path or "")

    await db.execute(delete(ProcessingJob).where(ProcessingJob.document_id == document_id))
    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    await db.execute(delete(Document).where(Document.id == document_id))
    await db.commit()

    if resolved_path is not None:
        try:
            resolved_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to delete stored file for document %s", document_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{document_id}/download",
    summary="Download original document file",
)
async def download_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Stream the original uploaded file for a document."""
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    storage_root = Path(settings.storage_path)
    stored_path = document.path or ""
    resolved_path = resolve_stored_file_path(storage_root, stored_path)
    if resolved_path is None or not resolved_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document file not found")

    return FileResponse(
        path=resolved_path,
        media_type=document.mime_type or "application/octet-stream",
        filename=sanitize_filename(document.filename),
    )
