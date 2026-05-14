"""HTTP endpoints for document management.

Exposes:
  POST /documents/upload      – multipart file upload
  GET  /documents             – paginated document list
  GET  /documents/{id}        – document detail
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.domain.status import DocumentStatus
from app.models.document import Document
from app.models.processing_job import ProcessingJob
from app.schemas.document import (
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    ProcessingJobSummary,
)
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
        escaped_q = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(Document.filename.ilike(f"%{escaped_q}%", escape="\\"))

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
