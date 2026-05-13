"""Shared status and stage vocabulary for PDA persistence and schemas.

These enums centralize all status/stage string constants so that models,
schemas, services, and future routers share a single authoritative source.
Database columns remain ``String`` for portability; these enums provide
Python-layer validation and default constants only.
"""

from __future__ import annotations

from enum import Enum


class DocumentStatus(str, Enum):
    """Lifecycle statuses for a Document record."""

    awaiting = "awaiting"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class ProcessingJobStatus(str, Enum):
    """Lifecycle statuses for a ProcessingJob record."""

    awaiting = "awaiting"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class ProcessingJobStage(str, Enum):
    """Fine-grained pipeline stages for a ProcessingJob record."""

    queued = "queued"
    upload_received = "upload_received"
    ocr = "ocr"
    text_extraction = "text_extraction"
    chunking = "chunking"
    embedding = "embedding"
    indexing = "indexing"
    completed = "completed"
    failed = "failed"
