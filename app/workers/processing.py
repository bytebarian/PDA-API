"""Background runner that starts document processing jobs after ingest."""

from __future__ import annotations

import logging
import uuid

from fastapi import BackgroundTasks

from app.db.session import get_session_factory
from app.services.processing_orchestrator import process_job

logger = logging.getLogger(__name__)


async def run_processing_job(job_id: uuid.UUID) -> None:
    """Open a fresh DB session and run the processing pipeline for *job_id*.

    Failures are recorded on the job/document by ``process_job`` and then
    logged here so the HTTP request that enqueued the work is unaffected.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            await process_job(session, job_id)
        except Exception:
            logger.exception("Background processing failed for job %s", job_id)


def enqueue_processing_job(background_tasks: BackgroundTasks, job_id: uuid.UUID) -> None:
    """Schedule ``run_processing_job`` to run after the response is sent."""
    background_tasks.add_task(run_processing_job, job_id)
