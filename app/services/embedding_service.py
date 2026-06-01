"""Embedding generation service for persisted document chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embeddings import (
    EmbeddingDimensionMismatchError,
    EmbeddingProvider,
    EmbeddingProviderResponseError,
    FakeEmbeddingProvider,
    OllamaEmbeddingProvider,
)
from app.core.config import Settings, get_settings
from app.models.app_settings import AppSettings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.processing_job import ProcessingJob
from app.services.vector_validation import validate_embedding_vector

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Base class for embedding service failures."""


class DocumentNotFoundError(EmbeddingServiceError):
    """Raised when the document does not exist."""


class NoChunksToEmbedError(EmbeddingServiceError):
    """Raised when no chunk content exists for a document."""


class UnknownEmbeddingProviderError(EmbeddingServiceError):
    """Raised when a configured embedding provider is not registered."""


@dataclass(frozen=True)
class EmbeddingGenerationResult:
    """Result summary for one document embedding generation pass."""

    document_id: uuid.UUID
    embedded_chunk_count: int
    provider: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class _EmbeddingRuntimeConfig:
    provider: str
    model: str
    dimensions: int
    batch_size: int
    truncate: bool


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_embedding_providers(settings: Settings) -> dict[str, EmbeddingProvider]:
    return {
        "fake": FakeEmbeddingProvider(),
        "ollama": OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.embedding_timeout_seconds,
        ),
    }


class EmbeddingService:
    """Batch-generate and persist embeddings for document chunks."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        providers: dict[str, EmbeddingProvider] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._db = db
        self._settings = settings or get_settings()
        self._providers = providers if providers is not None else build_embedding_providers(self._settings)

    async def generate_embeddings_for_document(
        self,
        document_id: uuid.UUID,
        *,
        job_id: uuid.UUID | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
    ) -> EmbeddingGenerationResult:
        started = time.perf_counter()
        document = await self._db.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")

        if job_id is not None:
            job = await self._db.get(ProcessingJob, job_id)
            if job is None:
                raise EmbeddingServiceError(f"Processing job not found: {job_id}")
            if job.document_id != document.id:
                raise EmbeddingServiceError(
                    f"Processing job {job_id} does not belong to document {document.id}"
                )

        chunks = await self._load_chunks(document_id)
        if not chunks or not any(chunk.content.strip() for chunk in chunks):
            raise NoChunksToEmbedError(f"Document {document_id} has no chunks to embed")

        runtime = await self._resolve_runtime_config(
            provider_name=provider_name,
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
        )
        provider = self._providers.get(runtime.provider)
        if provider is None:
            raise UnknownEmbeddingProviderError(
                f"Unknown embedding provider '{runtime.provider}'"
            )

        pending_updates: list[tuple[DocumentChunk, list[float], str]] = []
        actual_model: str | None = None
        for start in range(0, len(chunks), runtime.batch_size):
            chunk_batch = chunks[start : start + runtime.batch_size]
            texts = [chunk.content for chunk in chunk_batch]
            results = await provider.embed_texts(
                texts,
                model=runtime.model,
                dimensions=runtime.dimensions,
                truncate=runtime.truncate,
            )
            if len(results) != len(chunk_batch):
                raise EmbeddingProviderResponseError(
                    f"Provider returned {len(results)} embeddings for {len(chunk_batch)} chunks"
                )
            indexed = {result.text_index: result for result in results}
            expected_indices = set(range(len(chunk_batch)))
            if set(indexed) != expected_indices:
                raise EmbeddingProviderResponseError(
                    "Provider returned embedding indices that do not match the input batch"
                )

            for index, chunk in enumerate(chunk_batch):
                embedding = indexed[index]
                if embedding.dimensions != runtime.dimensions:
                    raise EmbeddingDimensionMismatchError(
                        f"Embedding dimensions {embedding.dimensions} do not match expected {runtime.dimensions}"
                    )
                if actual_model is None:
                    actual_model = embedding.model
                elif actual_model != embedding.model:
                    raise EmbeddingProviderResponseError(
                        f"Provider returned inconsistent model names across batches: "
                        f"'{actual_model}' vs '{embedding.model}'"
                    )
                pending_updates.append((chunk, embedding.vector, embedding.model))

        if actual_model is None:
            raise EmbeddingProviderResponseError(
                "Provider returned no model name; cannot determine authoritative embedding model"
            )

        # Validate every vector before mutating any chunk so a validation
        # failure never leaves partial vectors in the session.
        validated_updates: list[tuple[DocumentChunk, list[float], str]] = [
            (
                chunk,
                validate_embedding_vector(
                    vector,
                    expected_dimensions=runtime.dimensions,
                ),
                model_name,
            )
            for chunk, vector, model_name in pending_updates
        ]

        # Apply mutations atomically only after all batches have been fetched
        # and validated, so a failed embedding run never leaves partial vectors.
        for chunk, vector, model_name in validated_updates:
            chunk.embedding = vector
            chunk.embedding_model = model_name
            chunk.embedding_provider = runtime.provider
            chunk.embedding_dimension = runtime.dimensions
            chunk.embedding_created_at = _utcnow()
        embedded_count = len(pending_updates)
        document.embedding_model = actual_model
        document.chunk_count = len(chunks)
        document.last_indexed_at = _utcnow()
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Document embeddings persisted",
            extra={
                "document_id": str(document.id),
                "chunk_count": len(chunks),
                "embedding_model": actual_model,
                "embedding_dimension": runtime.dimensions,
                "duration_ms": duration_ms,
            },
        )

        return EmbeddingGenerationResult(
            document_id=document.id,
            embedded_chunk_count=embedded_count,
            provider=runtime.provider,
            model=actual_model,
            dimensions=runtime.dimensions,
        )

    async def _resolve_runtime_config(
        self,
        *,
        provider_name: str | None,
        model: str | None,
        dimensions: int | None,
        batch_size: int | None,
    ) -> _EmbeddingRuntimeConfig:
        persisted = await self._load_app_settings()

        resolved_provider = (
            provider_name
            or (persisted.embedding_provider if persisted else None)
            or self._settings.embedding_provider
        )
        resolved_model = (
            model
            or (persisted.embedding_model if persisted else None)
            or self._settings.embedding_model
        )
        resolved_dimensions = (
            dimensions
            if dimensions is not None
            else (persisted.embedding_dimensions if persisted else None)
        )
        if resolved_dimensions is None:
            resolved_dimensions = self._settings.embedding_dimensions
        if resolved_dimensions <= 0:
            raise EmbeddingDimensionMismatchError("Embedding dimensions must be greater than 0")

        resolved_batch_size = batch_size if batch_size is not None else self._settings.embedding_batch_size
        if resolved_batch_size <= 0:
            raise EmbeddingServiceError("Embedding batch size must be greater than 0")

        if not resolved_provider or not resolved_model:
            raise EmbeddingServiceError("Embedding provider and model must be configured")

        return _EmbeddingRuntimeConfig(
            provider=resolved_provider,
            model=resolved_model,
            dimensions=resolved_dimensions,
            batch_size=resolved_batch_size,
            truncate=self._settings.embedding_truncate,
        )

    async def _load_chunks(self, document_id: uuid.UUID) -> list[DocumentChunk]:
        result = await self._db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def _load_app_settings(self) -> AppSettings | None:
        result = await self._db.execute(
            select(AppSettings).order_by(AppSettings.updated_at.desc()).limit(1)
        )
        return result.scalar_one_or_none()
