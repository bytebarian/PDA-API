"""Embedding generation service for persisted document chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

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
from app.models.document_chunk import DocumentChunk, EMBEDDING_DIMENSIONS


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
        del job_id
        document = await self._db.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")

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

        embedded_count = 0
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
                chunk.embedding = embedding.vector
                chunk.embedding_model = embedding.model
                embedded_count += 1

        document.embedding_model = runtime.model
        document.chunk_count = len(chunks)
        document.last_indexed_at = _utcnow()

        return EmbeddingGenerationResult(
            document_id=document.id,
            embedded_chunk_count=embedded_count,
            provider=runtime.provider,
            model=runtime.model,
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
        if resolved_dimensions != EMBEDDING_DIMENSIONS:
            raise EmbeddingDimensionMismatchError(
                f"Configured embedding dimensions {resolved_dimensions} do not match chunk vector dimensions {EMBEDDING_DIMENSIONS}"
            )

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
