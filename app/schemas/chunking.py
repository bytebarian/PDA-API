"""Pydantic schemas/DTOs for chunking configuration and results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from app.models.app_settings import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


class ChunkingConfig(BaseModel):
    """Validated chunking configuration loaded from persisted settings."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    @model_validator(mode="after")
    def validate_relationship(self) -> ChunkingConfig:
        if self.chunk_size <= 0:
            raise ValueError(f"chunkSize must be > 0, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunkOverlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunkOverlap ({self.chunk_overlap}) must be < chunkSize ({self.chunk_size})"
            )
        return self


class ChunkRead(BaseModel):
    """Read-only representation of a produced chunk (not persisted schema)."""

    model_config = ConfigDict(frozen=True)

    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    page_number: int | None = None
    source_section: str | None = None
    metadata: dict[str, Any] | None = None
