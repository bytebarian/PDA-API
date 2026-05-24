"""Pydantic schemas/DTOs for chunking configuration and results."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

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

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    chunk_index: int
    content: str
    source_start_offset: int = Field(
        validation_alias=AliasChoices("source_start_offset", "start_offset")
    )
    source_end_offset: int = Field(
        validation_alias=AliasChoices("source_end_offset", "end_offset")
    )
    page_number: int | None = None
    source_section: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def start_offset(self) -> int:
        return self.source_start_offset

    @property
    def end_offset(self) -> int:
        return self.source_end_offset
