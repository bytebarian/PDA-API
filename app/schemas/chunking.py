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
        from app.services.chunking_service import (
            ChunkingValidationError,
            validate_chunk_settings,
        )

        try:
            validate_chunk_settings(self.chunk_size, self.chunk_overlap)
        except ChunkingValidationError as exc:
            raise ValueError(str(exc)) from exc
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
