"""DTOs for vector similarity search."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel


class SimilarityResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str | None = None
    chunk_index: int
    excerpt: str
    page_number: int | None = None
    similarity: float
    distance: float
    embedding_model: str | None = None
    metadata: dict[str, Any] | None = None
