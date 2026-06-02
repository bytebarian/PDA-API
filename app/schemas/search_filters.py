"""Reusable semantic-search metadata filters."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.status import DocumentStatus

_METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_METADATA_FILTER_KEYS = 20
_MAX_METADATA_KEY_LENGTH = 100
_MAX_METADATA_VALUE_LENGTH = 500


class SearchFilters(BaseModel):
    """Document-level filters applied before vector similarity ordering."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        validation_alias=AliasChoices("document_ids", "documentIds"),
    )
    categories: list[str] | None = None
    file_types: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("file_types", "fileTypes"),
    )
    statuses: list[str] | None = None
    filename_contains: str | None = Field(
        default=None,
        max_length=255,
        validation_alias=AliasChoices("filename_contains", "filenameContains"),
    )
    created_from: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("created_from", "createdFrom"),
    )
    created_to: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("created_to", "createdTo"),
    )
    modified_from: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("modified_from", "modifiedFrom"),
    )
    modified_to: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("modified_to", "modifiedTo"),
    )
    metadata: dict[str, str] | None = None

    @field_validator("categories", "file_types", "statuses", mode="before")
    @classmethod
    def _normalize_non_empty_list(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list):
            return value
        normalized = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return normalized or None

    @field_validator("filename_contains")
    @classmethod
    def _normalize_filename_contains(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("metadata filter must be an object of key/value pairs")
        if len(value) > _MAX_METADATA_FILTER_KEYS:
            raise ValueError(
                f"metadata filter supports at most {_MAX_METADATA_FILTER_KEYS} keys"
            )
        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise ValueError("metadata keys must be strings")
            key = raw_key.strip()
            if not key:
                raise ValueError("metadata keys must not be blank")
            if len(key) > _MAX_METADATA_KEY_LENGTH:
                raise ValueError(
                    f"metadata key '{key[:20]}...' exceeds {_MAX_METADATA_KEY_LENGTH} characters"
                )
            if not _METADATA_KEY_RE.fullmatch(key):
                raise ValueError(
                    "metadata keys may contain only letters, numbers, underscores, and hyphens"
                )
            if not isinstance(raw_value, str):
                raise ValueError(f"metadata value for '{key}' must be a string")
            item = raw_value.strip()
            if len(item) > _MAX_METADATA_VALUE_LENGTH:
                raise ValueError(
                    f"metadata value for '{key}' exceeds {_MAX_METADATA_VALUE_LENGTH} characters"
                )
            normalized[key] = item
        return normalized or None

    @model_validator(mode="after")
    def _validate_ranges_and_statuses(self) -> SearchFilters:
        if self.created_from and self.created_to and self.created_from > self.created_to:
            raise ValueError("created_from must be less than or equal to created_to")
        if self.modified_from and self.modified_to and self.modified_from > self.modified_to:
            raise ValueError("modified_from must be less than or equal to modified_to")
        if self.statuses is not None:
            allowed = {status.value for status in DocumentStatus}
            invalid = [status for status in self.statuses if status not in allowed]
            if invalid:
                raise ValueError(f"unsupported statuses: {', '.join(sorted(set(invalid)))}")
        return self

    def resolved_statuses(self) -> list[str]:
        """Return explicit statuses or the default searchable status."""
        if self.statuses:
            return self.statuses
        return [DocumentStatus.ready.value]

    def filters_applied(self) -> dict[str, Any]:
        """Safe diagnostics payload for API responses and logs."""
        payload: dict[str, Any] = {}
        if self.document_ids:
            payload["document_ids"] = [str(value) for value in self.document_ids]
        if self.categories:
            payload["categories"] = self.categories
        if self.file_types:
            payload["file_types"] = self.file_types
        if self.statuses:
            payload["statuses"] = self.statuses
        if self.filename_contains:
            payload["filename_contains"] = self.filename_contains
        if self.created_from:
            payload["created_from"] = self.created_from.isoformat()
        if self.created_to:
            payload["created_to"] = self.created_to.isoformat()
        if self.modified_from:
            payload["modified_from"] = self.modified_from.isoformat()
        if self.modified_to:
            payload["modified_to"] = self.modified_to.isoformat()
        if self.metadata:
            payload["metadata_keys"] = sorted(self.metadata.keys())
        return payload
