"""Local file storage utilities.

Provides helpers for sanitizing filenames, computing SHA-256 checksums,
and persisting uploaded bytes under a configured storage directory.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """Return a safe filename stripped of path-traversal components.

    * Strips directory separators and leading dots so callers cannot write
      outside the target storage directory.
    * Replaces runs of whitespace or control characters with underscores.
    * Falls back to a random UUID-based name when the result would be empty.
    """
    # Take only the final path component to block traversal sequences like
    # "../../../etc/passwd" or "foo/bar".
    name = Path(filename).name

    # Strip leading dots (hidden-file trick on Unix).
    name = name.lstrip(".")

    # Replace characters that are problematic on most file systems.
    name = re.sub(r'[\x00-\x1f\x7f/\\:*?"<>|]', "_", name)

    # Collapse runs of whitespace or repeated underscores.
    name = re.sub(r"\s+", "_", name)

    if not name:
        name = f"{uuid.uuid4().hex}"

    return name


def compute_sha256(data: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def save_file(storage_path: Path, filename: str, data: bytes) -> tuple[Path, str]:
    """Persist *data* under *storage_path* and return ``(file_path, checksum)``.

    The storage directory is created on first use.  The filename is sanitized
    before being written so that callers never need to pre-sanitize.

    Returns:
        A tuple of the absolute ``Path`` where the file was written and its
        SHA-256 hex digest.
    """
    safe_name = sanitize_filename(filename)
    storage_path.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    dest = storage_path / unique_name
    dest.write_bytes(data)

    checksum = compute_sha256(data)
    return dest, checksum
