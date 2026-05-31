"""Text normalization service for the PDA document-processing pipeline.

This module provides a deterministic, versioned text normalization stage that
runs after OCR/text-extraction and before chunking.  It cleans encoding
artifacts, normalises whitespace and line endings, removes control characters,
and applies conservative OCR-artifact cleanup while preserving meaningful
Unicode content (including non-ASCII letters and diacritics).

All normalization is local – no content is sent to external services.

Rule-set version: ``pda-normalization-v1``
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal, cast

# ---------------------------------------------------------------------------
# Rule-set version
# ---------------------------------------------------------------------------

RULE_SET_VERSION = "pda-normalization-v1"

# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class TextNormalizationError(Exception):
    """Base class for text normalization errors."""


class TextNormalizationEmptyInputError(TextNormalizationError):
    """Raised when the normalization stage receives empty or absent input."""


class TextNormalizationEmptyOutputError(TextNormalizationError):
    """Raised when normalization produces empty output from non-empty input."""


class TextNormalizationConfigurationError(TextNormalizationError):
    """Raised when normalization options are invalid."""


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextNormalizationOptions:
    """Configurable knobs for the normalization rule set.

    All flags default to the conservative production values.  Tests may
    override individual flags without affecting other behaviour.
    """

    unicode_form: str = "NFKC"
    """Unicode normalization form applied to the input (default NFKC)."""

    normalize_line_endings: bool = True
    """Convert ``\\r\\n`` and ``\\r`` to ``\\n``."""

    collapse_horizontal_whitespace: bool = True
    """Replace runs of spaces/tabs on a single line with a single space."""

    collapse_blank_lines: bool = True
    """Collapse repeated blank lines to at most ``max_blank_lines``."""

    trim_lines: bool = True
    """Strip leading and trailing whitespace from each line."""

    remove_control_characters: bool = True
    """Remove non-content ASCII control characters (keeps ``\\n``)."""

    dehyphenate_line_breaks: bool = True
    """Join words split by line-end hyphenation when safe."""

    max_blank_lines: int = 1
    """Maximum number of consecutive blank lines preserved."""


@dataclass(frozen=True)
class TextNormalizationWarning:
    """A single diagnostic warning produced during normalization."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextNormalizationResult:
    """Immutable result produced by :func:`normalize_text`."""

    normalized_text: str
    input_character_count: int
    output_character_count: int
    input_line_count: int
    output_line_count: int
    changed: bool
    rule_set_version: str
    warnings: list[TextNormalizationWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal normalization helpers (pure functions, easy to unit-test)
# ---------------------------------------------------------------------------

# Matches ASCII control characters except \n (0x0a).
# Excludes: 0x09 (tab, handled separately in whitespace normalization),
#           0x0a (newline, kept as paragraph separator).
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

# Matches lines that contain only 5+ non-word characters (no alphanumeric
# content). Conservative OCR artifact cleanup.
_SEPARATOR_LINE_RE = re.compile(
    r"(?m)^[^\w\n]{5,}$"
)


def _remove_bom(text: str) -> str:
    """Remove a leading UTF-8 BOM character if present."""
    return text[1:] if text.startswith("\ufeff") else text


def _apply_unicode_normalization(text: str, form: str) -> str:
    """Apply Unicode normalization *form* to *text*."""
    if not form:
        return text
    try:
        return unicodedata.normalize(
            cast(Literal["NFC", "NFD", "NFKC", "NFKD"], form), text
        )
    except (ValueError, TypeError) as exc:
        raise TextNormalizationConfigurationError(
            f"Invalid Unicode normalization form '{form}': {exc}"
        ) from exc


def _normalize_line_endings(text: str) -> str:
    """Convert ``\\r\\n`` and standalone ``\\r`` to ``\\n``."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _remove_control_chars(text: str) -> str:
    """Strip non-content ASCII control characters, keeping ``\\n``."""
    return _CONTROL_CHAR_RE.sub("", text)


def _remove_soft_hyphens(text: str) -> str:
    """Remove soft-hyphen characters (U+00AD)."""
    return text.replace("\u00ad", "")


def _dehyphenate_line_breaks(text: str) -> str:
    """Join words that were split across lines by a hyphen.

    Only joins when both sides are lowercase letters to avoid corrupting
    proper nouns, compound words, or list markers.

    Example: ``"docu-\\nment"`` → ``"document"``
    """
    return re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)


def _normalize_lines(text: str, *, collapse_whitespace: bool, trim: bool) -> str:
    """Apply per-line whitespace normalisation.

    Args:
        text: Input text with normalised line endings (only ``\\n``).
        collapse_whitespace: Replace runs of spaces/tabs with a single space.
        trim: Strip leading and trailing whitespace from each line.
    """
    if not collapse_whitespace and not trim:
        return text

    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        if collapse_whitespace:
            line = re.sub(r"[ \t]+", " ", line)
        if trim:
            line = line.strip()
        result.append(line)
    return "\n".join(result)


def _collapse_blank_lines(text: str, max_blank_lines: int) -> str:
    """Reduce runs of blank lines to at most *max_blank_lines*.

    A "blank line" is a line containing only whitespace (empty after per-line
    trim).  ``max_blank_lines=1`` allows a single blank line between
    paragraphs; ``max_blank_lines=0`` removes all blank lines.
    """
    # (max_blank_lines + 1) consecutive ``\\n`` means max_blank_lines blank
    # lines between paragraphs; we collapse anything longer.
    threshold = max_blank_lines + 2  # number of \\n chars that means "too many"
    replacement = "\n" * (max_blank_lines + 1)
    pattern = re.compile(r"\n{" + str(threshold) + r",}")
    return pattern.sub(replacement, text)


def _remove_separator_lines(text: str) -> str:
    """Remove lines consisting solely of 5+ repeated separator characters.

    This is a very conservative OCR artifact cleanup.  Lines with any
    alphanumeric content are left untouched.
    """
    return _SEPARATOR_LINE_RE.sub("", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_text(
    text: str,
    options: TextNormalizationOptions | None = None,
    warn_removal_ratio: float = 0.30,
) -> TextNormalizationResult:
    """Normalize *text* according to the PDA conservative rule set.

    The function is **deterministic**: identical *text* + *options* always
    produces the same output.  No external APIs are called.

    Args:
        text: Source string from a text-extraction or OCR stage.
        options: Normalization options.  Defaults to
            :class:`TextNormalizationOptions` conservative defaults.
        warn_removal_ratio: Fraction of input characters that may be removed
            before a ``HIGH_REMOVAL_RATIO`` warning is emitted.

    Returns:
        A frozen :class:`TextNormalizationResult` containing the normalised
        text and diagnostic metadata.

    Raises:
        TextNormalizationError: When *text* is not a ``str``.
        TextNormalizationConfigurationError: When *options* contains invalid
            settings, for example an unknown Unicode form.

        Note:
            Empty/whitespace input enforcement (raising
            :class:`TextNormalizationEmptyInputError` /
            :class:`TextNormalizationEmptyOutputError`) is performed by the
            processing pipeline stage, not by this pure function.
    """
    if not isinstance(text, str):
        raise TextNormalizationError(
            f"normalize_text expects str input, got {type(text).__name__}"
        )

    if options is None:
        options = TextNormalizationOptions()

    if options.max_blank_lines < 0:
        raise TextNormalizationConfigurationError(
            f"max_blank_lines must be >= 0, got {options.max_blank_lines}"
        )
    if not 0.0 <= warn_removal_ratio <= 1.0:
        raise TextNormalizationConfigurationError(
            f"warn_removal_ratio must be between 0 and 1, got {warn_removal_ratio}"
        )

    input_char_count = len(text)
    line_count_source = (
        text.replace("\r\n", "\n").replace("\r", "\n") if text else ""
    )
    input_line_count = line_count_source.count("\n") + 1 if line_count_source else 0
    warnings: list[TextNormalizationWarning] = []
    result = text

    # 1. Remove BOM
    result = _remove_bom(result)

    # 2. Unicode normalization
    if options.unicode_form:
        result = _apply_unicode_normalization(result, options.unicode_form)

    # 3. Line ending normalization
    if options.normalize_line_endings:
        result = _normalize_line_endings(result)

    # 4. Control character cleanup (keeps \n)
    if options.remove_control_characters:
        result = _remove_control_chars(result)

    # 5. Soft-hyphen removal
    result = _remove_soft_hyphens(result)

    # 6. Dehyphenate line breaks (before per-line whitespace normalization)
    if options.dehyphenate_line_breaks:
        result = _dehyphenate_line_breaks(result)

    # 7. Per-line horizontal whitespace normalization
    result = _normalize_lines(
        result,
        collapse_whitespace=options.collapse_horizontal_whitespace,
        trim=options.trim_lines,
    )

    # 8. Blank-line normalization
    if options.collapse_blank_lines:
        result = _collapse_blank_lines(result, options.max_blank_lines)

    # 9. Conservative OCR separator-line cleanup; re-run blank-line collapse
    # afterward so that gaps left by removed separator lines are also bounded.
    result = _remove_separator_lines(result)
    if options.collapse_blank_lines:
        result = _collapse_blank_lines(result, options.max_blank_lines)

    # 10. Final trim
    result = result.strip()

    output_char_count = len(result)
    output_line_count = result.count("\n") + 1 if result else 0

    # Diagnostics
    if input_char_count > 0:
        removal_ratio = (input_char_count - output_char_count) / input_char_count
        if removal_ratio > warn_removal_ratio:
            warnings.append(
                TextNormalizationWarning(
                    code="HIGH_REMOVAL_RATIO",
                    message=(
                        f"Normalization removed {removal_ratio:.1%} of input text"
                    ),
                    details={
                        "removal_ratio": round(removal_ratio, 4),
                        "input_char_count": input_char_count,
                        "output_char_count": output_char_count,
                    },
                )
            )

        if output_char_count == 0:
            warnings.append(
                TextNormalizationWarning(
                    code="EMPTY_OUTPUT",
                    message="Normalization produced empty output from non-empty input",
                    details={"input_char_count": input_char_count},
                )
            )

    return TextNormalizationResult(
        normalized_text=result,
        input_character_count=input_char_count,
        output_character_count=output_char_count,
        input_line_count=input_line_count,
        output_line_count=output_line_count,
        changed=(result != text),
        rule_set_version=RULE_SET_VERSION,
        warnings=warnings,
    )
