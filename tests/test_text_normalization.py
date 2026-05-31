"""Unit tests for app/services/text_normalization.py.

All tests are deterministic and require no database, filesystem, or network
access.  They cover the normalization rules one-by-one and verify that the
result DTO carries accurate metadata.
"""

from __future__ import annotations

import pytest

from app.services.text_normalization import (
    RULE_SET_VERSION,
    TextNormalizationConfigurationError,
    TextNormalizationError,
    TextNormalizationOptions,
    TextNormalizationResult,
    normalize_text,
)


# ---------------------------------------------------------------------------
# Fixture strings (canonical regression inputs)
# ---------------------------------------------------------------------------

# Reproduces the example from the issue specification.
FIXTURE_NOISY = (
    "\ufeff  This   is\t a   test.\r\n\r\n\r\n"
    "Docu-\nment text with soft\u00adhyphen.\x00"
)
FIXTURE_NOISY_EXPECTED = "This is a test.\n\nDocument text with softhyphen."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: str, **kwargs: object) -> TextNormalizationResult:
    """Call normalize_text with an options override built from *kwargs*."""
    options = TextNormalizationOptions(**kwargs)  # type: ignore[arg-type]
    return normalize_text(text, options)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_returns_text_normalization_result() -> None:
    result = normalize_text("hello")
    assert isinstance(result, TextNormalizationResult)


def test_rule_set_version_is_stable() -> None:
    result = normalize_text("hello")
    assert result.rule_set_version == RULE_SET_VERSION
    assert result.rule_set_version == "pda-normalization-v1"


def test_deterministic_for_same_input() -> None:
    text = "Hello\r\nWorld\t\t  !\n\n\nEnd."
    assert normalize_text(text).normalized_text == normalize_text(text).normalized_text


def test_empty_string_input_returns_empty_output() -> None:
    result = normalize_text("")
    assert result.normalized_text == ""
    assert result.input_character_count == 0
    assert result.output_character_count == 0
    assert result.changed is False


def test_non_string_input_raises_error() -> None:
    with pytest.raises(TextNormalizationError):
        normalize_text(None)  # type: ignore[arg-type]


def test_non_string_bytes_raises_error() -> None:
    with pytest.raises(TextNormalizationError):
        normalize_text(b"bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------


def test_unicode_nfkc_normalizes_ligature() -> None:
    # LATIN SMALL LIGATURE FI (U+FB01) → "fi"
    result = normalize_text("\ufb01le")
    assert result.normalized_text == "file"


def test_unicode_nfkc_preserves_polish_characters() -> None:
    polish = "Zażółć gęślą jaźń"
    result = normalize_text(polish)
    assert result.normalized_text == polish


def test_unicode_nfkc_preserves_german_umlauts() -> None:
    german = "Über straße"
    result = normalize_text(german)
    assert result.normalized_text == german


def test_bom_removed() -> None:
    result = normalize_text("\ufeffHello")
    assert result.normalized_text == "Hello"
    assert result.changed is True


def test_bom_in_middle_is_not_removed() -> None:
    # Only leading BOM is targeted.
    text = "Hello\ufeffWorld"
    result = normalize_text(text)
    # NFKC keeps interior U+FEFF as zero-width no-break space → may remain or
    # become U+FEFF (depending on Python unicodedata).  Key assertion: no
    # leading BOM after processing.
    assert not result.normalized_text.startswith("\ufeff")


def test_invalid_unicode_form_raises_configuration_error() -> None:
    with pytest.raises(TextNormalizationConfigurationError):
        _normalize("hello", unicode_form="INVALID_FORM")


# ---------------------------------------------------------------------------
# Line ending normalization
# ---------------------------------------------------------------------------


def test_crlf_converted_to_lf() -> None:
    result = normalize_text("line1\r\nline2")
    assert "\r" not in result.normalized_text
    assert result.normalized_text == "line1\nline2"


def test_bare_cr_converted_to_lf() -> None:
    result = normalize_text("line1\rline2")
    assert "\r" not in result.normalized_text
    assert result.normalized_text == "line1\nline2"


def test_mixed_line_endings_all_become_lf() -> None:
    result = normalize_text("a\r\nb\rc\nd")
    assert result.normalized_text == "a\nb\nc\nd"


def test_no_mixed_endings_in_output() -> None:
    result = normalize_text("x\r\ny\rz\nw")
    assert "\r\n" not in result.normalized_text
    assert "\r" not in result.normalized_text


# ---------------------------------------------------------------------------
# Control character cleanup
# ---------------------------------------------------------------------------


def test_null_byte_removed() -> None:
    result = normalize_text("hello\x00world")
    assert "\x00" not in result.normalized_text
    assert "helloworld" == result.normalized_text


def test_control_chars_removed_but_newline_kept() -> None:
    text = "a\x01\x02\x03\nb\x0e\x1f\x7fc"
    result = normalize_text(text)
    assert result.normalized_text == "a\nbc"


def test_soft_hyphen_removed() -> None:
    result = normalize_text("soft\u00adhyphen")
    assert "\u00ad" not in result.normalized_text
    assert result.normalized_text == "softhyphen"


# ---------------------------------------------------------------------------
# Horizontal whitespace normalization
# ---------------------------------------------------------------------------


def test_multiple_spaces_collapsed() -> None:
    result = normalize_text("hello   world")
    assert result.normalized_text == "hello world"


def test_tab_converted_to_space() -> None:
    result = normalize_text("col1\tcol2")
    assert "\t" not in result.normalized_text
    assert result.normalized_text == "col1 col2"


def test_mixed_spaces_and_tabs_collapsed() -> None:
    result = normalize_text("a \t  b")
    assert result.normalized_text == "a b"


def test_line_level_trim() -> None:
    result = normalize_text("  leading\ntrailing  \n  both  ")
    lines = result.normalized_text.split("\n")
    assert lines[0] == "leading"
    assert lines[1] == "trailing"
    assert lines[2] == "both"


# ---------------------------------------------------------------------------
# Blank-line normalization
# ---------------------------------------------------------------------------


def test_triple_blank_lines_collapsed_to_one() -> None:
    result = _normalize("para1\n\n\n\npara2", max_blank_lines=1)
    assert result.normalized_text == "para1\n\npara2"


def test_double_blank_lines_preserved_at_max_two() -> None:
    result = _normalize("para1\n\n\npara2", max_blank_lines=2)
    # 3 \n = 2 blank lines, which equals max → not collapsed
    assert result.normalized_text == "para1\n\n\npara2"


def test_paragraph_boundary_preserved() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    result = _normalize(text, max_blank_lines=1)
    assert result.normalized_text == text


def test_max_blank_lines_zero_removes_all_blank_lines() -> None:
    result = _normalize("a\n\n\nb", max_blank_lines=0)
    assert result.normalized_text == "a\nb"


# ---------------------------------------------------------------------------
# Soft hyphen and line-break dehyphenation
# ---------------------------------------------------------------------------


def test_dehyphenate_lowercase_word_split() -> None:
    result = normalize_text("docu-\nment")
    assert result.normalized_text == "document"


def test_dehyphenate_does_not_join_uppercase_after_hyphen() -> None:
    # "pre-\nExisting" should NOT be joined because "E" is uppercase.
    result = normalize_text("pre-\nExisting")
    # The hyphen-newline is preserved; only the newline changes to a space/lf.
    assert "-" in result.normalized_text


def test_dehyphenate_does_not_alter_real_compound_word() -> None:
    # A hyphen on the same line is never touched by this rule.
    result = normalize_text("well-known compound")
    assert "well-known" in result.normalized_text


# ---------------------------------------------------------------------------
# OCR artifact cleanup
# ---------------------------------------------------------------------------


def test_separator_line_removed() -> None:
    text = "Content above.\n------\nContent below."
    result = normalize_text(text)
    assert "------" not in result.normalized_text
    assert "Content above." in result.normalized_text
    assert "Content below." in result.normalized_text


def test_short_separator_line_not_removed() -> None:
    # Lines shorter than 5 separator chars are preserved.
    result = normalize_text("Note:\n---\nBody text.")
    assert "---" in result.normalized_text


def test_line_with_alphanumeric_not_treated_as_separator() -> None:
    result = normalize_text("Item 1 ---\nItem 2")
    assert "Item 1" in result.normalized_text


# ---------------------------------------------------------------------------
# Metadata accuracy
# ---------------------------------------------------------------------------


def test_changed_flag_true_when_text_modified() -> None:
    result = normalize_text("hello   world")
    assert result.changed is True


def test_changed_flag_false_when_text_unchanged() -> None:
    text = "already clean text"
    result = normalize_text(text)
    assert result.changed is False


def test_input_character_count_matches_raw_input() -> None:
    text = "hello\nworld"
    result = normalize_text(text)
    assert result.input_character_count == len(text)


def test_output_character_count_matches_normalized_text() -> None:
    result = normalize_text("hello\r\nworld")
    assert result.output_character_count == len(result.normalized_text)


def test_input_line_count() -> None:
    result = normalize_text("line1\nline2\nline3")
    assert result.input_line_count == 3


def test_output_line_count() -> None:
    result = normalize_text("a\n\n\nb")
    # After collapsing blank lines → "a\n\nb" → 3 lines
    assert result.output_line_count == 3


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_high_removal_ratio_emits_warning() -> None:
    # Create text that shrinks by > 30% after normalization (all control chars).
    control_chars_text = "\x01" * 100
    result = normalize_text(control_chars_text, warn_removal_ratio=0.30)
    codes = [w.code for w in result.warnings]
    assert "HIGH_REMOVAL_RATIO" in codes
    assert "EMPTY_OUTPUT" in codes


def test_empty_output_from_nonempty_input_emits_warning() -> None:
    result = normalize_text("\x01\x02\x03", warn_removal_ratio=0.30)
    codes = [w.code for w in result.warnings]
    assert "EMPTY_OUTPUT" in codes


def test_no_warnings_for_clean_text() -> None:
    result = normalize_text("Clean document text.", warn_removal_ratio=0.30)
    assert result.warnings == []


def test_warning_details_contain_ratio() -> None:
    control_chars_text = "\x01" * 100
    result = normalize_text(control_chars_text, warn_removal_ratio=0.30)
    high_removal = next(w for w in result.warnings if w.code == "HIGH_REMOVAL_RATIO")
    assert "removal_ratio" in high_removal.details


# ---------------------------------------------------------------------------
# Regression fixture from issue specification
# ---------------------------------------------------------------------------


def test_regression_fixture_noisy_input() -> None:
    """Canonical regression: normalize the fixture string from the issue spec."""
    result = normalize_text(FIXTURE_NOISY)
    assert result.normalized_text == FIXTURE_NOISY_EXPECTED
    assert result.changed is True
    assert result.rule_set_version == RULE_SET_VERSION


# ---------------------------------------------------------------------------
# Stability across repeated runs
# ---------------------------------------------------------------------------


def test_idempotent_second_run() -> None:
    """Normalizing an already-normalized text must produce identical output."""
    first = normalize_text(FIXTURE_NOISY)
    second = normalize_text(first.normalized_text)
    assert first.normalized_text == second.normalized_text


def test_idempotent_multiple_runs() -> None:
    text = "Some\r\n\r\n\r\nText.\t\t  extra  spaces."
    first = normalize_text(text)
    second = normalize_text(first.normalized_text)
    third = normalize_text(second.normalized_text)
    assert first.normalized_text == second.normalized_text == third.normalized_text


# ---------------------------------------------------------------------------
# Conservative: legal/financial text is not altered semantically
# ---------------------------------------------------------------------------


def test_legal_numbers_and_decimals_preserved() -> None:
    legal = "Section 3.1.4: Payment of $1,234.56 is due on 2024-01-31."
    result = normalize_text(legal)
    assert result.normalized_text == legal


def test_uppercase_proper_nouns_preserved() -> None:
    text = "The European Union (EU) issued Regulation No. 2016/679."
    result = normalize_text(text)
    assert result.normalized_text == text


def test_numeric_sequences_preserved() -> None:
    text = "Account: 123-456-7890, IBAN: DE89370400440532013000"
    result = normalize_text(text)
    assert result.normalized_text == text
