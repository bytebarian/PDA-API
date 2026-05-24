"""Unit tests for the chunking service algorithm and settings validation."""

from __future__ import annotations

import pytest

from app.services.chunking_service import (
    ChunkingValidationError,
    chunk_text,
    validate_chunk_settings,
)


# ---------------------------------------------------------------------------
# validate_chunk_settings
# ---------------------------------------------------------------------------


def test_validate_chunk_settings_valid() -> None:
    validate_chunk_settings(1000, 150)  # No exception expected.


def test_validate_chunk_settings_chunk_size_zero_raises() -> None:
    with pytest.raises(ChunkingValidationError, match="chunkSize must be > 0"):
        validate_chunk_settings(0, 0)


def test_validate_chunk_settings_chunk_size_negative_raises() -> None:
    with pytest.raises(ChunkingValidationError, match="chunkSize must be > 0"):
        validate_chunk_settings(-1, 0)


def test_validate_chunk_settings_negative_overlap_raises() -> None:
    with pytest.raises(ChunkingValidationError, match="chunkOverlap must be >= 0"):
        validate_chunk_settings(1000, -1)


def test_validate_chunk_settings_overlap_equals_size_raises() -> None:
    with pytest.raises(ChunkingValidationError, match="must be < chunkSize"):
        validate_chunk_settings(100, 100)


def test_validate_chunk_settings_overlap_exceeds_size_raises() -> None:
    with pytest.raises(ChunkingValidationError, match="must be < chunkSize"):
        validate_chunk_settings(100, 200)


def test_validate_chunk_settings_overlap_zero_is_valid() -> None:
    validate_chunk_settings(100, 0)  # No exception expected.


# ---------------------------------------------------------------------------
# chunk_text – empty/whitespace input
# ---------------------------------------------------------------------------


def test_chunk_text_empty_string_returns_empty() -> None:
    result = chunk_text("", 1000, 150)
    assert result == []


def test_chunk_text_whitespace_only_returns_empty() -> None:
    result = chunk_text("   \n\t  ", 1000, 150)
    assert result == []


def test_chunk_text_newline_only_returns_empty() -> None:
    result = chunk_text("\n\n\n", 1000, 150)
    assert result == []


# ---------------------------------------------------------------------------
# chunk_text – small document (text <= chunk_size)
# ---------------------------------------------------------------------------


def test_chunk_text_short_text_single_chunk() -> None:
    text = "Hello, world!"
    result = chunk_text(text, 1000, 150)
    assert len(result) == 1
    chunk = result[0]
    assert chunk.chunk_index == 0
    assert chunk.content == "Hello, world!"
    assert chunk.start_offset == 0
    assert chunk.end_offset == len(text)


def test_chunk_text_text_equal_to_chunk_size() -> None:
    text = "A" * 100
    result = chunk_text(text, 100, 10)
    assert len(result) == 1
    assert result[0].chunk_index == 0
    assert result[0].start_offset == 0
    assert result[0].end_offset == 100


# ---------------------------------------------------------------------------
# chunk_text – chunk ordering and indexing
# ---------------------------------------------------------------------------


def test_chunk_text_indexes_are_sequential() -> None:
    text = "word " * 500  # 2500 chars
    result = chunk_text(text, 200, 20)
    for i, chunk in enumerate(result):
        assert chunk.chunk_index == i


def test_chunk_text_no_duplicate_indices() -> None:
    text = "alpha beta gamma delta " * 100
    result = chunk_text(text, 200, 50)
    indices = [c.chunk_index for c in result]
    assert indices == list(range(len(result)))


# ---------------------------------------------------------------------------
# chunk_text – determinism
# ---------------------------------------------------------------------------


def test_chunk_text_is_deterministic() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 50
    first = chunk_text(text, 100, 20)
    second = chunk_text(text, 100, 20)
    assert first == second


def test_chunk_text_same_settings_same_output() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    r1 = chunk_text(text, 30, 5)
    r2 = chunk_text(text, 30, 5)
    assert r1 == r2


# ---------------------------------------------------------------------------
# chunk_text – valid offsets / content reconstruction
# ---------------------------------------------------------------------------


def test_chunk_text_offsets_reconstruct_content() -> None:
    """Slicing normalized text by [start_offset:end_offset] must yield .content (stripped)."""
    text = "First sentence. Second sentence.\nThird sentence.\n\nFourth paragraph."
    # Normalise to match the service's internal normalisation.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    result = chunk_text(text, 30, 5)
    for chunk in result:
        sliced = normalized[chunk.start_offset : chunk.end_offset].strip()
        assert sliced == chunk.content


def test_chunk_text_offsets_non_overlapping_order() -> None:
    """Each chunk's start_offset must be >= previous chunk's start_offset."""
    text = "word " * 300
    result = chunk_text(text, 100, 20)
    for i in range(1, len(result)):
        assert result[i].start_offset >= result[i - 1].start_offset


def test_chunk_text_end_offset_after_start() -> None:
    text = "Hello world. " * 20
    result = chunk_text(text, 50, 10)
    for chunk in result:
        assert chunk.end_offset > chunk.start_offset


# ---------------------------------------------------------------------------
# chunk_text – overlap handling
# ---------------------------------------------------------------------------


def test_chunk_text_overlap_zero_no_shared_content() -> None:
    text = "ABCDEFGHIJ" * 10  # 100 chars, no whitespace
    result = chunk_text(text, 30, 0)
    # Each chunk's start should be >= previous chunk's end (no overlap with 0 overlap)
    for i in range(1, len(result)):
        assert result[i].start_offset >= result[i - 1].end_offset


def test_chunk_text_adjacent_chunks_overlap_with_positive_overlap() -> None:
    """Adjacent chunks should share content when chunk_overlap > 0."""
    text = "The quick brown fox jumps over the lazy dog. " * 10
    result = chunk_text(text, 80, 20)
    if len(result) >= 2:
        # The second chunk should start before the first chunk ends.
        assert result[1].start_offset < result[0].end_offset


def test_chunk_text_overlap_does_not_cause_infinite_loop() -> None:
    """High overlap should still terminate and produce ordered chunks."""
    text = "word " * 200  # 1000 chars
    result = chunk_text(text, 50, 49)  # overlap = chunk_size - 1
    assert len(result) > 0
    for i in range(1, len(result)):
        assert result[i].start_offset > result[i - 1].start_offset


# ---------------------------------------------------------------------------
# chunk_text – boundary-aware splitting
# ---------------------------------------------------------------------------


def test_chunk_text_splits_on_paragraph_boundary() -> None:
    """A double newline within the window should be preferred."""
    para1 = "A" * 80
    para2 = "B" * 80
    text = para1 + "\n\n" + para2
    result = chunk_text(text, 100, 0)
    # The split should happen at the paragraph boundary (after para1).
    assert all("\n\n" not in chunk.content for chunk in result)
    # First chunk should not contain para2 content.
    assert "B" not in result[0].content


def test_chunk_text_splits_on_sentence_boundary() -> None:
    """Sentence terminators ('. ') should be preferred over mid-word hard splits."""
    text = "First sentence. " + "Second sentence. " + "Third sentence. " * 5
    result = chunk_text(text, 40, 0)
def test_chunk_text_no_whitespace_text_hard_splits() -> None:
    """Text with no whitespace is hard-split at chunk_size."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    sentence_markers = (". ", "! ", "? ")
    text = "A" * 250
    # Every non-final chunk should end at a sentence boundary (marker includes trailing space).
    for chunk in result[:-1]:
        assert normalized[chunk.end_offset - 2 : chunk.end_offset] in sentence_markers


    result = chunk_text(text, 100, 0)
    assert len(result) == 3  # 100, 100, 50
    for chunk in result:
        assert len(chunk.content) <= 100


# ---------------------------------------------------------------------------
# chunk_text – CRLF normalisation
# ---------------------------------------------------------------------------


def test_chunk_text_crlf_normalised() -> None:
    text = "line one\r\nline two\r\nline three"
    result = chunk_text(text, 1000, 0)
    assert len(result) == 1
    assert "\r" not in result[0].content


# ---------------------------------------------------------------------------
# chunk_text – invalid settings propagation
# ---------------------------------------------------------------------------


def test_chunk_text_invalid_settings_raises_validation_error() -> None:
    with pytest.raises(ChunkingValidationError):
        chunk_text("some text", 0, 0)


def test_chunk_text_overlap_equals_size_raises() -> None:
    with pytest.raises(ChunkingValidationError):
        chunk_text("some text", 100, 100)
