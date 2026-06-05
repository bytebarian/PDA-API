"""Unit tests for CitationBuilder service."""

from __future__ import annotations

import uuid
from typing import Literal

from app.schemas.context import ContextSource, IncludedTextRange
from app.schemas.hybrid_search import HybridSearchResult
from app.services.citation_builder import CitationBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(
    source_id: str = "S1",
    *,
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    document_name: str = "contract.pdf",
    page_number: int | None = 3,
    chunk_index: int = 7,
    score: float | None = 0.9,
    included_text_range: IncludedTextRange | None = None,
) -> ContextSource:
    return ContextSource(
        source_id=source_id,
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        document_name=document_name,
        document_path="/docs/contract.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        score=score,
        included_text_range=included_text_range,
        excerpt_only=False,
    )


def _make_hybrid_result(
    chunk_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    text: str = "Sample chunk text.",
    excerpt: str | None = None,
    score: float = 0.9,
    matched_by: list[Literal["vector", "full_text"]] | None = None,
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name="contract.pdf",
        document_path="/docs/contract.pdf",
        category="contracts",
        file_type="pdf",
        page_number=3,
        chunk_index=7,
        start_offset=0,
        end_offset=len(text),
        text=text,
        excerpt=excerpt or text,
        score=score,
        matched_by=matched_by or ["vector", "full_text"],
        metadata={},
    )


# ---------------------------------------------------------------------------
# extract_source_markers
# ---------------------------------------------------------------------------

def test_extract_single_marker() -> None:
    builder = CitationBuilder()
    assert builder.extract_source_markers("See [S1] for details.") == ["S1"]


def test_extract_multiple_markers_in_order() -> None:
    builder = CitationBuilder()
    assert builder.extract_source_markers("See [S2] and [S1].") == ["S2", "S1"]


def test_extract_deduplicates_repeated_markers() -> None:
    builder = CitationBuilder()
    assert builder.extract_source_markers("[S1] foo [S2] bar [S1]") == ["S1", "S2"]


def test_extract_no_markers_returns_empty() -> None:
    builder = CitationBuilder()
    assert builder.extract_source_markers("No citations here.") == []


def test_extract_ignores_non_citation_brackets() -> None:
    builder = CitationBuilder()
    assert builder.extract_source_markers("[Note] and [TODO] are not markers") == []


def test_extract_handles_multidigit_source_ids() -> None:
    builder = CitationBuilder()
    result = builder.extract_source_markers("See [S12] and [S3].")
    assert result == ["S12", "S3"]


def test_extract_handles_adjacent_markers() -> None:
    """[S1][S2] without space must still parse both markers."""
    builder = CitationBuilder()
    assert builder.extract_source_markers("[S1][S2]") == ["S1", "S2"]


# ---------------------------------------------------------------------------
# build_from_sources – answer-based citations
# ---------------------------------------------------------------------------

def test_build_from_sources_maps_single_marker() -> None:
    builder = CitationBuilder()
    source = _make_source("S1")
    result_item = _make_hybrid_result(source.chunk_id, source.document_id, text="Notice period is three months.")

    citations, diag = builder.build_from_sources(
        [source],
        answer_text="The notice period is three months. [S1]",
        retrieval_results=[result_item],
    )

    assert len(citations) == 1
    assert citations[0].source_id == "S1"
    assert citations[0].citation_index == 1
    assert citations[0].document_id == source.document_id
    assert citations[0].document_name == source.document_name
    assert citations[0].chunk_id == source.chunk_id
    assert citations[0].chunk_index == source.chunk_index
    assert "three months" in citations[0].excerpt
    assert diag.citation_count == 1
    assert diag.source_markers_found == ["S1"]
    assert diag.unknown_source_markers == []


def test_build_from_sources_orders_citations_by_marker_appearance() -> None:
    builder = CitationBuilder()
    s1 = _make_source("S1")
    s2 = _make_source("S2")
    r1 = _make_hybrid_result(s1.chunk_id, s1.document_id, text="First.")
    r2 = _make_hybrid_result(s2.chunk_id, s2.document_id, text="Second.")

    citations, _ = builder.build_from_sources(
        [s1, s2],
        answer_text="See [S2] and then [S1].",
        retrieval_results=[r1, r2],
    )

    assert len(citations) == 2
    assert citations[0].source_id == "S2"
    assert citations[1].source_id == "S1"
    assert citations[0].citation_index == 1
    assert citations[1].citation_index == 2


def test_build_from_sources_deduplicates_duplicate_markers() -> None:
    builder = CitationBuilder()
    source = _make_source("S1")
    result_item = _make_hybrid_result(source.chunk_id, source.document_id)

    citations, diag = builder.build_from_sources(
        [source],
        answer_text="See [S1] and [S1] again.",
        retrieval_results=[result_item],
    )

    assert len(citations) == 1
    assert diag.duplicate_markers_ignored == ["S1"]


def test_build_from_sources_ignores_unknown_markers() -> None:
    builder = CitationBuilder()
    source = _make_source("S1")
    result_item = _make_hybrid_result(source.chunk_id, source.document_id)

    citations, diag = builder.build_from_sources(
        [source],
        answer_text="[S99] is unknown, [S1] is real.",
        retrieval_results=[result_item],
    )

    assert len(citations) == 1
    assert citations[0].source_id == "S1"
    assert "S99" in diag.unknown_source_markers


def test_build_from_sources_no_fabricated_citation_for_missing_source_id() -> None:
    """S99 marker with no matching source must return zero citations for S99."""
    builder = CitationBuilder()
    citations, diag = builder.build_from_sources(
        [],
        answer_text="See [S99].",
    )

    assert citations == []
    assert "S99" in diag.unknown_source_markers


def test_build_from_sources_returns_all_sources_when_no_answer_text() -> None:
    builder = CitationBuilder()
    s1 = _make_source("S1")
    s2 = _make_source("S2")

    citations, _ = builder.build_from_sources([s1, s2])

    assert len(citations) == 2
    assert citations[0].source_id == "S1"
    assert citations[1].source_id == "S2"


def test_build_from_sources_include_uncited_sources() -> None:
    builder = CitationBuilder()
    s1 = _make_source("S1")
    s2 = _make_source("S2")
    r1 = _make_hybrid_result(s1.chunk_id, s1.document_id)
    r2 = _make_hybrid_result(s2.chunk_id, s2.document_id)

    citations, _ = builder.build_from_sources(
        [s1, s2],
        answer_text="See [S1].",
        include_uncited_sources=True,
        retrieval_results=[r1, r2],
    )

    assert len(citations) == 2
    # Cited first
    assert citations[0].source_id == "S1"
    # Uncited appended
    assert citations[1].source_id == "S2"


def test_build_from_sources_preserves_document_metadata() -> None:
    builder = CitationBuilder()
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    source = _make_source(
        "S1",
        chunk_id=chunk_id,
        document_id=doc_id,
        document_name="employment-contract.pdf",
        page_number=3,
        chunk_index=7,
    )
    result_item = _make_hybrid_result(chunk_id, doc_id, text="Notice period.")

    citations, _ = builder.build_from_sources(
        [source],
        answer_text="See [S1].",
        retrieval_results=[result_item],
    )

    assert citations[0].document_id == doc_id
    assert citations[0].document_name == "employment-contract.pdf"
    assert citations[0].chunk_id == chunk_id
    assert citations[0].page_number == 3
    assert citations[0].chunk_index == 7


# ---------------------------------------------------------------------------
# Excerpt generation
# ---------------------------------------------------------------------------

def test_excerpt_uses_retrieval_text() -> None:
    builder = CitationBuilder()
    source = _make_source("S1")
    result_item = _make_hybrid_result(
        source.chunk_id,
        source.document_id,
        text="The notice period is three months from the date of resignation.",
    )

    citations, _ = builder.build_from_sources(
        [source],
        answer_text="[S1]",
        retrieval_results=[result_item],
    )

    assert "notice period" in citations[0].excerpt


def test_excerpt_respects_included_text_range() -> None:
    builder = CitationBuilder()
    text = "IGNORED_PREFIX The notice period is three months. IGNORED_SUFFIX"
    start = text.index("The notice")
    end = text.index(". IGNORED_SUFFIX")
    source = _make_source(
        "S1",
        included_text_range=IncludedTextRange(start=start, end=end),
    )
    result_item = _make_hybrid_result(source.chunk_id, source.document_id, text=text)

    citations, _ = builder.build_from_sources(
        [source],
        answer_text="[S1]",
        retrieval_results=[result_item],
    )

    assert "IGNORED_PREFIX" not in citations[0].excerpt
    assert "notice period" in citations[0].excerpt


def test_excerpt_truncated_with_ellipsis() -> None:
    builder = CitationBuilder()
    long_text = "word " * 200  # way over 500 chars
    source = _make_source("S1")
    result_item = _make_hybrid_result(source.chunk_id, source.document_id, text=long_text)

    citations, diag = builder.build_from_sources(
        [source],
        answer_text="[S1]",
        max_excerpt_characters=50,
        retrieval_results=[result_item],
    )

    assert len(citations[0].excerpt) <= 55  # allow ellipsis overhead
    assert citations[0].excerpt.endswith("…")
    assert diag.excerpts_truncated == 1


def test_excerpt_preserves_utf8_characters() -> None:
    builder = CitationBuilder()
    text = "Wypowiedzenie umowy pracowniczej wynosi trzy miesiące. " * 20
    source = _make_source("S1")
    result_item = _make_hybrid_result(source.chunk_id, source.document_id, text=text)

    citations, _ = builder.build_from_sources(
        [source],
        answer_text="[S1]",
        max_excerpt_characters=80,
        retrieval_results=[result_item],
    )

    # Excerpt must be valid Python string (no broken multibyte sequences)
    assert isinstance(citations[0].excerpt, str)
    citations[0].excerpt.encode("utf-8")  # must not raise


def test_excerpt_never_empty_when_text_exists() -> None:
    builder = CitationBuilder()
    source = _make_source("S1")
    result_item = _make_hybrid_result(source.chunk_id, source.document_id, text="Short text.")

    citations, _ = builder.build_from_sources(
        [source],
        answer_text="[S1]",
        retrieval_results=[result_item],
    )

    assert citations[0].excerpt != ""


# ---------------------------------------------------------------------------
# build_from_retrieval_results
# ---------------------------------------------------------------------------

def test_build_from_retrieval_results_assigns_sequential_source_ids() -> None:
    builder = CitationBuilder()
    doc_id = uuid.uuid4()
    r1 = _make_hybrid_result(uuid.uuid4(), doc_id, score=0.9)
    r2 = _make_hybrid_result(uuid.uuid4(), doc_id, score=0.7)

    citations, _ = builder.build_from_retrieval_results([r1, r2])

    assert citations[0].source_id == "S1"
    assert citations[1].source_id == "S2"
    assert citations[0].citation_index == 1
    assert citations[1].citation_index == 2


def test_build_from_retrieval_results_orders_by_score_desc() -> None:
    builder = CitationBuilder()
    doc_id = uuid.uuid4()
    low = _make_hybrid_result(uuid.uuid4(), doc_id, text="Low relevance.", score=0.4)
    high = _make_hybrid_result(uuid.uuid4(), doc_id, text="High relevance.", score=0.95)

    citations, _ = builder.build_from_retrieval_results([low, high])

    assert citations[0].score == 0.95
    assert citations[1].score == 0.4


def test_build_from_retrieval_results_deduplicates_chunks() -> None:
    builder = CitationBuilder()
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    dup1 = _make_hybrid_result(chunk_id, doc_id, text="Original.", score=0.9)
    dup2 = _make_hybrid_result(chunk_id, doc_id, text="Duplicate.", score=0.85)

    citations, _ = builder.build_from_retrieval_results([dup1, dup2])

    assert len(citations) == 1
    assert citations[0].chunk_id == chunk_id


def test_build_from_retrieval_results_includes_document_fields() -> None:
    builder = CitationBuilder()
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    result = _make_hybrid_result(chunk_id, doc_id, text="Content.", score=0.8)

    citations, _ = builder.build_from_retrieval_results([result])

    assert citations[0].document_id == doc_id
    assert citations[0].chunk_id == chunk_id
    assert citations[0].document_name == result.document_name
    assert citations[0].chunk_index == result.chunk_index


def test_build_from_retrieval_results_sets_hybrid_relevance_source() -> None:
    builder = CitationBuilder()
    result = _make_hybrid_result(uuid.uuid4(), uuid.uuid4(), matched_by=["vector", "full_text"])

    citations, _ = builder.build_from_retrieval_results([result])

    assert citations[0].relevance_source == "hybrid"


def test_build_from_retrieval_results_sets_vector_relevance_source() -> None:
    builder = CitationBuilder()
    result = _make_hybrid_result(uuid.uuid4(), uuid.uuid4(), matched_by=["vector"])

    citations, _ = builder.build_from_retrieval_results([result])

    assert citations[0].relevance_source == "vector"


def test_build_from_retrieval_results_sets_full_text_relevance_source() -> None:
    builder = CitationBuilder()
    result = _make_hybrid_result(uuid.uuid4(), uuid.uuid4(), matched_by=["full_text"])

    citations, _ = builder.build_from_retrieval_results([result])

    assert citations[0].relevance_source == "full_text"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_track_unknown_markers() -> None:
    builder = CitationBuilder()
    source = _make_source("S1")

    _, diag = builder.build_from_sources(
        [source],
        answer_text="[S1] and [S99].",
    )

    assert "S99" in diag.unknown_source_markers
    assert "S1" not in diag.unknown_source_markers


def test_diagnostics_track_duplicate_markers() -> None:
    builder = CitationBuilder()
    source = _make_source("S1")

    _, diag = builder.build_from_sources(
        [source],
        answer_text="[S1] [S1] [S1].",
    )

    assert diag.duplicate_markers_ignored == ["S1", "S1"]


def test_diagnostics_sources_available_count() -> None:
    builder = CitationBuilder()
    sources = [_make_source("S1"), _make_source("S2"), _make_source("S3")]

    _, diag = builder.build_from_sources(sources, answer_text="[S1].")

    assert diag.sources_available == 3


def test_diagnostics_citation_count() -> None:
    builder = CitationBuilder()
    sources = [_make_source("S1"), _make_source("S2")]

    _, diag = builder.build_from_sources(sources, answer_text="[S1] [S2].")

    assert diag.citation_count == 2


def test_diagnostics_excerpts_truncated_count() -> None:
    builder = CitationBuilder()
    long_text = "x " * 300  # > 500 chars
    source = _make_source("S1")
    result_item = _make_hybrid_result(source.chunk_id, source.document_id, text=long_text)

    _, diag = builder.build_from_sources(
        [source],
        answer_text="[S1]",
        max_excerpt_characters=100,
        retrieval_results=[result_item],
    )

    assert diag.excerpts_truncated == 1
