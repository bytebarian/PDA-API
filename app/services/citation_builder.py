"""Reusable citation builder service for document references and excerpts."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from app.schemas.citations import Citation, CitationDiagnostics
from app.services.search_service import _make_excerpt

if TYPE_CHECKING:
    from app.schemas.context import ContextSource
    from app.schemas.hybrid_search import HybridSearchResult
    from app.schemas.search import SemanticSearchResult

_SOURCE_MARKER_RE = re.compile(r"\[(S\d+)\]")

_DEFAULT_MAX_EXCERPT_CHARACTERS = 500

_RelevanceSource = Literal["vector", "full_text", "hybrid", "manual", "context"]


def _relevance_source_from_matched_by(
    matched_by: list[str] | None,
) -> _RelevanceSource | None:
    """Derive a relevance_source label from a HybridSearchResult matched_by list."""
    if not matched_by:
        return None
    has_vector = "vector" in matched_by
    has_full_text = "full_text" in matched_by
    if has_vector and has_full_text:
        return "hybrid"
    if has_vector:
        return "vector"
    if has_full_text:
        return "full_text"
    return None


class CitationBuilder:
    """Build normalized citations from context sources or retrieval results.

    This service is the single authoritative citation-building layer used by
    chat, report, and search-preview endpoints.  It replaces ad-hoc citation
    mapping that was previously embedded in individual endpoint handlers.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_source_markers(self, text: str) -> list[str]:
        """Return unique source marker IDs in the order they appear in *text*.

        Duplicate markers are silently deduplicated; only the first occurrence
        determines position in the returned list.

        Examples::

            >>> builder.extract_source_markers("See [S1] and [S2][S1].")
            ['S1', 'S2']
        """
        ordered: OrderedDict[str, None] = OrderedDict.fromkeys(
            match.group(1) for match in _SOURCE_MARKER_RE.finditer(text)
        )
        return list(ordered.keys())

    def build_from_sources(
        self,
        sources: Sequence[ContextSource],
        *,
        answer_text: str | None = None,
        include_uncited_sources: bool = False,
        max_excerpt_characters: int = _DEFAULT_MAX_EXCERPT_CHARACTERS,
        retrieval_results: Sequence[HybridSearchResult | SemanticSearchResult] | None = None,
    ) -> tuple[list[Citation], CitationDiagnostics]:
        """Build citations from context-builder sources.

        When *answer_text* is given, only markers present in the answer are
        returned (unless *include_uncited_sources* is ``True``).  When no
        answer text is provided, all sources are returned as citations.

        Args:
            sources: Ordered list of :class:`ContextSource` objects from the
                context builder.
            answer_text: Optional model-generated answer containing ``[S1]``
                style markers.
            include_uncited_sources: When ``True``, sources that are not
                referenced in *answer_text* are appended after cited sources.
            max_excerpt_characters: Maximum length of each citation excerpt.
            retrieval_results: Optional retrieval results used to look up
                chunk text when building excerpts (``ContextSource`` does not
                carry the raw text).

        Returns:
            A ``(citations, diagnostics)`` tuple.
        """
        source_map = {source.source_id: source for source in sources}
        result_lookup: dict[object, HybridSearchResult | SemanticSearchResult] = {}
        if retrieval_results is not None:
            result_lookup = {r.chunk_id: r for r in retrieval_results}

        diag = CitationDiagnostics(sources_available=len(source_map))

        if answer_text is not None:
            raw_markers = [m.group(1) for m in _SOURCE_MARKER_RE.finditer(answer_text)]
            seen: set[str] = set()
            unique_markers: list[str] = []
            duplicates: list[str] = []
            for marker in raw_markers:
                if marker in seen:
                    duplicates.append(marker)
                else:
                    seen.add(marker)
                    unique_markers.append(marker)

            diag.source_markers_found = unique_markers
            diag.duplicate_markers_ignored = duplicates
            diag.unknown_source_markers = [m for m in unique_markers if m not in source_map]

            cited_source_ids = [m for m in unique_markers if m in source_map]
            cited_sources = [source_map[sid] for sid in cited_source_ids]

            if include_uncited_sources:
                cited_set = set(cited_source_ids)
                uncited = [s for s in sources if s.source_id not in cited_set]
                ordered_sources = cited_sources + uncited
            else:
                ordered_sources = cited_sources
        else:
            ordered_sources = list(sources)

        citations: list[Citation] = []
        for idx, source in enumerate(ordered_sources, start=1):
            retrieval_result = result_lookup.get(source.chunk_id)
            excerpt, truncated = self._excerpt_from_source(
                source,
                retrieval_result=retrieval_result,
                max_excerpt_characters=max_excerpt_characters,
            )
            if truncated:
                diag.excerpts_truncated += 1

            metadata = {}
            relevance_source: _RelevanceSource | None = "context"
            if retrieval_result is not None:
                metadata = getattr(retrieval_result, "metadata", {}) or {}
                if hasattr(retrieval_result, "matched_by"):
                    matched_by = getattr(retrieval_result, "matched_by", None)
                    relevance_source = _relevance_source_from_matched_by(matched_by) or "manual"
                else:
                    relevance_source = "vector"

            citations.append(
                Citation(
                    source_id=source.source_id,
                    citation_index=idx,
                    document_id=source.document_id,
                    document_name=source.document_name,
                    document_path=source.document_path,
                    chunk_id=source.chunk_id,
                    page_number=source.page_number,
                    chunk_index=source.chunk_index,
                    start_offset=source.start_offset,
                    end_offset=source.end_offset,
                    excerpt=excerpt,
                    score=source.score,
                    relevance_source=relevance_source,
                    metadata=metadata,
                )
            )

        diag.citation_count = len(citations)
        return citations, diag

    def build_from_retrieval_results(
        self,
        results: Sequence[HybridSearchResult | SemanticSearchResult],
        *,
        max_excerpt_characters: int = _DEFAULT_MAX_EXCERPT_CHARACTERS,
    ) -> tuple[list[Citation], CitationDiagnostics]:
        """Build citations directly from retrieval results.

        Duplicate chunks (same ``chunk_id``) are deduplicated; the first
        occurrence wins.  Results are ordered by score descending, then
        document ID ascending, then chunk index ascending.

        Args:
            results: Retrieval results from hybrid or semantic search.
            max_excerpt_characters: Maximum length of each citation excerpt.

        Returns:
            A ``(citations, diagnostics)`` tuple.
        """
        seen_chunk_ids: set[object] = set()
        deduped: list[HybridSearchResult | SemanticSearchResult] = []
        for result in results:
            if result.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(result.chunk_id)
                deduped.append(result)

        sorted_results = sorted(
            deduped,
            key=lambda r: (
                -(r.score or 0.0),
                str(r.document_id),
                r.chunk_index,
            ),
        )

        diag = CitationDiagnostics(sources_available=len(sorted_results))
        citations: list[Citation] = []

        for idx, result in enumerate(sorted_results, start=1):
            source_id = f"S{idx}"
            text = getattr(result, "text", None) or result.excerpt or ""
            excerpt, truncated = self._truncate_excerpt(text, max_excerpt_characters)
            if truncated:
                diag.excerpts_truncated += 1

            matched_by: list[str] | None = getattr(result, "matched_by", None)
            relevance_source = _relevance_source_from_matched_by(matched_by)
            if relevance_source is None:
                relevance_source = "vector"

            citations.append(
                Citation(
                    source_id=source_id,
                    citation_index=idx,
                    document_id=result.document_id,
                    document_name=result.document_name,
                    document_path=getattr(result, "document_path", None),
                    chunk_id=result.chunk_id,
                    page_number=result.page_number,
                    chunk_index=result.chunk_index,
                    start_offset=result.start_offset,
                    end_offset=result.end_offset,
                    excerpt=excerpt,
                    score=result.score,
                    relevance_source=relevance_source,
                    metadata=getattr(result, "metadata", {}),
                )
            )

        diag.citation_count = len(citations)
        return citations, diag

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _excerpt_from_source(
        self,
        source: ContextSource,
        *,
        retrieval_result: HybridSearchResult | SemanticSearchResult | None = None,
        max_excerpt_characters: int,
    ) -> tuple[str, bool]:
        """Return ``(excerpt_text, was_truncated)`` for a context source.

        Tries, in order:
        1. Raw text from the matching retrieval result, sliced by
           ``source.included_text_range``.
        2. Excerpt string from the retrieval result.
        3. Any ``excerpt`` or ``text`` attribute on the source itself
           (for forward-compatible callers that annotate sources with text).
        4. Empty string as a safe fallback.
        """
        included_range = source.included_text_range
        raw: str = ""

        if retrieval_result is not None:
            raw = getattr(retrieval_result, "text", None) or getattr(retrieval_result, "excerpt", None) or ""
        if not raw:
            raw = getattr(source, "text", None) or getattr(source, "excerpt", None) or ""

        if included_range is not None and raw:
            raw = raw[included_range.start : included_range.end]

        return self._truncate_excerpt(raw, max_excerpt_characters)

    @staticmethod
    def _truncate_excerpt(text: str, max_chars: int) -> tuple[str, bool]:
        """Return ``(excerpt, truncated)`` enforcing *max_chars* limit.

        Truncation honours word boundaries and appends ``…`` (U+2026).
        """
        text = " ".join(text.split())  # normalise whitespace
        if not text:
            return "", False

        if len(text) <= max_chars:
            return text, False

        truncated = _make_excerpt(text, max_chars)
        return truncated, True
