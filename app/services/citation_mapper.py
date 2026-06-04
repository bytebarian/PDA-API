"""Utilities for extracting and mapping citation markers."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING

from app.schemas.chat import ChatCitation
from app.services.search_service import _make_excerpt

if TYPE_CHECKING:
    from app.schemas.context import ContextSource
    from app.schemas.hybrid_search import HybridSearchResult
    from app.schemas.search import SemanticSearchResult

_SOURCE_ID_RE = re.compile(r"\[(S\d+)\]")


class CitationMapper:
    """Map model citation markers back to included retrieval sources."""

    def extract_source_ids(self, answer: str) -> list[str]:
        """Return source IDs in stable answer order without duplicates."""
        ordered = OrderedDict.fromkeys(match.group(1) for match in _SOURCE_ID_RE.finditer(answer))
        return list(ordered.keys())

    def map_to_citations(
        self,
        source_ids: list[str],
        source_map: Sequence[ContextSource],
        retrieval_results: Sequence[HybridSearchResult | SemanticSearchResult],
    ) -> list[ChatCitation]:
        """Return citations for source IDs that exist in the built context."""
        source_lookup = {source.source_id: source for source in source_map}
        result_lookup = {result.chunk_id: result for result in retrieval_results}
        citations: list[ChatCitation] = []

        for source_id in source_ids:
            source = source_lookup.get(source_id)
            if source is None:
                continue

            result = result_lookup.get(source.chunk_id)
            excerpt = self._build_excerpt(result, source) if result is not None else ""
            citations.append(
                ChatCitation(
                    source_id=source.source_id,
                    document_id=source.document_id,
                    document_name=source.document_name,
                    document_path=source.document_path,
                    chunk_id=source.chunk_id,
                    page_number=source.page_number,
                    chunk_index=source.chunk_index,
                    excerpt=excerpt,
                    start_offset=source.start_offset,
                    end_offset=source.end_offset,
                    score=source.score,
                )
            )

        return citations

    def _build_excerpt(
        self,
        result: HybridSearchResult | SemanticSearchResult,
        source: ContextSource,
    ) -> str:
        text = result.text or result.excerpt
        if not text:
            return ""

        included_range = source.included_text_range
        if included_range is not None:
            text = text[included_range.start : included_range.end]
        return _make_excerpt(text)
