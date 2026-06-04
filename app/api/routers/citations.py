"""HTTP endpoint for the citation builder utility."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.citations import CitationBuildRequest, CitationBuildResponse, CitationSourceInput
from app.services.citation_builder import CitationBuilder

router = APIRouter(prefix="/citations", tags=["citations"])

_builder = CitationBuilder()


def _make_context_source_from_input(inp: CitationSourceInput):  # type: ignore[return]
    """Convert a CitationSourceInput to a lightweight duck-typed object for the builder."""
    from app.schemas.context import ContextSource, IncludedTextRange

    return ContextSource(
        source_id=inp.source_id,
        chunk_id=inp.chunk_id,
        document_id=inp.document_id,
        document_name=inp.document_name,
        document_path=inp.document_path,
        page_number=inp.page_number,
        chunk_index=inp.chunk_index,
        start_offset=inp.start_offset,
        end_offset=inp.end_offset,
        score=inp.score,
        # Provide the full text as a single inclusive range so the builder
        # can produce an excerpt even without a separate retrieval result.
        included_text_range=IncludedTextRange(start=0, end=len(inp.text or inp.excerpt or "")) if (inp.text or inp.excerpt) else None,
        excerpt_only=inp.excerpt is not None and inp.text is None,
    )


def _make_retrieval_result_from_input(inp: CitationSourceInput):
    """Return a duck-typed retrieval result used for text lookup."""
    from types import SimpleNamespace
    return SimpleNamespace(
        chunk_id=inp.chunk_id,
        document_id=inp.document_id,
        document_name=inp.document_name,
        document_path=inp.document_path,
        page_number=inp.page_number,
        chunk_index=inp.chunk_index,
        start_offset=inp.start_offset,
        end_offset=inp.end_offset,
        text=inp.text,
        excerpt=inp.excerpt,
        score=inp.score,
        matched_by=None,
        metadata=inp.metadata,
    )


@router.post(
    "/build",
    response_model=CitationBuildResponse,
    summary="Build normalized citations from sources and optional answer text",
    description=(
        "Accept context source objects and an optional model answer, parse "
        "[S1]-style markers, and return normalized citation objects with "
        "excerpts and diagnostics."
    ),
)
def build_citations(request: CitationBuildRequest) -> CitationBuildResponse:
    """Normalize citations from source objects and optional answer text."""
    context_sources = [_make_context_source_from_input(s) for s in request.sources]
    retrieval_results = [_make_retrieval_result_from_input(s) for s in request.sources]

    citations, diagnostics = _builder.build_from_sources(
        context_sources,
        answer_text=request.answer_text,
        include_uncited_sources=request.include_uncited_sources,
        max_excerpt_characters=request.max_excerpt_characters,
        retrieval_results=retrieval_results,
    )
    return CitationBuildResponse(citations=citations, diagnostics=diagnostics)
