"""Service for assembling retrieval chunks into deterministic model prompt context."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from app.schemas.context import (
    BuiltContext,
    ContextSource,
    ExcludedContextChunk,
    IncludedTextRange,
    ModelContextPayload,
    RetrievalResultForContext,
)
from app.schemas.hybrid_search import HybridSearchResult
from app.schemas.search import SemanticSearchResult

DEFAULT_MAX_CHUNKS = 12
DEFAULT_MAX_CHARACTERS = 12000


class ApproximateTokenEstimator:
    """Deterministic fallback token estimator."""

    def estimate(self, text: str) -> int:
        return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class _ContextStyleTemplate:
    system_instruction: str
    include_wrappers: bool


def _style_template(style: Literal["chat", "report", "raw"]) -> _ContextStyleTemplate:
    if style == "report":
        return _ContextStyleTemplate(
            system_instruction=(
                "You are given private document excerpts to support report drafting. "
                "Use only this context and cite source IDs like [S1], [S2]."
            ),
            include_wrappers=True,
        )
    if style == "raw":
        return _ContextStyleTemplate(system_instruction="", include_wrappers=False)
    return _ContextStyleTemplate(
        system_instruction=(
            "You are given excerpts from the user's private document library. "
            "Use only this context when answering document-specific questions. "
            "Cite sources using their source IDs, for example [S1], [S2]."
        ),
        include_wrappers=True,
    )


def _normalize_text_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _sort_results(
    results: list[RetrievalResultForContext], *, group_by_document: bool
) -> list[RetrievalResultForContext]:
    if not group_by_document:
        return sorted(
            results,
            key=lambda item: (
                -(item.score or 0.0),
                str(item.document_id),
                item.chunk_index,
                item.page_number if item.page_number is not None else 10**9,
                item.start_offset if item.start_offset is not None else 10**9,
                str(item.chunk_id),
            ),
        )

    grouped: dict[str, list[RetrievalResultForContext]] = defaultdict(list)
    for item in results:
        grouped[str(item.document_id)].append(item)

    ordered_document_ids = sorted(
        grouped.keys(),
        key=lambda doc_id: (
            -max((candidate.score or 0.0) for candidate in grouped[doc_id]),
            doc_id,
        ),
    )

    ordered: list[RetrievalResultForContext] = []
    for doc_id in ordered_document_ids:
        ordered.extend(
            sorted(
                grouped[doc_id],
                key=lambda item: (
                    item.chunk_index,
                    item.page_number if item.page_number is not None else 10**9,
                    item.start_offset if item.start_offset is not None else 10**9,
                    str(item.chunk_id),
                ),
            )
        )
    return ordered


def _resolve_text(item: RetrievalResultForContext) -> tuple[str | None, bool]:
    if item.text and item.text.strip():
        return item.text.strip(), False
    if item.excerpt and item.excerpt.strip():
        return item.excerpt.strip(), True
    return None, False


def _truncate_on_boundary(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    working = text[:max_chars]
    boundary_start = max(max_chars // 2, 1)
    for marker in ("\n\n", ". ", "\n", " "):
        boundary = working.rfind(marker, boundary_start)
        if boundary > 0:
            return working[:boundary].rstrip()
    return working.rstrip()


class ContextBuilderService:
    """Build prompt-ready context text and citation source maps."""

    def __init__(self, *, token_estimator: ApproximateTokenEstimator | None = None) -> None:
        self._token_estimator = token_estimator or ApproximateTokenEstimator()

    async def build_context(
        self,
        retrieval_results: list[
            RetrievalResultForContext | SemanticSearchResult | HybridSearchResult | dict[str, Any]
        ],
        *,
        query: str | None = None,
        max_tokens: int | None = None,
        max_characters: int | None = None,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        include_metadata: bool = True,
        include_scores: bool = False,
        group_by_document: bool = True,
        context_style: Literal["chat", "report", "raw"] = "chat",
    ) -> BuiltContext:
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        max_characters = max_characters or DEFAULT_MAX_CHARACTERS
        if max_characters < 1:
            raise ValueError("max_characters must be >= 1")

        normalized, excluded = self._normalize_and_dedupe(retrieval_results)
        ordered = _sort_results(normalized, group_by_document=group_by_document)
        template = _style_template(context_style)

        prefix, suffix = self._context_wrappers(template, context_style)

        included_blocks: list[str] = []
        sources: list[ContextSource] = []
        truncated = False

        for index, item in enumerate(ordered):
            if len(sources) >= max_chunks:
                for rest in ordered[index:]:
                    excluded.append(
                        ExcludedContextChunk(
                            chunk_id=rest.chunk_id,
                            reason="max_chunks_exceeded",
                        )
                    )
                break

            resolved_text, excerpt_only = _resolve_text(item)
            if resolved_text is None:
                excluded.append(
                    ExcludedContextChunk(chunk_id=item.chunk_id, reason="missing_content")
                )
                continue

            source_id = f"S{len(sources) + 1}"
            block = self._format_block(
                source_id,
                item,
                resolved_text,
                include_metadata=include_metadata,
                include_scores=include_scores,
                excerpt_only=excerpt_only,
            )

            if self._fits_budget(
                prefix,
                included_blocks,
                block,
                suffix,
                max_characters=max_characters,
                max_tokens=max_tokens,
            ):
                included_blocks.append(block)
                sources.append(
                    self._build_source(
                        source_id,
                        item,
                        text_length=len(resolved_text),
                        excerpt_only=excerpt_only,
                    )
                )
                continue

            truncated_block, truncated_len = self._truncate_block_to_fit(
                prefix,
                included_blocks,
                suffix,
                source_id,
                item,
                resolved_text,
                max_characters=max_characters,
                max_tokens=max_tokens,
                include_metadata=include_metadata,
                include_scores=include_scores,
                excerpt_only=excerpt_only,
            )
            if truncated_block is None:
                excluded.append(
                    ExcludedContextChunk(chunk_id=item.chunk_id, reason="budget_exceeded")
                )
                continue

            included_blocks.append(truncated_block)
            sources.append(
                self._build_source(
                    source_id,
                    item,
                    text_length=truncated_len,
                    excerpt_only=excerpt_only,
                )
            )
            truncated = True

        context_text = self._render_context(prefix, included_blocks, suffix)
        estimated_tokens = self._token_estimator.estimate(context_text)

        return BuiltContext(
            context_text=context_text,
            sources=sources,
            included_chunk_count=len(sources),
            excluded_chunk_count=len(excluded),
            estimated_tokens=estimated_tokens,
            character_count=len(context_text),
            truncated=truncated,
            excluded=excluded,
        )

    def build_model_context_payload(
        self,
        built_context: BuiltContext,
        *,
        context_style: Literal["chat", "report", "raw"] = "chat",
        query: str | None = None,
    ) -> ModelContextPayload:
        template = _style_template(context_style)
        return ModelContextPayload(
            system_instruction=template.system_instruction,
            context_text=built_context.context_text,
            source_map=built_context.sources,
            estimated_tokens=built_context.estimated_tokens,
            metadata={
                "query": query,
                "included_chunk_count": built_context.included_chunk_count,
                "excluded_chunk_count": built_context.excluded_chunk_count,
                "character_count": built_context.character_count,
                "truncated": built_context.truncated,
                "context_style": context_style,
            },
        )

    def _normalize_and_dedupe(
        self,
        retrieval_results: Iterable[
            RetrievalResultForContext | SemanticSearchResult | HybridSearchResult | dict[str, Any]
        ],
    ) -> tuple[list[RetrievalResultForContext], list[ExcludedContextChunk]]:
        deduped: list[RetrievalResultForContext] = []
        excluded: list[ExcludedContextChunk] = []

        seen_chunk_ids: dict[str, int] = {}
        seen_document_chunk: dict[tuple[str, int], int] = {}
        seen_text: dict[str, int] = {}

        for raw in retrieval_results:
            try:
                payload: Any = raw.model_dump(mode="python") if hasattr(raw, "model_dump") else raw
                normalized = RetrievalResultForContext.model_validate(payload)
            except ValidationError:
                continue

            chunk_key = str(normalized.chunk_id)
            doc_chunk_key = (str(normalized.document_id), normalized.chunk_index)
            resolved_text, _ = _resolve_text(normalized)
            text_key = _normalize_text_for_dedupe(resolved_text) if resolved_text else None

            duplicate_reason: str | None = None
            duplicate_index: int | None = None
            if chunk_key in seen_chunk_ids:
                duplicate_reason = "duplicate_chunk_id"
                duplicate_index = seen_chunk_ids[chunk_key]
            elif doc_chunk_key in seen_document_chunk:
                duplicate_reason = "duplicate_document_chunk"
                duplicate_index = seen_document_chunk[doc_chunk_key]
            elif text_key and text_key in seen_text:
                duplicate_reason = "duplicate_text"
                duplicate_index = seen_text[text_key]

            if duplicate_reason is not None and duplicate_index is not None:
                self._merge_preferred(deduped[duplicate_index], normalized)
                excluded.append(
                    ExcludedContextChunk(
                        chunk_id=normalized.chunk_id,
                        reason=duplicate_reason,
                    )
                )
                continue

            position = len(deduped)
            deduped.append(normalized)
            seen_chunk_ids[chunk_key] = position
            seen_document_chunk[doc_chunk_key] = position
            if text_key:
                seen_text[text_key] = position

        return deduped, excluded

    def _merge_preferred(
        self, current: RetrievalResultForContext, incoming: RetrievalResultForContext
    ) -> None:
        if (incoming.score or 0.0) > (current.score or 0.0):
            current.score = incoming.score

        for attr in (
            "document_path",
            "category",
            "file_type",
            "page_number",
            "start_offset",
            "end_offset",
            "text",
            "excerpt",
        ):
            if getattr(current, attr) in (None, "") and getattr(incoming, attr) not in (
                None,
                "",
            ):
                setattr(current, attr, getattr(incoming, attr))

        if incoming.metadata:
            current.metadata = {**incoming.metadata, **current.metadata}

    def _context_wrappers(
        self,
        template: _ContextStyleTemplate,
        style: Literal["chat", "report", "raw"],
    ) -> tuple[str, str]:
        if not template.include_wrappers:
            return "", ""

        opening = (
            f"{template.system_instruction}\n\n"
            "<document_context>\n"
        )
        if style == "report":
            opening = (
                f"{template.system_instruction}\n\n"
                "<report_document_context>\n"
            )
            return opening, "\n</report_document_context>"
        return opening, "\n</document_context>"

    def _render_context(self, prefix: str, blocks: list[str], suffix: str) -> str:
        body = "\n\n".join(blocks)
        return f"{prefix}{body}{suffix}" if (prefix or suffix) else body

    def _fits_budget(
        self,
        prefix: str,
        blocks: list[str],
        candidate_block: str,
        suffix: str,
        *,
        max_characters: int,
        max_tokens: int | None,
    ) -> bool:
        rendered = self._render_context(prefix, [*blocks, candidate_block], suffix)
        if len(rendered) > max_characters:
            return False
        if max_tokens is not None and self._token_estimator.estimate(rendered) > max_tokens:
            return False
        return True

    def _truncate_block_to_fit(
        self,
        prefix: str,
        blocks: list[str],
        suffix: str,
        source_id: str,
        item: RetrievalResultForContext,
        text: str,
        *,
        max_characters: int,
        max_tokens: int | None,
        include_metadata: bool,
        include_scores: bool,
        excerpt_only: bool,
    ) -> tuple[str | None, int]:
        low = 0
        high = len(text)
        best: tuple[str | None, int] = (None, 0)

        while low <= high:
            mid = (low + high) // 2
            maybe_text = _truncate_on_boundary(text, mid)
            if not maybe_text:
                low = mid + 1
                continue
            if len(maybe_text) < len(text):
                maybe_text = f"{maybe_text}\n[...]"
            candidate_block = self._format_block(
                source_id,
                item,
                maybe_text,
                include_metadata=include_metadata,
                include_scores=include_scores,
                excerpt_only=excerpt_only,
            )
            if self._fits_budget(
                prefix,
                blocks,
                candidate_block,
                suffix,
                max_characters=max_characters,
                max_tokens=max_tokens,
            ):
                best = (candidate_block, len(maybe_text))
                low = mid + 1
            else:
                high = mid - 1

        return best

    def _format_block(
        self,
        source_id: str,
        item: RetrievalResultForContext,
        text: str,
        *,
        include_metadata: bool,
        include_scores: bool,
        excerpt_only: bool,
    ) -> str:
        lines = [f"[{source_id}]"]

        if include_metadata:
            lines.append(f"Document: {item.document_name}")
            lines.append(f"Document ID: {item.document_id}")
            if item.category:
                lines.append(f"Category: {item.category}")
            if item.file_type:
                lines.append(f"File type: {item.file_type}")
            if item.page_number is not None:
                lines.append(f"Page: {item.page_number}")
            lines.append(f"Chunk: {item.chunk_index}")
            if include_scores and item.score is not None:
                lines.append(f"Score: {item.score:.6f}")

        lines.append("Content (excerpt only):" if excerpt_only else "Content:")
        lines.append(text)
        return "\n".join(lines)

    def _build_source(
        self,
        source_id: str,
        item: RetrievalResultForContext,
        *,
        text_length: int,
        excerpt_only: bool,
    ) -> ContextSource:
        return ContextSource(
            source_id=source_id,
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            document_name=item.document_name,
            document_path=item.document_path,
            page_number=item.page_number,
            chunk_index=item.chunk_index,
            start_offset=item.start_offset,
            end_offset=item.end_offset,
            score=item.score,
            included_text_range=IncludedTextRange(start=0, end=text_length),
            excerpt_only=excerpt_only,
        )
