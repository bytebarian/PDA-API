"""Tests for context assembly and model handoff payload construction."""

from __future__ import annotations

import math
import uuid
from typing import Any

import pytest

from app.schemas.hybrid_search import HybridSearchResult
from app.services.context_builder import ApproximateTokenEstimator, ContextBuilderService


@pytest.fixture
def service() -> ContextBuilderService:
    return ContextBuilderService()


def _result(
    *,
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    document_name: str = "employment-contract.pdf",
    chunk_index: int = 0,
    text: str | None = "The notice period is three months.",
    excerpt: str | None = "The notice period is three months.",
    score: float = 0.9,
    page_number: int | None = 3,
    category: str | None = "contracts",
    file_type: str | None = "pdf",
) -> dict[str, Any]:
    return {
        "chunkId": str(chunk_id or uuid.uuid4()),
        "documentId": str(document_id or uuid.uuid4()),
        "documentName": document_name,
        "documentPath": f"/documents/{document_name}",
        "category": category,
        "fileType": file_type,
        "pageNumber": page_number,
        "chunkIndex": chunk_index,
        "startOffset": chunk_index * 100,
        "endOffset": (chunk_index * 100) + 50,
        "text": text,
        "excerpt": excerpt,
        "score": score,
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_empty_results_return_empty_context(service: ContextBuilderService) -> None:
    built = await service.build_context([])

    assert built.included_chunk_count == 0
    assert built.excluded_chunk_count == 0
    assert built.sources == []
    assert built.truncated is False
    assert "<document_context>" in built.context_text


@pytest.mark.asyncio
async def test_source_ids_are_stable_and_sequential(service: ContextBuilderService) -> None:
    doc = uuid.uuid4()
    built = await service.build_context(
        [
            _result(document_id=doc, chunk_index=1, score=0.9, text="a", excerpt="a"),
            _result(document_id=doc, chunk_index=2, score=0.8, text="b", excerpt="b"),
        ]
    )

    assert [source.source_id for source in built.sources] == ["S1", "S2"]
    assert "[S1]" in built.context_text
    assert "[S2]" in built.context_text


@pytest.mark.asyncio
async def test_duplicate_chunk_id_is_included_once(service: ContextBuilderService) -> None:
    chunk = uuid.uuid4()
    doc = uuid.uuid4()
    built = await service.build_context(
        [
            _result(chunk_id=chunk, document_id=doc, chunk_index=1, score=0.8),
            _result(chunk_id=chunk, document_id=doc, chunk_index=1, score=0.95),
        ]
    )

    assert built.included_chunk_count == 1
    assert built.excluded_chunk_count == 1
    assert built.excluded[0].reason == "duplicate_chunk_id"
    assert built.sources[0].score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_ordering_by_score_is_deterministic_without_grouping(
    service: ContextBuilderService,
) -> None:
    doc_a = uuid.UUID("00000000-0000-0000-0000-000000000010")
    doc_b = uuid.UUID("00000000-0000-0000-0000-000000000020")

    built = await service.build_context(
        [
            _result(document_id=doc_b, chunk_index=3, score=0.7, text="B-3"),
            _result(document_id=doc_a, chunk_index=2, score=0.9, text="A-2"),
            _result(document_id=doc_a, chunk_index=1, score=0.9, text="A-1"),
        ],
        group_by_document=False,
    )

    assert [source.document_id for source in built.sources] == [doc_a, doc_a, doc_b]
    assert [source.chunk_index for source in built.sources] == [1, 2, 3]


@pytest.mark.asyncio
async def test_group_by_document_preserves_chunk_order(service: ContextBuilderService) -> None:
    doc_best = uuid.UUID("00000000-0000-0000-0000-000000000001")
    doc_other = uuid.UUID("00000000-0000-0000-0000-000000000002")

    built = await service.build_context(
        [
            _result(document_id=doc_other, chunk_index=5, score=0.7, text="other-5"),
            _result(document_id=doc_best, chunk_index=3, score=0.95, text="best-3"),
            _result(document_id=doc_best, chunk_index=1, score=0.8, text="best-1"),
        ],
        group_by_document=True,
    )

    assert [source.document_id for source in built.sources][:2] == [doc_best, doc_best]
    assert [source.chunk_index for source in built.sources][:2] == [1, 3]


@pytest.mark.asyncio
async def test_max_chunk_count_is_enforced(service: ContextBuilderService) -> None:
    doc = uuid.uuid4()
    built = await service.build_context(
        [
            _result(
                document_id=doc,
                chunk_index=i,
                score=1.0 - i * 0.01,
                text=f"text-{i}",
                excerpt=f"text-{i}",
            )
            for i in range(4)
        ],
        max_chunks=2,
    )

    assert built.included_chunk_count == 2
    assert built.excluded_chunk_count == 2
    assert all(item.reason == "max_chunks_exceeded" for item in built.excluded)


@pytest.mark.asyncio
async def test_max_character_budget_is_enforced(service: ContextBuilderService) -> None:
    built = await service.build_context(
        [
            _result(chunk_index=1, text="A" * 200, excerpt="A" * 200, score=0.9),
            _result(chunk_index=2, text="B" * 200, excerpt="B" * 200, score=0.8),
        ],
        max_characters=450,
        group_by_document=False,
    )

    assert built.character_count <= 450
    assert built.excluded_chunk_count >= 1


@pytest.mark.asyncio
async def test_oversized_single_chunk_is_truncated_safely(
    service: ContextBuilderService,
) -> None:
    built = await service.build_context(
        [_result(text="Sentence one. " * 200, excerpt="Sentence one. " * 200)],
        max_characters=500,
    )

    assert built.included_chunk_count == 1
    assert built.truncated is True
    assert built.character_count <= 500
    assert built.sources[0].included_text_range is not None
    assert built.sources[0].included_text_range.end < len("Sentence one. " * 200)


@pytest.mark.asyncio
async def test_source_map_matches_source_ids_in_context(
    service: ContextBuilderService,
) -> None:
    built = await service.build_context([_result(chunk_index=1), _result(chunk_index=2)])
    for source in built.sources:
        assert f"[{source.source_id}]" in built.context_text


@pytest.mark.asyncio
async def test_token_estimation_is_deterministic(service: ContextBuilderService) -> None:
    built = await service.build_context([_result(text="abcd" * 25, excerpt="abcd" * 25)])

    assert built.estimated_tokens == math.ceil(len(built.context_text) / 4)


@pytest.mark.asyncio
async def test_context_includes_core_metadata_and_content(
    service: ContextBuilderService,
) -> None:
    built = await service.build_context([_result(page_number=7, chunk_index=9)])

    assert "Document: employment-contract.pdf" in built.context_text
    assert "Page: 7" in built.context_text
    assert "Chunk: 9" in built.context_text
    assert "The notice period is three months." in built.context_text


@pytest.mark.asyncio
async def test_scores_not_in_text_by_default_but_available_in_sources(
    service: ContextBuilderService,
) -> None:
    built = await service.build_context([_result(score=0.876543)])

    assert "Score:" not in built.context_text
    assert built.sources[0].score == pytest.approx(0.876543)


@pytest.mark.asyncio
async def test_scores_can_be_included_when_requested(service: ContextBuilderService) -> None:
    built = await service.build_context([_result(score=0.876543)], include_scores=True)

    assert "Score: 0.876543" in built.context_text


@pytest.mark.asyncio
async def test_uses_excerpt_when_text_missing_and_marks_it(
    service: ContextBuilderService,
) -> None:
    built = await service.build_context(
        [_result(text=None, excerpt="excerpt-only content")]
    )

    assert "Content (excerpt only):" in built.context_text
    assert built.sources[0].excerpt_only is True


@pytest.mark.asyncio
async def test_hybrid_search_result_schema_is_accepted(
    service: ContextBuilderService,
) -> None:
    hybrid_result = HybridSearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_name="energy-supplier-contract.pdf",
        document_path="/documents/energy-supplier-contract.pdf",
        category="contracts",
        file_type="pdf",
        page_number=2,
        chunk_index=4,
        start_offset=10,
        end_offset=50,
        text="The G11 tariff applies.",
        excerpt="The G11 tariff applies.",
        score=0.91,
        vector_score=0.9,
        full_text_score=0.92,
        vector_rank=1,
        full_text_rank=1,
        matched_by=["vector", "full_text"],
        metadata={},
    )

    built = await service.build_context([hybrid_result])

    assert built.included_chunk_count == 1
    assert "energy-supplier-contract.pdf" in built.context_text


@pytest.mark.asyncio
async def test_model_payload_can_be_passed_to_mock_model_client(
    service: ContextBuilderService,
) -> None:
    built = await service.build_context([_result(chunk_index=2)])
    payload = service.build_model_context_payload(
        built,
        context_style="chat",
        query="What is the notice period?",
    )

    captured: list[dict[str, str]] = []

    async def mock_model_call(model_input: list[dict[str, str]]) -> dict[str, str]:
        captured.extend(model_input)
        return {"status": "ok"}

    model_input = [
        {"role": "system", "content": payload.system_instruction},
        {
            "role": "user",
            "content": (
                f"Context:\n{payload.context_text}\n\n"
                "Question:\nWhat is the notice period?"
            ),
        },
    ]

    response = await mock_model_call(model_input)

    assert response == {"status": "ok"}
    assert captured == model_input
    assert captured[0]["content"] == payload.system_instruction
    assert payload.source_map[0].source_id == "S1"


def test_approximate_token_estimator() -> None:
    estimator = ApproximateTokenEstimator()
    assert estimator.estimate("abcd") == 1
    assert estimator.estimate("abcde") == 2
