"""Tests for application settings API endpoints and model-selection logic."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app as fastapi_app
from app.models.app_settings import AppSettings
from app.repositories.settings_repository import SettingsRepository
from app.schemas.app_settings import (
    DEFAULT_LLM_MODEL,
    SUPPORTED_LLM_MODELS,
    AppSettingsUpdate,
)
from app.services.settings_service import SettingsService

import app.models  # noqa: F401 – ensure all models are registered


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(
        storage_path=tmp_path,  # type: ignore[arg-type]
        embedding_provider="fake",
        embedding_model="test-fake-model",
        embedding_dimensions=8,
        model_provider="mock",
        model_name=DEFAULT_LLM_MODEL,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def api_client(
    db_session: AsyncSession,
    api_settings: Settings,
) -> Generator[TestClient, None, None]:
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    fastapi_app.dependency_overrides[get_db] = override_get_db
    fastapi_app.dependency_overrides[get_settings] = lambda: api_settings

    with TestClient(fastapi_app) as client:
        yield client

    fastapi_app.dependency_overrides.clear()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# SettingsRepository tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_repository_get_returns_none_when_empty(
    db_session: AsyncSession,
) -> None:
    repo = SettingsRepository(db_session)
    result = await repo.get()
    assert result is None


@pytest.mark.asyncio
async def test_settings_repository_get_or_create_creates_row(
    db_session: AsyncSession,
) -> None:
    repo = SettingsRepository(db_session)
    row = await repo.get_or_create()
    assert row is not None
    assert row.llm_model == DEFAULT_LLM_MODEL
    assert row.llm_provider == "local"


@pytest.mark.asyncio
async def test_settings_repository_get_or_create_idempotent(
    db_session: AsyncSession,
) -> None:
    repo = SettingsRepository(db_session)
    first = await repo.get_or_create()
    second = await repo.get_or_create()
    assert first.id == second.id


@pytest.mark.asyncio
async def test_settings_repository_update(
    db_session: AsyncSession,
) -> None:
    repo = SettingsRepository(db_session)
    row = await repo.get_or_create()
    updated = await repo.update(row, {"llm_model": "llama3.1:8b"})
    assert updated.llm_model == "llama3.1:8b"


# ---------------------------------------------------------------------------
# SettingsService tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_service_get_creates_default_on_first_call(
    db_session: AsyncSession,
) -> None:
    service = SettingsService(db_session)
    row = await service.get_settings()
    assert row.llm_model == DEFAULT_LLM_MODEL


@pytest.mark.asyncio
async def test_settings_service_update_llm_model(
    db_session: AsyncSession,
) -> None:
    service = SettingsService(db_session)
    update = AppSettingsUpdate(llm_model="llama3.2:3b")
    row = await service.update_settings(update)
    assert row.llm_model == "llama3.2:3b"


@pytest.mark.asyncio
async def test_settings_service_update_preserves_unset_fields(
    db_session: AsyncSession,
) -> None:
    service = SettingsService(db_session)
    # First set an initial state
    await service.update_settings(AppSettingsUpdate(llm_model="llama3.2:3b"))
    # Then update a different field
    row = await service.update_settings(AppSettingsUpdate(telemetry_enabled=True))
    assert row.llm_model == "llama3.2:3b"
    assert row.telemetry_enabled is True


# ---------------------------------------------------------------------------
# AppSettingsUpdate validation
# ---------------------------------------------------------------------------


def test_app_settings_update_rejects_unsupported_model() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        AppSettingsUpdate(llm_model="gpt-4o-mini")

    errors = exc_info.value.errors()
    assert any("llm_model" in str(e["loc"]) for e in errors)


def test_app_settings_update_accepts_all_supported_models() -> None:
    for model in SUPPORTED_LLM_MODELS:
        update = AppSettingsUpdate(llm_model=model)
        assert update.llm_model == model


def test_app_settings_update_accepts_none_llm_model() -> None:
    update = AppSettingsUpdate(llm_model=None)
    assert update.llm_model is None


# ---------------------------------------------------------------------------
# GET /settings endpoint
# ---------------------------------------------------------------------------


def test_get_settings_returns_200(api_client: TestClient) -> None:
    response = api_client.get("/settings")
    assert response.status_code == 200


def test_get_settings_returns_default_llm_model(api_client: TestClient) -> None:
    response = api_client.get("/settings")
    data = response.json()
    assert data["llm_model"] == DEFAULT_LLM_MODEL


def test_get_settings_response_has_required_fields(api_client: TestClient) -> None:
    response = api_client.get("/settings")
    data = response.json()
    assert "id" in data
    assert "llm_model" in data
    assert "llm_provider" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_get_settings_is_idempotent(api_client: TestClient) -> None:
    first = api_client.get("/settings").json()
    second = api_client.get("/settings").json()
    assert first["id"] == second["id"]
    assert first["llm_model"] == second["llm_model"]


# ---------------------------------------------------------------------------
# PATCH /settings endpoint
# ---------------------------------------------------------------------------


def test_patch_settings_updates_llm_model(api_client: TestClient) -> None:
    response = api_client.patch("/settings", json={"llm_model": "llama3.2:3b"})
    assert response.status_code == 200
    assert response.json()["llm_model"] == "llama3.2:3b"


def test_patch_settings_rejects_unsupported_model(api_client: TestClient) -> None:
    response = api_client.patch("/settings", json={"llm_model": "gpt-4o-mini"})
    assert response.status_code == 422


def test_patch_settings_rejects_empty_model_string(api_client: TestClient) -> None:
    response = api_client.patch("/settings", json={"llm_model": ""})
    assert response.status_code == 422


@pytest.mark.parametrize("model", sorted(SUPPORTED_LLM_MODELS))
def test_patch_settings_accepts_all_supported_models(
    api_client: TestClient, model: str
) -> None:
    response = api_client.patch("/settings", json={"llm_model": model})
    assert response.status_code == 200
    assert response.json()["llm_model"] == model


def test_patch_settings_persists_across_get(api_client: TestClient) -> None:
    api_client.patch("/settings", json={"llm_model": "gemma3:1b"})
    get_response = api_client.get("/settings")
    assert get_response.json()["llm_model"] == "gemma3:1b"


# ---------------------------------------------------------------------------
# Chat model resolution from persisted settings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_uses_persisted_model_when_no_request_override(
    db_session: AsyncSession,
) -> None:
    from app.adapters.llm.mock import MockChatModelProvider
    from app.schemas.chat import ChatAskRequest
    from app.schemas.hybrid_search import HybridSearchResponse, HybridSearchResult
    from app.services.chat_service import ChatService
    import uuid

    # Persist a non-default model
    service = SettingsService(db_session)
    await service.update_settings(AppSettingsUpdate(llm_model="llama3.2:3b"))

    captured: list[str] = []

    class CapturingMockProvider(MockChatModelProvider):
        async def generate(self, messages, *, model, **kwargs):  # type: ignore[override]
            captured.append(model)
            return await super().generate(messages, model=model, **kwargs)

    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=HybridSearchResponse(
            query="test",
            top_k=8,
            fusion_strategy="rrf",
            embedding_model="fake",
            filters_applied={},
            candidate_document_count=1,
            candidate_chunk_count=1,
            vector_candidate_count=1,
            full_text_candidate_count=1,
            result_count=1,
            results=[
                HybridSearchResult(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name="test.pdf",
                    document_path="/test.pdf",
                    category="other",
                    file_type="pdf",
                    page_number=1,
                    chunk_index=0,
                    start_offset=0,
                    end_offset=10,
                    text="Some relevant context.",
                    excerpt="Some relevant context.",
                    score=0.9,
                    vector_score=0.9,
                    full_text_score=None,
                    vector_rank=1,
                    full_text_rank=None,
                    matched_by=["vector"],
                    metadata={},
                )
            ],
        )
    )

    chat_svc = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        model_provider=CapturingMockProvider(answer="Context says something. [S1]"),
        settings=Settings(model_provider="mock", model_name=DEFAULT_LLM_MODEL, _env_file=None),  # type: ignore[call-arg]
    )

    await chat_svc.ask_question(ChatAskRequest(question="What does the doc say?"))

    # Model resolved from persisted settings, not env default
    assert captured == ["llama3.2:3b"]


@pytest.mark.asyncio
async def test_chat_request_model_override_takes_precedence(
    db_session: AsyncSession,
) -> None:
    from app.adapters.llm.mock import MockChatModelProvider
    from app.schemas.chat import ChatAskRequest
    from app.schemas.hybrid_search import HybridSearchResponse, HybridSearchResult
    from app.services.chat_service import ChatService
    import uuid

    # Persist one model
    service = SettingsService(db_session)
    await service.update_settings(AppSettingsUpdate(llm_model="llama3.2:3b"))

    captured: list[str] = []

    class CapturingMockProvider(MockChatModelProvider):
        async def generate(self, messages, *, model, **kwargs):  # type: ignore[override]
            captured.append(model)
            return await super().generate(messages, model=model, **kwargs)

    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=HybridSearchResponse(
            query="test",
            top_k=8,
            fusion_strategy="rrf",
            embedding_model="fake",
            filters_applied={},
            candidate_document_count=1,
            candidate_chunk_count=1,
            vector_candidate_count=1,
            full_text_candidate_count=1,
            result_count=1,
            results=[
                HybridSearchResult(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name="test.pdf",
                    document_path="/test.pdf",
                    category="other",
                    file_type="pdf",
                    page_number=1,
                    chunk_index=0,
                    start_offset=0,
                    end_offset=10,
                    text="Some relevant context.",
                    excerpt="Some relevant context.",
                    score=0.9,
                    vector_score=0.9,
                    full_text_score=None,
                    vector_rank=1,
                    full_text_rank=None,
                    matched_by=["vector"],
                    metadata={},
                )
            ],
        )
    )

    chat_svc = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        model_provider=CapturingMockProvider(answer="Context says something. [S1]"),
        settings=Settings(model_provider="mock", model_name=DEFAULT_LLM_MODEL, _env_file=None),  # type: ignore[call-arg]
    )

    # Override with a different supported model
    await chat_svc.ask_question(
        ChatAskRequest(question="What does the doc say?", model="llama3.1:8b")
    )

    # The per-request override takes precedence
    assert captured == ["llama3.1:8b"]


@pytest.mark.asyncio
async def test_chat_rejects_unsupported_request_model(
    db_session: AsyncSession,
) -> None:
    from app.adapters.llm.mock import MockChatModelProvider
    from app.schemas.chat import ChatAskRequest
    from app.schemas.hybrid_search import HybridSearchResponse, HybridSearchResult
    from app.services.chat_service import ChatModelNotSupportedError, ChatService
    import uuid
    import pytest

    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=HybridSearchResponse(
            query="test",
            top_k=8,
            fusion_strategy="rrf",
            embedding_model="fake",
            filters_applied={},
            candidate_document_count=1,
            candidate_chunk_count=1,
            vector_candidate_count=1,
            full_text_candidate_count=1,
            result_count=1,
            results=[
                HybridSearchResult(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name="test.pdf",
                    document_path="/test.pdf",
                    category="other",
                    file_type="pdf",
                    page_number=1,
                    chunk_index=0,
                    start_offset=0,
                    end_offset=10,
                    text="Some relevant context.",
                    excerpt="Some relevant context.",
                    score=0.9,
                    vector_score=0.9,
                    full_text_score=None,
                    vector_rank=1,
                    full_text_rank=None,
                    matched_by=["vector"],
                    metadata={},
                )
            ],
        )
    )

    chat_svc = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        model_provider=MockChatModelProvider(),
        settings=Settings(model_provider="mock", model_name=DEFAULT_LLM_MODEL, _env_file=None),  # type: ignore[call-arg]
    )

    with pytest.raises(ChatModelNotSupportedError):
        await chat_svc.ask_question(
            ChatAskRequest(question="test?", model="gpt-4o-mini")
        )


@pytest.mark.asyncio
async def test_chat_falls_back_to_env_default_when_no_db_row(
    db_session: AsyncSession,
) -> None:
    from app.adapters.llm.mock import MockChatModelProvider
    from app.schemas.chat import ChatAskRequest
    from app.schemas.hybrid_search import HybridSearchResponse, HybridSearchResult
    from app.services.chat_service import ChatService
    import uuid

    captured: list[str] = []

    class CapturingMockProvider(MockChatModelProvider):
        async def generate(self, messages, *, model, **kwargs):  # type: ignore[override]
            captured.append(model)
            return await super().generate(messages, model=model, **kwargs)

    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    hybrid_service = MagicMock()
    hybrid_service.hybrid_search = AsyncMock(
        return_value=HybridSearchResponse(
            query="test",
            top_k=8,
            fusion_strategy="rrf",
            embedding_model="fake",
            filters_applied={},
            candidate_document_count=1,
            candidate_chunk_count=1,
            vector_candidate_count=1,
            full_text_candidate_count=1,
            result_count=1,
            results=[
                HybridSearchResult(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name="test.pdf",
                    document_path="/test.pdf",
                    category="other",
                    file_type="pdf",
                    page_number=1,
                    chunk_index=0,
                    start_offset=0,
                    end_offset=10,
                    text="Some relevant context.",
                    excerpt="Some relevant context.",
                    score=0.9,
                    vector_score=0.9,
                    full_text_score=None,
                    vector_rank=1,
                    full_text_rank=None,
                    matched_by=["vector"],
                    metadata={},
                )
            ],
        )
    )

    env_model = "llama3.1:8b-instruct-q8_0"
    chat_svc = ChatService(
        db_session,
        hybrid_search_service=hybrid_service,
        search_service=MagicMock(),
        model_provider=CapturingMockProvider(answer="Context says something. [S1]"),
        settings=Settings(
            model_provider="mock",
            model_name=env_model,
            _env_file=None,  # type: ignore[call-arg]
        ),
    )

    # No app_settings row in DB → falls back to env model_name
    await chat_svc.ask_question(ChatAskRequest(question="What does the doc say?"))

    assert captured == [env_model]
