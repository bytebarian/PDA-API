"""Tests for the async database session layer (app/db)."""

from collections.abc import Generator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db import Base, get_db, get_engine, get_session_factory


@pytest.fixture(autouse=True)
def reset_db_singletons() -> Generator[None, None, None]:
    """Reset module-level engine/factory singletons between tests."""
    import app.db.session as db_session

    original_engine = db_session._engine
    original_factory = db_session._session_factory
    yield
    db_session._engine = original_engine
    db_session._session_factory = original_factory


def test_get_engine_returns_async_engine() -> None:
    engine = get_engine()
    assert isinstance(engine, AsyncEngine)


def test_get_engine_is_singleton() -> None:
    engine1 = get_engine()
    engine2 = get_engine()
    assert engine1 is engine2


def test_get_session_factory_returns_async_sessionmaker() -> None:
    factory = get_session_factory()
    assert isinstance(factory, async_sessionmaker)


def test_get_session_factory_is_singleton() -> None:
    factory1 = get_session_factory()
    factory2 = get_session_factory()
    assert factory1 is factory2


def test_base_metadata_accessible() -> None:
    """Base.metadata should be importable and have a tables mapping."""
    assert hasattr(Base, "metadata")
    assert hasattr(Base.metadata, "tables")


@pytest.mark.asyncio
async def test_get_db_yields_async_session() -> None:
    """get_db dependency should yield an AsyncSession."""
    gen = get_db()
    session = await gen.__anext__()
    assert isinstance(session, AsyncSession)
    # Exhaust the generator (triggers cleanup)
    try:
        await gen.aclose()
    except StopAsyncIteration:
        pass


def test_engine_uses_configured_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine URL should reflect the configured database_url setting."""
    import app.db.session as db_session

    # Reset singleton so a new engine is created with the patched env var.
    db_session._engine = None
    db_session._session_factory = None

    monkeypatch.setenv("PDA_DATABASE_URL", "sqlite+aiosqlite:///./test_override.db")

    from app.core.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    engine = get_engine()
    assert "test_override" in str(engine.url)

    get_settings.cache_clear()  # type: ignore[attr-defined]
    db_session._engine = None
    db_session._session_factory = None
