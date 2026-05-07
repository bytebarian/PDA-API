"""Tests for /health/live and /health/ready endpoints."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health/live
# ---------------------------------------------------------------------------


def test_live_returns_200(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200


def test_live_response_shape(client: TestClient) -> None:
    response = client.get("/health/live")
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "pda-api"
    assert "version" in body


def test_live_does_not_depend_on_database(client: TestClient) -> None:
    """Live endpoint must succeed even when the DB engine raises."""
    with patch("app.api.routers.health.get_engine", side_effect=RuntimeError("no db")):
        response = client.get("/health/live")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /health/ready – happy path
# ---------------------------------------------------------------------------


def _make_mock_engine() -> MagicMock:
    """Return a mock async engine whose connect() context manager executes SELECT 1."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=None)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.connect = MagicMock(return_value=mock_ctx)
    return mock_engine


def test_ready_returns_200_when_db_ok(client: TestClient) -> None:
    with patch("app.api.routers.health.get_engine", return_value=_make_mock_engine()):
        response = client.get("/health/ready")
    assert response.status_code == 200


def test_ready_response_shape_when_db_ok(client: TestClient) -> None:
    with patch("app.api.routers.health.get_engine", return_value=_make_mock_engine()):
        response = client.get("/health/ready")
    body = response.json()
    assert body["status"] == "ready"
    assert body["dependencies"]["database"] == "ok"


# ---------------------------------------------------------------------------
# /health/ready – failure path
# ---------------------------------------------------------------------------


def _make_failing_engine() -> MagicMock:
    """Return a mock engine whose connect() context manager raises on execute."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(
        side_effect=OperationalError("connection refused", None, Exception())
    )

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.connect = MagicMock(return_value=mock_ctx)
    return mock_engine


def test_ready_returns_503_when_db_unavailable(client: TestClient) -> None:
    with patch("app.api.routers.health.get_engine", return_value=_make_failing_engine()):
        response = client.get("/health/ready")
    assert response.status_code == 503


def test_ready_response_body_when_db_unavailable(client: TestClient) -> None:
    with patch("app.api.routers.health.get_engine", return_value=_make_failing_engine()):
        response = client.get("/health/ready")
    body = response.json()
    assert body["detail"]["status"] == "not ready"
    assert body["detail"]["dependencies"]["database"] == "unavailable"
