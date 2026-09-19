from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as client:
        yield client


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    payload = response.json()
    assert payload["info"]["title"] == app.title
    assert payload["info"]["version"] == app.version


def test_swagger_ui_available(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Swagger UI" in response.text


def test_cors_allows_local_frontend_origin(client: TestClient) -> None:
    response = client.options(
        "/health/live",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "https://localhost:4173",
    ],
)
def test_cors_allows_loopback_frontend_origins(
    client: TestClient, origin: str
) -> None:
    response = client.options(
        "/health/live",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin


def test_cors_rejects_non_loopback_origin(client: TestClient) -> None:
    response = client.options(
        "/health/live",
        headers={
            "Origin": "http://localhost.example.com:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
