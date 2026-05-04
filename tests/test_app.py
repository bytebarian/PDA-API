import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
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


def test_request_logs_are_structured_with_request_id(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="pda.request")

    response = client.get("/", headers={"x-request-id": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "test-request-id"

    request_logs = [record for record in caplog.records if record.name == "pda.request"]
    assert request_logs

    payload = json.loads(request_logs[-1].getMessage())
    assert payload["message"] == "request.completed"
    assert payload["request_id"] == "test-request-id"
    assert payload["method"] == "GET"
    assert payload["path"] == "/"
    assert payload["status"] == 200
    assert isinstance(payload["duration_ms"], float)
