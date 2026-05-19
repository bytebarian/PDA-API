"""OpenAPI smoke tests for stable route and schema contracts."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


def test_openapi_exposes_expected_gamma_routes() -> None:
    with TestClient(app) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    paths = schema["paths"]
    expected_paths = {
        "/documents/upload",
        "/documents",
        "/documents/{document_id}",
        "/documents/{document_id}/download",
        "/documents/{document_id}/reprocess",
        "/jobs/{job_id}",
        "/health/live",
        "/health/ready",
    }
    assert expected_paths.issubset(paths)


def test_openapi_operation_ids_are_unique(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    operation_ids = [
        operation["operationId"]
        for path_item in schema["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids))


def test_openapi_uuid_path_parameters_are_typed_as_uuid(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    document_id_schema = schema["paths"]["/documents/{document_id}"]["get"]["parameters"][0]["schema"]
    assert document_id_schema["type"] == "string"
    assert document_id_schema["format"] == "uuid"

    job_id_schema = schema["paths"]["/jobs/{job_id}"]["get"]["parameters"][0]["schema"]
    assert job_id_schema["type"] == "string"
    assert job_id_schema["format"] == "uuid"
