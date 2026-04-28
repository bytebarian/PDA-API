# Gamma – Document ingest, storage, and registry endpoints

## Repository
bytebarian/PDA-API

## Objective
Deliver this Foundation slice as an independent Codex task with tests and deterministic verification.

## Implementation scope
- POST /api/v1/documents/upload
- GET /api/v1/documents
- GET /api/v1/documents/{id}
- DELETE /api/v1/documents/{id}
- POST /api/v1/documents/{id}/reprocess stub
- Local file storage service
- Validation for extension and size

## Files and directories that may be touched
- `app/api/routers/documents.py`
- `app/services/document_service.py`
- `app/services/storage_service.py`
- `app/schemas/document.py`
- `tests`

## Out of scope
- Do not redesign the existing UI unless this task explicitly says so.
- Do not add public cloud dependencies for tests.
- Do not remove privacy-first local execution assumptions.

## Commands Codex must run
- `make test`
- `make smoke`

## Definition of Done
- Real upload stores file and DB record
- List supports filters and pagination
- Detail endpoint returns full document
- Delete cleans DB and file state
- Invalid upload is rejected

## Acceptance notes
Implementation is complete only when the code, tests, and documentation are committed together and the required commands pass in a clean local environment.
