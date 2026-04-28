# Beta – Metadata model, migrations, and persistence

## Repository
bytebarian/PDA-API

## Objective
Deliver this Foundation slice as an independent Codex task with tests and deterministic verification.

## Implementation scope
- Implement Document model
- Implement DocumentChunk model with vector column
- Implement ProcessingJob model
- Implement AppSettings model
- Add enums compatible with frontend
- Add repository/service CRUD tests

## Files and directories that may be touched
- `app/models`
- `app/schemas`
- `app/db`
- `app/services`
- `alembic/versions`
- `tests`

## Out of scope
- Do not redesign the existing UI unless this task explicitly says so.
- Do not add public cloud dependencies for tests.
- Do not remove privacy-first local execution assumptions.

## Commands Codex must run
- `make migrate`
- `make test`

## Definition of Done
- Migrations upgrade and downgrade cleanly
- Models persist and reload correctly
- Frontend-compatible DTOs are produced
- pgvector extension is enabled

## Acceptance notes
Implementation is complete only when the code, tests, and documentation are committed together and the required commands pass in a clean local environment.
