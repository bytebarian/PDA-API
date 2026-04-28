# Epsilon – Retrieval, chat, reports, dashboard, and settings APIs

## Repository
bytebarian/PDA-API

## Objective
Deliver this Foundation slice as an independent Codex task with tests and deterministic verification.

## Implementation scope
- Dashboard summary endpoint
- GET/PUT settings
- Vector search
- Metadata filtering
- Hybrid retrieval
- POST /api/v1/chat/ask
- Citation builder
- POST /api/v1/reports/generate

## Files and directories that may be touched
- `app/api/routers`
- `app/services/retrieval`
- `app/services/chat`
- `app/services/reports`
- `tests`

## Out of scope
- Do not redesign the existing UI unless this task explicitly says so.
- Do not add public cloud dependencies for tests.
- Do not remove privacy-first local execution assumptions.

## Commands Codex must run
- `make test`
- `make smoke`

## Definition of Done
- Dashboard counts are real
- Settings persist across restarts
- Chat returns answer with citations
- Report returns markdown and citations
- Lexical and semantic retrieval tests pass

## Acceptance notes
Implementation is complete only when the code, tests, and documentation are committed together and the required commands pass in a clean local environment.
