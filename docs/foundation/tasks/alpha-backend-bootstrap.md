# Alpha – Backend workspace bootstrap and agent contract

## Repository
bytebarian/PDA-API

## Objective
Deliver this Foundation slice as an independent Codex task with tests and deterministic verification.

## Implementation scope
- Create FastAPI skeleton
- Configure settings and logging
- Add Docker Compose with PostgreSQL and pgvector
- Setup SQLAlchemy async and Alembic
- Add health endpoints
- Add AGENTS.md and Makefile

## Files and directories that may be touched
- `app/main.py`
- `app/api/routers`
- `app/core`
- `app/db`
- `alembic`
- `docker-compose.yml`
- `Makefile`
- `AGENTS.md`
- `tests`

## Out of scope
- Do not redesign the existing UI unless this task explicitly says so.
- Do not add public cloud dependencies for tests.
- Do not remove privacy-first local execution assumptions.

## Commands Codex must run
- `make setup`
- `make lint`
- `make typecheck`
- `make test`
- `make migrate`

## Definition of Done
- API boots locally
- /health/live returns 200
- /health/ready checks database
- Alembic is initialized
- Codex commands are documented

## Acceptance notes
Implementation is complete only when the code, tests, and documentation are committed together and the required commands pass in a clean local environment.
