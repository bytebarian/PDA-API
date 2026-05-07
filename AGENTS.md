# AGENTS.md – bytebarian/PDA-API

This file is the repository-level contract for future Codex runs in the backend API repository.

## Repository purpose

PDA-API is the FastAPI backend for the Personal Documents Assistant. The current Foundation state is a backend shell with configuration, database wiring, Docker-based local infrastructure, Alembic migrations, and operational health endpoints. Business features such as document ingestion, OCR, embeddings, chat, reporting, and richer domain models are intentionally out of scope unless the task explicitly asks for them.

## Backend architecture

- `app/main.py` creates the FastAPI application and applies the configured API prefix.
- `app/api/router.py` is the top-level router aggregator.
- `app/api/routers/` contains HTTP endpoints. Current operational endpoints include `/`, `/health/live`, and `/health/ready`.
- `app/core/config.py` is the authoritative runtime settings module. Settings use `pydantic-settings`, the `PDA_` env prefix, and `.env` loading.
- `app/db/base.py` defines the SQLAlchemy declarative base.
- `app/db/session.py` owns the async SQLAlchemy engine, async session factory, and `get_db` dependency.
- `alembic/` contains migration environment code and versioned revisions.
- `docker-compose.yml` defines the local development stack with PostgreSQL + pgvector and the API container.
- `docker/initdb/01-enable-pgvector.sql` enables the `vector` extension for local PostgreSQL containers.
- `tests/` contains pytest coverage for the current Foundation surface area.

Empty or near-empty packages such as `app/models`, `app/services`, `app/adapters`, `app/schemas`, and `app/workers` are placeholders for later phases. Do not invent new production architecture during Foundation unless a task explicitly requires it.

## Directory structure

```text
app/
  api/
    router.py
    routers/
  core/
  db/
  adapters/
  models/
  schemas/
  services/
  workers/
alembic/
  versions/
docker/
  initdb/
tests/
```

## Authoritative command contract

Use these Make targets exactly as written:

- `make setup` — install the project in editable mode with development dependencies.
- `make lint` — run Ruff against `app` and `tests`.
- `make typecheck` — run MyPy against `app` and `tests`.
- `make test` — run the pytest suite.
- `make migrate` — run `alembic upgrade head`.
- `make smoke` — reserved command name for smoke validation. It currently fails intentionally until smoke coverage is implemented.

If AGENTS.md and the Makefile ever disagree, update them together so the command names and behavior stay aligned.

## Local development with Docker

The repository ships with Docker Compose for local PostgreSQL + pgvector development.

### First-time setup

```bash
cp .env.example .env
```

### Start PostgreSQL + API

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- PostgreSQL: `localhost:5432`

### Start only PostgreSQL

```bash
docker compose up db
```

When running the API outside Docker, set `PDA_DATABASE_URL` to the PostgreSQL DSN from `.env.example`, for example:

```bash
PDA_DATABASE_URL=postgresql+asyncpg://pda:pda_dev@localhost:5432/pda
```

### Stop the stack

```bash
docker compose down
```

To also remove the database volume:

```bash
docker compose down -v
```

### Verify pgvector

```bash
docker compose exec db sh -lc 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT extname FROM pg_extension WHERE extname = '\''vector'\'';"'
```

## Alembic workflow

- Alembic configuration lives in `alembic.ini` and `alembic/env.py`.
- `alembic/env.py` reads the database URL from application settings, so `.env` and `PDA_DATABASE_URL` apply to migrations too.
- Apply the latest revision with:

```bash
make migrate
```

- New migration revisions should only be created when a task explicitly changes schema.

## Operational endpoints

- `GET /health/live` — liveness probe for the API process.
- `GET /health/ready` — readiness probe that checks database connectivity.

Keep these exact paths stable unless a task explicitly changes the operational contract.

## Coding and change rules

- Keep changes tightly scoped to the assigned task.
- One task per PR; do not bundle unrelated work.
- Do not perform unrelated refactors while addressing a focused issue.
- Do not expand business scope during infrastructure or documentation tasks.
- Preserve privacy-first, local-first behavior.
- Do not add external cloud dependencies for local development or tests unless the task explicitly requires them.
- Reuse the existing architecture and module boundaries instead of redesigning the app structure.
- Update documentation when code changes would otherwise make it inaccurate.
- Add or update tests when production behavior changes. Documentation-only tasks do not need new tests unless existing test infrastructure already covers the changed contract.

## Validation rules for Codex runs

Before finishing a non-trivial task, run the canonical validation commands that apply to the changes:

```bash
make lint
make typecheck
make test
make migrate
```

Run targeted tests early when possible, then rerun the relevant full validation before completion. Use `make smoke` only when smoke tests exist for the task being implemented.
