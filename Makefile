.PHONY: setup lint typecheck test migrate smoke

setup:
	pip install -e ".[dev]"

lint:
	ruff check app tests

typecheck:
	mypy app tests

test:
	pytest

migrate:
	@echo "No migrations defined yet (Alembic setup is a future task)."

smoke:
	@echo "Smoke tests not yet defined."
