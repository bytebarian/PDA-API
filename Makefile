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
	alembic upgrade head

smoke:
	$(error smoke: No smoke tests defined yet. Add smoke tests before using this target.)
