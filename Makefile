PYTHON ?= python3

.PHONY: setup lint typecheck test migrate smoke

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check app tests

typecheck:
	$(PYTHON) -m mypy app tests

test:
	$(PYTHON) -m pytest

migrate:
	$(PYTHON) -c "print('No Alembic migrations configured for Alpha.6 yet.')"

smoke:
	$(PYTHON) -m pytest tests/test_app.py::test_root_endpoint
