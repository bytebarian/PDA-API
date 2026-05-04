PYTHON ?= python3

.PHONY: setup upgrade-pip lint typecheck test migrate smoke

setup:
	$(PYTHON) -m pip install -e .[dev]

upgrade-pip:
	$(PYTHON) -m pip install --upgrade pip

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
