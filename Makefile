# Dev shortcuts. `make check` is the pre-commit habit and exactly what CI runs:
# lint (ruff), types (mypy), tests (pytest). If check passes locally, CI passes.
.PHONY: install test lint fmt type check

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff format src tests

type:
	mypy

check: lint type test
