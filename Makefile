# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := default

.PHONY: default install lint test upgrade build clean docs

default: install lint test

install:
	uv sync --all-extras --dev

lint:
	uv run python devtools/lint.py

test:
	uv run pytest

upgrade:
	uv sync --upgrade --all-extras --dev

build:
	uv build

docs:
	uv sync --group docs
	cd docs && uv run sphinx-build -b html . _build/html

clean:
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf .venv/
	-rm -rf docs/_build/
	-find . -type d -name "__pycache__" -exec rm -rf {} +