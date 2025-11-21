# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := default

.PHONY: default install install-dev install-docs install-all lint test upgrade build clean docs docs-autoapi

default: install lint test

# Install only production dependencies
install:
	uv sync --all-extras

# Install production + dev dependencies
install-dev:
	uv sync --all-extras --group dev

# Install production + docs dependencies
install-docs:
	uv sync --all-extras --group docs

# Install all dependencies (production + all groups)
install-all:
	uv sync --all-extras --all-groups

lint:
	uv run python devtools/lint.py

test:
	uv run pytest

# Upgrade all dependencies
upgrade:
	uv sync --upgrade --all-extras --all-groups

# Upgrade specific group
upgrade-dev:
	uv sync --upgrade --all-extras --group dev

build:
	uv build

docs:
	uv sync --group docs
	cd docs && uv run sphinx-build -b html . _build/html

docs-autoapi:
	@echo "🔄 Removing old autoapi files..."
	-rm -rf docs/autoapi
	@echo "📚 Syncing documentation dependencies..."
	uv sync --group docs
	@echo "🔨 Rebuilding documentation with fresh autoapi..."
	cd docs && uv run sphinx-build -b html . _build/html
	@echo "✅ Done! Documentation updated at docs/_build/html/index.html"

clean:
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf .venv/
	-rm -rf docs/_build/
	-find . -type d -name "__pycache__" -exec rm -rf {} +