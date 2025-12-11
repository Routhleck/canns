# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := default

.PHONY: default install install-dev install-docs install-all install-cuda12 install-cuda13 install-tpu lint test upgrade upgrade-dev build clean docs docs-autoapi

default: install lint test

# Install only production dependencies (CPU-only by default)
install:
	uv sync --extra cpu

# Install production + dev dependencies (CPU-only)
install-dev:
	uv sync --extra cpu --group dev

# Install production + docs dependencies (CPU-only)
install-docs:
	uv sync --extra cpu --group docs

# Install all dependencies (production + all groups, CPU-only)
install-all:
	uv sync --extra cpu --all-groups

# Install with CUDA 12 support (Linux only)
install-cuda12:
	uv sync --extra cuda12 --group dev

# Install with CUDA 13 support (Linux only)
install-cuda13:
	uv sync --extra cuda13 --group dev

# Install with TPU support (Linux only)
install-tpu:
	uv sync --extra tpu --group dev

lint:
	uv run python devtools/lint.py

test:
	uv run pytest

# Upgrade all dependencies (CPU-only)
upgrade:
	uv sync --upgrade --extra cpu --all-groups

# Upgrade specific group (CPU-only)
upgrade-dev:
	uv sync --upgrade --extra cpu --group dev

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