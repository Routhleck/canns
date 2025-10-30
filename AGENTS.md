# Repository Guidelines

## Project Structure & Modules
- Source: `src/canns/` (core packages: `models/`, `trainer/`, `analyzer/`, `task/`, `pipeline/`, `data/`, `utils/`).
- Tests: `tests/` (e.g., `tests/trainer/test_*.py`, `tests/analyzer/test_*.py`).
- Docs: `docs/` (Sphinx), examples in `examples/`, dev utilities in `devtools/`, CI in `.github/`.
- Package entry: `pyproject.toml` (dynamic versioning via `uv-dynamic-versioning`).

## Build, Test, and Dev Commands
- `make install`: Install with dev + extras using `uv sync`.
- `make docs`: Build the documentation via `uv run sphinx-build`.
- `make lint`: Run `devtools/lint.py` (codespell, ruff check --fix, ruff format).
- `make test`: Run `pytest` via `uv`.
- `make build`: Build wheel/sdist with `uv build`.
- `make clean`: Remove build caches and virtual env.
Examples: `uv run pytest -q`, `uv run python devtools/lint.py`.

## Coding Style & Naming
- Python ≥ 3.11. Follow PEP 8 with ruff line length `100`.
- Indentation: 4 spaces. Use type hints where practical.
- Naming: modules and functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- Imports: keep sorted (ruff `I`), no unused imports (ruff `F`).
- Format with ruff; avoid manual formatting that fights the tool.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/` and name files `test_*.py`.
- Keep tests deterministic; prefer small synthetic arrays over large random data.
- Run fast checks locally: `make test` or `uv run pytest tests/trainer -q`.
- If adding visual outputs, assert file creation and basic shapes rather than pixels (see `tests/analyzer/test_visualize.py`).

## Commit & Pull Requests
- Commits: clear, imperative subject; include scope where helpful (e.g., `[analyzer] ...`), reference PR/issue (`#123`).
- PRs: concise description, motivation, and summary of changes; link issues; include screenshots or artifacts for visual/plot changes; ensure CI green (`lint` + `test`).
- Versioning is tag-driven; do not manually bump versions—use Git tags and follow `RELEASE.md`.

## Security & Configuration
- Use `uv` workflows; avoid committing large artifacts. Prefer `examples/` or `docs/` for notebooks.
- Optional extras: `canns[cuda12]` (Linux), `canns[tpu]` (Linux). Default CPU works everywhere.
- Keep new CLI or APIs wired via `pyproject.toml` and `src/canns/__init__.py`.

## Architecture Notes
- High-level API centered in `src/canns/`:
  - Models in `models/` (e.g., `CANN1D/2D`, brain-inspired variants).
  - Training via `trainer/` (e.g., `HebbianTrainer`).
  - Analysis and visualization in `analyzer/` with config-first plotting.
  - Dataset management in `data/` (dataset registry, downloading, loaders).
  - General utilities in `utils/` (benchmarking, etc.).

## API Policy (Development)
- Prefer trainer-led APIs. Do not call `model.predict(...)` directly; use `HebbianTrainer.predict(...)`.
- Progress: Trainer uses `tqdm` directly for training and prediction progress (no reporter abstraction).
- Models should expose minimal interfaces for trainer:
  - Weights param `W.value` (or override `weight_attr`).
  - State vector `s.value` for prediction (or override `predict_state_attr`).
  - Methods: `update(prev_energy)`, property: `energy`.
- Hebbian training is generic in trainer; avoid duplicating `apply_hebbian_learning` unless necessary.
