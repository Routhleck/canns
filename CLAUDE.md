# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CANNS is a Python library for Continuous Attractor Neural Networks and brain-inspired computational models. Built on BrainX (formerly BrainPy) and BrainState, it provides a unified API for experimenting with CANN architectures in computational neuroscience research.

- **Language**: Python 3.11+ with type hints
- **Package Manager**: UV (modern, fast alternative to pip)
- **Status**: Beta development stage

## Development Commands

**Setup and Dependencies:**
```bash
uv sync --all-extras --dev    # Install all dependencies
make install                  # Same as above
```

**Linting and Code Quality:**
```bash
make lint                     # Run all linting (codespell, ruff check --fix, ruff format)
uv run python devtools/lint.py  # Direct linting script
uv run ruff check src/        # Check linting only
uv run ruff format src/       # Format code only
uv run basedpyright src/      # Type checking (optional, commented out in lint script)
```

**Testing:**
```bash
make test                     # Run all tests
uv run pytest                # Direct pytest
uv run pytest tests/analyzer/  # Run specific test directory
uv run pytest -k "test_name"   # Run specific test
```

**Build and Distribution:**
```bash
make build                    # Build distribution packages
uv build                     # Direct build command
make clean                    # Clean build artifacts
```

**Development Workflow:**
```bash
make                         # Default: install + lint + test
make upgrade                 # Upgrade all dependencies
```

## Architecture Overview

### Core Structure
- **`src/canns/models/`**: Neural network implementations
  - `basic/`: Core CANN models (1D, 2D, hierarchical)
  - `brain_inspired/`: Brain-inspired architectures
  - `BaseCANN`: Abstract base class for all models
- **`src/canns/task/`**: Task definitions with standardized data handling
  - `SmoothTracking1D/2D`: Tracking tasks for testing
  - `path_integration/`: Spatial navigation tasks
- **`src/canns/analyzer/`**: Analysis and visualization tools
  - Firing rate analysis, tuning curves, raster plots
  - Animation support for network dynamics
- **`src/canns/trainer/`**: Training utilities
- **`src/canns/pipeline/`**: Data processing pipelines

### Key Dependencies
- **BrainX[cpu]**: Core neural simulation framework (JAX-based)
- **BrainState**: Neural dynamics and compilation system
- **ratinabox**: Spatial navigation modeling

### Design Patterns
- Models inherit from `brainstate.nn.Dynamics` for neural dynamics
- Tasks use abstract base classes with standardized data save/load (`.npz` format)
- Compilation system via BrainState for performance optimization
- Type hints throughout for better development experience

### Testing Organization
- Tests mirror source structure in `tests/` directory
- Visual tests generate plots/animations for verification
- Pytest configuration in `pyproject.toml` with custom test paths

### Code Quality Standards
- Line length: 100 characters (Ruff configuration)
- BasedPyright for type checking (configured to be less strict)
- Codespell for spell checking with custom ignore words
- Modern Python features (3.11+ required)

## Examples and Usage
The `examples/` directory contains practical demonstrations:
- 1D/2D tracking examples
- Oscillatory behavior demonstrations  
- Hierarchical model implementations
- Integration with BrainState's compilation system

Generated outputs (GIFs, plots) demonstrate network behavior and are part of the testing/validation process.