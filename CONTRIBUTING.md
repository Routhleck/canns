# Contributing to CANNs

Thank you for your interest in contributing to CANNs! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/routhleck/canns.git
cd canns
```

2. Install development dependencies:
```bash
make install-dev
```

This will install the package with CPU support and all development tools (pytest, ruff, codespell, etc.).

For GPU support (Linux only):
```bash
make install-cuda12  # For CUDA 12
make install-cuda13  # For CUDA 13
make install-tpu     # For TPU
```

## Development Workflow

### Code Quality

Before submitting a pull request, ensure your code passes all quality checks:

```bash
make lint  # Run formatting and linting checks
make test  # Run the test suite
```

The `make lint` command runs:
- **ruff**: Code formatting and linting
- **codespell**: Spell checking
- **basedpyright**: Type checking

### Running Tests

```bash
make test
```

Tests are located in the `tests/` directory and use pytest.

### Building Documentation

```bash
make docs         # Build documentation
make docs-autoapi # Rebuild with fresh API documentation
```

Documentation will be generated in `docs/_build/html/`.

## Contribution Guidelines

### Reporting Issues

- Check existing issues before creating a new one
- Use the appropriate issue template (bug report, feature request, or custom)
- Provide clear reproduction steps for bugs
- Include relevant code snippets, error messages, and environment details

### Proposing Changes

1. **Open an issue or discussion** if you plan significant changes
2. **Fork the repository** and create a new branch for your changes
3. **Follow the existing code style** - the project uses ruff for formatting
4. **Write tests** for new functionality
5. **Update documentation** if you're adding or changing features
6. **Ensure all checks pass** by running `make lint && make test`

### Pull Request Process

1. Create a pull request with a clear title and description
2. Reference any related issues
3. Ensure CI checks pass
4. Wait for review from maintainers
5. Address any feedback or requested changes

### Commit Messages

Follow conventional commit format when possible:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions or modifications
- `refactor:` for code refactoring
- `chore:` for maintenance tasks

Example:
```
feat: add support for hierarchical CANNs
fix: resolve memory leak in CANN2D training
docs: update quick start guide with new examples
```

## Code Style

### Python Style Guide

- Follow PEP 8 conventions
- Line length: 100 characters (configured in pyproject.toml)
- Use type hints where appropriate
- Write clear, self-documenting code

### Formatting

The project uses ruff for automatic formatting. Run `make lint` to format your code.

### Naming Conventions

- Classes: `PascalCase` (e.g., `CANN1D`, `SmoothTracking1D`)
- Functions/methods: `snake_case` (e.g., `get_data`, `run_simulation`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMESTEP`)
- Private members: prefix with `_` (e.g., `_internal_state`)

## Project Structure

```
src/canns/          # Core library modules
├── models/         # Neural network models
│   ├── basic/      # Basic CANN models (1D, 2D, SFA, hierarchical)
│   └── brain_inspired/  # Brain-inspired models (Hopfield, etc.)
├── task/           # Task generators
├── analyzer/       # Analysis tools
├── trainer/        # Training utilities
├── pipeline/       # End-to-end pipelines
└── visualization/  # Plotting and visualization

docs/               # Sphinx documentation and notebooks
examples/           # Ready-to-run example scripts
tests/              # Test suite
devtools/           # Development utilities
scripts/            # Utility scripts
```

## Adding New Features

### Adding a New Model

1. Create your model class in `src/canns/models/basic/` or `src/canns/models/brain_inspired/`
2. Inherit from `BasicModel` base class
3. Implement required methods
4. Add tests in `tests/`
5. Add examples in `examples/`
6. Update documentation

### Adding a New Task

1. Create your task class in `src/canns/task/`
2. Inherit from `Task` base class
3. Implement task-specific data generation
4. Add tests and examples
5. Update documentation

### Adding Analysis Tools

1. Add analysis functions to `src/canns/analyzer/`
2. Follow existing patterns for consistency
3. Include visualization capabilities where appropriate
4. Add tests and examples

## Testing

- Write unit tests for new functionality
- Ensure tests are deterministic and reproducible
- Use fixtures for common test setup
- Test edge cases and error conditions
- Aim for good test coverage

## Documentation

- Update docstrings for new or modified functions/classes
- Follow NumPy docstring format
- Add examples to docstrings where helpful
- Update relevant notebooks in `docs/en/notebooks/`
- Add new examples to `examples/` directory

## Getting Help

- Open a [discussion](https://github.com/routhleck/canns/discussions) for questions
- Check existing [issues](https://github.com/routhleck/canns/issues) and documentation
- Reach out to maintainers if you need guidance

## License

By contributing to CANNs, you agree that your contributions will be licensed under the Apache License 2.0.

## Recognition

Contributors will be acknowledged in the project. Significant contributions may be highlighted in release notes.

Thank you for contributing to CANNs!