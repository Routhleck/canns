# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
make install          # Install all dependencies with uv sync --all-extras --dev
```

### Code Quality
```bash
make lint             # Run ruff check, ruff format, and codespell
make check            # Run basedpyright type checking
```

### Testing
```bash
make test             # Run full pytest test suite
pytest tests/models/  # Run specific test directory
pytest -v -k "test_cann1d"  # Run specific test pattern
```

### Build
```bash
make build            # Build package with uv build
```

### Documentation
```bash
make docs             # Build Sphinx documentation
```

## Architecture Overview

CANNs is a Python library for Continuous Attractor Neural Networks focused on spatial cognition and neural dynamics. Built on JAX/BrainX for high-performance computation with GPU/TPU support.

### Core Model Hierarchy
```
BaseCANN (abstract)
├── BaseCANN1D → CANN1D, CANN1D_SFA
├── BaseCANN2D → CANN2D, CANN2D_SFA
└── HierarchicalNetwork (grid cells, place cells, band cells)
```

### Key Directories
- `src/canns/models/basic/` - Core CANN implementations
- `src/canns/models/brain_inspired/` - Bio-inspired models (under development)
- `src/canns/task/` - Task definitions for tracking, navigation, population coding
- `src/canns/analyzer/` - Visualization and analysis tools with unified PlotConfig system
- `src/canns/trainer/` - Training framework
- `examples/` - Usage demonstrations

### Standard Computation Pattern
All simulations follow this JAX-compiled loop pattern:
```python
# 1. Initialize environment and model
brainstate.environ.set(dt=0.1)
model.init_state()

# 2. Define step function
def run_step(t, inputs):
    model(inputs)
    return model.u.value, model.r.value

# 3. Run compiled loop
results = brainstate.transform.for_loop(
    run_step, time_steps, data,
    pbar=brainstate.transform.ProgressBar(10)
)
```

### Visualization System
Uses unified PlotConfig dataclasses:
```python
# Modern approach (preferred)
config = PlotConfig.for_animation(figsize=(8, 6), interval=50)
analyzer.animate_dynamics(cann, config=config)

# Legacy function calls still supported
analyzer.animate_dynamics(cann, figsize=(8, 6), interval=50)
```

## Development Guidelines

### Model Development
- New basic models go in `models/basic/`
- Brain-inspired models go in `models/brain_inspired/`
- Follow the BaseCANN inheritance pattern
- Always implement required abstract methods: `cell_coords()`, `f_r()`, `f_u()`, `f_r_given_u()`

### Testing
- Place tests in corresponding `tests/` subdirectories
- Test model initialization, forward pass, and key behaviors
- Use pytest fixtures for common setup

### Dependencies
- Core: JAX, BrainX, NumPy
- Visualization: matplotlib
- Progress: tqdm
- Build: uv (not pip/conda)

### File Organization
- Models are organized by capability level: basic → brain_inspired → hybrid
- Tasks are organized by function: tracking, navigation, population coding
- Keep related functionality in the same module where possible