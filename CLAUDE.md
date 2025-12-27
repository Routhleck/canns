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
- `src/canns/analyzer/` - Analysis and visualization tools
  - `metrics/` - Computational analysis (spatial metrics, experimental data analysis)
  - `visualization/` - Plotting and animation (unified PlotConfig system)
  - `slow_points/` - Fixed point analysis
  - `model_specific/` - Specialized model analyzers
- `src/canns/trainer/` - Training framework
- `examples/` - Usage demonstrations

### Standard Computation Pattern

All simulations follow this JAX-compiled loop pattern:

```python
# 1. Initialize environment and model
import brainpy.math as bm
bm.set_dt(0.1)

# 2. Define step function
def run_step(t, inputs):
    model(inputs)
    return model.u.value, model.r.value

# 3. Run compiled loop
results = bm.for_loop(
    run_step, time_steps, data
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

### Animation Performance Best Practices

The visualization system has been optimized for high-performance animation generation. Follow these guidelines for best performance:

#### 1. Use MP4 Format (36.8x Faster Encoding)

**Always prefer MP4 over GIF** for animation output:

```python
# ✅ Recommended: MP4 format (fast encoding, full color, smaller files)
create_1d_bump_animation(data, save_path='output.mp4', fps=30)

# ❌ Avoid: GIF format (36.8x slower encoding, 256 colors, larger files)
create_1d_bump_animation(data, save_path='output.gif', fps=30)
```

**Performance comparison (100 frames)**:
- MP4 (H.264): 986 FPS encoding (0.1s total)
- GIF: 27 FPS encoding (3.7s total)

**When to use GIF**: Only if you need inline display in GitHub README or other platforms that don't support video.

#### 2. Use Blitting for 2D Animations (4.6x Faster Rendering)

For custom animations, use the optimized blitting pattern:

```python
from canns.analyzer.visualization.core import OptimizedAnimationBase, AnimationConfig

class MyAnimation(OptimizedAnimationBase):
    def create_artists(self):
        """Pre-create all artist objects (called once)"""
        self.line, = self.ax.plot([], [], animated=True)
        self.scatter = self.ax.scatter([], [], animated=True)
        return [self.line, self.scatter]

    def update_frame(self, frame_idx):
        """Update artist data only (no ax.clear!)"""
        x, y = self.compute_frame_data(frame_idx)
        self.line.set_data(x, y)
        self.scatter.set_offsets(self.points[frame_idx])
        return (self.line, self.scatter)

# Configure and render
config = AnimationConfig(fps=30, enable_blitting=True)
animation = MyAnimation(fig, ax, config)
animation.render_animation(nframes=200, save_path='output.mp4')
```

**Key principles**:
- ❌ Never use `ax.clear()` in update function
- ✅ Pre-create artists with `animated=True`
- ✅ Update data via `set_data()`, `set_offsets()`, etc.
- ✅ Enable blitting: `blit=True` in FuncAnimation

#### 3. Use 2D Projections for Fast Previews (10-20x Faster)

For CANN2D visualization, choose format based on use case:

```python
# Fast 2D projection mode (recommended for daily analysis)
plot_2d_bump_on_manifold(
    cann2d, data,
    mode='fast',  # 2D heatmap with blitting
    save_path='analysis.mp4'
)

# 3D torus mode (for presentations/publications)
plot_3d_bump_on_torus(
    cann2d, data,
    save_path='presentation.mp4'
)
```

**Performance**: 2D mode achieves 40-50 FPS rendering vs 2-3 FPS for 3D surfaces.

#### 4. Use OptimizedAnimationWriter

For fine-grained control, use the optimized writer directly:

```python
from canns.analyzer.visualization.core import create_optimized_writer

# Automatically selects best format based on file extension
writer = create_optimized_writer('output.mp4', fps=30)

# Or explicitly configure
from canns.analyzer.visualization.core import OptimizedAnimationWriter
writer = OptimizedAnimationWriter(
    format='mp4',
    fps=30,
    quality='high',  # Options: 'draft', 'medium', 'high'
    codec='h264',
    bitrate=5000
)
```

#### 5. Configuration Options

Use `AnimationConfig` for advanced control:

```python
from canns.analyzer.visualization.core import AnimationConfig

# High-quality production rendering
config = AnimationConfig(
    fps=30,
    enable_blitting=True,
    quality='high',
    use_parallel=False  # Auto-enabled for >500 frames
)

# Fast draft mode for quick iteration
config = AnimationConfig(
    fps=15,
    quality='draft',  # Reduces resolution by 50%
    use_parallel=True,
    num_workers=4
)
```

#### 6. Parallel Rendering (3-4x Faster for Long Animations)

For animations with >500 frames, parallel rendering is automatically enabled:

```python
# Automatically uses parallel rendering for long animations
create_grid_cell_tracking_animation(
    position, activity, rate_map,
    config=PlotConfigs.grid_cell_tracking_animation(
        fps=20,
        save_path='long_trajectory.mp4'
    )
)
```

Manual control:

```python
from canns.analyzer.visualization.core import ParallelAnimationRenderer

renderer = ParallelAnimationRenderer(num_workers=4)
renderer.render_parallel(animation_func, nframes=1000, output_path='output.mp4')
```

#### 7. Jupyter Notebook Display Optimization

For Jupyter notebooks, use the optimized HTML5 video display (default since v2.0):

```python
# Automatic (uses html5 by default)
create_animation(data, show=True)  # Shows in notebook with HTML5 video

# Manual control (if needed)
from canns.analyzer.visualization import display_animation_in_jupyter

ani = create_animation(data, show=False)
display_animation_in_jupyter(ani, format='html5')  # Recommended: 2x faster
# or
display_animation_in_jupyter(ani, format='jshtml')  # Fallback (no FFmpeg needed)
```

**Performance comparison (100 frames)**:

| Format | Time | Size | Notes |
|--------|------|------|-------|
| html5 (default) | 1.3s | 134 KB | Fast, small, smooth playback |
| jshtml (fallback) | 2.6s | 4837 KB | 2x slower, 36x larger |

**Recommendation**: Use `show=False` when saving files to avoid double rendering:

```python
# ✅ Best practice: Save only, display later from file
create_animation(data, save_path='output.mp4', show=False)

# ⚠️ Avoid: Both save and show (renders twice)
create_animation(data, save_path='output.mp4', show=True)
```

#### Performance Summary

Optimization techniques and their speedups:

| Technique | Speedup | Use Case |
|-----------|---------|----------|
| MP4 vs GIF encoding | 36.8x | All animations |
| Notebook html5 vs jshtml | 2.0x | Jupyter display |
| Blitting (2D) | 4.6x | Custom 2D animations |
| 2D projection vs 3D | 10-20x | CANN2D visualization |
| Parallel rendering | 3-4x | Long animations (>500 frames) |
| **Combined workflow** | **11.2x** | Complete animation pipeline |

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
