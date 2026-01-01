# Grid Cell Examples

This directory contains 3 core examples demonstrating grid cell models:

## 1. Position → Spatial Analysis
**File**: `grid_cell_position_spatial_analysis.py`

Demonstrates spatial structure analysis using **GridCell2DPosition**:
- Trajectory generation (OpenLoopNavigationTask)
- Firing field computation and smoothing
- Spatial autocorrelation analysis
- Grid score and spacing measurement
- Tracking animation generation

**Run**:
```bash
python grid_cell_position_spatial_analysis.py
```

**Expected**: Grid score ~0.3-0.5, hexagonal firing patterns

---

## 2. Velocity → Path Integration
**File**: `grid_cell_velocity_path_integration.py`

Verifies path integration accuracy of **GridCell2DVelocity**:
- Pure velocity-based path integration
- Position estimation from bump activity
- Error quantification over time
- Trajectory comparison visualization

**Run**:
```bash
python grid_cell_velocity_path_integration.py
```

**Expected**: Mean error <0.1m, low accumulation rate

---

## 3. Velocity → Spatial Analysis
**File**: `grid_cell_velocity_spatial_analysis.py`

Reproduces **Burak & Fiete (2009)** grid cell model with systematic spatial sampling:
- GridCell2DVelocity with healing initialization
- Systematic rate map computation (100% coverage)
- High-quality grid score analysis (>0.6)
- Detailed spatial property visualization

**Run**:
```bash
python grid_cell_velocity_spatial_analysis.py
```

**Expected**: Grid score >0.6, clear hexagonal autocorrelation

---

## Key Differences

| Feature | Position → Spatial | Velocity → Path Int. | Velocity → Spatial |
|---------|-------------------|---------------------|--------------------|
| **Model** | GridCell2DPosition | GridCell2DVelocity | GridCell2DVelocity |
| **Input** | Coordinates (x, y) | Velocity (vx, vy) | Velocity (vx, vy) |
| **Analysis** | Spatial structure | Error quantification | Spatial structure |
| **Sampling** | Trajectory-based | Trajectory-based | Systematic grid |
| **Grid Score** | ~0.3-0.5 | N/A | >0.6 |
| **Focus** | General workflow | Path integration accuracy | High-quality spatial metrics |

---

## Quick Start

```bash
# Position input - spatial analysis
python grid_cell_position_spatial_analysis.py

# Velocity input - path integration verification
python grid_cell_velocity_path_integration.py

# Velocity input - spatial analysis (high quality)
python grid_cell_velocity_spatial_analysis.py
```

Each example generates visualizations in `outputs/` directory.

---

## Understanding the Models

### GridCell2DPosition
- **Input**: Direct position coordinates (x, y)
- **Mechanism**: Position decoding from external input
- **Use case**: When absolute position is available

### GridCell2DVelocity
- **Input**: Velocity vectors (vx, vy)
- **Mechanism**: Path integration (continuous attractor dynamics)
- **Use case**: Navigation from self-motion cues only

### Systematic Spatial Sampling (grid_cell_velocity_spatial_analysis.py)
This method achieves higher grid scores by:
- Systematically sampling all spatial locations (100% coverage)
- Using state restoration for efficient uniform sampling
- Not strictly continuous trajectory (see docstring for details)
- Trading strict continuity for uniform spatial coverage
