# Tutorial 6: Theta Sweep System Model (HD + Grid Cell)

> **Reading Time**: ~40-45 minutes
> **Difficulty**: Advanced
> **Prerequisites**: [Tutorials 1-5](./01_build_cann_model.md)

This tutorial introduces theta sweep systems combining head direction (HD) cells and grid cells with theta-modulated dynamics for realistic neural navigation.

---

## Table of Contents

1. [Introduction to Theta Sweeps](#1-introduction-to-theta-sweeps)
2. [Model Architecture](#2-model-architecture)
3. [Complete Example: Theta-Modulated Navigation](#3-complete-example-theta-modulated-navigation)
4. [Visualization and Analysis](#4-visualization-and-analysis)
5. [Next Steps](#5-next-steps)

---

## 1. Introduction to Theta Sweeps

### 1.1 What are Theta Sweeps?

**Theta oscillations** (4-12 Hz) are rhythmic brain waves prominent during navigation. **Theta sweeps** refer to systematic phase shifts in neural firing relative to the theta cycle:

- **Phase precession**: Neurons fire progressively earlier in the theta cycle as an animal moves
- **Anticipatory coding**: Neural activity "looks ahead" beyond current position
- **Speed modulation**: Theta frequency and amplitude vary with movement speed

### 1.2 Biological Basis

Experimental observations in rodents:
- **Head direction (HD) cells** show phase precession relative to turning angle
- **Grid cells** exhibit theta sweeps during locomotion
- **Theta power** increases with running speed
- **Phase relationships** coordinate information across brain regions

### 1.3 Computational Significance

Theta sweeps provide:
- **Predictive coding**: Anticipate future states during rapid movement
- **Temporal compression**: Encode sequences within single theta cycles
- **Error correction**: Compare predicted vs actual positions
- **Memory consolidation**: Facilitate hippocampal replay during rest

### 1.4 Recent Research

This tutorial implements models from:
- **Ji et al. (2025)** - "Phase Precession Relative to Turning Angle in Theta-Modulated Head Direction Cells", *Hippocampus*, 35(2)
- **Ji et al. (2025)** - "A systems model of alternating theta sweeps via firing rate adaptation", *Current Biology*, 35(4)

---

## 2. Model Architecture

### 2.1 Component Overview

The theta sweep system consists of two coupled networks:

```python
from canns.models.basic.theta_sweep_model import (
    DirectionCellNetwork,
    GridCellNetwork,
    calculate_theta_modulation
)

# Create head direction network
dc_net = DirectionCellNetwork(
    num=100,                   # Number of direction cells
    adaptation_strength=15.0,  # Spike-frequency adaptation
    noise_strength=0.0        # Activity noise
)

# Create grid cell network
gc_net = GridCellNetwork(
    num_dc=100,               # Direction cells (matches dc_net)
    num_gc_x=100,             # Grid cells per dimension
    adaptation_strength=8.0,   # SFA strength
    mapping_ratio=5,          # Controls grid spacing
    noise_strength=0.0
)
```

### 2.2 Key Parameters

**DirectionCellNetwork**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num` | 100 | Number of direction cells |
| `adaptation_strength` | 15.0 | Spike-frequency adaptation (SFA) strength |
| `noise_strength` | 0.1 | Activity noise level |
| `tau` | 10.0 | Membrane time constant |
| `tau_v` | 100.0 | Adaptation time constant |
| `k` | 0.2 | Global inhibition strength |

**GridCellNetwork**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_dc` | 100 | Number of direction cell inputs |
| `num_gc_x` | 100 | Grid cells per spatial dimension |
| `adaptation_strength` | 15.0 | SFA strength |
| `mapping_ratio` | 1 | Grid spacing control (larger = smaller spacing) |
| `phase_offset` | 1.0/20 | Drives theta sweeps (fraction of [-π,π]) |
| `noise_strength` | 0.1 | Activity noise |

**Parameter Guidelines**:
- Higher `adaptation_strength`: Stronger theta sweeps, more phase precession
- `mapping_ratio`: Determines grid cell spacing (5 = ~30cm grid spacing for typical environments)
- `phase_offset`: Controls sweep magnitude (1/20 works well for most cases)

### 2.3 Theta Modulation Mechanism

**Spike-frequency adaptation** (SFA) creates theta-like oscillations:
1. Active neurons accumulate adaptation current (`v`)
2. Adaptation current inhibits firing
3. Creates oscillatory bump dynamics
4. Speed and turning modulate oscillation amplitude

---

## 3. Complete Example: Theta-Modulated Navigation

### 3.1 Setup and Task Creation

```python
import numpy as np
import brainstate
import brainunit as u
from canns.task.open_loop_navigation import OpenLoopNavigationTask
from canns.models.basic.theta_sweep_model import (
    DirectionCellNetwork,
    GridCellNetwork,
    calculate_theta_modulation
)

# Setup
np.random.seed(10)
env_size = 1.5  # meters
simulate_time = 2.0  # seconds
dt = 0.001  # simulation time step

brainstate.environ.set(dt=1.0)

# Create navigation task
task = OpenLoopNavigationTask(
    duration=simulate_time,
    initial_head_direction=11/12 * u.math.pi,  # Starting angle
    width=env_size,
    height=env_size,
    start_pos=[env_size * 15/16, env_size * 1/16],  # Top-right corner
    speed_mean=2.0,           # m/s
    speed_std=0.0,            # Constant speed
    dt=dt,
    speed_coherence_time=10,
    rotational_velocity_std=40 * np.pi / 180,  # 40 deg/s std
)

# Generate trajectory
task.get_data()

# IMPORTANT: Calculate theta sweep parameters
task.calculate_theta_sweep_data()

print(f"Trajectory: {task.run_steps} steps")
print(f"Duration: {simulate_time}s")
```

**Key Method**: `calculate_theta_sweep_data()` computes:
- `linear_speed_gains`: Normalized forward speed [0, 1]
- `ang_speed_gains`: Normalized angular speed [-1, 1]
- These modulate theta oscillation amplitude

### 3.2 Create Networks

```python
# Create direction cell network
dc_net = DirectionCellNetwork(
    num=100,
    adaptation_strength=15,  # Strong SFA for clear theta
    noise_strength=0.0,      # No noise for clean demo
)

# Create grid cell network
mapping_ratio = 5  # Controls grid spacing
gc_net = GridCellNetwork(
    num_dc=dc_net.num,
    num_gc_x=100,
    adaptation_strength=8,   # Moderate SFA
    mapping_ratio=mapping_ratio,
    noise_strength=0.0,
)

# Initialize states
dc_net.init_state()
gc_net.init_state()

print(f"Direction cells: {dc_net.num}")
print(f"Grid cells: {gc_net.num_gc_x} x {gc_net.num_gc_x}")
```

### 3.3 Main Simulation with Theta Modulation

```python
# Extract task data
position = task.data.position
direction = task.data.hd_angle
linear_speed_gains = task.data.linear_speed_gains
ang_speed_gains = task.data.ang_speed_gains

def run_step(i, pos, hd_angle, linear_gain, ang_gain):
    """Single step with theta modulation"""

    # Calculate theta phase and modulation
    theta_phase, theta_mod_hd, theta_mod_gc = calculate_theta_modulation(
        time_step=i,
        linear_gain=linear_gain,      # Forward speed influence
        ang_gain=ang_gain,             # Turning speed influence
        theta_strength_hd=1.0,         # HD theta modulation strength
        theta_strength_gc=0.5,         # Grid theta modulation strength
        theta_cycle_len=100.0,         # Theta cycle = 100 time steps
        dt=dt,
    )

    # Update direction cells with theta modulation
    dc_net(hd_angle, theta_mod_hd)
    dc_activity = dc_net.r.value

    # Update grid cells with direction input and theta modulation
    gc_net(pos, dc_activity, theta_mod_gc)
    gc_activity = gc_net.r.value

    # Return state for analysis
    return (
        gc_net.center_position.value,  # Decoded grid position
        dc_net.center.value,            # Decoded head direction
        gc_activity,                     # Grid cell activities
        gc_net.gc_bump.value,           # Grid bump state
        dc_activity,                     # Direction cell activities
        theta_phase,                     # Current theta phase
        theta_mod_hd,                    # HD theta modulation
        theta_mod_gc,                    # Grid theta modulation
    )

# Run simulation
print("Running theta-modulated simulation...")
results = brainstate.transform.for_loop(
    run_step,
    u.math.arange(len(position)),
    position,
    direction,
    linear_speed_gains,
    ang_speed_gains,
    pbar=brainstate.transform.ProgressBar(10),
)

# Unpack results
(
    internal_position,
    internal_direction,
    gc_activity,
    gc_bump,
    dc_activity,
    theta_phase,
    theta_mod_hd,
    theta_mod_gc,
) = results

print("Simulation complete!")
print(f"Internal position tracking shape: {internal_position.shape}")
print(f"Grid cell activity shape: {gc_activity.shape}")
```

**Understanding theta modulation**:
- `theta_phase`: Oscillates between -π and π at theta frequency
- `theta_mod_hd/gc`: Multiplicative modulation (typically 0.5 to 1.5)
- **Speed dependence**: Faster movement → stronger modulation → bigger sweeps

---

## 4. Visualization and Analysis

### 4.1 Population Activity with Theta Overlay

Visualize how direction cells fire relative to theta phase:

```python
from canns.analyzer.theta_sweep import plot_population_activity_with_theta
from canns.analyzer.plotting import PlotConfigs

# Configure plot
config_pop = PlotConfigs.theta_population_activity_static(
    title="Direction Cell Population Activity with Theta",
    xlabel="Time (s)",
    ylabel="Direction (°)",
    figsize=(10, 4),
    show=True,
    save_path=None
)

# Plot population activity
plot_population_activity_with_theta(
    time_steps=task.run_steps * dt,  # Convert to seconds
    theta_phase=theta_phase,
    net_activity=dc_activity,
    direction=direction,
    config=config_pop,
    add_lines=True,     # Add theta phase lines
    atol=5e-2,          # Tolerance for line detection
)
```

**What to observe**:
- **Diagonal stripes**: Phase precession as direction changes
- **Theta lines**: Overlaid phase markers show oscillation
- **Speed effects**: Faster turning → clearer phase precession

### 4.2 Grid Cell Activity on Twisted Torus

Grid cells live on a twisted torus manifold. Visualize activity in this space:

```python
from canns.analyzer.theta_sweep import plot_grid_cell_manifold

# Transform grid positions to twisted torus coordinates
value_grid_twisted = np.dot(
    gc_net.coor_transform_inv,
    gc_net.value_grid.T
).T

# Reshape grid cell activity
grid_cell_activity = gc_activity.reshape(
    -1,
    gc_net.num_gc_1side,
    gc_net.num_gc_1side
)

# Select a frame to visualize
frame_idx = 900

# Configure plot
config_manifold = PlotConfigs.grid_cell_manifold_static(
    title="Grid Cell Activity on Twisted Torus Manifold",
    figsize=(6, 5),
    show=True,
    save_path=None
)

# Plot manifold
plot_grid_cell_manifold(
    value_grid_twisted=value_grid_twisted / mapping_ratio,
    grid_cell_activity=grid_cell_activity[frame_idx],
    config=config_manifold,
)
```

**Interpretation**:
- **Bump location**: Where the activity peak sits on the torus
- **Torus topology**: Grid cells wrap around edges periodically
- **Hexagonal structure**: Emerges from twisted torus geometry

### 4.3 Theta Sweep Animation

Create comprehensive animation showing all components:

```python
from canns.analyzer.theta_sweep import create_theta_sweep_grid_cell_animation

# Configure animation
config_anim = PlotConfigs.theta_sweep_animation(
    figsize=(12, 3),
    fps=10,
    save_path="theta_sweep_animation.gif",
    show=False
)

# Create animation
animation = create_theta_sweep_grid_cell_animation(
    position_data=position,
    direction_data=direction,
    dc_activity_data=dc_activity,
    gc_activity_data=gc_activity,
    gc_network=gc_net,
    env_size=env_size,
    mapping_ratio=mapping_ratio,
    dt=dt,
    config=config_anim,
    n_step=10,                # Sample every 10 frames
    show_progress_bar=True,
    render_backend="auto",    # Automatically choose best backend
    output_dpi=120,
)

print(f"Animation saved to: {config_anim.save_path}")
```

**Animation shows**:
1. **Trajectory panel**: Animal's path in environment
2. **Direction cells**: Ring attractor tracking head direction
3. **Grid cells**: 2D bump activity
4. **Theta phase**: Current position in theta cycle

---

## 5. Next Steps

Congratulations! You've learned how to implement and analyze theta sweep systems with head direction and grid cells.

### Key Takeaways

1. **Spike-frequency adaptation** creates theta oscillations without explicit oscillators
2. **Phase precession** emerges from speed-modulated adaptation
3. **Direction and grid cells** interact through conjunctive inputs
4. **Theta modulation** depends on both linear and angular speed
5. **Visualization tools** reveal phase relationships and manifold structure

### When to Use Theta Sweep Models

- **Realistic navigation dynamics**: Match experimental observations
- **Predictive coding research**: Study anticipatory neural activity
- **Multi-timescale processing**: Combine fast dynamics (theta) with slow memory
- **Phase coding**: Investigate temporal aspects of spatial coding
- **Biological validation**: Compare with rodent navigation data

### Parameter Tuning Tips

- **Stronger sweeps**: Increase `adaptation_strength`
- **Faster theta**: Decrease `theta_cycle_len`
- **Grid spacing**: Adjust `mapping_ratio` (larger = smaller grids)
- **Sweep magnitude**: Tune `phase_offset` in GridCellNetwork
- **Noise robustness**: Add `noise_strength` for realistic conditions

### Continue Learning

- **Next**: [Tutorial 7: Theta Sweep Place Cell Network](./07_theta_sweep_place_cell.md) - Apply theta sweeps to complex environments
- **Alternative**: Explore other scenarios for data analysis or brain-inspired learning

### Advanced Topics

- **Alternating sweeps**: Implement bidirectional phase precession
- **Multi-scale theta**: Combine with hierarchical networks (Tutorial 5)
- **Replay sequences**: Use theta sweeps for memory consolidation
- **Experimental comparison**: Validate against neural recordings

### Related Research

The models in this tutorial implement findings from:
- **Ji, Z., Ji, J., Du, J., & Wu, S. (2025)**. Phase Precession Relative to Turning Angle in Theta-Modulated Head Direction Cells. *Hippocampus*, 35(2).
- **Ji, Z., Chu, H., Wang, R., Du, J., & Wu, S. (2025)**. A systems model of alternating theta sweeps via firing rate adaptation. *Current Biology*, 35(4).

For implementation details, see the source code in `canns.models.basic.theta_sweep_model` and visualization tools in `canns.analyzer.theta_sweep`.
