# Tutorial 7: Theta Sweep Place Cell Network

> **Reading Time**: ~35-40 minutes
> **Difficulty**: Advanced
> **Prerequisites**: [Tutorials 1-6](./01_build_cann_model.md)

This tutorial introduces theta-modulated place cell networks using geodesic distances for complex environments with obstacles and non-convex geometries.

---

## Table of Contents

1. [Introduction to Place Cell Networks](#1-introduction-to-place-cell-networks)
2. [Geodesic Distance and Complex Environments](#2-geodesic-distance-and-complex-environments)
3. [Complete Example: T-Maze Navigation](#3-complete-example-t-maze-navigation)
4. [Visualization and Analysis](#4-visualization-and-analysis)
5. [Next Steps](#5-next-steps)

---

## 1. Introduction to Place Cell Networks

### 1.1 What are Place Cells?

**Place cells** are hippocampal neurons that fire when an animal occupies specific locations in its environment:

- **Localized firing fields**: Each cell fires in one or more specific regions
- **Population coverage**: Different cells tile the entire environment
- **Context dependence**: Firing patterns can differ across environments
- **Theta modulation**: Firing shows phase precession relative to theta oscillations

### 1.2 Why Geodesic Distances?

Traditional CANN models use Euclidean distance, which fails in complex environments:

**Euclidean distance problems**:
- Cannot handle obstacles or walls
- Inappropriate for non-convex spaces (mazes, T-mazes)
- Ignores environment topology

**Geodesic distance solution**:
- Computes shortest path along accessible surfaces
- Respects barriers and obstacles
- Naturally handles complex geometries
- More biologically realistic connectivity

### 1.3 Graph-Based Continuous Attractors

Place cell networks use **graph-based attractors**:

1. Discretize environment into grid
2. Compute geodesic distances between all grid points
3. Build connectivity matrix based on geodesic (not Euclidean) distances
4. Continuous attractor emerges on this graph structure

---

## 2. Geodesic Distance and Complex Environments

### 2.1 Conceptual Overview

**Geodesic distance** is the shortest path between two points constrained to accessible space:

```
In open field:     geodesic ≈ Euclidean
With wall between: geodesic >> Euclidean (must go around)
In maze:           geodesic follows corridor structure
```

### 2.2 Computing Geodesic Distances

The CANNs library handles geodesic computation automatically:

```python
from canns.task.open_loop_navigation import TMazeRecessOpenLoopNavigationTask

# Create task with T-maze geometry
task = TMazeRecessOpenLoopNavigationTask(
    w=0.84,           # Corridor width
    l_s=3.64,         # Stem length
    l_arm=2.36,       # Arm length
    t=1.0,            # Junction thickness
    recess_width=0.2,
    recess_depth=0.2,
    # ... other parameters
)

task.get_data()

# CRITICAL: Set grid resolution before computing geodesics
task.set_grid_resolution(0.05, 0.05)  # 5cm x 5cm grid

# Compute geodesic distance matrix
geodesic_result = task.compute_geodesic_distance_matrix()
```

**Key steps**:
1. `set_grid_resolution()`: Defines spatial discretization (smaller = more accurate but slower)
2. `compute_geodesic_distance_matrix()`: Returns connectivity information for network

### 2.3 Supported Environment Types

**TMazeRecessOpenLoopNavigationTask**: T-maze with recesses (Chu et al. 2024)
- Customizable arm lengths and widths
- Recesses for turn-around behavior
- Biologically realistic geometry

**Future support**: Custom polygon environments, multi-room structures, open field with obstacles

---

## 3. Complete Example: T-Maze Navigation

### 3.1 Setup and Task Creation

```python
import numpy as np
import brainstate
import brainunit as u
from canns.models.basic.theta_sweep_model import PlaceCellNetwork
from canns.task.open_loop_navigation import TMazeRecessOpenLoopNavigationTask

# Setup
np.random.seed(10)
simulate_time = 3.0  # seconds
dt = 0.001

brainstate.environ.set(dt=1.0)

# Create T-maze navigation task
# Parameters from Chu et al. (2024) Table 3
task = TMazeRecessOpenLoopNavigationTask(
    duration=simulate_time,
    w=0.84,                    # Corridor width (m)
    l_s=3.64,                  # Stem length (m)
    l_arm=2.36,                # Arm length (m)
    t=1.0,                     # T-junction thickness (m)
    start_pos=(0.0, 0.6),      # Start in stem
    recess_width=0.2,          # Recess dimensions
    recess_depth=0.2,
    initial_head_direction=1/2 * u.math.pi,  # Facing up
    speed_mean=1.2,            # m/s
    speed_std=0.0,             # Constant speed
    rotational_velocity_std=0,  # No rotation noise
    dt=dt,
)

# Generate trajectory
task.get_data()

# Calculate theta sweep parameters
task.calculate_theta_sweep_data()

# Set grid resolution (CRITICAL STEP)
task.set_grid_resolution(0.05, 0.05)  # 5cm grid

# Compute geodesic distances
print("Computing geodesic distance matrix...")
geodesic_result = task.compute_geodesic_distance_matrix()
print("Geodesic computation complete!")

print(f"Trajectory: {task.run_steps} steps")
print(f"Grid resolution: 5cm x 5cm")
```

**Grid resolution trade-offs**:
- Smaller (e.g., 0.02): More accurate, but slower computation
- Larger (e.g., 0.10): Faster, but coarser connectivity
- **Recommended**: 0.05m (5cm) for most applications

### 3.2 Create Place Cell Network

```python
# Create place cell network with T-maze parameters
# Parameters from Chu et al. (2024) Table 3
pc_net = PlaceCellNetwork(
    geodesic_result,           # From task.compute_geodesic_distance_matrix()
    tau=3.0,                   # Fast membrane time constant (ms)
    tau_v=150.0,               # Slow adaptation time constant (ms)
    noise_strength=0.05,       # Small activity noise
    k=1.40,                    # Global inhibition
    m=1.1,                     # Adaptation strength
    a=0.3,                     # Local excitation range (grid units)
    A=2.3,                     # Excitation amplitude
    J0=0.25,                   # Baseline synaptic strength
    g=20.0,                    # Firing rate gain
    conn_noise=0.0,            # No connectivity noise
)

# Initialize state
pc_net.init_state()

print(f"Place cells: {pc_net.num}")
print("Network initialized")
```

**Parameter notes**:
- `tau` and `tau_v`: Create timescale separation for theta oscillations
- `a`: Connection width in grid units (0.3 = ~6 grid points)
- `k` and `m`: Balance excitation/inhibition and adaptation
- These values are optimized for T-maze geometry from published research

### 3.3 Warmup Phase

**CRITICAL**: Run warmup before main simulation to stabilize network state:

```python
# Warmup period: initialize at starting position
warmup_time = 0.1  # seconds
warmup_steps = int(warmup_time / dt)

position = task.data.position
linear_speed_gains = task.data.linear_speed_gains

print(f"Running warmup for {warmup_time}s ({warmup_steps} steps)...")

def warmup_step(i):
    """Warmup step without theta modulation"""
    pc_net(position[0], 1.0)  # theta_modulation = 1.0 (no oscillation)
    return None

brainstate.transform.for_loop(
    warmup_step,
    u.math.arange(warmup_steps),
    pbar=brainstate.transform.ProgressBar(10),
)

print("Warmup complete")
```

**Why warmup is necessary**:
- Allows network to form stable bump at starting location
- Without warmup, transients can cause incorrect initial tracking
- Typically 100-300ms sufficient (matches biological timescales)

### 3.4 Main Simulation with Theta Modulation

```python
def run_step(i, pos, vel_gain, theta_strength=0.1, theta_cycle_len=100):
    """Single step with theta modulation"""

    # Calculate theta phase
    t = i * brainstate.environ.get_dt()
    theta_phase = u.math.mod(t, theta_cycle_len) / theta_cycle_len
    theta_phase = theta_phase * 2 * u.math.pi - u.math.pi  # [-π, π]

    # Compute theta modulation
    theta_modulation = 1 + theta_strength * vel_gain * u.math.cos(theta_phase)

    # Update network
    pc_net(pos, theta_modulation)

    return (
        pc_net.center.value,       # Decoded position
        pc_net.r.value,            # Place cell activities
        theta_phase,                # Current theta phase
        theta_modulation,           # Theta modulation value
    )

# Run main simulation
print("Running theta-modulated simulation...")
results = brainstate.transform.for_loop(
    run_step,
    u.math.arange(len(position)),
    position,
    linear_speed_gains,
    pbar=brainstate.transform.ProgressBar(10),
)

# Unpack results
(
    internal_position,
    net_activity,
    theta_phase,
    theta_modulation,
) = results

print("Simulation complete!")
print(f"Network activity shape: {net_activity.shape}")
```

**Theta parameters**:
- `theta_strength=0.1`: Moderate theta modulation (10% amplitude)
- `theta_cycle_len=100`: Theta cycle = 100ms (10 Hz frequency)
- `vel_gain`: Speed-dependent modulation (faster = stronger theta)

---

## 4. Visualization and Analysis

### 4.1 Environment and Trajectory

Visualize the T-maze geometry and trajectory:

```python
# Show trajectory overlaid on environment
task.show_data(
    show=True,
    overlay_movement_cost=True,  # Show accessible regions
    save_path=None
)
```

**What to observe**:
- T-maze structure with stem and two arms
- Trajectory following corridors
- Movement cost showing walls vs accessible space

### 4.2 Geodesic Distance Matrix

Visualize the computed geodesic distances:

```python
# Show geodesic distance heatmap
task.show_geodesic_distance_matrix(
    show=True,
    save_path=None
)
```

**Interpretation**:
- **Block structure**: Reflects maze topology
- **Gradual transitions**: Within corridors
- **Sharp boundaries**: At walls and turns
- Distances respect maze geometry (not straight-line)

### 4.3 Population Activity Heatmap

Show all place cells' activity over time:

```python
from canns.analyzer.plotting.spikes import population_activity_heatmap

# Create population activity heatmap
population_activity_heatmap(
    activity_data=net_activity,
    dt=dt,
    title="Place Cell Population Activity",
    figsize=(10, 6),
    cmap="viridis",
    save_path=None,
    show=True
)
```

**Expected patterns**:
- **Sequential activation**: As animal moves through maze
- **Repeating patterns**: If trajectory revisits locations
- **Theta sweeps**: Fine temporal structure within bumps

### 4.4 Theta Sweep Animation

Create comprehensive animation showing place cell dynamics:

```python
from canns.analyzer.theta_sweep import create_theta_sweep_place_cell_animation

# Create animation (this may take time)
print("Creating theta sweep animation...")
create_theta_sweep_place_cell_animation(
    position_data=position,
    pc_activity_data=net_activity,
    pc_network=pc_net,
    navigation_task=task,
    n_step=20,              # Sample every 20 frames
    fps=10,                  # 10 frames per second
    figsize=(14, 5),
    save_path="place_cell_theta_sweep.gif",
    show=False               # Don't display to avoid errors
)

print("Animation saved to: place_cell_theta_sweep.gif")
```

**Animation panels**:
1. **Environment**: T-maze with current position
2. **Place cell activity**: Network bump in graph space
3. **Theta phase**: Current position in oscillation cycle

---

## 5. Next Steps

Congratulations! You've completed all seven foundation and advanced tutorials for CANN modeling.

### Key Takeaways

1. **Geodesic distances** enable place cells in complex environments
2. **Graph-based attractors** handle arbitrary geometries
3. **Grid resolution** trades accuracy for computation speed
4. **Warmup phase** is critical for stable initialization
5. **Theta modulation** creates realistic place cell dynamics

### When to Use Place Cell Networks

- **Complex environments**: Mazes, multi-room structures, obstacles
- **Realistic spatial coding**: Match hippocampal recordings
- **Navigation research**: Study path planning and memory
- **Environment topology**: When geometry matters (not just coordinates)
- **Biological models**: Hippocampal place cell simulations

### Parameter Tuning Guidelines

**Grid resolution**:
- Start with 0.05m (5cm) for most environments
- Use 0.02m for high-precision studies
- Increase to 0.10m for faster prototyping

**Network parameters**:
- `k` and `m`: Balance bump stability vs responsiveness
- `a`: Adjust for environment size (larger environments may need larger `a`)
- `tau` / `tau_v`: Create timescale separation for oscillations
- `theta_strength`: Increase for more pronounced sweeps

**Environment types**:
- T-maze: Use `TMazeRecessOpenLoopNavigationTask`
- Custom geometries: Will be supported in future versions
- Open field with obstacles: Coming soon

### Completed Tutorial Series

You now have completed:

**Foundation (Tutorials 1-4)**:
- Basic CANN models and parameters
- Task generation and simulation
- Visualization methods
- Parameter exploration

**Advanced Applications (Tutorials 5-7)**:
- Hierarchical path integration (multi-scale grids)
- Theta sweeps with HD + grid cells
- Place cells in complex environments

### Continue Your Journey

Explore other scenarios:
- **Scenario 2: Data Analysis** - Analyze experimental neural recordings
- **Scenario 3: Brain-Inspired Learning** - Train CANNs on memory tasks
- **Scenario 4: End-to-End Pipeline** - Complete research workflows

### Advanced Research Directions

- **Multi-compartment models**: Combine hierarchical networks with theta sweeps
- **Learning dynamics**: Train place cells from sensory inputs
- **Memory consolidation**: Implement replay sequences during rest
- **Experimental validation**: Compare with rodent hippocampal data
- **Neural decoding**: Extract position from place cell populations

### Related Publications

This tutorial implements models from:
- **Chu, H., Ji, Z., Wang, R., Du, J., & Wu, S. (2024)**. Theta sweeps in the hippocampus. *Journal of Neuroscience Research* (parameters from Table 3).

For implementation details, see:
- Source code: `canns.models.basic.theta_sweep_model.PlaceCellNetwork`
- Visualization: `canns.analyzer.theta_sweep`
- Navigation tasks: `canns.task.open_loop_navigation`

### Community and Support

- **Documentation**: [https://canns.readthedocs.io](https://canns.readthedocs.io)
- **GitHub**: [https://github.com/PKU-NIP-Lab/CANNs](https://github.com/PKU-NIP-Lab/CANNs)
- **Examples**: See `examples/cann/` directory for more usage patterns

Thank you for completing the CANN modeling tutorial series!
