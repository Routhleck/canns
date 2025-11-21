# Tutorial 4: CANN Parameter Effects

> **Reading Time**: ~30-35 minutes
> **Difficulty**: Intermediate
> **Prerequisites**: [Tutorial 1-3](./01_build_cann_model.md)

This tutorial systematically explores how different CANN1D parameters affect model dynamics.

---

## Table of Contents

1. [Parameter Overview](#1-parameter-overview)
2. [Experimental Setup](#2-experimental-setup)
3. [Parameter Exploration](#3-parameter-exploration)
4. [Next Steps](#4-next-steps)

---

## 1. Parameter Overview

CANN1D has the following main parameters:

| Parameter | Default | Physical Meaning |
|-----------|---------|------------------|
| `num` | 256 | Number of neurons (network resolution) |
| `tau` | 1.0 | Time constant (dynamics speed) |
| `k` | 8.1 | Global inhibition strength |
| `a` | 0.5 | Connection width (local excitation range) |
| `A` | 10 | External input amplitude |
| `J0` | 4.0 | Synaptic connection strength |
| `tau_v` | 50.0 | SFA time constant (CANN1D_SFA only) |

We'll use **SmoothTracking1D** task and **energy_landscape_1d_animation** visualization to observe each parameter's effect.

---

## 2. Experimental Setup

### 2.1 Base Code Framework

```python
import brainstate
from canns.models.basic import CANN1D, CANN1D_SFA
from canns.task import SmoothTracking1D
from canns.analyzer.plotting import (
    PlotConfigs,
    energy_landscape_1d_animation
)

# Setup environment
brainstate.environ.set(dt=0.1)

def run_experiment(model, title="", save_path=None):
    """Run standard experiment and visualize results"""
    model.init_state()

    # Create smooth tracking task
    task = SmoothTracking1D(
        cann_instance=model,
        Iext=[-2.0, 2.0],
        duration=[50.0],
        time_step=0.1,
    )

    # Get task data
    task.get_data()

    def run_step(t, inp):
        model.update(inp)
        return model.u.value, model.r.value

    u_history, r_history = brainstate.transform.for_loop(
        run_step,
        task.run_steps,
        task.data,
        pbar=brainstate.transform.ProgressBar(10)
    )

    # Configure and create visualization
    config = PlotConfigs.energy_landscape_1d_animation(
        time_steps_per_second=100,
        fps=20,
        title=title,
        xlabel='Position',
        ylabel='Firing Rate',
        repeat=True,
        show=True,
        save_path=save_path
    )

    energy_landscape_1d_animation(
        data_sets={'u': (model.x, u_history)},
        config=config
    )

    return r_history
```

### 2.2 Default Parameter Baseline

First, establish a baseline with default parameters:

```python
# Default parameters
model_default = CANN1D(
    num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0
)
r_default = run_experiment(
    model_default,
    title="Default Parameters",
    save_path=None  # Set to 'default.gif' to save
)
```

**What to observe**: The bump should smoothly track the stimulus from left to right with a stable, well-defined shape.

---

## 3. Parameter Exploration

For each parameter, we'll create side-by-side comparisons using multiple experiments. This approach allows you to directly compare different parameter values.

### 3.1 Number of Neurons `num`

`num` controls network resolution. More neurons mean finer feature representation but higher computational cost.

```python
# Test different neuron counts
for num_val in [64, 256, 512]:
    model = CANN1D(num=num_val, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)
    run_experiment(
        model,
        title=f"Number of Neurons: {num_val}",
        save_path=None
    )
```

**Observations**:
- `num=64`: Lower resolution, coarser bump representation
- `num=256`: Balanced resolution, standard choice
- `num=512`: Finer resolution, smoother bump, but increased computation

**Key Insight**: Resolution affects bump smoothness but not fundamental dynamics. Use higher `num` when precise spatial representation matters.

### 3.2 Time Constant `tau`

`tau` controls the dynamics timescale. Larger `tau` means slower response to inputs.

```python
# Test different time constants
for tau_val in [0.5, 1.0, 2.0]:
    model = CANN1D(num=256, tau=tau_val, k=8.1, a=0.5, A=10, J0=4.0)
    run_experiment(
        model,
        title=f"Time Constant: tau={tau_val}",
        save_path=None
    )
```

**Observations**:
- `tau=0.5`: Fast response, bump tightly follows stimulus with minimal lag
- `tau=1.0`: Standard response, balanced tracking speed
- `tau=2.0`: Slow response, bump lags behind moving stimulus

**Key Insight**: `tau` controls the network's temporal filtering. Smaller `tau` means faster adaptation to changing inputs, larger `tau` means smoother, slower dynamics.

### 3.3 Global Inhibition `k`

`k` controls global inhibition strength, critical for maintaining single-bump attractor state.

```python
# Test different inhibition strengths
for k_val in [4.0, 8.1, 16.0]:
    model = CANN1D(num=256, tau=1.0, k=k_val, a=0.5, A=10, J0=4.0)
    run_experiment(
        model,
        title=f"Global Inhibition: k={k_val}",
        save_path=None
    )
```

**Observations**:
- `k=4.0`: Weak inhibition, bump may be unstable, wider, or split into multiple bumps
- `k=8.1`: Balanced inhibition, stable single bump
- `k=16.0`: Strong inhibition, sharper, narrower bump

**Key Insight**: `k` balances excitation and inhibition. Too weak: instability or multiple bumps. Too strong: bump may fail to form or be overly narrow.

### 3.4 Connection Width `a`

`a` controls the spatial range of local excitatory connections, directly affecting bump width.

```python
# Test different connection widths
for a_val in [0.3, 0.5, 0.8]:
    model = CANN1D(num=256, tau=1.0, k=8.1, a=a_val, A=10, J0=4.0)
    run_experiment(
        model,
        title=f"Connection Width: a={a_val}",
        save_path=None
    )
```

**Observations**:
- `a=0.3`: Narrow connections, produces narrower bump with sharper peak
- `a=0.5`: Standard width, balanced bump shape
- `a=0.8`: Wide connections, produces wider, more distributed bump

**Key Insight**: `a` directly controls the spatial receptive field size. Match `a` to your application's required spatial precision.

### 3.5 External Input Amplitude `A`

`A` controls the amplitude of external stimulus input.

```python
# Test different input amplitudes
for A_val in [5, 10, 20]:
    model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=A_val, J0=4.0)
    run_experiment(
        model,
        title=f"Input Amplitude: A={A_val}",
        save_path=None
    )
```

**Observations**:
- `A=5`: Weaker input, bump has lower peak firing rate
- `A=10`: Standard amplitude, balanced response
- `A=20`: Stronger input, bump has higher peak firing rate and potentially wider spread

**Key Insight**: `A` controls how strongly external inputs drive the network. Higher `A` makes the bump more responsive to external stimuli versus internal recurrent dynamics.

### 3.6 Synaptic Connection Strength `J0`

`J0` controls the maximum strength of recurrent synaptic connections.

```python
# Test different connection strengths
for J0_val in [2.0, 4.0, 8.0]:
    model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=J0_val)
    run_experiment(
        model,
        title=f"Synaptic Strength: J0={J0_val}",
        save_path=None
    )
```

**Observations**:
- `J0=2.0`: Weak recurrent connections, bump may not maintain stability without input
- `J0=4.0`: Standard connections, stable bump maintenance
- `J0=8.0`: Strong recurrent connections, very stable bump that may become too rigid

**Key Insight**: `J0` determines the strength of the attractor. Higher `J0` means stronger self-sustaining dynamics and better memory maintenance, but may reduce responsiveness to changing inputs.

### 3.7 SFA Time Constant `tau_v`

For `CANN1D_SFA`, `tau_v` controls the Spike Frequency Adaptation timescale. SFA creates an adaptation current that can cause the bump to drift even without external velocity input.

```python
# Test different SFA time constants
for tau_v_val in [20.0, 50.0, 100.0]:
    model = CANN1D_SFA(
        num=256,
        tau=1.0,
        tau_v=tau_v_val,
        k=8.1,
        a=0.3,
        A=0.2,
        J0=1.0,
        m=0.3
    )
    run_experiment(
        model,
        title=f"SFA Time Constant: tau_v={tau_v_val}",
        save_path=None
    )
```

**Observations**:
- `tau_v=20.0`: Fast adaptation, bump may drift faster or lead the stimulus
- `tau_v=50.0`: Standard adaptation, moderate drift behavior
- `tau_v=100.0`: Slow adaptation, weaker drift effect

**Key Insight**: SFA introduces anticipatory dynamics. The adaptation current creates an asymmetry that can make the bump "predict" motion direction. This is useful for path integration models.

---

## 4. Next Steps

Congratulations on completing all foundation tutorials! You now understand:
- All major CANN parameters and their physical meanings
- Each parameter's effect on bump dynamics
- How to conduct systematic parameter sweep experiments
- The trade-offs involved in parameter selection

### Parameter Selection Guidelines

When configuring CANN models for your applications:

1. **Start with defaults** - The default parameters provide stable, well-behaved dynamics
2. **Match resolution to needs** - Higher `num` for precise spatial coding, lower for efficiency
3. **Tune temporal dynamics** - Adjust `tau` based on required response speed
4. **Balance stability and flexibility** - `J0` and `k` control attractor strength
5. **Match spatial scale** - Set `a` based on required bump width

### Continue Learning

From here, choose advanced application tutorials:

- **[Tutorial 5: Hierarchical Path Integration Network](./05_hierarchical_network.md)** - Learn hierarchical path integration with multiple spatial scales
- **[Tutorial 6: Theta Sweep System Model](./06_theta_sweep_hd_grid.md)** - Learn Theta Sweep systems combining head direction and grid cells
- **[Tutorial 7: Theta Sweep Place Cell Network](./07_theta_sweep_place_cell.md)** - Learn place cell networks and spatial memory

Or explore other scenarios:
- **Scenario 2: Data Analysis** - Analyze experimental neural data with CANNs
- **Scenario 3: Brain-Inspired Learning** - Brain-inspired training methods for CANNs
- **Scenario 4: End-to-End Pipeline** - Complete research workflow from data to publication

### Key Takeaways

1. **Parameters are interconnected** - Changing one parameter may require adjusting others to maintain stability
2. **Visualize to understand** - Always use energy landscape visualizations when exploring parameters
3. **Start simple** - Begin with basic CANN1D before moving to SFA or 2D variants
4. **Document your choices** - Keep notes on why you selected specific parameter values for reproducibility
