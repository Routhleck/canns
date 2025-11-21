# Tutorial 2: Task Generation and CANN Simulation

> **Reading Time**: ~20-25 minutes
> **Difficulty**: Beginner
> **Prerequisites**: [Tutorial 1](./01_build_cann_model.md)

This tutorial teaches you how to generate task data using the Task module and run simulations with CANN models.

---

## Table of Contents

1. [Task Module Overview](#1-task-module-overview)
2. [PopulationCoding1D in Detail](#2-populationcoding1d-in-detail)
3. [Running Simulations with brainstate.for_loop](#3-running-simulations-with-brainstatefor_loop)
4. [Complete Example](#4-complete-example)
5. [Next Steps](#5-next-steps)

---

## 1. Task Module Overview

The CANNs Task module generates experimental paradigms and input data. The relationship between Task and Model:

- **Task**: Generates external stimulus sequences (input data)
- **Model**: Consumes input data, runs in simulation loop

### Task Categories

CANNs provides two main task types:

**Tracking Tasks**:
- `PopulationCoding1D/2D` - Population coding
- `TemplateMatching1D/2D` - Template matching
- `SmoothTracking1D/2D` - Smooth tracking

**Navigation Tasks**:
- `ClosedLoopNavigation` - Closed-loop navigation
- `OpenLoopNavigation` - Open-loop navigation

> This tutorial uses the simplest `PopulationCoding1D` as example. Other tasks follow similar usage patterns with different initialization parameters. We'll demonstrate different tasks in later tutorials.

---

## 2. PopulationCoding1D in Detail

`PopulationCoding1D` is a simple population coding task: no stimulus → stimulus → no stimulus. This tests the network's ability to form and maintain a memory bump.

### 2.1 Import and Create Task

```python
from canns.task import PopulationCoding1D
from canns.models.basic import CANN1D
import brainstate

# First create model instance
brainstate.environ.set(dt=0.1)
model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)
model.init_state()

# Create task
task = PopulationCoding1D(
    cann_instance=model,      # CANN model instance
    before_duration=10.0,     # Duration before stimulus
    after_duration=50.0,      # Duration after stimulus
    Iext=0.0,                 # Stimulus position in feature space
    duration=10.0,            # Stimulus duration
    time_step=0.1,            # Time step
)
```

### 2.2 Parameter Descriptions

| Parameter | Type | Description |
|-----------|------|-------------|
| `cann_instance` | BaseCANN1D | CANN model instance, task calls its `get_stimulus_by_pos()` |
| `before_duration` | float | Duration before stimulus presentation (no input period) |
| `after_duration` | float | Duration after stimulus ends (observe bump maintenance) |
| `Iext` | float | Stimulus position in feature space, typically in `[z_min, z_max]` |
| `duration` | float | Duration of stimulus presentation |
| `time_step` | float | Simulation time step, should match `brainstate.environ.set(dt=...)` |

**Why these parameters matter**:
- `cann_instance` is required because the task needs to call the model's `get_stimulus_by_pos()` method to generate appropriate stimulus
- `before_duration` and `after_duration` allow observing bump formation and maintenance
- `Iext` determines where the bump will form
- All durations use the same unit as `time_step`

### 2.3 Getting Task Data

After creating a task, call `get_data()` to generate and store input data in `task.data`:

```python
# Generate task data
task.get_data()

# Access task properties
print(f"Total time steps: {task.run_steps}")
print(f"Total duration: {task.total_duration}")
print(f"Data shape: {task.data.shape}")
```

> **Important**: `get_data()` does not return a value. It modifies `task.data` in-place. Access the data via `task.data`.

---

## 3. Running Simulations with brainstate.for_loop

### 3.1 Why use for_loop?

BrainState provides `brainstate.transform.for_loop` for efficient simulation loops. Compared to Python `for` loops, it offers:

- **JIT Compilation**: Entire loop compiled to efficient machine code
- **GPU Acceleration**: Automatic GPU utilization
- **Auto-vectorization**: Better memory access patterns

> **Learn More**: See [BrainState Loops Tutorial](https://brainstate.readthedocs.io/tutorials/transforms/05_loops_conditions.html) for detailed `for_loop` usage.

### 3.2 Basic Usage

```python
import brainstate
import brainunit as u

# Define step function
def run_step(t, inp):
    """
    Single simulation step.

    Args:
        t: Current time step index
        inp: Input data at current time step

    Returns:
        State variables to record
    """
    model(inp)  # Or model.update(inp)
    return model.u.value, model.r.value

# Run simulation using task.data
results = brainstate.transform.for_loop(
    run_step,           # Step function
    task.run_steps,     # Number of time steps
    task.data,          # Input data (from task)
    pbar=brainstate.transform.ProgressBar(10)  # Optional progress bar
)
```

### 3.3 Handling Return Values

`for_loop` returns values corresponding to step function returns:

```python
# results is a tuple of return values across all time steps
u_history, r_history = results

print(f"Membrane potential history shape: {u_history.shape}")  # (run_steps, num)
print(f"Firing rate history shape: {r_history.shape}")  # (run_steps, num)
```

### 3.4 JIT Compilation Benefits

First run includes compilation time (few seconds), but subsequent runs are much faster:

```python
import time

# First run (includes compilation)
start = time.time()
results = brainstate.transform.for_loop(run_step, task.run_steps, task.data)
print(f"First run: {time.time() - start:.2f}s")

# Re-initialize state
model.init_state()

# Second run (already compiled)
start = time.time()
results = brainstate.transform.for_loop(run_step, task.run_steps, task.data)
print(f"Second run: {time.time() - start:.2f}s")
```

---

## 4. Complete Example

Here's a complete example from model creation to simulation:

```python
import brainstate
import brainunit as u
from canns.models.basic import CANN1D
from canns.task import PopulationCoding1D

# ============================================================
# Step 1: Setup environment and create model
# ============================================================
brainstate.environ.set(dt=0.1)

model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)
model.init_state()

# ============================================================
# Step 2: Create task
# ============================================================
task = PopulationCoding1D(
    cann_instance=model,
    before_duration=10.0,
    after_duration=50.0,
    Iext=0.0,
    duration=10.0,
    time_step=0.1,
)

# Get task data
task.get_data()

print("Task Information:")
print(f"  Total time steps: {task.run_steps}")
print(f"  Total duration: {task.total_duration}")
print(f"  Data shape: {task.data.shape}")

# ============================================================
# Step 3: Define simulation step function
# ============================================================
def run_step(t, inp):
    model.update(inp)
    return model.u.value, model.r.value

# ============================================================
# Step 4: Run simulation
# ============================================================
time_steps = u.math.arange(task.run_steps)
u_history, r_history = brainstate.transform.for_loop(
    run_step,
    time_steps,
    task.data,
)

# ============================================================
# Step 5: Inspect results
# ============================================================
print("\nSimulation Results:")
print(f"  Membrane potential history shape: {u_history.shape}")
print(f"  Firing rate history shape: {r_history.shape}")

# Check states at different phases
before_steps = int(10.0 / 0.1)  # Before stimulus
stim_end = int(20.0 / 0.1)      # End of stimulus
after_steps = int(70.0 / 0.1)   # End of simulation

print(f"\nBefore stimulus (t={before_steps-1}) max firing rate: {u.math.max(r_history[before_steps-1]):.6f}")
print(f"During stimulus (t={stim_end-1}) max firing rate: {u.math.max(r_history[stim_end-1]):.6f}")
print(f"After stimulus (t={after_steps-1}) max firing rate: {u.math.max(r_history[after_steps-1]):.6f}")
```

**Expected output**:
- Before stimulus: firing rate ~0
- During stimulus: firing rate increases (bump forms)
- After stimulus: firing rate maintained (memory persists)

---

## 5. Next Steps

Congratulations on completing Tutorial 2! You now know:
- How to use the Task module to generate data
- All parameters of PopulationCoding1D
- How to run efficient simulations with `brainstate.for_loop`

### Continue Learning

- **Next**: [Tutorial 3: Analysis and Visualization](./03_analysis_visualization.md) - Learn how to visualize simulation results
- **Other Task Types**: We'll demonstrate different tasks' effects in Tutorial 3
- **More on for_loop**: [BrainState Loops Tutorial](https://brainstate.readthedocs.io/tutorials/transforms/05_loops_conditions.html)
