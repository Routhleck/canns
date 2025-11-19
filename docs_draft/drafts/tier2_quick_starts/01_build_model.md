# How to Build CANN Model?

**Goal**: By the end of this guide, you'll be able to create and run a basic CANN model.

**Estimated Reading Time**: 10 minutes

---

## Introduction

Building a CANN model in this library is straightforward thanks to the integration with **BrainState**, a framework for dynamical systems built on JAX. This guide shows you how to:

1. Set up the BrainState environment
2. Create a CANN1D model instance
3. Initialize the model state
4. Run a simple forward pass

## The Basics: BrainState Framework

CANNs models are built using [BrainState](https://brainstate.readthedocs.io), which provides:
- **Unified time-step management** via `brainstate.environ`
- **State containers** (`State`, `HiddenState`, `ParamState`) for managing neural dynamics
- **JIT compilation** through `brainstate.transform.for_loop` for high performance
- **Automatic differentiation** support for gradient-based analysis

All CANN models inherit from `brainstate.nn.Dynamics`, which means they follow a consistent interface across the library.

## Step-by-Step: Creating Your First CANN

### 1. Set the Time Step

Before creating any model, you must set the simulation time step:

```python
import brainstate

# Set time step to 0.1 ms (or your preferred value)
brainstate.environ.set(dt=0.1)
```

**Why this matters**: The time step `dt` controls the granularity of your simulation. All models in your session will use this value for their dynamics updates.

### 2. Import and Create the Model

```python
from canns.models.basic import CANN1D

# Create a 1D CANN with 512 neurons
cann = CANN1D(num=512)
```

**What's happening here**:
- `num=512` specifies the number of neurons in the network
- The model automatically sets up connection weights, neuron positions, and dynamics parameters
- Default parameters (e.g., connection strength `k`, time constant `tau`) are used unless you specify otherwise

### 3. Initialize the Model State

```python
# Initialize all state variables
cann.init_state()
```

**Critical step**: This allocates and initializes the internal state variables (`u` for synaptic input, `r` for neural activity). **Forgetting this step is the most common beginner mistake.**

### 4. Run a Forward Pass

Now you can call the model to update its state:

```python
import jax.numpy as jnp

# Create a simple external input (stimulus at position 0)
external_input = jnp.zeros(512)

# Run one time step
cann(external_input)

# Access the model's current state
print("Synaptic input:", cann.u.value)
print("Neural activity:", cann.r.value)
```

**What's happening**:
- The model takes external input and updates its internal dynamics
- `cann.u` stores synaptic input (membrane potential)
- `cann.r` stores neural firing rates (activity)
- Each call to `cann(...)` advances the model by one time step (`dt`)

## Complete Working Example

Here's a minimal, runnable example that puts it all together:

```python
import brainstate
import jax.numpy as jnp
from canns.models.basic import CANN1D

# Step 1: Set time step
brainstate.environ.set(dt=0.1)

# Step 2: Create model
cann = CANN1D(num=512)

# Step 3: Initialize state
cann.init_state()

# Step 4: Create a Gaussian bump stimulus centered at position 0
positions = cann.x  # Neuron positions from -pi to pi
stimulus = jnp.exp(-0.5 * (positions - 0.0)**2 / 0.25**2)

# Step 5: Run one forward pass
cann(stimulus)

# Step 6: Check the output
print(f"Activity shape: {cann.r.value.shape}")
print(f"Max activity: {jnp.max(cann.r.value):.3f}")
```

**Expected output**:
```
Activity shape: (512,)
Max activity: 0.895
```

## Understanding CANN1D Parameters

When creating a CANN model, you can customize several parameters:

```python
cann = CANN1D(
    num=512,           # Number of neurons
    k=1.0,             # Global connection strength
    tau=1.0,           # Time constant (ms)
    a=0.5,             # Width of excitatory connections
    A=10.0,            # Amplitude of excitatory connections
    J_ext=1.0,         # External input strength
)
```

**Key parameters**:
- `num`: Number of neurons (higher = finer spatial resolution, but slower)
- `k`: Controls overall connection strength (higher = stronger self-organization)
- `tau`: Time constant of dynamics (higher = slower changes)
- `a`: Width of connection kernel (controls bump width)
- `A`: Amplitude of connections (affects stability)

For most applications, **the defaults work well**. We'll explore parameter tuning in the Core Concepts section.

## Running Multiple Time Steps

In practice, you'll run many time steps in a loop. BrainState provides optimized tools for this:

```python
def step_function(t, stimulus):
    """Run one time step of the model."""
    cann(stimulus)
    return cann.r.value  # Return activity for each step

# Create stimuli for 100 time steps (here, constant stimulus)
stimuli = jnp.tile(stimulus, (100, 1))

# Run optimized loop with progress bar
activities = brainstate.transform.for_loop(
    step_function,
    operands=100,           # Number of steps
    dyn_vars=stimuli,       # Input data
    pbar=brainstate.transform.ProgressBar(10)  # Show progress
)

print(f"Recorded activities shape: {activities.shape}")  # (100, 512)
```

**What's happening**:
- `brainstate.transform.for_loop` JIT-compiles the loop for speed
- Progress bar shows simulation progress (updates every 10%)
- The result is a JAX array of all recorded activities

## Common Mistakes and How to Avoid Them

### ❌ Mistake 1: Forgetting to Initialize State

```python
cann = CANN1D(num=512)
cann(stimulus)  # ERROR! State not initialized
```

**✅ Solution**: Always call `init_state()` before first use:
```python
cann = CANN1D(num=512)
cann.init_state()  # Initialize first!
cann(stimulus)     # Now it works
```

### ❌ Mistake 2: Not Setting the Time Step

```python
from canns.models.basic import CANN1D
cann = CANN1D(num=512)  # Uses whatever dt was set before (or default)
```

**✅ Solution**: Explicitly set `dt` at the start of your script:
```python
import brainstate
brainstate.environ.set(dt=0.1)  # Set dt first
cann = CANN1D(num=512)
```

### ❌ Mistake 3: Wrong Input Dimensions

```python
cann = CANN1D(num=512)
cann.init_state()
cann(jnp.zeros(256))  # ERROR! Input size doesn't match num neurons
```

**✅ Solution**: Input must have same size as `num`:
```python
cann = CANN1D(num=512)
cann.init_state()
cann(jnp.zeros(512))  # Correct size
```

## What About 2D CANNs?

The same principles apply to 2D models:

```python
from canns.models.basic import CANN2D

brainstate.environ.set(dt=0.1)

# Create 2D CANN with 32x32 neurons
cann2d = CANN2D(num=32)
cann2d.init_state()

# Input must be (32, 32) shaped
stimulus_2d = jnp.zeros((32, 32))
cann2d(stimulus_2d)

print(f"2D activity shape: {cann2d.r.value.shape}")  # (32, 32)
```

The API is nearly identical—just adapt your input dimensions!

## Next Steps

Now that you know how to create and run CANN models, you're ready to:

1. **[Generate task data](03_how_to_generate_task_data.md)** - Learn how to create smooth tracking inputs, navigation tasks, and more
2. **[Explore Model Collections](link-to-core-concepts-models)** - Discover other model variants (SFA-CANN, hierarchical networks, brain-inspired models)
3. **[Learn BrainState basics](https://brainstate.readthedocs.io/en/latest/tutorials/)** - If you want to build custom models or understand the framework deeply

---

**Quick Reference**:
```python
# Template for creating any CANN model
import brainstate
from canns.models.basic import CANN1D

brainstate.environ.set(dt=0.1)      # 1. Set time step
cann = CANN1D(num=512)              # 2. Create model
cann.init_state()                   # 3. Initialize
cann(stimulus)                      # 4. Run forward pass
result = cann.r.value               # 5. Access activity
```

---

*Have questions? Check the [FAQ](link) or open a [GitHub Discussion](https://github.com/routhleck/canns/discussions).*
