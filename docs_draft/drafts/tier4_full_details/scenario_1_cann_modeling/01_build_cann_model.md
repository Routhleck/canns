# Tutorial 1: Building and Using CANN Models

> **Reading Time**: ~25-30 minutes
> **Difficulty**: Beginner
> **Prerequisites**: Python basics, NumPy/JAX array operations

This tutorial will help you understand how models are constructed in the CANNs library and how to use the built-in CANN models.

---

## Table of Contents

1. [Introduction to BrainState Framework](#1-introduction-to-brainstate-framework)
2. [CANN1D Implementation Analysis](#2-cann1d-implementation-analysis)
3. [How to Use Built-in CANN Models](#3-how-to-use-built-in-cann-models)
4. [Overview of Built-in Models](#4-overview-of-built-in-models)
5. [Next Steps](#5-next-steps)

> **Quick Jump**: If you're already familiar with the BrainState framework, skip to [How to Use Built-in CANN Models](#3-how-to-use-built-in-cann-models).

---

## 1. Introduction to BrainState Framework

All models in the CANNs library are built on the [BrainState](https://brainstate.readthedocs.io/) framework. BrainState is a core framework for dynamical systems in the Brain Simulation Ecosystem, built on JAX with JIT compilation and automatic differentiation support.

### 1.1 Core Concepts

Before we begin, you need to understand these key concepts:

#### Dynamics Abstraction

All CANN models inherit from `brainstate.nn.Dynamics`, a base class for defining dynamical systems. It provides:
- State management mechanisms
- Time step management
- JIT compilation support

```python
import brainstate

class MyModel(brainstate.nn.Dynamics):
    def init_state(self):
        # Initialize state variables
        pass

    def update(self, inp):
        # Define single-step dynamics update
        pass
```

#### State Containers

BrainState provides three types of state containers for managing different types of variables:

| Container Type | Purpose | Example |
|---------------|---------|---------|
| `brainstate.State` | External inputs or observable states | External stimulus `inp` |
| `brainstate.HiddenState` | Internal hidden states | Membrane potential `u`, firing rate `r` |
| `brainstate.ParamState` | Learnable parameters | Synaptic weights `W` |

```python
def init_state(self):
    # Hidden state: neuron membrane potential
    self.u = brainstate.HiddenState(u.math.zeros(self.num))
    # Hidden state: neuron firing rate
    self.r = brainstate.HiddenState(u.math.zeros(self.num))
    # External input state
    self.inp = brainstate.State(u.math.zeros(self.num))
```

#### Time Step Management

BrainState manages simulation time steps uniformly through `brainstate.environ`:

```python
import brainstate

# Set simulation time step (unit: milliseconds)
brainstate.environ.set(dt=0.1)

# Get current time step in the model
dt = brainstate.environ.get_dt()
```

> **Important**: You must set the time step `dt` before running any simulation, otherwise errors will occur.

#### Further Learning

To learn more about the BrainState framework, see:
- [BrainState Official Documentation](https://brainstate.readthedocs.io/)
- [Loops and Conditions Tutorial](https://brainstate.readthedocs.io/tutorials/transforms/05_loops_conditions.html)

---

## 2. CANN1D Implementation Analysis

Let's use `CANN1D` as an example to understand how a complete CANN model is implemented.

### 2.1 Model Inheritance Structure

```
brainstate.nn.Dynamics
    └── BasicModel
        └── BaseCANN
            └── BaseCANN1D
                └── CANN1D
```

### 2.2 Initialization Method `__init__`

The `CANN1D` initialization method defines all model parameters:

```python
class CANN1D(BaseCANN1D):
    def __init__(
        self,
        num: int,           # Number of neurons
        tau: float = 1.0,   # Time constant
        k: float = 8.1,     # Global inhibition strength
        a: float = 0.5,     # Connection width
        A: float = 10,      # External input amplitude
        J0: float = 4.0,    # Synaptic connection strength
        z_min: float = -π,  # Feature space minimum
        z_max: float = π,   # Feature space maximum
        **kwargs,
    ):
        # ...
```

These parameters control the network's dynamical behavior. We will explore each parameter's effect in detail in [Tutorial 4: Parameter Effects](./04_parameter_effects.md).

### 2.3 Connection Matrix Generation `make_conn`

The `make_conn` method generates the connectivity matrix between neurons. CANN uses a Gaussian connection kernel so that neurons with similar feature preferences have stronger excitatory connections:

```python
def make_conn(self):
    # Calculate distances between all neuron pairs
    x_left = u.math.reshape(self.x, (-1, 1))
    x_right = u.math.repeat(self.x.reshape((1, -1)), len(self.x), axis=0)
    d = self.dist(x_left - x_right)

    # Compute connection strength using Gaussian function
    return (
        self.J0
        * u.math.exp(-0.5 * u.math.square(d / self.a))
        / (u.math.sqrt(2 * u.math.pi) * self.a)
    )
```

### 2.4 Stimulus Generation `get_stimulus_by_pos`

`get_stimulus_by_pos` generates external stimulus (a Gaussian-shaped bump) based on a given position in feature space:

```python
def get_stimulus_by_pos(self, pos):
    return self.A * u.math.exp(
        -0.25 * u.math.square(self.dist(self.x - pos) / self.a)
    )
```

This method is called by the task module to generate input data.

### 2.5 State Initialization `init_state`

The `init_state` method initializes all state variables of the model:

```python
def init_state(self, *args, **kwargs):
    # Firing rate
    self.r = brainstate.HiddenState(u.math.zeros(self.varshape))
    # Membrane potential (synaptic input)
    self.u = brainstate.HiddenState(u.math.zeros(self.varshape))
    # External input
    self.inp = brainstate.State(u.math.zeros(self.varshape))
```

> **Important**: You must call `model.init_state()` to initialize states before running simulations, otherwise errors will occur.

### 2.6 Dynamics Update `update`

The `update` method defines the single-step dynamics update of the network:

```python
def update(self, inp):
    self.inp.value = inp

    # Compute firing rate (divisive normalization)
    r1 = u.math.square(self.u.value)
    r2 = 1.0 + self.k * u.math.sum(r1)
    self.r.value = r1 / r2

    # Compute recurrent input
    Irec = u.math.dot(self.conn_mat, self.r.value)

    # Update membrane potential using Euler method
    self.u.value += (
        (-self.u.value + Irec + self.inp.value)
        / self.tau * brainstate.environ.get_dt()
    )
```

---

## 3. How to Use Built-in CANN Models

Now let's learn how to actually use the built-in CANN models.

### 3.1 Basic Usage Workflow

```python
import brainstate
import brainunit as u
from canns.models.basic import CANN1D

# Step 1: Set time step
brainstate.environ.set(dt=0.1)

# Step 2: Create model instance
model = CANN1D(
    num=256,      # 256 neurons
    tau=1.0,      # Time constant
    k=8.1,        # Global inhibition
    a=0.5,        # Connection width
    A=10,         # Input amplitude
    J0=4.0,       # Connection strength
)

# Step 3: Initialize state
model.init_state()

# Step 4: View model information
print(f"Number of neurons: {model.num}")
print(f"Feature space range: [{model.z_min}, {model.z_max}]")
print(f"Connection matrix shape: {model.conn_mat.shape}")
```

### 3.2 Running a Single Step Update

```python
# Generate external stimulus at pos=0
pos = 0.0
stimulus = model.get_stimulus_by_pos(pos)

# Run single step update
model(stimulus)	# or you can explicitly call model.update(stimulus)

# View current state
print(f"Firing rate shape: {model.r.value.shape}")
print(f"Max firing rate: {u.math.max(model.r.value):.4f}")
print(f"Max membrane potential: {u.math.max(model.u.value):.4f}")
```

### 3.3 Complete Example

Here's a complete example of creating and testing a CANN1D model:

```python
import brainstate
import brainunit as u
from canns.models.basic import CANN1D

# Setup environment
brainstate.environ.set(dt=0.1)

# Create model
model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)
model.init_state()

# Print basic model information
print("=" * 50)
print("CANN1D Model Information")
print("=" * 50)
print(f"Number of neurons: {model.num}")
print(f"Time constant tau: {model.tau}")
print(f"Global inhibition k: {model.k}")
print(f"Connection width a: {model.a}")
print(f"Input amplitude A: {model.A}")
print(f"Connection strength J0: {model.J0}")
print(f"Feature space: [{model.z_min:.2f}, {model.z_max:.2f}]")
print(f"Neural density rho: {model.rho:.2f}")

# Test stimulus generation
pos = 0.5
stimulus = model.get_stimulus_by_pos(pos)
print(f"\nStimulus position: {pos}")
print(f"Stimulus shape: {stimulus.shape}")
print(f"Max stimulus value: {u.math.max(stimulus):.4f}")

# Run several update steps
print("\nRunning 100 update steps...")
for _ in range(100):
    model.update(stimulus)

print(f"Max firing rate: {u.math.max(model.r.value):.6f}")
print(f"Max membrane potential: {u.math.max(model.u.value):.6f}")
```

---

## 4. Overview of Built-in Models

The CANNs library provides three categories of built-in models:

### Basic Models

Standard CANN implementations and variants:
- `CANN1D` - 1D continuous attractor neural network
- `CANN1D_SFA` - CANN1D with Spike Frequency Adaptation
- `CANN2D` - 2D continuous attractor neural network
- `CANN2D_SFA` - CANN2D with SFA
- Hierarchical Path Integration networks (Grid Cell, Place Cell, Band Cell, etc.)
- Theta Sweep models

### Brain-Inspired Models

Learning models based on neuroscience principles:
- Hopfield networks
- ...

### Hybrid Models

Combinations of CANN with artificial neural networks (under development).

> **Detailed Information**: See [Tier 3 Core Concepts - Model Collections](../../docs/en/2_core_concepts/02_model_collections.rst) for a complete list of models and use cases.

---

## 5. Next Steps

Congratulations on completing the first tutorial! You now understand:
- Core concepts of the BrainState framework
- The implementation structure of CANN1D
- How to create and initialize built-in models

### Continue Learning

- **Next Tutorial**: [Tutorial 2: Task Generation and CANN Simulation](./02_task_and_simulation.md) - Learn how to generate task data and run complete simulations
- **To learn more about BrainState**: Visit [BrainState ReadTheDocs](https://brainstate.readthedocs.io/)
- **To see all available models**: Check [Model Collections](../../docs/en/2_core_concepts/02_model_collections.rst)
