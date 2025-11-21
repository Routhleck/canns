# Tutorial: Feature Learning and Temporal Patterns

> **Reading Time**: ~35-40 minutes
> **Difficulty**: Intermediate to Advanced
> **Prerequisites**: [Tutorial 1: Pattern Storage and Recall](./01_pattern_storage_recall.md)

This tutorial explores advanced brain-inspired learning rules for feature extraction and temporal credit assignment: BCM plasticity, Oja's rule, and STDP.

---

## Table of Contents

1. [BCM Plasticity for Receptive Field Formation](#1-bcm-plasticity-for-receptive-field-formation)
2. [Oja's Rule for Principal Component Analysis](#2-ojas-rule-for-principal-component-analysis)
3. [STDP for Temporal Learning](#3-stdp-for-temporal-learning)
4. [Next Steps](#4-next-steps)

---

## 1. BCM Plasticity for Receptive Field Formation

### 1.1 What is BCM Plasticity?

**BCM (Bienenstock-Cooper-Munro)** plasticity extends Hebbian learning with a **sliding threshold** that maintains homeostatic balance:

$$
\Delta W_{ij} = \eta \cdot y_j \cdot (y_j - \theta_j) \cdot x_i
$$

Where:
- $y_j$: Post-synaptic activity
- $\theta_j$: Sliding threshold (adapts based on history)
- $x_i$: Pre-synaptic activity

**Key features**:
- **LTP (Long-Term Potentiation)**: When $y_j > \theta_j$, weights increase
- **LTD (Long-Term Depression)**: When $y_j < \theta_j$, weights decrease
- **Homeostatic regulation**: Threshold prevents runaway excitation
- **Selectivity**: Neurons become tuned to specific features

### 1.2 Biological Motivation

BCM explains:
- **Orientation selectivity** in visual cortex (Hubel & Wiesel)
- **Ocular dominance** formation
- **Experience-dependent plasticity** in development
- **Homeostatic mechanisms** in neural circuits

### 1.3 Complete Example: Orientation Selectivity

Train neurons to become selective to oriented bars (like simple cells in V1):

```python
import numpy as np
import brainstate
from canns.models.brain_inspired import LinearLayer
from canns.trainer import BCMTrainer
from canns.analyzer.plotting import PlotConfigs, tuning_curve

np.random.seed(42)
brainstate.random.seed(42)
```

#### Step 1: Generate Oriented Stimuli

```python
def create_oriented_bar(angle, size=12):
    """Create oriented bar stimulus"""
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x = x - size / 2
    y = y - size / 2

    # Rotate coordinates
    x_rot = x * np.cos(angle) + y * np.sin(angle)

    # Create Gaussian bar perpendicular to orientation
    pattern = np.exp(-((x_rot / 2) ** 2))

    # Flatten and normalize to [0, 1]
    pattern_flat = pattern.flatten()
    pattern_flat = (pattern_flat - pattern_flat.min()) / (
        pattern_flat.max() - pattern_flat.min() + 1e-8
    )

    return pattern_flat.astype(np.float32)

def generate_training_data(n_samples=1000, size=12, n_orientations=8):
    """Generate random oriented bars"""
    orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)
    stimuli = []

    for _ in range(n_samples):
        angle = np.random.choice(orientations)
        stimuli.append(create_oriented_bar(angle, size))

    return stimuli, orientations

# Generate training data
size = 12  # 12x12 images
n_samples = 1000
train_data, orientations = generate_training_data(n_samples, size=size)

print(f"Generated {len(train_data)} oriented bar stimuli")
print(f"Input dimension: {size * size}")
```

#### Step 2: Create Model with BCM Support

```python
# Create linear layer with BCM sliding threshold
n_neurons = 4  # Learn 4 different orientation detectors

model = LinearLayer(
    input_size=size * size,
    output_size=n_neurons,
    use_bcm_threshold=True,  # Enable BCM threshold mechanism
    threshold_tau=50.0        # Time constant for threshold adaptation
)
model.init_state()

print(f"\nModel created:")
print(f"  Input size: {model.input_size}")
print(f"  Output size: {model.output_size}")
print(f"  BCM threshold: Enabled")
```

#### Step 3: Train with BCM Rule

```python
# Create BCM trainer
trainer = BCMTrainer(
    model,
    learning_rate=0.00001  # Small learning rate for stable learning
)

# Train for multiple epochs
n_epochs = 100
checkpoint_interval = 20

threshold_history = []
weight_history = []

print(f"\nTraining for {n_epochs} epochs...")

for epoch in range(n_epochs):
    # Train on all stimuli (batch learning)
    trainer.train(train_data)

    # Track evolution
    if (epoch + 1) % checkpoint_interval == 0:
        threshold_history.append(model.theta.value.copy())
        weight_history.append(model.W.value.copy())
        print(
            f"Epoch {epoch+1}/{n_epochs}: "
            f"Thresholds: [{model.theta.value.min():.4f}, {model.theta.value.max():.4f}], "
            f"Weight range: [{model.W.value.min():.3f}, {model.W.value.max():.3f}]"
        )

print("\nTraining complete!")
```

#### Step 4: Visualize Learned Receptive Fields

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.ravel()

final_weights = model.W.value

for i in range(n_neurons):
    receptive_field = final_weights[i].reshape(size, size)
    im = axes[i].imshow(receptive_field, cmap='RdBu_r', interpolation='nearest')
    axes[i].set_title(f'Neuron {i+1} Receptive Field')
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046)

plt.tight_layout()
plt.savefig('bcm_receptive_fields.png', dpi=150)
plt.show()
```

**Expected result**: Each neuron develops selectivity to a different orientation (0°, 45°, 90°, 135°, etc.)

#### Step 5: Test Orientation Tuning

```python
# Generate test stimuli at many orientations
n_test_angles = 16
test_orientations = np.linspace(0, np.pi, n_test_angles, endpoint=False)

# Get responses
test_stimuli = [create_oriented_bar(angle, size) for angle in test_orientations]
test_responses = [trainer.predict(stimulus) for stimulus in test_stimuli]

# Organize for tuning curve plot
stimulus_angles = test_orientations
firing_rates = np.array(test_responses)  # Shape: (n_angles, n_neurons)

# Plot tuning curves
config = PlotConfigs.tuning_curve(
    num_bins=n_test_angles,
    title='Orientation Tuning Curves',
    xlabel='Orientation (radians)',
    ylabel='Neuron Response',
    figsize=(10, 6),
    show=True,
    save_path=None,
    kwargs={'linewidth': 2, 'marker': 'o', 'markersize': 6}
)

tuning_curve(
    stimulus=stimulus_angles,
    firing_rates=firing_rates,
    neuron_indices=np.arange(n_neurons),
    config=config
)

# Find preferred orientations
for i in range(n_neurons):
    preferred_idx = np.argmax(firing_rates[:, i])
    preferred_angle_deg = np.degrees(stimulus_angles[preferred_idx])
    print(f"Neuron {i+1} prefers: {preferred_angle_deg:.1f}°")
```

**Observations**:
- Each neuron shows bell-shaped tuning curve
- Different neurons prefer different orientations
- Tuning width depends on learning rate and training duration

---

## 2. Oja's Rule for Principal Component Analysis

### 2.1 What is Oja's Rule?

**Oja's rule** is a normalized Hebbian learning rule that extracts principal components:

$$
\Delta W = \eta \cdot (y \cdot x - y^2 \cdot W)
$$

Where:
- $\eta$: Learning rate
- $y$: Post-synaptic activity
- $x$: Pre-synaptic input
- $W$: Weight vector

**Key properties**:
- **Weight normalization**: The $-y^2 \cdot W$ term prevents unbounded growth
- **PCA**: First neuron extracts first principal component
- **Unsupervised dimensionality reduction**: Like biological sensory processing

### 2.2 Complete Example: PCA on High-Dimensional Data

```python
from canns.models.brain_inspired import LinearLayer
from canns.trainer import OjaTrainer

# Generate correlated data
np.random.seed(42)

n_samples = 1000
n_features = 50

# Create data with 2 main principal components
mean = np.zeros(n_features)
# Covariance: first 2 dimensions have high variance, rest is noise
cov = np.eye(n_features) * 0.1  # Background noise
cov[0, 0] = 10.0  # First PC: high variance
cov[1, 1] = 5.0   # Second PC: medium variance
cov[0, 1] = cov[1, 0] = 3.0  # Correlation

data = np.random.multivariate_normal(mean, cov, size=n_samples)
data = data.astype(np.float32)

print(f"Generated {n_samples} samples with {n_features} features")
```

#### Train Oja Network

```python
# Create model
n_components = 5  # Extract 5 principal components
model = LinearLayer(input_size=n_features, output_size=n_components)
model.init_state()

# Create Oja trainer
trainer = OjaTrainer(model, learning_rate=0.001)

# Train
n_epochs = 50
print("\nTraining Oja network...")

for epoch in range(n_epochs):
    trainer.train(data)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}")

print("Training complete!")
```

#### Compare with Standard PCA

```python
from sklearn.decomposition import PCA

# Standard PCA from sklearn
sklearn_pca = PCA(n_components=n_components)
sklearn_pca.fit(data)

# Get first component from both methods
oja_pc1 = model.W.value[0]
sklearn_pc1 = sklearn_pca.components_[0]

# Compute correlation (they may differ by sign)
correlation = np.abs(np.corrcoef(oja_pc1, sklearn_pc1)[0, 1])
print(f"\nCorrelation between Oja PC1 and sklearn PC1: {correlation:.4f}")

# Visualize explained variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Oja explained variance (approximate)
oja_projections = data @ model.W.value.T
oja_var = np.var(oja_projections, axis=0)
oja_var_ratio = oja_var / np.sum(oja_var)

ax1.bar(range(n_components), oja_var_ratio)
ax1.set_title('Oja Rule - Explained Variance')
ax1.set_xlabel('Component')
ax1.set_ylabel('Variance Ratio')
ax1.set_ylim([0, 1])

# sklearn PCA explained variance
ax2.bar(range(n_components), sklearn_pca.explained_variance_ratio_)
ax2.set_title('sklearn PCA - Explained Variance')
ax2.set_xlabel('Component')
ax2.set_ylabel('Variance Ratio')
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('oja_vs_sklearn_pca.png', dpi=150)
plt.show()
```

**Observations**:
- Oja's rule closely approximates PCA
- First component captures most variance
- Biologically plausible (local learning rule)
- Can be computed online (no batch required)

---

## 3. STDP for Temporal Learning

### 3.1 What is STDP?

**Spike-Timing-Dependent Plasticity (STDP)** adjusts weights based on precise spike timing:

- **LTP (potentiation)**: Pre-synaptic spike before post-synaptic → strengthen
- **LTD (depression)**: Post-synaptic spike before pre-synaptic → weaken

$$
\Delta W = \begin{cases}
A_{plus} \cdot \exp(-\Delta t / \tau_{plus}) & \text{if } \Delta t > 0 \text{ (pre before post)} \\
-A_{minus} \cdot \exp(\Delta t / \tau_{minus}) & \text{if } \Delta t < 0 \text{ (post before pre)}
\end{cases}
$$

Where:
- $\Delta t = t_{post} - t_{pre}$: Spike time difference
- $A_{plus}, A_{minus}$: LTP and LTD amplitudes
- $\tau_{plus}, \tau_{minus}$: Time constants for exponential decay

**Temporal credit assignment**: Learn which inputs predict outputs

### 3.2 Complete Example: Learning Temporal Sequences

```python
from canns.models.brain_inspired import SpikingLayer
from canns.trainer import STDPTrainer

np.random.seed(42)
brainstate.random.seed(42)

# Network dimensions
n_input = 20
n_output = 5
n_patterns = 100
n_timesteps = 50

print("Creating temporal spike patterns...")
```

#### Generate Temporal Patterns

```python
spike_patterns = []

for _ in range(n_patterns):
    pattern_sequence = []
    for t in range(n_timesteps):
        spikes = np.zeros(n_input, dtype=np.float32)

        # Early group (neurons 0-4) spikes at t < 10
        if t < 10:
            spikes[0:5] = (np.random.rand(5) < 0.3).astype(np.float32)

        # Middle group (neurons 5-9) spikes at 10 < t < 20
        if 10 < t < 20:
            spikes[5:10] = (np.random.rand(5) < 0.3).astype(np.float32)

        # Late group (neurons 10-14) spikes at 20 < t < 30
        if 20 < t < 30:
            spikes[10:15] = (np.random.rand(5) < 0.3).astype(np.float32)

        # Background noise
        spikes[15:] = (np.random.rand(5) < 0.05).astype(np.float32)

        pattern_sequence.append(spikes)

    spike_patterns.append(pattern_sequence)

print(f"Generated {len(spike_patterns)} temporal patterns")
```

#### Create Spiking Model

```python
model = SpikingLayer(
    input_size=n_input,
    output_size=n_output,
    threshold=0.8,       # Spike threshold
    v_reset=0.0,         # Reset potential
    leak=0.95,           # Membrane leak (0.95 = 95% retention)
    trace_decay=0.90,    # Trace decay (determines STDP window)
    dt=1.0,
)
model.init_state()

initial_weights = model.W.value.copy()
```

#### Train with STDP

```python
# Create STDP trainer
trainer = STDPTrainer(
    model,
    learning_rate=0.02,
    A_plus=0.005,         # LTP amplitude
    A_minus=0.00525,      # LTD amplitude (slightly larger)
    w_min=0.0,            # Minimum weight
    w_max=1.0,            # Maximum weight
    compiled=True,        # JIT compilation for speed
)

print("\nTraining with STDP...")
n_epochs = 20
weight_history = [initial_weights.copy()]

for epoch in range(n_epochs):
    model.reset_state()

    # Train on all patterns
    for pattern_seq in spike_patterns:
        trainer.train(pattern_seq)

    # Record weights
    weight_history.append(model.W.value.copy())

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} complete")

print("Training complete!")
```

#### Visualize Weight Changes

```python
# Compare initial vs final weights
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Initial weights
im1 = ax1.imshow(initial_weights, cmap='RdBu_r', vmin=0, vmax=1)
ax1.set_title('Initial Weights')
ax1.set_xlabel('Input Neurons')
ax1.set_ylabel('Output Neurons')
plt.colorbar(im1, ax=ax1)

# Final weights
final_weights = model.W.value
im2 = ax2.imshow(final_weights, cmap='RdBu_r', vmin=0, vmax=1)
ax2.set_title('Final Weights (After STDP)')
ax2.set_xlabel('Input Neurons')
ax2.set_ylabel('Output Neurons')
plt.colorbar(im2, ax=ax2)

# Weight change
weight_change = final_weights - initial_weights
im3 = ax3.imshow(weight_change, cmap='RdBu_r',
                vmin=-np.abs(weight_change).max(),
                vmax=np.abs(weight_change).max())
ax3.set_title('Weight Change (Δ)')
ax3.set_xlabel('Input Neurons')
ax3.set_ylabel('Output Neurons')
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.savefig('stdp_weight_changes.png', dpi=150)
plt.show()
```

**Expected observations**:
- **Early inputs (0-4)** have strongest connections (fire first → predict output)
- **Late inputs (10-14)** have weaker connections (less predictive)
- **Weight structure** reflects temporal statistics of input patterns

---

## 4. Next Steps

Congratulations! You've mastered advanced brain-inspired learning rules for feature extraction and temporal credit assignment.

### Key Takeaways

1. **BCM plasticity** creates orientation-selective neurons through homeostatic regulation
2. **Oja's rule** extracts principal components with local, online learning
3. **STDP** implements temporal credit assignment based on precise spike timing
4. All rules are **biologically plausible** (local, unsupervised)
5. **JAX compilation** accelerates training (10-100x speedup)

### When to Use Each Rule

**BCM (Bienenstock-Cooper-Munro)**:
- Sensory feature learning (orientation, frequency selectivity)
- Homeostatic regulation needed
- Experience-dependent development
- When neurons need to become selective to distinct features

**Oja's Rule**:
- Dimensionality reduction
- Feature extraction from high-dimensional data
- Unsupervised learning of statistical structure
- When you need PCA-like behavior with biological constraints

**STDP (Spike-Timing-Dependent Plasticity)**:
- Temporal sequence learning
- Predictive coding
- Spike-based networks
- When timing information is critical

### Comparison with Backpropagation

| Aspect | Brain-Inspired Rules | Backpropagation |
|--------|---------------------|----------------|
| Supervision | Unsupervised | Supervised |
| Learning signal | Local activity | Global error |
| Biological plausibility | High | Low |
| Temporal credit | STDP handles naturally | Requires BPTT |
| Computation | Local | Non-local |

### Continue Learning

- **Previous**: [Tutorial 1: Pattern Storage & Recall](./01_pattern_storage_recall.md) - Hebbian learning basics
- **Next**: [Scenario 4: End-to-End Research Workflow](../scenario_4_end_to_end_pipeline/01_theta_sweep_pipeline.md) - Apply models to real data

### Advanced Topics

- **Sanger's rule**: Multiple principal components with deflation
- **Competitive learning**: Winner-take-all with anti-Hebbian lateral connections
- **Three-factor rules**: Combining Hebbian learning with neuromodulation
- **Deep networks**: Stacking multiple layers with local learning

### Related Models and Trainers

Available in CANNs library:

**Models**:
- `LinearLayer` - Feedforward layer with BCM support
- `SpikingLayer` - Leaky integrate-and-fire neurons
- `AmariHopfieldNetwork` - Recurrent associative memory

**Trainers**:
- `HebbianTrainer` / `AntiHebbianTrainer` - Correlation-based learning
- `BCMTrainer` - Homeostatic threshold mechanism
- `OjaTrainer` / `SangerTrainer` - PCA extraction
- `STDPTrainer` - Spike-timing-dependent plasticity

For more details, see [API documentation](../../api_reference.md) or explore source code in `canns.models.brain_inspired` and `canns.trainer`.
