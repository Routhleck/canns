# Tutorial: Pattern Storage and Recall with Hebbian Learning

> **Reading Time**: ~30-35 minutes
> **Difficulty**: Intermediate
> **Prerequisites**: Basic understanding of neural networks and Python

This tutorial introduces brain-inspired learning through Hebbian plasticity and associative memory using Hopfield networks.

---

## Table of Contents

1. [Introduction to Hebbian Learning](#1-introduction-to-hebbian-learning)
2. [Hopfield Networks for Associative Memory](#2-hopfield-networks-for-associative-memory)
3. [Complete Example: MNIST Digit Memory](#3-complete-example-mnist-digit-memory)
4. [Anti-Hebbian Learning](#4-anti-hebbian-learning)
5. [Next Steps](#5-next-steps)

---

## 1. Introduction to Hebbian Learning

### 1.1 The Hebbian Principle

**"Neurons that fire together, wire together"**

Hebbian learning is a biologically-inspired learning rule where synaptic strength increases when pre- and post-synaptic neurons are simultaneously active:

$$
\Delta W_{ij} = \eta \cdot x_i \cdot x_j
$$

Where:
- $W_{ij}$: Synaptic weight from neuron i to neuron j
- $\eta$: Learning rate
- $x_i, x_j$: Activities of neurons i and j

### 1.2 Why Hebbian Learning?

**Biological realism**:
- Local learning rule (no global error signal needed)
- Activity-dependent plasticity
- Matches experimental observations in cortex

**Computational advantages**:
- Unsupervised learning
- Pattern completion and noise resistance
- Attractor dynamics for memory

### 1.3 Hebbian Learning in CANNs

```python
from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer

# Create network
model = AmariHopfieldNetwork(num_neurons=784)  # 28x28 images
model.init_state()

# Create trainer with Hebbian learning
trainer = HebbianTrainer(model, compiled_prediction=True)

# Train on patterns
trainer.train(pattern_list)

# Retrieve from corrupted input
output = trainer.predict(noisy_pattern)
```

---

## 2. Hopfield Networks for Associative Memory

### 2.1 What is a Hopfield Network?

A **Hopfield network** is a recurrent neural network that stores patterns as stable attractors in its energy landscape:

- **Storage**: Hebbian learning creates attractors at training patterns
- **Retrieval**: Network dynamics converge to nearest stored pattern
- **Capacity**: Can store approximately $0.138 \times N$ patterns ($N$ = number of neurons)

### 2.2 Network Dynamics

The Hopfield network updates its state to minimize energy:

$$
E = -\frac{1}{2} \sum_{i,j} W_{ij} \cdot x_i \cdot x_j
$$

**Update rules**:
- **Asynchronous**: Update one neuron at a time
- **Synchronous**: Update all neurons simultaneously

```python
from canns.models.brain_inspired import AmariHopfieldNetwork

# Discrete activation (sign function): x ∈ {-1, +1}
model = AmariHopfieldNetwork(
    num_neurons=784,
    threshold=80.0,      # Convergence threshold
    asyn=False,          # Synchronous updates
    activation="sign"    # Binary states {-1, +1}
)
```

### 2.3 Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_neurons` | int | Network size (e.g., 784 for 28×28 images) |
| `threshold` | float | Convergence criterion (max iterations) |
| `asyn` | bool | Asynchronous (True) or synchronous (False) updates |
| `activation` | str | "sign" (discrete) or "tanh"/"sigmoid" (continuous) |

---

## 3. Complete Example: MNIST Digit Memory

### 3.1 Setup and Data Loading

```python
import numpy as np
from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer

# Load MNIST data (automatically tries multiple sources)
def load_mnist_data():
    """Load MNIST using available library (datasets/torchvision/keras/sklearn)"""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("mnist", split="train")
        x_train = np.stack([np.array(img, dtype=np.float32) for img in ds_train["image"]])
        y_train = np.array(ds_train["label"], dtype=np.int64)
        return x_train, y_train
    except:
        # Fallback to other sources...
        from torchvision.datasets import MNIST
        ds_train = MNIST(root="~/.cache/torchvision", train=True, download=True)
        x_train = ds_train.data.numpy().astype(np.float32)
        y_train = ds_train.targets.numpy().astype(np.int64)
        return x_train, y_train

x_train, y_train = load_mnist_data()
print(f"Loaded {len(x_train)} training images")
```

### 3.2 Data Preprocessing

Hopfield networks work best with binary patterns {-1, +1}:

```python
def threshold_to_binary(image, use_mean=True):
    """Convert grayscale image to {-1, +1}"""
    if use_mean:
        threshold = image.mean()
    else:
        from skimage.filters import threshold_mean
        threshold = threshold_mean(image)

    binary = image > threshold
    return np.where(binary, 1.0, -1.0).astype(np.float32)

def flatten_image(image_2d):
    """Flatten 2D image to 1D vector"""
    return image_2d.reshape(-1)

# Select patterns to store (e.g., digits 0, 1, 2)
classes = [0, 1, 2]

# Get one training example per class
train_patterns = []
for digit in classes:
    # Find first occurrence of this digit
    idx = np.where(y_train == digit)[0][0]
    img_2d = x_train[idx]

    # Convert to binary and flatten
    binary = threshold_to_binary(img_2d)
    flat = flatten_image(binary)
    train_patterns.append(flat)

print(f"Prepared {len(train_patterns)} training patterns")
print(f"Pattern dimensions: {train_patterns[0].shape}")
```

### 3.3 Create and Train Hopfield Network

```python
# Create Hopfield network
n_neurons = train_patterns[0].size  # 784 for 28×28 images
model = AmariHopfieldNetwork(
    num_neurons=n_neurons,
    threshold=80.0,        # Max iterations for convergence
    asyn=False,            # Synchronous updates
    activation="sign"      # Binary activation
)
model.init_state()

# Create Hebbian trainer
trainer = HebbianTrainer(
    model,
    compiled_prediction=True  # Use JIT compilation for speed
)

# Train on patterns (one-shot learning!)
print("Training network...")
trainer.train(train_patterns)
print("Training complete!")
```

**Training details**:
- **One-shot learning**: Patterns stored in single pass
- **Weight update**: $W = \frac{1}{P} \sum_{\mu} \mathbf{p}_\mu \mathbf{p}_\mu^T$ (normalized Hebbian)
- **Diagonal zeroing**: $W_{ii} = 0$ (no self-connections)
- **Mean subtraction**: Optional for better storage

### 3.4 Pattern Retrieval

Test the network's ability to retrieve stored patterns:

```python
# Get test patterns (different examples of same digits)
test_patterns = []
for digit in classes:
    # Find second occurrence (different from training)
    idx = np.where(y_train == digit)[0][1]
    img_2d = x_train[idx]
    binary = threshold_to_binary(img_2d)
    flat = flatten_image(binary)
    test_patterns.append(flat)

# Retrieve patterns
print("Retrieving patterns...")
retrieved = trainer.predict_batch(
    test_patterns,
    show_sample_progress=True  # Show convergence progress
)

print(f"Retrieved {len(retrieved)} patterns")
```

### 3.5 Visualization

Compare training, input, and retrieved patterns:

```python
import matplotlib.pyplot as plt

def reshape_to_image(flat_vector):
    """Reshape 1D vector back to 2D image"""
    dim = int(np.sqrt(flat_vector.size))
    return flat_vector.reshape(dim, dim)

# Create visualization
fig, axes = plt.subplots(len(classes), 3, figsize=(6, 2*len(classes)))

for i in range(len(classes)):
    # Training pattern
    axes[i, 0].imshow(reshape_to_image(train_patterns[i]), cmap='gray')
    axes[i, 0].axis('off')
    if i == 0:
        axes[i, 0].set_title('Stored Pattern')

    # Test input
    axes[i, 1].imshow(reshape_to_image(test_patterns[i]), cmap='gray')
    axes[i, 1].axis('off')
    if i == 0:
        axes[i, 1].set_title('Input Pattern')

    # Retrieved pattern
    axes[i, 2].imshow(reshape_to_image(retrieved[i]), cmap='gray')
    axes[i, 2].axis('off')
    if i == 0:
        axes[i, 2].set_title('Retrieved Pattern')

plt.tight_layout()
plt.savefig('hopfield_mnist_retrieval.png', dpi=150)
plt.show()
```

**Expected results**:
- Retrieved patterns closely match stored patterns
- Network completes patterns despite differences in input
- Some distortions possible (depends on pattern similarity)

### 3.6 Testing with Noise

Add noise to test robustness:

```python
def add_noise(pattern, noise_level=0.1):
    """Flip random bits with given probability"""
    noise_mask = np.random.random(pattern.shape) < noise_level
    noisy = pattern.copy()
    noisy[noise_mask] *= -1  # Flip sign
    return noisy

# Test with noisy inputs
noise_levels = [0.0, 0.1, 0.2, 0.3]

for noise in noise_levels:
    print(f"\nTesting with {int(noise*100)}% noise:")
    noisy_inputs = [add_noise(p, noise) for p in test_patterns]
    recovered = trainer.predict_batch(noisy_inputs)

    # Calculate accuracy (proportion of correct bits)
    accuracies = []
    for orig, recov in zip(train_patterns, recovered):
        accuracy = np.mean(orig == recov)
        accuracies.append(accuracy)

    print(f"  Mean accuracy: {np.mean(accuracies):.2%}")
```

**Observations**:
- Low noise (< 20%): Near-perfect retrieval
- Medium noise (20-30%): Degraded but recognizable
- High noise (> 40%): May converge to wrong pattern

---

## 4. Anti-Hebbian Learning

### 4.1 What is Anti-Hebbian Learning?

**Anti-Hebbian learning** uses negative correlations:

$$
\Delta W_{ij} = -\eta \cdot x_i \cdot x_j
$$

**Purpose**:
- **Decorrelation**: Reduce redundancy in representations
- **Sparse coding**: Learn efficient, distributed codes
- **Competitive learning**: Winner-take-all dynamics

### 4.2 When to Use Anti-Hebbian

- **Lateral inhibition**: Between neurons in same layer
- **Feature decorrelation**: Remove statistical redundancies
- **Sparse representations**: Encourage few active neurons

### 4.3 Example: Comparing Hebbian vs Anti-Hebbian

```python
from canns.trainer import AntiHebbianTrainer

# Create two networks with same architecture
model_hebb = AmariHopfieldNetwork(num_neurons=784, activation="tanh")
model_anti = AmariHopfieldNetwork(num_neurons=784, activation="tanh")

model_hebb.init_state()
model_anti.init_state()

# Train with different rules
trainer_hebb = HebbianTrainer(model_hebb)
trainer_anti = AntiHebbianTrainer(model_anti)

trainer_hebb.train(train_patterns)
trainer_anti.train(train_patterns)

# Compare weight matrices
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.imshow(model_hebb.W.value, cmap='RdBu', vmin=-1, vmax=1)
ax1.set_title('Hebbian Weights')
ax1.axis('off')

ax2.imshow(model_anti.W.value, cmap='RdBu', vmin=-1, vmax=1)
ax2.set_title('Anti-Hebbian Weights')
ax2.axis('off')

plt.tight_layout()
plt.show()
```

**Observations**:
- **Hebbian**: Positive correlations, reinforce patterns
- **Anti-Hebbian**: Negative correlations, decorrelate features
- **Combined**: Often use both (Hebbian feedforward, anti-Hebbian lateral)

---

## 5. Next Steps

Congratulations! You've learned the basics of brain-inspired learning with Hebbian plasticity and associative memory.

### Key Takeaways

1. **Hebbian learning** is local, unsupervised, and biologically realistic
2. **Hopfield networks** store patterns as energy minima
3. **One-shot learning** possible with Hebbian rule
4. **Pattern completion** works with partial/noisy inputs
5. **Anti-Hebbian learning** provides decorrelation and sparse coding

### When to Use Hebbian Learning

- **Associative memory**: Store and retrieve patterns
- **Unsupervised learning**: No labels required
- **One-shot learning**: Learn from few examples
- **Pattern completion**: Robust to noise and missing data
- **Biological modeling**: Match neural plasticity

### Limitations

- **Capacity limits**: $\sim 0.138 \times N$ patterns for $N$ neurons
- **Spurious attractors**: Network may converge to unwanted states
- **No error minimization**: Unlike backpropagation
- **Binary patterns**: Works best with $\{-1, +1\}$ states

### Continue Learning

- **Next**: [Tutorial: Feature Learning & Temporal Patterns](./02_feature_learning_temporal.md) - Learn BCM, Oja, and STDP plasticity
- **Related**: Tutorials 1-7 for CANN modeling foundations

### Advanced Topics

- **Storage capacity**: Theoretical limits and practical considerations
- **Energy landscape**: Understanding attractor basins
- **Continuous Hopfield**: Using continuous activations (tanh/sigmoid)
- **Modern Hopfield**: Recent advances with exponential capacity

### Related Models

These brain-inspired models are available in the CANNs library:
- `AmariHopfieldNetwork` - Pattern storage and associative memory
- `LinearLayer` - Generic feedforward layer with Hebbian learning
- `SpikingLayer` - Leaky integrate-and-fire neurons for STDP

For more details, see the [API documentation](../../api_reference.md) or explore `canns.models.brain_inspired` and `canns.trainer`.
