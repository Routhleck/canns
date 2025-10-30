# Brain-Inspired Models Examples

This directory contains examples demonstrating various brain-inspired learning rules and neural network models implemented in CANNS.

## Hebbian Learning Examples

### Hopfield Networks

#### `hopfield_train.py`
Basic Hopfield network training on real images from scikit-image.
- **Trainer**: `HebbianTrainer`
- **Model**: `AmariHopfieldNetwork`
- **Features**: Pattern storage, corruption recovery, batch prediction

#### `hopfield_train_1d.py`
1D pattern storage and retrieval with Hopfield networks.
- **Trainer**: `HebbianTrainer`
- **Model**: `AmariHopfieldNetwork`
- **Features**: 1D binary patterns, visualization of recall

#### `hopfield_train_mnist.py`
MNIST digit storage and retrieval using Hopfield networks.
- **Trainer**: `HebbianTrainer`
- **Model**: `AmariHopfieldNetwork`
- **Features**: Real-world dataset, high-dimensional patterns

#### `hopfield_hebbian_vs_antihebbian.py`
Comparison of Hebbian vs Anti-Hebbian learning rules.
- **Trainers**: `HebbianTrainer`, `AntiHebbianTrainer`
- **Model**: `AmariHopfieldNetwork`
- **Features**: Side-by-side comparison, decorrelation effects

#### `hopfield_energy_diagnostics.py` ⭐ NEW
Advanced Hopfield network with energy-based diagnostics and capacity analysis.
- **Trainer**: `HopfieldEnergyTrainer`
- **Model**: `AmariHopfieldNetwork`
- **Features**:
  - Pattern storage capacity estimation (N/4ln(N))
  - Energy landscape visualization
  - Pattern recall with overlap metrics
  - Capacity limit testing
  - Noise robustness analysis
- **Output**: `hopfield_energy_diagnostics.png`, `hopfield_capacity_test.png`

### Oja's Rule for PCA

#### `oja_pca_extraction.py` ⭐ NEW
Principal component analysis using Oja's normalized Hebbian learning.
- **Trainer**: `OjaTrainer`
- **Model**: `LinearHebbLayer`
- **Features**:
  - Normalized Hebbian learning with automatic weight stabilization
  - PCA extraction from high-dimensional data
  - Weight norm convergence tracking
  - Comparison with sklearn PCA
  - Variance explained visualization
- **Output**: `oja_pca_extraction.png`

### BCM Plasticity

#### `bcm_receptive_fields.py` ⭐ NEW
Receptive field development using BCM sliding-threshold plasticity.
- **Trainer**: `BCMTrainer`
- **Model**: `BCMLayer`
- **Features**:
  - Sliding threshold adaptation (θ = ⟨y²⟩)
  - LTP/LTD regime switching
  - Oriented bar stimuli
  - Receptive field visualization
  - Orientation tuning curves
- **Output**: `bcm_receptive_fields.png`, `bcm_orientation_tuning.png`

## Temporal Learning Examples

### Spike-Timing-Dependent Plasticity (STDP)

#### `stdp_spiking_plasticity.py` ⭐ NEW
Temporal learning in spiking neural networks using STDP.
- **Trainer**: `STDPTrainer`
- **Model**: `LIFSpikingNetwork`
- **Features**:
  - Leaky integrate-and-fire (LIF) neurons
  - Exponential STDP timing window
  - Trace-based synaptic updates
  - Spike raster visualization
  - Weight change analysis (LTP/LTD)
  - STDP window visualization
- **Output**: `stdp_spiking_plasticity.png`, `stdp_weight_analysis.png`

## Energy-Based and Generative Models

### Restricted Boltzmann Machine (RBM)

#### `rbm_mnist_features.py` ⭐ NEW
Feature learning from MNIST digits using RBM with contrastive divergence.
- **Trainer**: `ContrastiveDivergenceTrainer`
- **Model**: `RestrictedBoltzmannModel`
- **Features**:
  - CD-k algorithm (k=1)
  - Unsupervised feature extraction
  - Learned feature visualization
  - Reconstruction quality
  - Feature activation analysis
  - Sparsity metrics
- **Output**: `rbm_mnist_features.png`, `rbm_feature_analysis.png`

### Helmholtz Machine (Wake-Sleep)

#### `wake_sleep_generative.py` ⭐ NEW
Generative modeling using the wake-sleep algorithm.
- **Trainer**: `WakeSleepTrainer`
- **Model**: `HelmholtzMachine`
- **Features**:
  - Dual-phase learning (wake and sleep)
  - Recognition network (bottom-up)
  - Generative network (top-down)
  - Sample generation from prior
  - Reconstruction visualization
  - Hidden representation analysis
- **Output**: `wake_sleep_generative.png`, `wake_sleep_representations.png`

## Running the Examples

All examples can be run independently:

```bash
# Navigate to examples directory
cd examples/brain_inspired/

# Run any example
python oja_pca_extraction.py
python bcm_receptive_fields.py
python hopfield_energy_diagnostics.py
python stdp_spiking_plasticity.py
python rbm_mnist_features.py
python wake_sleep_generative.py
```

### Dependencies

The examples require:
- `numpy`
- `matplotlib`
- `scikit-learn` (optional, for some examples)
- `scikit-image` (optional, for Hopfield image examples)

Install with:
```bash
pip install numpy matplotlib scikit-learn scikit-image
```

## Example Categories

| Learning Rule | Model | Trainer | Example File |
|--------------|-------|---------|-------------|
| **Hebbian** | AmariHopfieldNetwork | HebbianTrainer | hopfield_train*.py |
| **Anti-Hebbian** | AmariHopfieldNetwork | AntiHebbianTrainer | hopfield_hebbian_vs_antihebbian.py |
| **Oja** | LinearHebbLayer | OjaTrainer | oja_pca_extraction.py |
| **BCM** | BCMLayer | BCMTrainer | bcm_receptive_fields.py |
| **STDP** | LIFSpikingNetwork | STDPTrainer | stdp_spiking_plasticity.py |
| **Hopfield Energy** | AmariHopfieldNetwork | HopfieldEnergyTrainer | hopfield_energy_diagnostics.py |
| **Contrastive Divergence** | RestrictedBoltzmannModel | ContrastiveDivergenceTrainer | rbm_mnist_features.py |
| **Wake-Sleep** | HelmholtzMachine | WakeSleepTrainer | wake_sleep_generative.py |

## Key Concepts

### Hebbian Learning
"Neurons that fire together, wire together" - strengthens connections between co-active neurons.

### Anti-Hebbian Learning
"Neurons that fire together, wire apart" - weakens connections for decorrelation and competitive learning.

### Oja's Rule
Normalized Hebbian learning with automatic weight stabilization, extracting principal components.

### BCM Rule
Sliding-threshold plasticity with dynamic LTP/LTD switching based on postsynaptic activity history.

### STDP
Synaptic plasticity depends on precise timing: pre-before-post → LTP, post-before-pre → LTD.

### Energy-Based Models
Models that minimize energy functions (Hopfield, RBM) for pattern completion and feature learning.

### Generative Models
Models that learn to generate new samples (Helmholtz Machine) using dual recognition/generative networks.

## References

- **Hopfield (1982)**: Neural networks and physical systems with emergent collective computational abilities
- **Oja (1982)**: Simplified neuron model as a principal component analyzer
- **Bienenstock et al. (1982)**: Theory for the development of neuron selectivity
- **Bi & Poo (1998)**: Synaptic modifications in cultured hippocampal neurons
- **Hinton (2002)**: Training products of experts by minimizing contrastive divergence
- **Hinton et al. (1995)**: The wake-sleep algorithm for unsupervised neural networks

## Tips

1. **Start simple**: Try `oja_pca_extraction.py` or `bcm_receptive_fields.py` first
2. **Adjust learning rates**: Different datasets may require different learning rates
3. **Visualization**: All examples save plots automatically - check the output files
4. **Epochs**: Increase n_epochs for better convergence if needed
5. **Random seeds**: Examples use fixed seeds for reproducibility - change for different runs
