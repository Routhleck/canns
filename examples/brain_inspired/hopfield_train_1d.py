"""
Continuous Hopfield training on 1D vector patterns.

This example demonstrates:
- Continuous Hopfield dynamics with tanh activation
- Generic Hebbian training (trainer.train) with continuous-valued 1D vectors
- Batch prediction (trainer.predict_batch)
- Simple progress reporting
"""

import numpy as np
from matplotlib import pyplot as plt

from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer

np.random.seed(42)

# Create continuous-valued 1D patterns (values in [-1, +1])
# Increase dimension for better storage capacity
# Hopfield capacity ≈ N/(4*ln(N)), so 500 neurons → ~40 patterns theoretical capacity
dim = 500

# Pattern 1: sinusoidal pattern
pattern1 = np.sin(np.linspace(0, 4 * np.pi, dim)).astype(np.float32)

# Pattern 2: cosine pattern with different frequency
pattern2 = np.cos(np.linspace(0, 6 * np.pi, dim)).astype(np.float32)

# Pattern 3: smooth random pattern (smoothed with moving average)
random_vals = np.random.randn(dim).astype(np.float32)
window = 10
pattern3 = np.convolve(random_vals, np.ones(window) / window, mode='same')
pattern3 = np.tanh(pattern3)  # Normalize to [-1, 1]

# Pattern 4: triangular wave pattern
t = np.linspace(0, 1, dim)
pattern4 = (2 * np.abs(2 * (t * 4 - np.floor(t * 4 + 0.5))) - 1).astype(np.float32)

data_list = [pattern1, pattern2, pattern3, pattern4]

# Print storage capacity info
print(f"Network size: {dim} neurons")
print(f"Theoretical storage capacity: ~{dim // (4 * int(np.log(dim)))} patterns")
print(f"Storing {len(data_list)} patterns\n")

# Create model with continuous activation (tanh)
# temperature controls the smoothness of the activation
# Lower temperature makes activation sharper (more binary-like)
model = AmariHopfieldNetwork(
    num_neurons=data_list[0].shape[0],
    asyn=False,
    activation="tanh",
    temperature=0.5  # Lower temperature for sharper transitions
)
trainer = HebbianTrainer(model)
trainer.train(data_list)


# Generate testset with Gaussian noise
def add_gaussian_noise(pattern, noise_std=0.3):
    """Add Gaussian noise to continuous pattern."""
    noisy = pattern + np.random.normal(0, noise_std, size=pattern.shape).astype(np.float32)
    # Clip to [-1, 1] range
    return np.clip(noisy, -1.0, 1.0)


tests = [add_gaussian_noise(d, noise_std=0.3) for d in data_list]

# Predict corrupted patterns (compiled for speed, show sample-level progress)
predicted = trainer.predict_batch(tests, show_sample_progress=True)


# Display predict results for 1D vectors
def plot_1d_patterns(data, test, predicted, figsize=(12, 8)):
    """Plot 1D patterns as line plots and heatmaps."""
    fig, axes = plt.subplots(len(data), 3, figsize=figsize)

    for i in range(len(data)):
        if i == 0:
            axes[i, 0].set_title('Original Pattern', fontsize=12)
            axes[i, 1].set_title("Corrupted Input", fontsize=12)
            axes[i, 2].set_title('Recovered Pattern', fontsize=12)

        # Plot as line plots
        axes[i, 0].plot(data[i], 'b-', linewidth=1)
        axes[i, 0].set_ylim([-1.5, 1.5])
        axes[i, 0].set_ylabel(f'Pattern {i + 1}', fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(test[i], 'r-', linewidth=1)
        axes[i, 1].set_ylim([-1.5, 1.5])
        axes[i, 1].grid(True, alpha=0.3)

        axes[i, 2].plot(predicted[i], 'g-', linewidth=1)
        axes[i, 2].set_ylim([-1.5, 1.5])
        axes[i, 2].grid(True, alpha=0.3)

        # Only show x-label on bottom row
        if i == len(data) - 1:
            axes[i, 0].set_xlabel('Neuron index')
            axes[i, 1].set_xlabel('Neuron index')
            axes[i, 2].set_xlabel('Neuron index')

    plt.tight_layout()
    plt.savefig("continuous_hopfield_train_1d.png", dpi=150)
    print("Figure saved to continuous_hopfield_train_1d.png")
    plt.show()


# Print recovery statistics for continuous patterns
print("\nRecovery Statistics:")
for i in range(len(data_list)):
    original = data_list[i]
    noisy = tests[i]
    recovered = predicted[i]

    # Calculate MSE and correlation
    noise_mse = np.mean((original - noisy) ** 2)
    recovery_mse = np.mean((original - recovered) ** 2)
    correlation = np.corrcoef(original, recovered)[0, 1]

    print(f"Pattern {i + 1}:")
    print(f"  Noise MSE: {noise_mse:.4f} → Recovery MSE: {recovery_mse:.4f}")
    print(f"  Correlation with original: {correlation:.4f}")

plot_1d_patterns(data_list, tests, predicted, figsize=(12, 8))
