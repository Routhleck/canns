"""
Hopfield network with energy-based diagnostics.

This example demonstrates:
- Hopfield network training with HebbianTrainer
- HopfieldAnalyzer for capacity estimation and diagnostics
- Pattern storage and energy landscape analysis
- Pattern recall with overlap metrics
"""

import numpy as np
from matplotlib import pyplot as plt

from canns.analyzer.model_specific import HopfieldAnalyzer
from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer

np.random.seed(42)


def generate_random_patterns(n_patterns, n_neurons):
    """Generate random binary patterns."""
    patterns = []
    for _ in range(n_patterns):
        pattern = np.random.choice([-1.0, 1.0], size=n_neurons).astype(np.float32)
        patterns.append(pattern)
    return patterns


def add_noise(pattern, noise_level):
    """Add noise by flipping bits with given probability."""
    noisy = pattern.copy()
    n_flips = int(len(pattern) * noise_level)
    flip_indices = np.random.choice(len(pattern), size=n_flips, replace=False)
    noisy[flip_indices] *= -1
    return noisy


# Network parameters
n_neurons = 100
n_patterns = 10  # Test capacity

print(f"Creating Hopfield network with {n_neurons} neurons")
print(f"Storing {n_patterns} patterns")

# Create model and trainer
model = AmariHopfieldNetwork(num_neurons=n_neurons, asyn=False, activation="sign")

trainer = HebbianTrainer(
    model, subtract_mean=True, zero_diagonal=True, normalize_by_patterns=True
)

# Generate and store patterns
patterns = generate_random_patterns(n_patterns, n_neurons)
trainer.train(patterns)

# Create analyzer for diagnostics
analyzer = HopfieldAnalyzer(model, stored_patterns=patterns)

# Get statistics
stats = analyzer.get_statistics()
print("\n" + "=" * 60)
print("Network Statistics")
print("=" * 60)
print(f"Number of stored patterns: {stats['num_patterns']}")
print(f"Theoretical capacity: {stats['capacity_estimate']} patterns")
print(f"Capacity usage: {stats['capacity_usage']:.1%}")
print(f"Mean pattern energy: {stats['mean_pattern_energy']:.2f}")
print(f"Energy std: {stats['std_pattern_energy']:.2f}")
print(f"Energy range: [{stats['min_pattern_energy']:.2f}, {stats['max_pattern_energy']:.2f}]")

# Test pattern recall with varying noise levels
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
recall_results = []

print("\n" + "=" * 60)
print("Pattern Recall Performance")
print("=" * 60)

for noise_level in noise_levels:
    overlaps = []

    for i, pattern in enumerate(patterns):
        # Create noisy version
        noisy = add_noise(pattern, noise_level)

        # Recall pattern using trainer
        recalled = trainer.predict(noisy, num_iter=20)

        # Analyze recall quality
        diag = analyzer.analyze_recall(noisy, recalled)

        # Record overlap
        overlaps.append(diag["best_match_overlap"])

    mean_overlap = np.mean(overlaps)
    recall_results.append(mean_overlap)
    print(f"Noise level {noise_level:.1%}: Mean overlap = {mean_overlap:.3f}")

# Visualize results
fig = plt.figure(figsize=(14, 10))

# Plot 1: Pattern energies
ax1 = plt.subplot(2, 3, 1)
energies = analyzer.pattern_energies
ax1.bar(range(len(energies)), energies)
ax1.set_xlabel("Pattern Index")
ax1.set_ylabel("Energy")
ax1.set_title("Energy of Stored Patterns")
ax1.axhline(y=stats["mean_pattern_energy"], color="r", linestyle="--", label="Mean")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# Plot 2: Recall performance vs noise
ax2 = plt.subplot(2, 3, 2)
ax2.plot(noise_levels, recall_results, marker="o", linewidth=2)
ax2.set_xlabel("Noise Level")
ax2.set_ylabel("Mean Overlap with Stored Pattern")
ax2.set_title("Recall Performance vs Noise")
ax2.set_ylim([0, 1.1])
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Perfect recall")
ax2.axhline(y=0.0, color="r", linestyle="--", alpha=0.5, label="Random")
ax2.legend()

# Plot 3: Weight matrix visualization
ax3 = plt.subplot(2, 3, 3)
W = model.W.value
im = ax3.imshow(W, cmap="RdBu_r", aspect="auto")
ax3.set_xlabel("Neuron j")
ax3.set_ylabel("Neuron i")
ax3.set_title("Learned Weight Matrix")
plt.colorbar(im, ax=ax3)

# Plot 4-6: Example recall sequences
example_indices = [0, n_patterns // 2, n_patterns - 1]
noise_level = 0.3

for plot_idx, pattern_idx in enumerate(example_indices):
    ax = plt.subplot(2, 3, 4 + plot_idx)

    pattern = patterns[pattern_idx]
    noisy = add_noise(pattern, noise_level)
    recalled = trainer.predict(noisy, num_iter=20)
    diag = analyzer.analyze_recall(noisy, recalled)

    # Reshape for visualization (if square-ish)
    size = int(np.sqrt(n_neurons))
    if size * size == n_neurons:
        pattern_2d = pattern.reshape(size, size)
        noisy_2d = noisy.reshape(size, size)
        recalled_2d = recalled.reshape(size, size)

        # Create composite image
        composite = np.hstack([pattern_2d, noisy_2d, recalled_2d])
        ax.imshow(composite, cmap="gray", interpolation="nearest")
        ax.set_title(
            f"Pattern {pattern_idx}: Overlap={diag['best_match_overlap']:.2f}\n"
            f"Original | Noisy ({noise_level:.0%}) | Recalled"
        )
    else:
        # Just plot as line plots if not square
        x = np.arange(n_neurons)
        ax.plot(x, pattern, "g-", alpha=0.7, label="Original")
        ax.plot(x, noisy, "r-", alpha=0.5, label="Noisy")
        ax.plot(x, recalled, "b-", alpha=0.7, label="Recalled")
        ax.set_title(f"Pattern {pattern_idx}: Overlap={diag['best_match_overlap']:.2f}")
        ax.set_ylim([-1.5, 1.5])
        ax.legend(fontsize=8)

    ax.axis("off")

plt.tight_layout()
plt.savefig("hopfield_energy_diagnostics.png")
print("\nPlot saved as 'hopfield_energy_diagnostics.png'")
plt.show()

# Test capacity limit
print("\n" + "=" * 60)
print("Testing Capacity Limit")
print("=" * 60)

capacity_test_results = []
pattern_counts = [5, 10, 15, 20, 25]

for n_test_patterns in pattern_counts:
    # Create fresh network
    test_model = AmariHopfieldNetwork(num_neurons=n_neurons, asyn=False, activation="sign")
    test_trainer = HebbianTrainer(test_model)

    # Store patterns
    test_patterns = generate_random_patterns(n_test_patterns, n_neurons)
    test_trainer.train(test_patterns)

    # Create analyzer for this test
    test_analyzer = HopfieldAnalyzer(test_model, stored_patterns=test_patterns)

    # Test recall
    noise_level = 0.2
    overlaps = []
    for pattern in test_patterns:
        noisy = add_noise(pattern, noise_level)
        recalled = test_trainer.predict(noisy, num_iter=20)
        diag = test_analyzer.analyze_recall(noisy, recalled)
        overlaps.append(diag["best_match_overlap"])

    mean_overlap = np.mean(overlaps)
    capacity_test_results.append(mean_overlap)

    capacity_usage = n_test_patterns / test_analyzer.estimate_capacity()
    print(
        f"Patterns: {n_test_patterns:2d} "
        f"(Capacity: {capacity_usage:5.1%}) -> "
        f"Mean overlap: {mean_overlap:.3f}"
    )

# Plot capacity test
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pattern_counts, capacity_test_results, marker="o", linewidth=2)
ax.axvline(
    x=analyzer.estimate_capacity(), color="r", linestyle="--", label="Theoretical capacity"
)
ax.set_xlabel("Number of Stored Patterns")
ax.set_ylabel("Mean Recall Overlap (20% noise)")
ax.set_title("Recall Quality vs Number of Stored Patterns")
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("hopfield_capacity_test.png")
print("\nCapacity test plot saved as 'hopfield_capacity_test.png'")
plt.show()
