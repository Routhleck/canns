"""
BCM rule for receptive field development.

This example demonstrates:
- BCMTrainer with sliding threshold plasticity
- Receptive field formation from natural stimuli
- LTP/LTD regime visualization
- JAX-accelerated training with brainpy
- Tuning curve visualization using canns.analyzer
"""

import brainpy.math as bm
import numpy as np
from matplotlib import pyplot as plt

from canns.analyzer.visualization import PlotConfigs, tuning_curve
from canns.models.brain_inspired import LinearLayer
from canns.trainer import BCMTrainer

np.random.seed(42)
bm.random.seed(42)


def create_oriented_bar(angle, size=16):
    """Create a single oriented bar stimulus at given angle."""
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x = x - size / 2
    y = y - size / 2

    # Rotate coordinates
    x_rot = x * np.cos(angle) + y * np.sin(angle)

    # Create bar pattern (Gaussian in rotated x direction)
    pattern = np.exp(-((x_rot / 2) ** 2))

    # Flatten and normalize to [0, 1]
    pattern_flat = pattern.flatten()
    pattern_flat = (pattern_flat - pattern_flat.min()) / (
        pattern_flat.max() - pattern_flat.min() + 1e-8
    )

    return pattern_flat.astype(np.float32)


def generate_oriented_stimuli(n_samples, size=16, n_orientations=8):
    """Generate oriented bar stimuli (simplified Gabor-like patterns)."""
    stimuli = []
    orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)

    for _ in range(n_samples):
        # Random orientation
        angle = np.random.choice(orientations)
        stimuli.append(create_oriented_bar(angle, size))

    return stimuli, orientations


# Generate training data
size = 12  # 12x12 images
n_samples = 1000
train_data, orientations = generate_oriented_stimuli(n_samples, size=size)

print(f"Generated {len(train_data)} oriented bar stimuli")
print(f"Input dimension: {len(train_data[0])}")

# Create linear layer with BCM threshold support
n_neurons = 4  # Learn 4 different receptive fields
model = LinearLayer(
    input_size=size * size, output_size=n_neurons, use_bcm_threshold=True, threshold_tau=50.0
)

# Create BCMTrainer
trainer = BCMTrainer(model, learning_rate=0.00001)

# Track evolution of receptive fields and thresholds
n_epochs = 100  # More epochs with JIT-accelerated training
checkpoint_interval = 20  # Save checkpoints every N epochs
threshold_history = []
weight_history = []

print(f"\nTraining for {n_epochs} epochs...")
print(f"Checkpoint interval: {checkpoint_interval} epochs")

for epoch in range(n_epochs):
    # Train on full dataset (batch learning)
    trainer.train(train_data)

    # Record state at checkpoints
    if (epoch + 1) % checkpoint_interval == 0:
        threshold_history.append(model.theta.value.copy())
        weight_history.append(model.W.value.copy())
        print(
            f"Epoch {epoch + 1}/{n_epochs}: "
            f"Thresholds: [{model.theta.value.min():.4f}, {model.theta.value.max():.4f}], "
            f"Weight range: [{model.W.value.min():.3f}, {model.W.value.max():.3f}]"
        )

print("\nTraining complete!")

# Visualize results
fig = plt.figure(figsize=(14, 10))

# Plot 1: Threshold evolution
ax1 = plt.subplot(3, 2, 1)
threshold_history_arr = np.array(threshold_history)
epochs_checkpointed = np.arange(checkpoint_interval, n_epochs + 1, checkpoint_interval)
for i in range(n_neurons):
    ax1.plot(epochs_checkpointed, threshold_history_arr[:, i], label=f"Neuron {i + 1}", marker='o', markersize=3)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Threshold θ")
ax1.set_title("BCM Threshold Evolution")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Weight magnitude evolution
ax2 = plt.subplot(3, 2, 2)
weight_norms = [np.linalg.norm(w, axis=1) for w in weight_history]
weight_norms_arr = np.array(weight_norms)
for i in range(n_neurons):
    ax2.plot(epochs_checkpointed, weight_norms_arr[:, i], label=f"Neuron {i + 1}", marker='o', markersize=3)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Weight Norm")
ax2.set_title("Weight Magnitude Evolution")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3-6: Final learned receptive fields
final_weights = model.W.value
for i in range(n_neurons):
    ax = plt.subplot(3, 2, 3 + i)
    rf = final_weights[i].reshape(size, size)
    im = ax.imshow(rf, cmap="RdBu_r", interpolation="nearest")
    ax.set_title(f"Neuron {i + 1} Receptive Field")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("bcm_receptive_fields.png")
print("\nPlot saved as 'bcm_receptive_fields.png'")
plt.show()

# Test neuron selectivity to different orientations using tuning_curve analyzer
print("\n" + "=" * 60)
print("Testing Orientation Selectivity")
print("=" * 60)

# Generate dense set of test stimuli at different orientations
n_test_angles = 16  # More angles for smoother tuning curves
test_orientations = np.linspace(0, np.pi, n_test_angles, endpoint=False)

# Create test stimuli and get responses
test_stimuli = [create_oriented_bar(angle, size) for angle in test_orientations]
test_responses = [trainer.predict(stimulus) for stimulus in test_stimuli]

# Convert to arrays for tuning curve analysis
stimulus_angles = test_orientations  # Shape: (n_test_angles,)
firing_rates = np.array(test_responses)  # Shape: (n_test_angles, n_neurons)

# Plot orientation tuning curves using analyzer
config = PlotConfigs.tuning_curve(
    num_bins=n_test_angles,
    title="Orientation Tuning Curves",
    xlabel="Orientation (radians)",
    ylabel="Neuron Response",
    figsize=(10, 6),
    save_path="bcm_orientation_tuning.png",
    show=True,
    kwargs={"linewidth": 2, "marker": "o", "markersize": 6}
)

tuning_curve(
    stimulus=stimulus_angles,
    firing_rates=firing_rates,
    neuron_indices=np.arange(n_neurons),  # Plot all neurons
    config=config
)

print("Tuning curve plot saved as 'bcm_orientation_tuning.png'")

# Find preferred orientation for each neuron
for i in range(n_neurons):
    preferred_idx = np.argmax(firing_rates[:, i])
    preferred_angle_rad = stimulus_angles[preferred_idx]
    preferred_angle_deg = np.degrees(preferred_angle_rad)
    print(f"Neuron {i + 1} prefers orientation: {preferred_angle_deg:.1f}° ({preferred_angle_rad:.3f} rad)")
