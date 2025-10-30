"""
BCM rule for receptive field development.

This example demonstrates:
- BCMTrainer with sliding threshold plasticity
- Receptive field formation from natural stimuli
- LTP/LTD regime visualization
"""

import numpy as np
from matplotlib import pyplot as plt

from canns.models.brain_inspired import BCMLayer
from canns.trainer import BCMTrainer

np.random.seed(42)


def generate_oriented_stimuli(n_samples, size=16, n_orientations=8):
    """Generate oriented bar stimuli (simplified Gabor-like patterns)."""
    stimuli = []
    orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)

    for _ in range(n_samples):
        # Random orientation
        angle = np.random.choice(orientations)

        # Create oriented bar
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x = x - size / 2
        y = y - size / 2

        # Rotate coordinates
        x_rot = x * np.cos(angle) + y * np.sin(angle)

        # Create bar pattern
        pattern = np.exp(-((x_rot / 2) ** 2))

        # Flatten and normalize to [0, 1]
        pattern_flat = pattern.flatten()
        pattern_flat = (pattern_flat - pattern_flat.min()) / (pattern_flat.max() - pattern_flat.min() + 1e-8)

        stimuli.append(pattern_flat.astype(np.float32))

    return stimuli, orientations


# Generate training data
size = 12  # 12x12 images
n_samples = 1000
train_data, orientations = generate_oriented_stimuli(n_samples, size=size)

print(f"Generated {len(train_data)} oriented bar stimuli")
print(f"Input dimension: {len(train_data[0])}")

# Create BCM layer
n_neurons = 4  # Learn 4 different receptive fields
model = BCMLayer(input_size=size * size, output_size=n_neurons, threshold_tau=50.0)
model.init_state()

trainer = BCMTrainer(model, learning_rate=0.00001)  # Small learning rate for stability

# Track evolution of receptive fields and thresholds
n_epochs = 100  # More epochs with smaller learning rate
threshold_history = []
weight_history = []

for epoch in range(n_epochs):
    # Shuffle data
    shuffled = [train_data[i] for i in np.random.permutation(len(train_data))]

    # Train
    trainer.train(shuffled)

    # Record state
    threshold_history.append(model.theta.value.copy())
    weight_history.append(model.W.value.copy())

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1}/{n_epochs}: "
            f"Thresholds: {model.theta.value}, "
            f"Weight range: [{model.W.value.min():.3f}, {model.W.value.max():.3f}]"
        )

# Visualize results
fig = plt.figure(figsize=(14, 10))

# Plot 1: Threshold evolution
ax1 = plt.subplot(3, 2, 1)
threshold_history_arr = np.array(threshold_history)
for i in range(n_neurons):
    ax1.plot(threshold_history_arr[:, i], label=f"Neuron {i+1}")
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
    ax2.plot(weight_norms_arr[:, i], label=f"Neuron {i+1}")
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
    ax.set_title(f"Neuron {i+1} Receptive Field")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("bcm_receptive_fields.png")
print("\nPlot saved as 'bcm_receptive_fields.png'")
plt.show()

# Test neuron selectivity to different orientations
print("\n" + "=" * 60)
print("Testing Orientation Selectivity")
print("=" * 60)

test_orientations = np.linspace(0, np.pi, 8, endpoint=False)
selectivity_matrix = np.zeros((n_neurons, len(test_orientations)))

for j, angle in enumerate(test_orientations):
    # Create test stimulus
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x = x - size / 2
    y = y - size / 2
    x_rot = x * np.cos(angle) + y * np.sin(angle)
    pattern = np.exp(-((x_rot / 2) ** 2))
    pattern_flat = pattern.flatten()
    pattern_flat = (pattern_flat - pattern_flat.min()) / (pattern_flat.max() - pattern_flat.min() + 1e-8)

    # Get neuron responses
    response = trainer.predict(pattern_flat.astype(np.float32))
    selectivity_matrix[:, j] = response

# Plot orientation tuning curves
fig, ax = plt.subplots(figsize=(10, 6))
angles_deg = np.degrees(test_orientations)
for i in range(n_neurons):
    ax.plot(angles_deg, selectivity_matrix[i], marker="o", label=f"Neuron {i+1}")

ax.set_xlabel("Orientation (degrees)")
ax.set_ylabel("Response")
ax.set_title("Orientation Tuning Curves")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bcm_orientation_tuning.png")
print("Tuning curve plot saved as 'bcm_orientation_tuning.png'")
plt.show()

# Find preferred orientation for each neuron
for i in range(n_neurons):
    preferred_idx = np.argmax(selectivity_matrix[i])
    preferred_angle = angles_deg[preferred_idx]
    print(f"Neuron {i+1} prefers orientation: {preferred_angle:.1f}°")
