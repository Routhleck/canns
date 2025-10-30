"""
Wake-Sleep algorithm for generative modeling.

This example demonstrates:
- WakeSleepTrainer with Helmholtz Machine
- Dual-phase learning (wake and sleep)
- Generation of new samples from learned model
"""

import numpy as np
from matplotlib import pyplot as plt

from canns.models.brain_inspired import HelmholtzMachine
from canns.trainer import WakeSleepTrainer

np.random.seed(42)


def generate_synthetic_data(n_samples, n_visible):
    """Generate synthetic binary data with structure."""
    # Create patterns with correlations
    data = []

    # Pattern 1: First half active
    pattern1 = np.zeros(n_visible)
    pattern1[: n_visible // 2] = 1.0

    # Pattern 2: Second half active
    pattern2 = np.zeros(n_visible)
    pattern2[n_visible // 2 :] = 1.0

    # Pattern 3: Alternating
    pattern3 = np.zeros(n_visible)
    pattern3[::2] = 1.0

    # Pattern 4: Sparse
    pattern4 = np.zeros(n_visible)
    pattern4[: n_visible // 4] = 1.0

    patterns = [pattern1, pattern2, pattern3, pattern4]

    for _ in range(n_samples):
        # Select random base pattern
        base = patterns[np.random.randint(len(patterns))].copy()

        # Add noise
        noise_mask = np.random.random(n_visible) < 0.2
        base[noise_mask] = 1 - base[noise_mask]

        data.append(base.astype(np.float32))

    return data


# Generate training data
n_visible = 16
n_hidden = 8
n_train = 200

train_data = generate_synthetic_data(n_train, n_visible)

print(f"Generated {len(train_data)} training samples")
print(f"Visible units: {n_visible}, Hidden units: {n_hidden}")

# Create Helmholtz Machine
model = HelmholtzMachine(num_visible=n_visible, num_hidden=n_hidden)
model.init_state()

trainer = WakeSleepTrainer(model, learning_rate=0.1, num_sleep_samples=20)

# Train with wake-sleep algorithm
n_epochs = 50
wake_errors_history = []
sleep_errors_history = []

print("\nTraining Helmholtz Machine...")
for epoch in range(n_epochs):
    # Shuffle data
    shuffled = [train_data[i] for i in np.random.permutation(len(train_data))]

    # Train (performs both wake and sleep phases)
    trainer.train(shuffled)

    # Get statistics
    stats = trainer.get_training_statistics()
    wake_errors_history.append(stats.get("mean_wake_error", 0))
    sleep_errors_history.append(stats.get("mean_sleep_error", 0))

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}/{n_epochs}: "
            f"Wake error = {stats.get('mean_wake_error', 0):.4f}, "
            f"Sleep error = {stats.get('mean_sleep_error', 0):.4f}"
        )

print("\nTraining complete!")

# Visualize training progress
fig = plt.figure(figsize=(15, 10))

# Plot 1: Wake and Sleep errors
ax1 = plt.subplot(3, 3, 1)
ax1.plot(wake_errors_history, label="Wake Phase Error", linewidth=2)
ax1.plot(sleep_errors_history, label="Sleep Phase Error", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Error")
ax1.set_title("Training Progress")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Recognition weights
ax2 = plt.subplot(3, 3, 2)
W_rec = model.W_rec.value
im = ax2.imshow(W_rec, cmap="RdBu_r", aspect="auto", interpolation="nearest")
ax2.set_xlabel("Visible Unit")
ax2.set_ylabel("Hidden Unit")
ax2.set_title("Recognition Weights (Bottom-Up)")
plt.colorbar(im, ax=ax2)

# Plot 3: Generative weights
ax3 = plt.subplot(3, 3, 3)
W_gen = model.W_gen.value
im = ax3.imshow(W_gen, cmap="RdBu_r", aspect="auto", interpolation="nearest")
ax3.set_xlabel("Hidden Unit")
ax3.set_ylabel("Visible Unit")
ax3.set_title("Generative Weights (Top-Down)")
plt.colorbar(im, ax=ax3)

# Plot 4-9: Test reconstructions and generations
size = int(np.sqrt(n_visible))
is_square = size * size == n_visible

# Test reconstructions
test_samples = train_data[:3]
for i, sample in enumerate(test_samples):
    ax = plt.subplot(3, 3, 4 + i)

    # Reconstruct
    reconstructed = trainer.predict(sample)

    if is_square:
        sample_2d = sample.reshape(size, size)
        recon_2d = reconstructed.reshape(size, size)
        composite = np.hstack([sample_2d, recon_2d])
        ax.imshow(composite, cmap="gray", interpolation="nearest")
        ax.set_title(f"Test {i+1}: Original | Reconstructed")
    else:
        x = np.arange(n_visible)
        ax.bar(x - 0.2, sample, width=0.4, alpha=0.7, label="Original")
        ax.bar(x + 0.2, reconstructed, width=0.4, alpha=0.7, label="Reconstructed")
        ax.set_title(f"Test Sample {i+1}")
        ax.set_ylim([0, 1.2])
        ax.legend(fontsize=8)

    ax.axis("off")

# Generate new samples from the model
print("\n" + "=" * 60)
print("Generating New Samples")
print("=" * 60)

for i in range(3):
    ax = plt.subplot(3, 3, 7 + i)

    # Sample from prior
    hidden_sample = model.sample_from_prior()

    # Generate visible from hidden
    generated = model.generate(hidden_sample, stochastic=True)

    if is_square:
        gen_2d = generated.reshape(size, size)
        ax.imshow(gen_2d, cmap="gray", interpolation="nearest")
        ax.set_title(f"Generated Sample {i+1}")
    else:
        ax.bar(np.arange(n_visible), generated, alpha=0.7)
        ax.set_title(f"Generated Sample {i+1}")
        ax.set_ylim([0, 1.2])

    ax.axis("off")

plt.tight_layout()
plt.savefig("wake_sleep_generative.png")
print("\nPlot saved as 'wake_sleep_generative.png'")
plt.show()

# Analyze learned representations
print("\n" + "=" * 60)
print("Representation Analysis")
print("=" * 60)

# Encode training data
encoded_representations = []
for sample in train_data:
    hidden = model.recognize(sample, stochastic=False)
    encoded_representations.append(hidden)

encoded_array = np.array(encoded_representations)

print(f"Encoded representation shape: {encoded_array.shape}")
print(f"Mean hidden activation: {encoded_array.mean():.3f}")
print(f"Hidden sparsity: {(encoded_array < 0.1).sum() / encoded_array.size:.1%}")

# Plot hidden activation patterns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of hidden activations
ax = axes[0]
ax.hist(encoded_array.flatten(), bins=30, alpha=0.7, edgecolor="black")
ax.set_xlabel("Hidden Unit Activation")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Hidden Activations")
ax.grid(True, alpha=0.3, axis="y")

# Hidden activation heatmap
ax = axes[1]
im = ax.imshow(encoded_array[:50].T, aspect="auto", cmap="hot", interpolation="nearest")
ax.set_xlabel("Training Sample")
ax.set_ylabel("Hidden Unit")
ax.set_title("Hidden Representations")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("wake_sleep_representations.png")
print("\nRepresentation plot saved as 'wake_sleep_representations.png'")
plt.show()

# Test generation quality
print("\n" + "=" * 60)
print("Generation Quality Test")
print("=" * 60)

n_generated = 100
generated_samples = []

for _ in range(n_generated):
    hidden = model.sample_from_prior()
    generated = model.generate(hidden, stochastic=True)
    generated_samples.append(generated)

generated_array = np.array(generated_samples)

print(f"Generated {len(generated_samples)} new samples")
print(f"Mean visible activation (generated): {generated_array.mean():.3f}")
print(f"Mean visible activation (training): {np.mean([s.mean() for s in train_data]):.3f}")

# Compare statistics
train_array = np.array(train_data)
print(f"\nTraining data std: {train_array.std():.3f}")
print(f"Generated data std: {generated_array.std():.3f}")
