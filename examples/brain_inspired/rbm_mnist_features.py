"""
Restricted Boltzmann Machine (RBM) for MNIST feature learning.

This example demonstrates:
- ContrastiveDivergenceTrainer for RBM training
- Feature extraction from MNIST digits
- Reconstruction visualization
"""

import numpy as np
from matplotlib import pyplot as plt

from canns.models.brain_inspired import RestrictedBoltzmannModel
from canns.trainer import ContrastiveDivergenceTrainer

np.random.seed(42)

# Load MNIST data (simplified - just a few digits)
try:
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data
    y = digits.target

    # Normalize to [0, 1]
    X = X / 16.0

    # Binarize for RBM
    X = (X > 0.5).astype(np.float32)

    print(f"Loaded {len(X)} digit images")
    print(f"Image size: {X.shape[1]} pixels")

except ImportError:
    print("Warning: scikit-learn not available, generating synthetic data")
    # Generate synthetic binary patterns
    n_samples = 100
    n_pixels = 64
    X = np.random.binomial(1, 0.3, size=(n_samples, n_pixels)).astype(np.float32)
    y = np.random.randint(0, 10, size=n_samples)

# Split into train and test
n_train = int(0.8 * len(X))
X_train = X[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]

# Create RBM
n_visible = X.shape[1]
n_hidden = 64  # Number of features to learn

print(f"\nCreating RBM: {n_visible} visible, {n_hidden} hidden units")

model = RestrictedBoltzmannModel(num_visible=n_visible, num_hidden=n_hidden)
model.init_state()

trainer = ContrastiveDivergenceTrainer(model, learning_rate=0.1, k=1)

# Train RBM
n_epochs = 20  # Reduced for faster execution
reconstruction_errors = []

print("\nTraining RBM...")
for epoch in range(n_epochs):
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    shuffled_data = X_train[indices]

    # Train
    trainer.train(shuffled_data)

    # Track reconstruction error
    mean_error = trainer.get_reconstruction_error()
    reconstruction_errors.append(mean_error)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}: Reconstruction error = {mean_error:.4f}")

print("\nTraining complete!")

# Visualize results
fig = plt.figure(figsize=(15, 10))

# Plot 1: Training curve
ax1 = plt.subplot(3, 3, 1)
ax1.plot(reconstruction_errors, linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Reconstruction Error")
ax1.set_title("Training Progress")
ax1.grid(True, alpha=0.3)

# Plot 2-3: Learned features (hidden unit weights)
for plot_idx in range(2):
    ax = plt.subplot(3, 3, 2 + plot_idx)

    # Select a subset of hidden units to visualize
    start_idx = plot_idx * 8
    end_idx = start_idx + 8

    features = model.W.value[start_idx:end_idx, :]

    # Reshape if data is square
    size = int(np.sqrt(n_visible))
    if size * size == n_visible:
        # Create grid of features (2x4 grid for 8 features)
        n_rows = 2
        n_cols = 4
        feature_grid = np.zeros((size * n_rows, size * n_cols))

        for i in range(min(8, end_idx - start_idx)):
            if start_idx + i >= n_hidden:
                break
            row = i // n_cols
            col = i % n_cols
            feature = features[i].reshape(size, size)
            feature_grid[row * size : (row + 1) * size, col * size : (col + 1) * size] = feature

        im = ax.imshow(feature_grid, cmap="RdBu_r", interpolation="nearest")
        ax.set_title(f"Learned Features {start_idx+1}-{min(start_idx+8, n_hidden)}")
    else:
        im = ax.imshow(features, aspect="auto", cmap="RdBu_r")
        ax.set_xlabel("Visible Unit")
        ax.set_ylabel("Hidden Unit")
        ax.set_title(f"Features {start_idx+1}-{min(end_idx, n_hidden)}")

    ax.axis("off")

# Plot 4-9: Test reconstructions
size = int(np.sqrt(n_visible))
is_square = size * size == n_visible

for i in range(6):
    ax = plt.subplot(3, 3, 4 + i)

    # Select test sample
    test_idx = i * (len(X_test) // 6)
    if test_idx >= len(X_test):
        break

    original = X_test[test_idx]
    label = y_test[test_idx]

    # Reconstruct
    reconstructed = trainer.predict(original, num_gibbs_steps=100)

    if is_square:
        # Show original and reconstructed side by side
        original_2d = original.reshape(size, size)
        reconstructed_2d = reconstructed.reshape(size, size)
        composite = np.hstack([original_2d, reconstructed_2d])

        ax.imshow(composite, cmap="gray", interpolation="nearest")
        ax.set_title(f"Digit {label}: Original | Reconstructed")
    else:
        # Plot as lines
        ax.plot(original, "g-", alpha=0.7, label="Original")
        ax.plot(reconstructed, "b-", alpha=0.7, label="Reconstructed")
        ax.set_title(f"Sample {i} (Label: {label})")
        ax.legend(fontsize=8)

    ax.axis("off")

plt.tight_layout()
plt.savefig("rbm_mnist_features.png")
print("\nPlot saved as 'rbm_mnist_features.png'")
plt.show()

# Analyze learned features
print("\n" + "=" * 60)
print("Feature Analysis")
print("=" * 60)

# Compute feature activation statistics
feature_activations = []
for sample in X_test:
    hidden_prob = model.hidden_prob(sample)
    feature_activations.append(hidden_prob)

feature_activations = np.array(feature_activations)
mean_activations = np.mean(feature_activations, axis=0)
std_activations = np.std(feature_activations, axis=0)

print(f"Mean feature activation: {mean_activations.mean():.3f} ± {mean_activations.std():.3f}")
print(f"Feature sparsity: {(mean_activations < 0.1).sum() / n_hidden:.1%} inactive")

# Find most and least active features
most_active = np.argsort(mean_activations)[-5:][::-1]
least_active = np.argsort(mean_activations)[:5]

print(f"\nMost active features: {most_active}")
print(f"Their mean activations: {mean_activations[most_active]}")

print(f"\nLeast active features: {least_active}")
print(f"Their mean activations: {mean_activations[least_active]}")

# Plot feature activation distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of mean activations
ax = axes[0]
ax.hist(mean_activations, bins=30, alpha=0.7, edgecolor="black")
ax.set_xlabel("Mean Activation")
ax.set_ylabel("Number of Features")
ax.set_title("Distribution of Feature Activations")
ax.grid(True, alpha=0.3, axis="y")

# Feature activation heatmap for test samples
ax = axes[1]
# Show subset of samples and features
n_show_samples = min(50, len(X_test))
n_show_features = min(32, n_hidden)
im = ax.imshow(
    feature_activations[:n_show_samples, :n_show_features].T,
    aspect="auto",
    cmap="hot",
    interpolation="nearest",
)
ax.set_xlabel("Test Sample")
ax.set_ylabel("Feature Index")
ax.set_title("Feature Activation Patterns")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("rbm_feature_analysis.png")
print("\nFeature analysis plot saved as 'rbm_feature_analysis.png'")
plt.show()
