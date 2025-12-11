"""
Oja's rule for principal component analysis.

This example demonstrates:
- OjaTrainer for normalized Hebbian learning with JIT compilation
- PCA extraction from high-dimensional data
- Weight normalization and convergence visualization
- Comparison with sklearn PCA
"""

import brainpy.math as bm
import numpy as np
from matplotlib import pyplot as plt

from canns.models.brain_inspired import LinearLayer
from canns.trainer import OjaTrainer

np.random.seed(42)
bm.random.seed(42)

# Generate synthetic data with clear principal components
n_samples = 500
n_features = 50
n_components = 3

# Create data with known structure: 3 principal components
# Component 1: strong variance along first 10 dimensions
component1 = np.random.randn(n_samples, 10) * 3.0
# Component 2: moderate variance along next 10 dimensions
component2 = np.random.randn(n_samples, 10) * 1.5
# Component 3: weak variance along next 10 dimensions
component3 = np.random.randn(n_samples, 10) * 0.8
# Noise in remaining dimensions
noise = np.random.randn(n_samples, 20) * 0.3

data = np.concatenate([component1, component2, component3, noise], axis=1)
print(f"Data shape: {data.shape}")

# Compute true PCA for comparison
from sklearn.decomposition import PCA

true_pca = PCA(n_components=n_components)
true_pca.fit(data)
print(f"\nTrue PCA explained variance ratio: {true_pca.explained_variance_ratio_}")

# Train Oja's rule to extract principal components
model = LinearLayer(input_size=n_features, output_size=n_components)

# Create OjaTrainer with JIT compilation enabled (default: compiled=True)
# JIT compilation provides significant speedup for training loops
trainer = OjaTrainer(
    model, learning_rate=0.001, normalize_weights=True, compiled=True
)

# Train over multiple epochs to track convergence
n_epochs = 20  # Fewer epochs needed with full batch training
checkpoint_interval = 2  # Track every N epochs
weight_norms_history = []
variance_explained = []

print(f"\nTraining for {n_epochs} epochs with JIT compilation...")
print(f"Checkpoint interval: {checkpoint_interval} epochs")

for epoch in range(n_epochs):
    # Train on all samples (full batch)
    trainer.train(data)

    # Track metrics at checkpoints
    if (epoch + 1) % checkpoint_interval == 0:
        # Weight norms (should stay at 1.0 with normalization)
        norms = np.linalg.norm(model.W.value, axis=1)
        weight_norms_history.append(norms.copy())

        # Compute variance explained by learned components
        outputs = np.array([trainer.predict(x) for x in data])
        var_explained = np.var(outputs, axis=0) / np.var(data)
        variance_explained.append(var_explained)

        print(
            f"Epoch {epoch + 1}/{n_epochs}: "
            f"Weight norms: [{norms.min():.4f}, {norms.max():.4f}], "
            f"Variance: [{var_explained.min():.2f}, {var_explained.max():.2f}]"
        )

print("\nTraining complete!")

# Final weights
print(f"\nFinal learned weights shape: {model.W.value.shape}")
print(f"Final weight norms: {np.linalg.norm(model.W.value, axis=1)}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Weight norm convergence
ax = axes[0, 0]
epochs_checkpointed = np.arange(checkpoint_interval, n_epochs + 1, checkpoint_interval)
for i in range(n_components):
    norms = [h[i] for h in weight_norms_history]
    ax.plot(epochs_checkpointed, norms, label=f"Component {i + 1}", marker='o', markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Weight Norm")
ax.set_title("Weight Norm Convergence (should be ~1.0)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Target")

# Plot 2: Variance explained over epochs
ax = axes[0, 1]
variance_explained_arr = np.array(variance_explained)
for i in range(n_components):
    ax.plot(epochs_checkpointed, variance_explained_arr[:, i], label=f"Component {i + 1}", marker='o', markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Variance Explained")
ax.set_title("Variance Explained by Each Component")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Learned weight patterns (heatmap)
ax = axes[1, 0]
im = ax.imshow(model.W.value, aspect="auto", cmap="RdBu_r")
ax.set_xlabel("Input Dimension")
ax.set_ylabel("Output Component")
ax.set_title("Learned Weight Matrix")
plt.colorbar(im, ax=ax)

# Plot 4: Comparison with true PCA (angle between weight vectors)
ax = axes[1, 1]
oja_weights = model.W.value
pca_components = true_pca.components_

# Compute cosine similarity (absolute value, as sign is arbitrary)
similarities = []
for i in range(n_components):
    oja_vec = oja_weights[i]
    pca_vec = pca_components[i]
    similarity = abs(np.dot(oja_vec, pca_vec) / (np.linalg.norm(oja_vec) * np.linalg.norm(pca_vec)))
    similarities.append(similarity)

ax.bar(range(n_components), similarities)
ax.set_xlabel("Component")
ax.set_ylabel("Cosine Similarity with PCA")
ax.set_title("Oja vs PCA Component Alignment")
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(similarities):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

plt.tight_layout()
plt.savefig("oja_pca_extraction.png")
print("\nPlot saved as 'oja_pca_extraction.png'")
plt.show()
