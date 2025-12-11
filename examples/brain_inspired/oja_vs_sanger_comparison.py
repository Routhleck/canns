"""
Comparison: Oja's Rule vs Sanger's Rule for PCA extraction.

This example demonstrates the fundamental difference between Oja and Sanger:
- Oja's Rule: Extracts only the first principal component correctly.
  Multiple neurons all converge to PC1 (no orthogonality enforcement).
- Sanger's Rule: Extracts multiple orthogonal principal components sequentially.
  Uses Gram-Schmidt orthogonalization to ensure different PCs.

This directly answers why in the Oja example, only component 1 aligns well
with PCA (>90%), while components 2-3 don't (~45%, ~30%).
"""

import brainpy.math as bm
import numpy as np
from matplotlib import pyplot as plt

from canns.models.brain_inspired import LinearLayer
from canns.trainer import OjaTrainer, SangerTrainer

np.random.seed(42)
bm.random.seed(42)

# Generate synthetic data with 3 clear principal components
n_samples = 500
n_features = 50
n_components = 3

# Component 1: strong variance (first 10 dims)
component1 = np.random.randn(n_samples, 10) * 3.0
# Component 2: moderate variance (next 10 dims)
component2 = np.random.randn(n_samples, 10) * 1.5
# Component 3: weak variance (next 10 dims)
component3 = np.random.randn(n_samples, 10) * 0.8
# Noise (remaining dims)
noise = np.random.randn(n_samples, 20) * 0.3

data = np.concatenate([component1, component2, component3, noise], axis=1)
print(f"Data shape: {data.shape}")

# Compute true PCA for comparison
from sklearn.decomposition import PCA

true_pca = PCA(n_components=n_components)
true_pca.fit(data)
print(f"True PCA explained variance ratio: {true_pca.explained_variance_ratio_}")

# ========================================================================
# Train Oja's Rule (will only extract PC1 correctly)
# ========================================================================
print("\n" + "=" * 70)
print("Training Oja's Rule")
print("=" * 70)

model_oja = LinearLayer(input_size=n_features, output_size=n_components)

trainer_oja = OjaTrainer(model_oja, learning_rate=0.001, normalize_weights=True, compiled=True)

n_epochs = 50
print(f"Training for {n_epochs} epochs with JIT compilation...")

for epoch in range(n_epochs):
    trainer_oja.train(data)
    if (epoch + 1) % 10 == 0:
        # Compute variance explained
        outputs = np.array([trainer_oja.predict(x) for x in data])
        var_explained = np.var(outputs, axis=0)
        print(f"Epoch {epoch + 1}: Variance explained by each neuron: {var_explained}")

print("\nOja training complete!")

# ========================================================================
# Train Sanger's Rule (will extract all 3 PCs correctly)
# ========================================================================
print("\n" + "=" * 70)
print("Training Sanger's Rule")
print("=" * 70)

model_sanger = LinearLayer(input_size=n_features, output_size=n_components)

trainer_sanger = SangerTrainer(
    model_sanger, learning_rate=0.001, normalize_weights=True, compiled=True
)

print(f"Training for {n_epochs} epochs with JIT compilation...")

for epoch in range(n_epochs):
    trainer_sanger.train(data)
    if (epoch + 1) % 10 == 0:
        # Compute variance explained
        outputs = np.array([trainer_sanger.predict(x) for x in data])
        var_explained = np.var(outputs, axis=0)
        print(f"Epoch {epoch + 1}: Variance explained by each neuron: {var_explained}")

print("\nSanger training complete!")

# ========================================================================
# Compare Results
# ========================================================================
print("\n" + "=" * 70)
print("Comparison: Oja vs Sanger vs PCA")
print("=" * 70)

# Compute alignment with true PCA components (cosine similarity)
oja_weights = model_oja.W.value
sanger_weights = model_sanger.W.value
pca_components = true_pca.components_


def compute_alignment(weights, pca_components):
    """Compute absolute cosine similarity between learned and PCA components."""
    similarities = []
    for i in range(len(weights)):
        vec1 = weights[i]
        vec2 = pca_components[i]
        similarity = abs(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        similarities.append(similarity)
    return similarities


oja_alignment = compute_alignment(oja_weights, pca_components)
sanger_alignment = compute_alignment(sanger_weights, pca_components)

print("\nAlignment with PCA (cosine similarity):")
print(f"  Oja:    {[f'{s:.3f}' for s in oja_alignment]}")
print(f"  Sanger: {[f'{s:.3f}' for s in sanger_alignment]}")


# Compute orthogonality between learned components


def compute_orthogonality(weights):
    """Compute dot products between all pairs of weight vectors."""
    n = len(weights)
    dots = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = abs(np.dot(weights[i], weights[j]))
            dots.append(dot)
    return dots


oja_ortho = compute_orthogonality(oja_weights)
sanger_ortho = compute_orthogonality(sanger_weights)

print("\nOrthogonality (|dot product| between components, should be ~0):")
print(f"  Oja:    {[f'{d:.3f}' for d in oja_ortho]}")
print(f"  Sanger: {[f'{d:.3f}' for d in sanger_ortho]}")

# ========================================================================
# Visualize Results
# ========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Oja's Rule
# Plot 1: Learned weight patterns (heatmap)
ax = axes[0, 0]
im = ax.imshow(oja_weights, aspect="auto", cmap="RdBu_r")
ax.set_xlabel("Input Dimension")
ax.set_ylabel("Component")
ax.set_title("Oja: Learned Weights")
plt.colorbar(im, ax=ax)

# Plot 2: Alignment with PCA
ax = axes[0, 1]
bars = ax.bar(range(n_components), oja_alignment, color=["#2ecc71", "#e74c3c", "#e74c3c"])
ax.set_xlabel("Component")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Oja: Alignment with PCA")
ax.set_ylim([0, 1.1])
ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="Good (>0.9)")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(oja_alignment):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
ax.legend()

# Plot 3: Orthogonality
ax = axes[0, 2]
labels = ["0-1", "0-2", "1-2"]
bars = ax.bar(range(len(oja_ortho)), oja_ortho, color="#e74c3c")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_xlabel("Component Pair")
ax.set_ylabel("|Dot Product|")
ax.set_title("Oja: Orthogonality (should be ~0)")
ax.set_ylim([0, 1.0])
ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Good (<0.1)")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(oja_ortho):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
ax.legend()

# Row 2: Sanger's Rule
# Plot 4: Learned weight patterns (heatmap)
ax = axes[1, 0]
im = ax.imshow(sanger_weights, aspect="auto", cmap="RdBu_r")
ax.set_xlabel("Input Dimension")
ax.set_ylabel("Component")
ax.set_title("Sanger: Learned Weights")
plt.colorbar(im, ax=ax)

# Plot 5: Alignment with PCA
ax = axes[1, 1]
bars = ax.bar(
    range(n_components), sanger_alignment, color=["#2ecc71", "#2ecc71", "#2ecc71"]
)
ax.set_xlabel("Component")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Sanger: Alignment with PCA")
ax.set_ylim([0, 1.1])
ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="Good (>0.9)")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(sanger_alignment):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
ax.legend()

# Plot 6: Orthogonality
ax = axes[1, 2]
bars = ax.bar(range(len(sanger_ortho)), sanger_ortho, color="#2ecc71")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_xlabel("Component Pair")
ax.set_ylabel("|Dot Product|")
ax.set_title("Sanger: Orthogonality (should be ~0)")
ax.set_ylim([0, 1.0])
ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Good (<0.1)")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(sanger_ortho):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
ax.legend()

plt.tight_layout()
plt.savefig("oja_vs_sanger_comparison.png")
print("\nPlot saved as 'oja_vs_sanger_comparison.png'")
plt.show()

# ========================================================================
# Summary
# ========================================================================
print("\n" + "=" * 70)
print("SUMMARY: Why Oja fails for multiple PCs")
print("=" * 70)
print(
    """
Oja's Rule limitation:
- Single Oja neuron → converges to PC1 (largest eigenvalue direction)
- Multiple Oja neurons → all independently converge to PC1
- No orthogonality constraint → neurons compete for the same direction
- Result: Only component 1 aligns well with PCA (>90%)
           Components 2-3 poorly aligned (~45%, ~30%)

Sanger's Rule solution:
- Adds Gram-Schmidt orthogonalization term
- ΔW_i = η * (y_i * x - y_i * Σ_{j≤i} y_j * W_j)
- Forces neuron i to be orthogonal to all previous neurons (j < i)
- Result: All components align well with PCA (>90% for all)

Key insight: For multiple PC extraction, use Sanger, not Oja!
"""
)
