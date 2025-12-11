"""
Hebbian learning followed by Anti-Hebbian unlearning on images.

This example demonstrates:
- Train a Hopfield network with Hebbian learning on real images
- Test pattern recovery performance
- Apply Anti-Hebbian learning to "forget" one specific image
- Compare recovery performance before and after unlearning
"""

import numpy as np
import skimage.data
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize

from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import AntiHebbianTrainer, HebbianTrainer

np.random.seed(42)


def preprocess_image(img, w=128, h=128) -> np.ndarray:
    """Resize, grayscale (if needed), threshold to binary, then map to {-1,+1}."""
    if img.ndim == 3:
        img = rgb2gray(img)
    img = resize(img, (w, h), anti_aliasing=True)
    img = img.astype(np.float32, copy=False)
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = np.where(binary, 1.0, -1.0).astype(np.float32)
    return shift.reshape(w * h)


# Training data from skimage
camera = preprocess_image(skimage.data.camera())
astronaut = preprocess_image(skimage.data.astronaut())
horse = preprocess_image(skimage.data.horse().astype(np.float32))
coffee = preprocess_image(skimage.data.coffee())

data_list = [camera, astronaut, horse, coffee]
image_names = ["Camera", "Astronaut", "Horse", "Coffee"]

print(f"Network size: {data_list[0].shape[0]} neurons")
print(f"Number of images: {len(data_list)}\n")

# Create model with sign activation for binary patterns
model = AmariHopfieldNetwork(
    num_neurons=data_list[0].shape[0],
    asyn=False,
    activation="sign"
)


# Add binary noise by flipping pixels
def get_corrupted_input(pattern, corruption_level):
    """Flip random pixels in the binary pattern."""
    corrupted = np.copy(pattern)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(pattern))
    corrupted[inv == 1] *= -1
    return corrupted


tests = [get_corrupted_input(d, 0.4) for d in data_list]

# Step 1: Hebbian learning on all images
print("=" * 60)
print("Step 1: Hebbian Learning on all 4 images")
print("=" * 60)
trainer_hebb = HebbianTrainer(model, subtract_mean=False, normalize_by_patterns=False)
trainer_hebb.train(data_list)

# Test recovery after Hebbian learning
predicted_after_hebb = trainer_hebb.predict_batch(tests, show_sample_progress=True)

print("\nRecovery after Hebbian learning:")
for i in range(len(data_list)):
    original = data_list[i]
    recovered = predicted_after_hebb[i]
    recovery_mse = np.mean((original - recovered) ** 2)
    correlation = np.corrcoef(original, recovered)[0, 1]
    print(f"  {image_names[i]}: MSE={recovery_mse:.4f}, Correlation={correlation:.4f}")

# Step 2: Demonstrate selective forgetting using Anti-Hebbian learning
print("\n" + "=" * 60)
print("Step 2: Anti-Hebbian Unlearning of Astronaut")
print("=" * 60)

# Apply anti-Hebbian learning to the astronaut pattern
# This demonstrates selective pattern unlearning
# Note: Use subtract_mean=False to match the Hebbian learning configuration
trainer_anti = AntiHebbianTrainer(
    model,
    subtract_mean=False,
    normalize_by_patterns=False
)
trainer_anti.train([data_list[1]])  # Apply anti-Hebbian to astronaut pattern

# Test recovery after Anti-Hebbian learning
predicted_after_anti = trainer_anti.predict_batch(tests, show_sample_progress=True)

print("\nRecovery after Anti-Hebbian unlearning of Astronaut:")
for i in range(len(data_list)):
    original = data_list[i]
    recovered = predicted_after_anti[i]
    recovery_mse = np.mean((original - recovered) ** 2)
    correlation = np.corrcoef(original, recovered)[0, 1]
    marker = " ← UNLEARNED" if i == 1 else ""
    print(f"  {image_names[i]}: MSE={recovery_mse:.4f}, Correlation={correlation:.4f}{marker}")


# Visualization
def plot_comparison(data, test, pred_hebb, pred_anti, names, figsize=(12, 10)):
    """Plot comparison of image recovery before and after anti-Hebbian unlearning."""

    def reshape(pattern):
        """Reshape 1D pattern back to 2D image."""
        dim = int(np.sqrt(len(pattern)))
        return np.reshape(pattern, (dim, dim))

    fig, axes = plt.subplots(len(data), 3, figsize=figsize)

    for i in range(len(data)):
        # Column 1: Original image
        axes[i, 0].imshow(reshape(data[i]), cmap='gray', vmin=-1, vmax=1)
        axes[i, 0].axis('off')
        axes[i, 0].set_ylabel(names[i], fontsize=11, fontweight='bold', rotation=0,
                              labelpad=40, ha='right', va='center')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=12, fontweight='bold')

        # Column 2: After Hebbian learning
        axes[i, 1].imshow(reshape(pred_hebb[i]), cmap='gray', vmin=-1, vmax=1)
        axes[i, 1].axis('off')

        mse_hebb = np.mean((data[i] - pred_hebb[i]) ** 2)
        corr_hebb = np.corrcoef(data[i], pred_hebb[i])[0, 1]

        if i == 0:
            axes[i, 1].set_title('After Hebbian Learning', fontsize=12, fontweight='bold')

        # Add text annotation
        axes[i, 1].text(0.5, -0.15, f'Corr: {corr_hebb:.3f}',
                        transform=axes[i, 1].transAxes, fontsize=9,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Column 3: After Anti-Hebbian on Astronaut
        axes[i, 2].imshow(reshape(pred_anti[i]), cmap='gray', vmin=-1, vmax=1)
        axes[i, 2].axis('off')

        mse_anti = np.mean((data[i] - pred_anti[i]) ** 2)
        corr_anti = np.corrcoef(data[i], pred_anti[i])[0, 1]

        if i == 0:
            axes[i, 2].set_title('After Unlearning Astronaut', fontsize=12, fontweight='bold')

        # Highlight the unlearned image
        facecolor = 'salmon' if i == 1 else 'lightblue'
        axes[i, 2].text(0.5, -0.15, f'Corr: {corr_anti:.3f}',
                        transform=axes[i, 2].transAxes, fontsize=9,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.7))

    plt.tight_layout()
    plt.savefig("hopfield_hebbian_vs_antihebbian.png", dpi=150, bbox_inches='tight')
    print("\nFigure saved to hopfield_hebbian_vs_antihebbian.png")
    plt.show()


plot_comparison(data_list, tests, predicted_after_hebb, predicted_after_anti, image_names)
