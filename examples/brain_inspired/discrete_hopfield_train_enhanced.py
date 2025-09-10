"""
Enhanced Discrete Hopfield Network Training Example

This example demonstrates the new progress reporting architecture with:
- Batch prediction with nested progress bars  
- Different progress reporting modes
- Compiled vs uncompiled prediction modes
- Real-time convergence monitoring
"""

import numpy as np
import skimage.data
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize

from canns.models import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer

np.random.seed(42)

# Load and preprocess data (same as original example)
def preprocess_image(img, w=128, h=128):
    img = resize(img, (w, h), mode='reflect')
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2 * (binary * 1) - 1
    return np.reshape(shift, w*h)

# Load sample images
camera = skimage.data.camera()
astronaut = rgb2gray(skimage.data.astronaut())
horse = skimage.data.horse()
coffee = rgb2gray(skimage.data.coffee())

data_list = [preprocess_image(d) for d in [camera, astronaut, horse, coffee]]

# Create corrupted test data
def get_corrupted_input(input_data, corruption_level):
    corrupted = np.copy(input_data)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input_data))
    for i, v in enumerate(input_data):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

tests = [get_corrupted_input(d, 0.3) for d in data_list]

print("=== Enhanced Discrete Hopfield Network Training ===\n")

# Create model and trainer with different configurations
model = AmariHopfieldNetwork(num_neurons=data_list[0].shape[0], asyn=False, activation="sign")
model.init_state()

print("1. Training with Hebbian learning...")
trainer = HebbianTrainer(
    model, 
    progress_mode="auto",
    show_iteration_progress=False,  # Clean display by default
    compiled_prediction=True
)
trainer.train(data_list)

print("\n2. Clean prediction with sample progress only:")
predicted_clean = trainer.predict_batch(
    tests,
    show_sample_progress=True,
    show_iteration_progress=False  # Clean display
)

print("\n3. Detailed prediction with convergence monitoring (for debugging):")
print("   Note: This shows detailed convergence info but may look cluttered")

# For users who want to see convergence details
trainer_detailed = HebbianTrainer(
    model,
    progress_mode="auto", 
    show_iteration_progress=True,
    compiled_prediction=False
)

predicted_detailed = trainer_detailed.predict_batch(
    tests[:2],  # Just first 2 samples to avoid too much output
    num_iter=20,  # Reduced iterations
    show_sample_progress=True,
    show_iteration_progress=True,
    compiled=False
)

print("\n4. Fast prediction with compiled mode:")
predicted_fast = trainer.predict_batch(
    tests,
    compiled=True,  # Use compiled mode for speed
    show_sample_progress=True
)

print("\n5. Silent mode (no progress bars):")
trainer_silent = HebbianTrainer(model, progress_mode="silent")
predicted_silent = trainer_silent.predict_batch(tests)

print("\n6. Individual pattern prediction with detailed monitoring:")
test_pattern = tests[0]
print(f"   Predicting single pattern (shape: {test_pattern.shape})...")

result = trainer_detailed.predict(
    test_pattern,
    num_iter=15,  # Reduced for cleaner output
    compiled=False,
    show_progress=True,
    convergence_threshold=1e-8
)

print(f"   Prediction completed. Final pattern energy: {model.energy:.0f}")

# Display results function (same as original)
def plot_results(data, test, predicted, title="Results", figsize=(5, 6)):
    def reshape(data):
        dim = int(np.sqrt(len(data)))
        return np.reshape(data, (dim, dim))

    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    
    for i in range(len(data)):
        if i == 0:
            axarr[i, 0].set_title('Original')
            axarr[i, 1].set_title("Corrupted (30%)")
            axarr[i, 2].set_title('Recovered')

        axarr[i, 0].imshow(data[i], cmap='gray')
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i], cmap='gray')
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i], cmap='gray')
        axarr[i, 2].axis('off')

    plt.tight_layout()
    return fig

# Plot results
fig1 = plot_results(data_list, tests, predicted_clean, 
                   "Clean Mode (Sample Progress Only)")
fig1.savefig("discrete_hopfield_clean.png", dpi=150, bbox_inches='tight')

fig2 = plot_results(data_list, tests, predicted_fast,
                   "Fast Mode (Compiled)")  
fig2.savefig("discrete_hopfield_fast.png", dpi=150, bbox_inches='tight')

plt.show()

print(f"\nResults saved:")
print(f"- discrete_hopfield_clean.png")
print(f"- discrete_hopfield_fast.png")

# Demonstrate trainer reconfiguration
print("\n7. Dynamic trainer reconfiguration:")
trainer.configure_progress(
    progress_mode="silent",
    compiled_prediction=True
)
print("   Trainer reconfigured to silent + compiled mode")

# Test backward compatibility
print("\n8. Backward compatibility test:")
print("   Using original trainer interface...")
old_style_trainer = HebbianTrainer(model)
old_style_result = old_style_trainer.predict(tests[0])
print(f"   Old-style prediction completed successfully")

print("\n=== Demo completed successfully! ===")