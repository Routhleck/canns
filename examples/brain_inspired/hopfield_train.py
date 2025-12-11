"""
Discrete Hopfield training on real images from scikit-image.

This example demonstrates:
- Generic Hebbian training (trainer.train)
- Batch prediction (trainer.predict_batch)
- Simple progress reporting
"""

import numpy as np
import skimage.data
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize

from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer

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

# Create model and unified trainer (discrete mode by default)
model = AmariHopfieldNetwork(num_neurons=data_list[0].shape[0], asyn=False, activation="sign")
trainer = HebbianTrainer(model)
trainer.train(data_list)


# Generate testset
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


tests = [get_corrupted_input(d, 0.3) for d in data_list]

# Predict corrupted patterns (compiled for speed, show sample-level progress)
predicted = trainer.predict_batch(tests, show_sample_progress=True)


# display predict results
def plot(data, test, predicted, figsize=(5, 6)):
    def reshape(data):
        dim = int(np.sqrt(len(data)))
        data = np.reshape(data, (dim, dim))
        return data

    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i == 0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i], cmap='gray')
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i], cmap='gray')
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i], cmap='gray')
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("discrete_hopfield_train.png")
    plt.show()


plot(data_list, tests, predicted, figsize=(5, 6))
