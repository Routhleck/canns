"""
Hopfield training on MNIST digits using the unified HebbianTrainer.

This example mirrors the reference in .ref/Hopfield-Network-master/train_mnist.py
but uses our API:
- Model: canns.models.brain_inspired.AmariHopfieldNetwork
- Trainer: canns.trainer.HebbianTrainer (generic Hebbian learning and prediction)

What it does
- Loads a few MNIST digit exemplars (one per chosen class) as training patterns
- Applies Hebbian learning to store these patterns
- Uses held-out clean samples for retrieval (no added noise)
- Saves a figure showing Train/Input/Output for each class

Data sources (no TensorFlow required)
- Prefer: Hugging Face `datasets` (mnist) if installed.
- Next: `torchvision.datasets.MNIST` (28x28), optional PyTorch dependency.
- Fallback: `keras` or `tensorflow` packaged keras.
- Last resort: scikit-learn `load_digits` (8x8 digits), which requires no network.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

try:
    # Optional: for thresholding convenience; otherwise fall back to mean threshold
    from skimage.filters import threshold_mean as skimage_threshold_mean  # type: ignore
except Exception:  # pragma: no cover - best effort optional dep
    skimage_threshold_mean = None  # type: ignore

from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer


# -------------------------
# Data loading and helpers
# -------------------------

def _load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST-like arrays (train_images, train_labels, test_images, test_labels).

    Preference order:
    1) Hugging Face datasets: `datasets.load_dataset('mnist')`
    2) TorchVision: `torchvision.datasets.MNIST`
    3) keras.datasets.mnist or tensorflow.keras.datasets.mnist
    4) scikit-learn load_digits (8x8) as a last-resort offline fallback
    """
    # 1) Hugging Face datasets (mnist)
    try:
        from datasets import load_dataset  # type: ignore

        ds_train = load_dataset("mnist", split="train")
        ds_test = load_dataset("mnist", split="test")
        x_train = np.stack([np.array(img, dtype=np.float32) for img in ds_train["image"]], axis=0)
        y_train = np.asarray(ds_train["label"], dtype=np.int64)
        x_test = np.stack([np.array(img, dtype=np.float32) for img in ds_test["image"]], axis=0)
        y_test = np.asarray(ds_test["label"], dtype=np.int64)
        print("Loaded MNIST from Hugging Face datasets.")
        return x_train, y_train, x_test, y_test
    except Exception:
        pass

    # 2) Keras (standalone), then TF packaged Keras
    # 2) TorchVision MNIST
    try:
        from torchvision.datasets import MNIST  # type: ignore

        # Default cache location; adjust as needed
        import os

        print("Loading MNIST from PyTorch.")
        root = os.path.expanduser("~/.cache/torchvision")
        ds_train = MNIST(root=root, train=True, download=True)
        ds_test = MNIST(root=root, train=False, download=True)

        # Use tensor attributes if available (CPU ByteTensor), convert to numpy
        if hasattr(ds_train, "data") and hasattr(ds_train, "targets"):
            x_train = ds_train.data.numpy().astype(np.float32)
            y_train = ds_train.targets.numpy().astype(np.int64)
            x_test = ds_test.data.numpy().astype(np.float32)
            y_test = ds_test.targets.numpy().astype(np.int64)
        else:
            # Fallback: iterate
            x_train = np.stack([np.array(img, dtype=np.float32) for img, _ in ds_train], axis=0)
            y_train = np.asarray([int(lbl) for _, lbl in ds_train], dtype=np.int64)
            x_test = np.stack([np.array(img, dtype=np.float32) for img, _ in ds_test], axis=0)
            y_test = np.asarray([int(lbl) for _, lbl in ds_test], dtype=np.int64)

        return x_train, y_train, x_test, y_test
    except Exception:
        pass

    # 3) Keras (standalone), then TF packaged Keras
    try:
        from keras.datasets import mnist as keras_mnist  # type: ignore

        print("Loading MNIST from Keras.")
        (x_train, y_train), (x_test, y_test) = keras_mnist.load_data()
        return (
            x_train.astype(np.float32),
            y_train.astype(np.int64),
            x_test.astype(np.float32),
            y_test.astype(np.int64),
        )
    except Exception:
        pass

    try:
        from tensorflow.keras.datasets import mnist as tf_mnist  # type: ignore

        (x_train, y_train), (x_test, y_test) = tf_mnist.load_data()
        return (
            x_train.astype(np.float32),
            y_train.astype(np.int64),
            x_test.astype(np.float32),
            y_test.astype(np.int64),
        )
    except Exception:
        pass

    # 4) scikit-learn digits (8x8) offline fallback
    try:
        from sklearn.datasets import load_digits  # type: ignore

        print("Loading MNIST from scikit-learn.")

        digits = load_digits()
        images = digits.images.astype(np.float32)  # shape: (n, 8, 8)
        labels = digits.target.astype(np.int64)
        # simple split
        split = int(0.8 * len(images))
        x_train, y_train = images[:split], labels[:split]
        x_test, y_test = images[split:], labels[split:]
        return x_train, y_train, x_test, y_test
    except Exception as e:  # pragma: no cover - no viable loader
        raise RuntimeError(
            "Unable to load MNIST-like data. Install one of: `datasets`, `keras`/`tensorflow`, or `scikit-learn`."
        ) from e


def _threshold_to_pm1(img2d: np.ndarray) -> np.ndarray:
    """Map a 2D grayscale image to {-1, +1} using a mean threshold.

    If scikit-image is available, use its threshold_mean for robustness; otherwise
    fall back to the image mean.
    """
    if skimage_threshold_mean is not None:
        thresh = float(skimage_threshold_mean(img2d))
    else:
        thresh = float(img2d.mean())
    binary = img2d > thresh
    pm1 = np.where(binary, 1.0, -1.0).astype(np.float32)
    return pm1


def _flatten_pm1(img2d: np.ndarray) -> np.ndarray:
    """Flatten a 2D {-1,+1} image to 1D vector."""
    return img2d.reshape(-1).astype(np.float32)


def select_by_classes(
    x: np.ndarray,
    y: np.ndarray,
    classes: list[int],
    *,
    index: int,
) -> list[np.ndarray]:
    """Pick one image per class at the specified index among that class' samples.

    Example: index=0 picks the first exemplar of each class; index=1 picks the next one.
    Returns a list of 2D images (dtype float32) without thresholding.
    """
    out: list[np.ndarray] = []
    for c in classes:
        xi = x[y == c]
        if len(xi) <= index:
            raise ValueError(f"Class {c} has fewer than {index + 1} samples in the dataset.")
        out.append(xi[index].astype(np.float32))
    return out


def reshape_square(vec: np.ndarray) -> np.ndarray:
    dim = int(np.sqrt(vec.size))
    return vec.reshape(dim, dim)


# ---------------
# Main procedure
# ---------------

def main():
    # Choose classes to store (match reference default more closely)
    classes = [0, 1, 2]

    # Load MNIST
    x_train, y_train, x_test, y_test = _load_mnist()
    # Training exemplars: first occurrence of each class
    train_imgs_2d = select_by_classes(x_train, y_train, classes, index=0)
    # Test exemplars: a different occurrence of each class
    test_imgs_2d = select_by_classes(x_train, y_train, classes, index=1)

    # Preprocess to {-1,+1} vectors
    train_pm1 = [_flatten_pm1(_threshold_to_pm1(img)) for img in train_imgs_2d]
    test_pm1 = [_flatten_pm1(_threshold_to_pm1(img)) for img in test_imgs_2d]

    # Build Hopfield model (discrete sign activation). n = image_size^2.
    n = train_pm1[0].size
    model = AmariHopfieldNetwork(num_neurons=n, threshold=80.0, asyn=False, activation="sign")

    # Trainer: generic Hebbian (subtract mean, zero diagonal, normalize by patterns)
    trainer = HebbianTrainer(model, compiled_prediction=True)
    trainer.train(train_pm1)

    # Predict recovered patterns on clean held-out inputs
    predicted = trainer.predict_batch(test_pm1, show_sample_progress=True)

    # Plot results: Train vs Input (corrupted) vs Output (recovered)
    _plot_triplets(train_pm1, test_pm1, predicted, save_path="result_mnist.png")


def _plot_triplets(
    train_vecs: list[np.ndarray],
    input_vecs: list[np.ndarray],
    output_vecs: list[np.ndarray],
    *,
    save_path: str,
):
    train_imgs = [reshape_square(v) for v in train_vecs]
    input_imgs = [reshape_square(v) for v in input_vecs]
    output_imgs = [reshape_square(v) for v in output_vecs]

    rows = len(train_imgs)
    fig, axarr = plt.subplots(rows, 3, figsize=(5, 1.6 * rows))

    if rows == 1:
        axarr = np.array([axarr])  # normalize to 2D grid

    for i in range(rows):
        if i == 0:
            axarr[i, 0].set_title("Train data")
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title("Output data")

        axarr[i, 0].imshow(train_imgs[i], cmap="gray")
        axarr[i, 0].axis("off")
        axarr[i, 1].imshow(input_imgs[i], cmap="gray")
        axarr[i, 1].axis("off")
        axarr[i, 2].imshow(output_imgs[i], cmap="gray")
        axarr[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()
