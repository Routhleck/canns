"""Oja's normalized Hebbian learning trainer."""

from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp

from ..models.brain_inspired import BrainInspiredModel
from ._base import Trainer
from .utils import normalize_weight_rows

__all__ = ["OjaTrainer"]


class OjaTrainer(Trainer):
    """
    Oja's normalized Hebbian learning trainer.

    Oja's rule stabilizes pure Hebbian growth by introducing a weight-dependent
    normalization term, enabling single-neuron principal component extraction
    without unbounded weight magnitudes.

    Learning Rule:
        ΔW_ij = η * (y_i * x_j - y_i^2 * W_ij)

    where:
        - W_ij is the weight from input j to output i
        - x_j is the input activity
        - y_i is the output activity (y = W @ x)
        - η is the learning rate

    The rule can be rewritten as:
        ΔW = η * (y @ x^T - diag(y^2) @ W)

    This naturally leads to weight normalization and PCA extraction.

    Reference:
        Oja, E. (1982). Simplified neuron model as a principal component analyzer.
        Journal of Mathematical Biology, 15(3), 267-273.
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        learning_rate: float = 0.01,
        normalize_weights: bool = True,
        weight_attr: str = "W",
        **kwargs,
    ):
        """
        Initialize Oja trainer.

        Args:
            model: The model to train (typically LinearHebbLayer)
            learning_rate: Learning rate η for weight updates
            normalize_weights: Whether to normalize weights to unit norm after each update
            weight_attr: Name of model attribute holding the connection weights
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate
        self.normalize_weights = normalize_weights
        self.weight_attr = weight_attr

    def train(self, train_data: Iterable):
        """
        Train the model using Oja's rule.

        Args:
            train_data: Iterable of input patterns (each of shape (input_size,))
        """
        # Get weight parameter
        weight_param = getattr(self.model, self.weight_attr, None)
        if weight_param is None or not hasattr(weight_param, "value"):
            raise AttributeError(
                f"Model does not have a '{self.weight_attr}' parameter with .value attribute"
            )

        W = weight_param.value

        # Process each pattern
        for pattern in train_data:
            x = jnp.asarray(pattern, dtype=jnp.float32)

            # Compute output: y = W @ x
            y = W @ x

            # Oja's rule: ΔW = η * (y @ x^T - diag(y^2) @ W)
            outer_product = jnp.outer(y, x)
            normalization = jnp.outer(y * y, jnp.ones_like(x)) * W

            delta_W = self.learning_rate * (outer_product - normalization)
            W = W + delta_W

            # Optional: normalize weights to unit norm
            if self.normalize_weights:
                W = normalize_weight_rows(W)

        # Update model weights
        weight_param.value = W

    def predict(self, pattern, *args, **kwargs):
        """
        Predict output for a single input pattern.

        Args:
            pattern: Input pattern of shape (input_size,)

        Returns:
            Output pattern of shape (output_size,)
        """
        if hasattr(self.model, "forward"):
            return self.model.forward(pattern)
        else:
            # Fallback: direct computation
            weight_param = getattr(self.model, self.weight_attr)
            x = jnp.asarray(pattern, dtype=jnp.float32)
            return weight_param.value @ x
