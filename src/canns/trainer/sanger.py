"""Sanger's rule for sequential principal component extraction."""

from __future__ import annotations

from collections.abc import Iterable

import brainpy.math as bm
import jax.numpy as jnp

from ..models.brain_inspired import BrainInspiredModel
from ._base import Trainer
from .utils import normalize_weight_rows

__all__ = ["SangerTrainer"]


class SangerTrainer(Trainer):
    """
    Sanger's rule (Generalized Hebbian Algorithm) for multiple PC extraction.

    Extends Oja's rule with Gram-Schmidt orthogonalization to extract multiple
    principal components. Each neuron learns to be orthogonal to all previous ones.

    Learning Rule:
        ΔW_i = η * (y_i * x - y_i * Σ_{j≤i} y_j * W_j)

    where:
        - W_i is the i-th neuron's weight vector
        - y = W @ x is the output vector
        - The sum enforces orthogonality (Gram-Schmidt process)

    This allows sequential extraction of orthogonal principal components,
    with neuron i converging to the i-th principal component.

    Reference:
        Sanger, T. D. (1989). Optimal unsupervised learning in a single-layer
        linear feedforward neural network. Neural Networks, 2(6), 459-473.
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        learning_rate: float = 0.01,
        normalize_weights: bool = True,
        weight_attr: str = "W",
        compiled: bool = True,
        **kwargs,
    ):
        """
        Initialize Sanger trainer.

        Args:
            model: The model to train (typically LinearLayer)
            learning_rate: Learning rate η for weight updates
            normalize_weights: Whether to normalize weights to unit norm after each update
            weight_attr: Name of model attribute holding the connection weights
            compiled: Whether to use JIT-compiled training loop (default: True)
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate
        self.normalize_weights = normalize_weights
        self.weight_attr = weight_attr
        self.compiled = compiled

    def train(self, train_data: Iterable):
        """
        Train the model using Sanger's rule.

        Args:
            train_data: Iterable of input patterns (each of shape (input_size,))
        """
        # Get weight parameter
        weight_param = getattr(self.model, self.weight_attr, None)
        if weight_param is None or not hasattr(weight_param, "value"):
            raise AttributeError(
                f"Model does not have a '{self.weight_attr}' parameter with .value attribute"
            )

        if self.compiled:
            self._train_compiled(train_data, weight_param)
        else:
            self._train_uncompiled(train_data, weight_param)

    def _train_compiled(self, train_data: Iterable, weight_param):
        """
        JIT-compiled training loop using bp.transform.scan.

        Args:
            train_data: Iterable of input patterns
            weight_param: Weight parameter object
        """
        # Convert patterns to array for JIT compilation
        patterns = jnp.stack([jnp.asarray(p, dtype=jnp.float32) for p in train_data])

        # Initial weights
        W_init = jnp.asarray(weight_param.value, dtype=jnp.float32)

        # Training step for single pattern
        def train_step(W, x):
            # Compute output: y = W @ x
            y = W @ x  # Shape: (n_neurons,)

            # Sanger's rule with Gram-Schmidt orthogonalization
            # For each neuron i: ΔW_i = η * (y_i * x - y_i * Σ_{j≤i} y_j * W_j)
            n_neurons = W.shape[0]

            # Vectorized Gram-Schmidt: use lower triangular mask
            # Create lower triangular matrix (including diagonal)
            mask = jnp.tril(jnp.ones((n_neurons, n_neurons)))

            # y_weighted: (n_neurons, n_neurons) where [i,j] = y_i * y_j if j<=i else 0
            y_outer = jnp.outer(y, y)  # (n_neurons, n_neurons)
            y_weighted = y_outer * mask  # Only keep j<=i terms

            # Gram-Schmidt term: Σ_{j≤i} y_j * W_j for each i
            # y_weighted @ W: (n_neurons, n_neurons) @ (n_neurons, input_size)
            #              -> (n_neurons, input_size)
            gs_term = y_weighted @ W

            # Hebbian term: y_i * x for all i
            hebbian_term = jnp.outer(y, x)  # (n_neurons, input_size)

            # Weight update
            delta_W = self.learning_rate * (hebbian_term - gs_term)
            W = W + delta_W

            # Optional: normalize weights to unit norm
            if self.normalize_weights:
                W = normalize_weight_rows(W)

            return W, None

        # Run compiled scan
        W_final, _ = bm.scan(train_step, W_init, patterns)

        # Update model parameters
        weight_param.value = W_final

    def _train_uncompiled(self, train_data: Iterable, weight_param):
        """
        Python loop training (fallback, slower but more flexible).

        Args:
            train_data: Iterable of input patterns
            weight_param: Weight parameter object
        """
        W = weight_param.value
        n_neurons = W.shape[0]

        # Process each pattern
        for pattern in train_data:
            x = jnp.asarray(pattern, dtype=jnp.float32)

            # Compute output: y = W @ x
            y = W @ x

            # Sanger's rule with Gram-Schmidt orthogonalization
            # Vectorized version (same as compiled)
            mask = jnp.tril(jnp.ones((n_neurons, n_neurons)))
            y_outer = jnp.outer(y, y)
            y_weighted = y_outer * mask
            gs_term = y_weighted @ W

            hebbian_term = jnp.outer(y, x)
            delta_W = self.learning_rate * (hebbian_term - gs_term)
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
