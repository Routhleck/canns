"""Contrastive Divergence trainer for Restricted Boltzmann Machines."""

from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp

from ..models.brain_inspired import BrainInspiredModel
from ._base import Trainer

__all__ = ["ContrastiveDivergenceTrainer"]


class ContrastiveDivergenceTrainer(Trainer):
    """
    Contrastive Divergence (CD-k) trainer for Restricted Boltzmann Machines.

    CD is an approximation to maximum likelihood learning for RBMs. Instead of
    computing the full partition function, it runs k steps of Gibbs sampling
    to approximate the gradient.

    Update Rules:
        ΔW = η * (⟨vh^T⟩_data - ⟨vh^T⟩_model)
        Δb = η * (⟨v⟩_data - ⟨v⟩_model)
        Δc = η * (⟨h⟩_data - ⟨h⟩_model)

    where ⟨·⟩_data is expectation over data distribution and
    ⟨·⟩_model is expectation over model distribution (approximated by k Gibbs steps).

    Reference:
        Hinton, G. E. (2002). Training products of experts by minimizing
        contrastive divergence. Neural Computation, 14(8), 1771-1800.
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        learning_rate: float = 0.01,
        k: int = 1,
        **kwargs,
    ):
        """
        Initialize Contrastive Divergence trainer.

        Args:
            model: The RBM model to train
            learning_rate: Learning rate for weight updates
            k: Number of Gibbs sampling steps (CD-k)
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate
        self.k = k
        self.reconstruction_errors = []

    def train(self, train_data: Iterable):
        """
        Train RBM using Contrastive Divergence.

        Args:
            train_data: Iterable of visible patterns (binary or continuous)
        """
        # Check required model methods
        required_methods = ["sample_hidden", "sample_visible", "hidden_prob", "visible_prob"]
        for method in required_methods:
            if not hasattr(self.model, method):
                raise AttributeError(f"Model must have '{method}' method for CD training")

        # Check required parameters
        if (
            not hasattr(self.model, "W")
            or not hasattr(self.model, "b")
            or not hasattr(self.model, "c")
        ):
            raise AttributeError("Model must have 'W', 'b', and 'c' parameters")

        W = self.model.W.value
        b = self.model.b.value
        c = self.model.c.value

        self.reconstruction_errors = []

        # Process each pattern
        for pattern in train_data:
            v_data = jnp.asarray(pattern, dtype=jnp.float32)

            # Positive phase: compute statistics from data
            h_data_prob = self.model.hidden_prob(v_data)

            positive_grad_W = jnp.outer(h_data_prob, v_data)
            positive_grad_b = v_data
            positive_grad_c = h_data_prob

            # Negative phase: run k steps of Gibbs sampling
            v_model = v_data
            for _ in range(self.k):
                h_model = self.model.sample_hidden(v_model)
                v_model = self.model.sample_visible(h_model)

            # Compute statistics from model
            h_model_prob = self.model.hidden_prob(v_model)

            negative_grad_W = jnp.outer(h_model_prob, v_model)
            negative_grad_b = v_model
            negative_grad_c = h_model_prob

            # Update parameters
            W = W + self.learning_rate * (positive_grad_W - negative_grad_W)
            b = b + self.learning_rate * (positive_grad_b - negative_grad_b)
            c = c + self.learning_rate * (positive_grad_c - negative_grad_c)

            # Track reconstruction error
            recon_error = jnp.mean((v_data - v_model) ** 2)
            self.reconstruction_errors.append(float(recon_error))

        # Update model parameters
        self.model.W.value = W
        self.model.b.value = b
        self.model.c.value = c

    def predict(self, pattern, num_gibbs_steps: int = 100, *args, **kwargs):
        """
        Reconstruct visible pattern through the RBM.

        Args:
            pattern: Input visible pattern
            num_gibbs_steps: Number of Gibbs steps for reconstruction

        Returns:
            Reconstructed visible pattern
        """
        v = jnp.asarray(pattern, dtype=jnp.float32)

        # Run Gibbs sampling
        for _ in range(num_gibbs_steps):
            h = self.model.sample_hidden(v)
            v = self.model.sample_visible(h)

        return v

    def get_reconstruction_error(self) -> float:
        """
        Get mean reconstruction error from last training epoch.

        Returns:
            Mean squared reconstruction error
        """
        if len(self.reconstruction_errors) == 0:
            return 0.0
        return float(jnp.mean(jnp.array(self.reconstruction_errors)))
