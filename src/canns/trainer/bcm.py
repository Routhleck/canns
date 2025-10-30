"""BCM (Bienenstock-Cooper-Munro) sliding-threshold plasticity trainer."""

from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp

from ..models.brain_inspired import BrainInspiredModel
from ._base import Trainer

__all__ = ["BCMTrainer"]


class BCMTrainer(Trainer):
    """
    BCM (Bienenstock-Cooper-Munro) sliding-threshold plasticity trainer.

    The BCM rule uses a dynamic postsynaptic threshold to switch between
    potentiation and depression based on recent activity, yielding stable
    receptive-field development and experience-dependent refinement.

    Learning Rule:
        ΔW_ij = η * y_i * (y_i - θ_i) * x_j

    where:
        - W_ij is the weight from input j to neuron i
        - x_j is the presynaptic activity
        - y_i is the postsynaptic activity
        - θ_i is the modification threshold for neuron i

    The threshold θ evolves as a sliding average:
        θ_i = <y_i^2>

    This creates two regimes:
        - If y > θ: potentiation (LTP, strengthen synapses)
        - If y < θ: depression (LTD, weaken synapses)

    Reference:
        Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982).
        Theory for the development of neuron selectivity. Journal of Neuroscience, 2(1), 32-48.
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        learning_rate: float = 0.01,
        weight_attr: str = "W",
        **kwargs,
    ):
        """
        Initialize BCM trainer.

        Args:
            model: The model to train (typically BCMLayer)
            learning_rate: Learning rate η for weight updates
            weight_attr: Name of model attribute holding the connection weights
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate
        self.weight_attr = weight_attr

    def train(self, train_data: Iterable):
        """
        Train the model using BCM rule.

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

        # Check if model has theta (sliding threshold)
        if not hasattr(self.model, "theta"):
            raise AttributeError("Model must have 'theta' attribute for BCM learning")

        # Process each pattern
        for pattern in train_data:
            x = jnp.asarray(pattern, dtype=jnp.float32)

            # Forward pass: y = W @ x
            if hasattr(self.model, "forward"):
                y = self.model.forward(x)
            else:
                y = W @ x

            # Get current threshold
            theta = self.model.theta.value

            # BCM rule: ΔW = η * y * (y - θ) * x^T
            # Broadcasting: (output_size,) * (output_size,) -> (output_size,)
            phi = y * (y - theta)  # BCM modulation function
            # Outer product: (output_size,) ⊗ (input_size,) -> (output_size, input_size)
            delta_W = self.learning_rate * jnp.outer(phi, x)

            W = W + delta_W

            # Update threshold using model's method
            if hasattr(self.model, "update_threshold"):
                self.model.update_threshold()

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
