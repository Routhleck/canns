"""BCM (Bienenstock-Cooper-Munro) sliding-threshold plasticity trainer."""

from __future__ import annotations

from collections.abc import Iterable

import brainpy.math as bm
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
        compiled: bool = True,
        **kwargs,
    ):
        """
        Initialize BCM trainer.

        Args:
            model: The model to train (typically LinearLayer with use_bcm_threshold=True)
            learning_rate: Learning rate η for weight updates
            weight_attr: Name of model attribute holding the connection weights
            compiled: Whether to use JIT-compiled training loop (default: True)
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate
        self.weight_attr = weight_attr
        self.compiled = compiled

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

        # Check if model has theta (sliding threshold)
        if not hasattr(self.model, "theta"):
            raise AttributeError("Model must have 'theta' attribute for BCM learning")

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

        # Get threshold tau from model
        threshold_tau = getattr(self.model, "threshold_tau", 100.0)

        # Initial state
        W_init = jnp.asarray(weight_param.value, dtype=jnp.float32)
        theta_init = jnp.asarray(self.model.theta.value, dtype=jnp.float32)

        # Training step for single pattern
        def train_step(carry, x):
            W, theta = carry

            # Forward pass
            y = W @ x

            # BCM rule: ΔW = η * y * (y - θ) * x^T
            phi = y * (y - theta)
            delta_W = self.learning_rate * jnp.outer(phi, x)
            W = W + delta_W

            # Clip weights
            W = jnp.clip(W, -10.0, 10.0)

            # Update threshold: θ ← θ + (1/τ) * (y² - θ)
            y_squared = y**2
            alpha = 1.0 / threshold_tau if threshold_tau > 0 else 1.0
            theta = theta + alpha * (y_squared - theta)

            return (W, theta), None

        # Run compiled scan
        (W_final, theta_final), _ = bm.scan(train_step, (W_init, theta_init), patterns)

        # Update model parameters
        weight_param.value = W_final
        self.model.theta.value = theta_final

    def _train_uncompiled(self, train_data: Iterable, weight_param):
        """
        Python loop training (fallback, slower but more flexible).

        Args:
            train_data: Iterable of input patterns
            weight_param: Weight parameter object
        """
        W = weight_param.value

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
            phi = y * (y - theta)  # BCM modulation function
            delta_W = self.learning_rate * jnp.outer(phi, x)

            W = W + delta_W

            # Clip weights to prevent divergence
            W = jnp.clip(W, -10.0, 10.0)

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
