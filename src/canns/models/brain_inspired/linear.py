"""Generic linear layer for brain-inspired learning algorithms."""

from __future__ import annotations

import brainpy.math as bm
import jax.numpy as jnp

from ._base import BrainInspiredModel

__all__ = ["LinearLayer"]


class LinearLayer(BrainInspiredModel):
    """
    Generic linear feedforward layer supporting multiple brain-inspired learning rules.

    This model provides a simple linear transformation with optional sliding threshold
    for BCM-style plasticity. It can be used with various trainers:
    - OjaTrainer: Normalized Hebbian learning for PCA
    - BCMTrainer: Sliding threshold plasticity (requires use_bcm_threshold=True)
    - HebbianTrainer: Standard Hebbian learning

    Computation:
        y = W @ x

    where W is the weight matrix, x is the input, and y is the output.

    For BCM learning, an optional sliding threshold θ tracks output activity:
        θ ← θ + (1/τ) * (y² - θ)

    References:
        - Oja (1982): Simplified neuron model as a principal component analyzer
        - Bienenstock et al. (1982): Theory for the development of neuron selectivity
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bcm_threshold: bool = False,
        threshold_tau: float = 100.0,
        **kwargs,
    ):
        """
        Initialize the linear layer.

        Args:
            input_size: Dimensionality of input vectors
            output_size: Number of output neurons (features to extract)
            use_bcm_threshold: Whether to maintain sliding threshold for BCM learning
            threshold_tau: Time constant for threshold sliding average (only used if use_bcm_threshold=True)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.use_bcm_threshold = use_bcm_threshold
        self.threshold_tau = threshold_tau

        # Weight matrix W: (output_size, input_size)
        # Initialize with small random values to break symmetry
        self.W = bm.Variable(
            bm.random.normal(size=(self.output_size, self.input_size), dtype=jnp.float32) * 0.01
        )
        # Input state (for training)
        self.x = bm.Variable(jnp.zeros(self.input_size, dtype=jnp.float32))
        # Output state
        self.y = bm.Variable(jnp.zeros(self.output_size, dtype=jnp.float32))

        # Optional sliding threshold for BCM learning
        if self.use_bcm_threshold:
            self.theta = bm.Variable(jnp.ones(self.output_size, dtype=jnp.float32) * 0.1)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the layer.

        Args:
            x: Input vector of shape (input_size,)

        Returns:
            Output vector of shape (output_size,)
        """
        self.x.value = jnp.asarray(x, dtype=jnp.float32)
        self.y.value = self.W.value @ self.x.value
        return self.y.value

    def update_threshold(self):
        """
        Update the sliding threshold based on recent activity (BCM only).

        This method should be called by BCMTrainer after each forward pass.
        Updates θ using: θ ← θ + (1/τ) * (y² - θ)
        """
        if not self.use_bcm_threshold:
            return

        y_squared = self.y.value**2
        alpha = 1.0 / self.threshold_tau if self.threshold_tau > 0 else 1.0
        self.theta.value = self.theta.value + alpha * (y_squared - self.theta.value)

    def update(self, prev_energy):
        """Update method for trainer compatibility (no-op for feedforward layer)."""
        pass

    @property
    def energy(self) -> float:
        """Energy for trainer compatibility (0 for feedforward layer)."""
        return 0.0

    @property
    def weight_attr(self) -> str:
        """Name of weight parameter for generic training."""
        return "W"

    @property
    def predict_state_attr(self) -> str:
        """Name of output state for prediction."""
        return "y"

    def resize(
        self, input_size: int, output_size: int | None = None, preserve_submatrix: bool = True
    ):
        """
        Resize layer dimensions.

        Args:
            input_size: New input dimension
            output_size: New output dimension (if None, keep current)
            preserve_submatrix: Whether to preserve existing weights
        """
        if output_size is None:
            output_size = self.output_size

        old_W = self.W.value if hasattr(self, "W") else None
        old_theta = None
        if self.use_bcm_threshold and hasattr(self, "theta"):
            old_theta = self.theta.value

        self.input_size = int(input_size)
        self.output_size = int(output_size)

        # Create new weight matrix
        new_W = jnp.zeros((self.output_size, self.input_size), dtype=jnp.float32)

        if preserve_submatrix and old_W is not None:
            min_out = min(old_W.shape[0], self.output_size)
            min_in = min(old_W.shape[1], self.input_size)
            new_W = new_W.at[:min_out, :min_in].set(old_W[:min_out, :min_in])

        # Update weight parameter
        if hasattr(self, "W"):
            self.W.value = new_W
        else:
            self.W = bm.Variable(new_W)

        # Update threshold if using BCM
        if self.use_bcm_threshold:
            new_theta = jnp.ones(self.output_size, dtype=jnp.float32) * 0.1
            if preserve_submatrix and old_theta is not None:
                min_out = min(old_theta.shape[0], self.output_size)
                new_theta = new_theta.at[:min_out].set(old_theta[:min_out])

            if hasattr(self, "theta"):
                self.theta.value = new_theta
            else:
                self.theta = bm.Variable(new_theta)

        # Update state variables
        if hasattr(self, "x"):
            self.x.value = jnp.zeros(self.input_size, dtype=jnp.float32)
        else:
            self.x = bm.Variable(jnp.zeros(self.input_size, dtype=jnp.float32))

        if hasattr(self, "y"):
            self.y.value = jnp.zeros(self.output_size, dtype=jnp.float32)
        else:
            self.y = bm.Variable(jnp.zeros(self.output_size, dtype=jnp.float32))
