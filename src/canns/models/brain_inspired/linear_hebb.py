"""Linear Hebbian layer for Oja's normalized learning."""

from __future__ import annotations

import brainstate
import jax
import jax.numpy as jnp

from ._base import BrainInspiredModel

__all__ = ["LinearHebbLayer"]


class LinearHebbLayer(BrainInspiredModel):
    """
    Linear feedforward layer with Hebbian plasticity support.

    This model exposes outputs and weight norms to support Oja-style normalized
    updates and PCA-like feature formation. The layer computes:

        y = W @ x

    where W is the weight matrix, x is the input, and y is the output.

    Oja's rule stabilizes pure Hebbian growth by introducing a weight-dependent
    normalization term, enabling single-neuron principal component extraction
    without unbounded weight magnitudes.

    Reference:
        Oja, E. (1982). Simplified neuron model as a principal component analyzer.
        Journal of Mathematical Biology, 15(3), 267-273.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        **kwargs,
    ):
        """
        Initialize the Linear Hebbian Layer.

        Args:
            input_size: Dimensionality of input vectors
            output_size: Number of output neurons (features to extract)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(in_size=input_size, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

    def init_state(self):
        """Initialize layer parameters."""
        # Weight matrix W: (output_size, input_size)
        # Initialize with small random values to break symmetry
        key = brainstate.random.get_key()
        self.W = brainstate.ParamState(
            jax.random.normal(key, (self.output_size, self.input_size), dtype=jnp.float32) * 0.01
        )
        # Input state (for training)
        self.x = brainstate.HiddenState(jnp.zeros(self.input_size, dtype=jnp.float32))
        # Output state
        self.y = brainstate.HiddenState(jnp.zeros(self.output_size, dtype=jnp.float32))

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

        self.input_size = int(input_size)
        self.output_size = int(output_size)

        # Create new weight matrix
        new_W = jnp.zeros((self.output_size, self.input_size), dtype=jnp.float32)

        if preserve_submatrix and old_W is not None:
            min_out = min(old_W.shape[0], self.output_size)
            min_in = min(old_W.shape[1], self.input_size)
            new_W = new_W.at[:min_out, :min_in].set(old_W[:min_out, :min_in])

        # Update parameters
        if hasattr(self, "W"):
            self.W.value = new_W
        else:
            self.W = brainstate.ParamState(new_W)

        if hasattr(self, "x"):
            self.x.value = jnp.zeros(self.input_size, dtype=jnp.float32)
        else:
            self.x = brainstate.HiddenState(jnp.zeros(self.input_size, dtype=jnp.float32))

        if hasattr(self, "y"):
            self.y.value = jnp.zeros(self.output_size, dtype=jnp.float32)
        else:
            self.y = brainstate.HiddenState(jnp.zeros(self.output_size, dtype=jnp.float32))
