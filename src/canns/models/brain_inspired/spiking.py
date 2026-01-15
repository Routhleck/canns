"""Simple spiking neuron layer for STDP learning."""

from __future__ import annotations

import brainpy.math as bm
import jax
import jax.numpy as jnp

from ._base import BrainInspiredModel

__all__ = ["SpikingLayer"]


class SpikingLayer(BrainInspiredModel):
    """Simple Leaky Integrate-and-Fire (LIF) spiking neuron layer.

    It supports STDP-style training by maintaining pre/post spike traces.

    Dynamics:
        v[t+1] = leak * v[t] + W @ x[t]
        spike = 1 if v >= threshold else 0
        v = v_reset if spike else v
        trace = decay * trace + spike

    Notes:
        - x[t] denotes the input current at time t. It can take arbitrary continuous
          values; binary {0, 1} spike trains are a special case of such inputs.
        - The layer does not internally binarize x; thresholding only applies to the
          membrane potential to generate output spikes.

    Examples:
        >>> import jax.numpy as jnp
        >>> from canns.models.brain_inspired import SpikingLayer
        >>>
        >>> layer = SpikingLayer(input_size=3, output_size=2, threshold=0.5)
        >>> # Continuous input currents (binary spikes {0,1} are a special case)
        >>> x = jnp.array([0.2, 0.5, 1.3], dtype=jnp.float32)
        >>> spikes = layer.forward(x)
        >>> spikes.shape
        (2,)

    References:
        - Gerstner & Kistler (2002): Spiking Neuron Models
        - Morrison et al. (2008): Phenomenological models of synaptic plasticity
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        threshold: float = 1.0,
        v_reset: float = 0.0,
        leak: float = 0.9,
        trace_decay: float = 0.95,
        dt: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the spiking layer.

        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            threshold: Spike threshold for membrane potential
            v_reset: Reset potential after spike
            leak: Membrane leak factor (0-1, closer to 1 = less leaky)
            trace_decay: Decay factor for spike traces (used in STDP)
            dt: Time step size
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.threshold = threshold
        self.v_reset = v_reset
        self.leak = leak
        self.trace_decay = trace_decay
        self.dt = dt

        # Weight matrix W: (output_size, input_size)
        # Initialize with small random values
        key = bm.random.get_key()
        self.W = bm.Variable(
            jax.random.normal(key, (self.output_size, self.input_size), dtype=jnp.float32) * 0.05
        )

        # Input spikes (for training)
        self.x = bm.Variable(jnp.zeros(self.input_size, dtype=jnp.float32))

        # Membrane potential
        self.v = bm.Variable(jnp.zeros(self.output_size, dtype=jnp.float32))

        # Output spikes
        self.spike = bm.Variable(jnp.zeros(self.output_size, dtype=jnp.float32))

        # Spike traces (exponentially decaying spike history)
        self.trace_pre = bm.Variable(jnp.zeros(self.input_size, dtype=jnp.float32))
        self.trace_post = bm.Variable(jnp.zeros(self.output_size, dtype=jnp.float32))

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute spikes for one time step.

        Args:
            x: Input currents of shape ``(input_size,)``. Can be continuous-valued
                (e.g., synaptic currents) or binary spikes {0, 1} as a special case.

        Returns:
            Output spikes of shape ``(output_size,)`` with values 0 or 1.
        """
        self.x.value = jnp.asarray(x, dtype=jnp.float32)

        # Update pre-synaptic traces
        self.trace_pre.value = self.trace_decay * self.trace_pre.value + self.x.value

        # Leaky integration
        input_current = self.W.value @ self.x.value
        self.v.value = self.leak * self.v.value + input_current

        # Generate spikes
        self.spike.value = (self.v.value >= self.threshold).astype(jnp.float32)

        # Reset membrane potential for neurons that spiked
        self.v.value = jnp.where(self.spike.value > 0, self.v_reset, self.v.value)

        # Update post-synaptic traces
        self.trace_post.value = self.trace_decay * self.trace_post.value + self.spike.value

        return self.spike.value

    def reset_state(self):
        """Reset membrane potentials and spike traces."""
        if hasattr(self, "v"):
            self.v.value = jnp.zeros(self.output_size, dtype=jnp.float32)
        if hasattr(self, "spike"):
            self.spike.value = jnp.zeros(self.output_size, dtype=jnp.float32)
        if hasattr(self, "trace_pre"):
            self.trace_pre.value = jnp.zeros(self.input_size, dtype=jnp.float32)
        if hasattr(self, "trace_post"):
            self.trace_post.value = jnp.zeros(self.output_size, dtype=jnp.float32)

    def update(self, prev_energy):
        """Update method for trainer compatibility (no-op for spiking layer)."""
        pass

    @property
    def energy(self) -> float:
        """Energy for trainer compatibility (0 for spiking layer)."""
        return 0.0

    @property
    def weight_attr(self) -> str:
        """Name of weight parameter for generic training."""
        return "W"

    @property
    def predict_state_attr(self) -> str:
        """Name of output state for prediction."""
        return "spike"
