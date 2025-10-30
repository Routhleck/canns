"""Spike-Timing-Dependent Plasticity (STDP) trainer."""

from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp

from ..models.brain_inspired import BrainInspiredModel
from ._base import Trainer

__all__ = ["STDPTrainer"]


class STDPTrainer(Trainer):
    """
    Spike-Timing-Dependent Plasticity (STDP) trainer.

    STDP is a biological learning rule where synaptic strength changes depend
    on the precise timing of pre- and postsynaptic spikes. The classic
    STDP window shows:
    - Potentiation (LTP) when presynaptic spike precedes postsynaptic spike
    - Depression (LTD) when postsynaptic spike precedes presynaptic spike

    Learning Rule:
        If pre-spike at t_pre and post-spike at t_post:
            Δt = t_post - t_pre

            If Δt > 0 (causality):
                ΔW = A+ * exp(-Δt / τ+)  (potentiation)

            If Δt < 0 (anti-causality):
                ΔW = -A- * exp(Δt / τ-)  (depression)

    where A+, A- are learning rate amplitudes and τ+, τ- are time constants.

    Reference:
        Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured
        hippocampal neurons: dependence on spike timing, synaptic strength, and
        postsynaptic cell type. Journal of Neuroscience, 18(24), 10464-10472.
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        a_plus: float = 0.01,
        a_minus: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        weight_attr: str = "W",
        **kwargs,
    ):
        """
        Initialize STDP trainer.

        Args:
            model: The model to train (typically LIFSpikingNetwork)
            a_plus: Learning rate for potentiation (LTP)
            a_minus: Learning rate for depression (LTD)
            tau_plus: Time constant for potentiation window (ms)
            tau_minus: Time constant for depression window (ms)
            weight_attr: Name of model attribute holding synaptic weights
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.weight_attr = weight_attr

    def train(self, train_data: Iterable):
        """
        Train the model using STDP.

        This method assumes the model has been simulated and spike times are recorded.
        It applies STDP updates based on the spike timing relationships.

        Args:
            train_data: Iterable of spike trains or stimulus patterns
        """
        # Get weight parameter
        weight_param = getattr(self.model, self.weight_attr, None)
        if weight_param is None or not hasattr(weight_param, "value"):
            raise AttributeError(
                f"Model does not have a '{self.weight_attr}' parameter with .value attribute"
            )

        W = weight_param.value

        # Check if model has spike time information
        if not hasattr(self.model, "spike_times"):
            raise AttributeError("Model must have 'spike_times' attribute for STDP learning")

        # For each training episode, simulate and apply STDP
        for pattern in train_data:
            # Set input pattern
            if hasattr(self.model, "set_input"):
                self.model.set_input(jnp.asarray(pattern, dtype=jnp.float32))

            # Run simulation for some steps to generate spikes
            num_steps = 100  # Default simulation length
            for _ in range(num_steps):
                self.model.update(0.0)

                # Apply STDP update based on current spikes
                if hasattr(self.model, "spikes") and hasattr(self.model, "x_trace"):
                    W = self._apply_stdp_update(W)

        # Update model weights
        weight_param.value = W

    def _apply_stdp_update(self, W: jnp.ndarray) -> jnp.ndarray:
        """
        Apply STDP weight update based on current spike state.

        Uses trace-based STDP for computational efficiency:
        - When presynaptic neuron j spikes: W[i,j] += A+ * x_post[i]
        - When postsynaptic neuron i spikes: W[i,j] -= A- * x_pre[j]

        Args:
            W: Current weight matrix

        Returns:
            Updated weight matrix
        """
        # Get spike indicators and traces
        spikes = self.model.spikes.value  # Current spikes (binary)
        x_trace = self.model.x_trace.value  # Synaptic traces

        # Potentiation: when post spikes, strengthen based on pre-trace
        # ΔW[i,j] += A+ * post_spike[i] * pre_trace[j]
        delta_W_ltp = self.a_plus * jnp.outer(spikes, x_trace)

        # Depression: when pre spikes, weaken based on post-trace
        # ΔW[i,j] -= A- * post_trace[i] * pre_spike[j]
        delta_W_ltd = -self.a_minus * jnp.outer(x_trace, spikes)

        # Total update
        W_new = W + delta_W_ltp + delta_W_ltd

        # Ensure weights stay within bounds [0, w_max]
        W_new = jnp.clip(W_new, 0.0, 10.0)

        return W_new

    def predict(self, pattern, num_steps: int = 100, *args, **kwargs):
        """
        Predict network response to input pattern.

        Args:
            pattern: Input stimulus pattern
            num_steps: Number of simulation timesteps

        Returns:
            Final spike pattern or membrane potential
        """
        # Set input
        if hasattr(self.model, "set_input"):
            self.model.set_input(jnp.asarray(pattern, dtype=jnp.float32))

        # Simulate
        for _ in range(num_steps):
            self.model.update(0.0)

        # Return spikes or membrane potential
        if hasattr(self.model, "spikes"):
            return self.model.spikes.value
        elif hasattr(self.model, "V"):
            return self.model.V.value
        else:
            return None
