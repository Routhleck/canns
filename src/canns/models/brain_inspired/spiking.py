"""Leaky Integrate-and-Fire (LIF) spiking neural network."""

from __future__ import annotations

import brainstate
import jax.numpy as jnp

from ._base import BrainInspiredModel

__all__ = ["LIFSpikingNetwork"]


class LIFSpikingNetwork(BrainInspiredModel):
    """
    Leaky Integrate-and-Fire (LIF) spiking neural network.
    
    This model implements a network of LIF neurons that communicate through
    discrete spikes. The membrane potential of each neuron evolves according to:
    
        τ * dV/dt = -(V - V_rest) + I_syn
    
    where:
        - V is the membrane potential
        - V_rest is the resting potential
        - I_syn is the synaptic input current
        - τ is the membrane time constant
    
    When V reaches threshold V_th, the neuron fires a spike and V resets to V_reset.
    
    This model exposes spike times, membrane potentials, and synaptic traces for
    STDP learning.
    """
    
    def __init__(
        self,
        num_neurons: int,
        tau_m: float = 20.0,
        tau_syn: float = 5.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh: float = -55.0,
        dt: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the LIF spiking network.
        
        Args:
            num_neurons: Number of neurons in the network
            tau_m: Membrane time constant (ms)
            tau_syn: Synaptic time constant (ms)
            v_rest: Resting potential (mV)
            v_reset: Reset potential after spike (mV)
            v_thresh: Spike threshold (mV)
            dt: Time step for simulation (ms)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(in_size=num_neurons, **kwargs)
        
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.tau_syn = tau_syn
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt
    
    def init_state(self):
        """Initialize network state variables."""
        # Synaptic weight matrix
        self.W = brainstate.ParamState(
            jnp.zeros((self.num_neurons, self.num_neurons), dtype=jnp.float32)
        )
        
        # Membrane potentials
        self.V = brainstate.HiddenState(
            jnp.ones(self.num_neurons, dtype=jnp.float32) * self.v_rest
        )
        
        # Spike indicators (1 if neuron spiked in current timestep, 0 otherwise)
        self.spikes = brainstate.HiddenState(
            jnp.zeros(self.num_neurons, dtype=jnp.float32)
        )
        
        # Last spike times (for STDP)
        self.spike_times = brainstate.HiddenState(
            jnp.full(self.num_neurons, -jnp.inf, dtype=jnp.float32)
        )
        
        # Synaptic traces (exponentially decaying, for STDP)
        self.x_trace = brainstate.HiddenState(
            jnp.zeros(self.num_neurons, dtype=jnp.float32)
        )
        
        # Current time
        self.t = brainstate.HiddenState(jnp.float32(0.0))
    
    def update(self, prev_energy):
        """
        Update network state for one time step.
        
        This performs:
        1. Compute synaptic input
        2. Update membrane potentials
        3. Check for spikes and reset
        4. Update synaptic traces
        """
        # Synaptic input from spikes
        I_syn = self.W.value @ self.spikes.value
        
        # Update membrane potential with leaky integration
        dV = (-(self.V.value - self.v_rest) + I_syn) * (self.dt / self.tau_m)
        V_new = self.V.value + dV
        
        # Check for spikes
        spiked = V_new >= self.v_thresh
        
        # Reset membrane potential for neurons that spiked
        V_new = jnp.where(spiked, self.v_reset, V_new)
        
        # Update spike times
        spike_times_new = jnp.where(spiked, self.t.value, self.spike_times.value)
        
        # Update synaptic traces (exponential decay + spike)
        decay_factor = jnp.exp(-self.dt / self.tau_syn)
        x_trace_new = self.x_trace.value * decay_factor + spiked.astype(jnp.float32)
        
        # Update state
        self.V.value = V_new
        self.spikes.value = spiked.astype(jnp.float32)
        self.spike_times.value = spike_times_new
        self.x_trace.value = x_trace_new
        self.t.value = self.t.value + self.dt
    
    def set_input(self, external_input: jnp.ndarray):
        """
        Set external input to the network (e.g., stimulus).
        
        Args:
            external_input: Input current for each neuron
        """
        # Add external input as spikes or direct current injection
        self.spikes.value = (external_input > 0).astype(jnp.float32)
    
    @property
    def energy(self) -> float:
        """Energy based on membrane potential deviation from rest."""
        return float(jnp.sum((self.V.value - self.v_rest) ** 2))
    
    @property
    def weight_attr(self) -> str:
        """Name of weight parameter for generic training."""
        return "W"
    
    @property
    def predict_state_attr(self) -> str:
        """Name of state for prediction (membrane potential)."""
        return "V"
    
    def resize(self, num_neurons: int, preserve_submatrix: bool = True):
        """
        Resize the network dimension.
        
        Args:
            num_neurons: New number of neurons
            preserve_submatrix: Whether to preserve existing weights
        """
        old_n = self.num_neurons
        old_W = self.W.value if hasattr(self, "W") else None
        
        self.num_neurons = int(num_neurons)
        
        # Create new weight matrix
        new_W = jnp.zeros((self.num_neurons, self.num_neurons), dtype=jnp.float32)
        
        if preserve_submatrix and old_W is not None:
            min_n = min(old_n, self.num_neurons)
            new_W = new_W.at[:min_n, :min_n].set(old_W[:min_n, :min_n])
        
        # Update parameters
        if hasattr(self, "W"):
            self.W.value = new_W
        else:
            self.W = brainstate.ParamState(new_W)
        
        # Reinitialize other state variables
        if hasattr(self, "V"):
            self.V.value = jnp.ones(self.num_neurons, dtype=jnp.float32) * self.v_rest
        else:
            self.V = brainstate.HiddenState(
                jnp.ones(self.num_neurons, dtype=jnp.float32) * self.v_rest
            )
        
        if hasattr(self, "spikes"):
            self.spikes.value = jnp.zeros(self.num_neurons, dtype=jnp.float32)
        else:
            self.spikes = brainstate.HiddenState(jnp.zeros(self.num_neurons, dtype=jnp.float32))
        
        if hasattr(self, "spike_times"):
            self.spike_times.value = jnp.full(self.num_neurons, -jnp.inf, dtype=jnp.float32)
        else:
            self.spike_times = brainstate.HiddenState(
                jnp.full(self.num_neurons, -jnp.inf, dtype=jnp.float32)
            )
        
        if hasattr(self, "x_trace"):
            self.x_trace.value = jnp.zeros(self.num_neurons, dtype=jnp.float32)
        else:
            self.x_trace = brainstate.HiddenState(
                jnp.zeros(self.num_neurons, dtype=jnp.float32)
            )
