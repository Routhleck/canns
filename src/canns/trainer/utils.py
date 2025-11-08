"""Shared utilities for brain-inspired trainers."""

from __future__ import annotations

import jax.numpy as jnp


def compute_running_average(
    current_avg: jnp.ndarray, new_value: jnp.ndarray, tau: float
) -> jnp.ndarray:
    """
    Compute exponential running average for BCM sliding thresholds.

    Args:
        current_avg: Current average value
        new_value: New value to incorporate
        tau: Time constant (higher = slower adaptation)

    Returns:
        Updated running average
    """
    alpha = 1.0 / tau if tau > 0 else 1.0
    return (1.0 - alpha) * current_avg + alpha * new_value


def normalize_weight_rows(W: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize each row of weight matrix to unit norm (for Oja's rule).

    Args:
        W: Weight matrix of shape (N, M)

    Returns:
        Normalized weight matrix with unit-norm rows
    """
    norms = jnp.linalg.norm(W, axis=1, keepdims=True)
    # Avoid division by zero
    norms = jnp.where(norms < 1e-10, 1.0, norms)
    return W / norms


def initialize_spike_buffer(num_neurons: int, buffer_size: int) -> jnp.ndarray:
    """
    Initialize spike time buffer for STDP learning.

    Args:
        num_neurons: Number of neurons in the network
        buffer_size: Number of recent spike times to store per neuron

    Returns:
        Spike buffer of shape (num_neurons, buffer_size) initialized to -inf
    """
    return jnp.full((num_neurons, buffer_size), -jnp.inf, dtype=jnp.float32)


def update_spike_buffer(buffer: jnp.ndarray, neuron_idx: int, spike_time: float) -> jnp.ndarray:
    """
    Update spike buffer with new spike time (circular buffer).

    Args:
        buffer: Current spike buffer of shape (num_neurons, buffer_size)
        neuron_idx: Index of neuron that spiked
        spike_time: Time of spike

    Returns:
        Updated spike buffer
    """
    # Roll buffer left and insert new spike time at end
    neuron_buffer = buffer[neuron_idx]
    new_buffer = jnp.roll(neuron_buffer, -1)
    new_buffer = new_buffer.at[-1].set(spike_time)
    return buffer.at[neuron_idx].set(new_buffer)


def stdp_kernel(dt: float, tau_plus: float = 20.0, tau_minus: float = 20.0) -> jnp.ndarray:
    """
    Compute STDP timing kernel for weight change.

    Args:
        dt: Time difference (post_spike_time - pre_spike_time)
        tau_plus: Time constant for potentiation (dt > 0)
        tau_minus: Time constant for depression (dt < 0)

    Returns:
        Weight change magnitude (positive for potentiation, negative for depression)
    """
    if dt > 0:
        # Potentiation: pre before post
        return jnp.exp(-dt / tau_plus)
    else:
        # Depression: post before pre
        return -jnp.exp(dt / tau_minus)


# Vectorized version for batch processing
stdp_kernel_vec = jnp.vectorize(stdp_kernel, excluded=[1, 2])
