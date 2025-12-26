import numpy as np
import pytest

from canns.analyzer.metrics.utils import spike_train_to_firing_rate, firing_rate_to_spike_train

def test_fr_to_st_zero_rate_produces_no_spikes():
    duration = 100
    dt_rate = 0.1
    dt_spike = 0.001
    num_neurons = 5

    num_timesteps_rate = int(duration / dt_rate)
    zero_rates = np.zeros((num_timesteps_rate, num_neurons))

    spike_train = firing_rate_to_spike_train(zero_rates, dt_spike=dt_spike, dt_rate=dt_rate)

    assert np.sum(spike_train) == 0, "zero firing rate should produce no spikes"

def test_fr_to_st_max_rate_produces_all_spikes():
    duration = 100
    dt_rate = 0.1
    dt_spike = 0.001
    num_neurons = 5

    num_timesteps_rate = int(duration / dt_rate)
    max_rates = np.full((num_timesteps_rate, num_neurons), dt_rate / dt_spike)

    spike_train = firing_rate_to_spike_train(max_rates, dt_spike=dt_spike, dt_rate=dt_rate)

    assert np.all(spike_train), "max firing rate should produce all spikes"


def test_fr_to_st_random():
    duration = 100
    dt_rate = 0.1
    dt_spike = 0.001
    num_neurons = 5
    prob = 0.5
    num_timesteps_rate = int(duration / dt_rate)
    rates = np.random.rand(num_timesteps_rate, num_neurons) * prob / dt_spike * dt_rate

    spike_train = firing_rate_to_spike_train(rates, dt_spike=dt_spike, dt_rate=dt_rate)
    assert spike_train.shape == (int(duration / dt_spike), num_neurons), "spike train shape mismatch"

    # Compute mean input rate per neuron (Hz)
    mean_input_rates = np.mean(rates, axis=0) / dt_rate  # rates are in spikes per dt_rate, so divide by dt_rate for Hz
    # Compute mean output rate per neuron (Hz)
    mean_output_rates = np.mean(spike_train, axis=0) / dt_spike  # spike_train is binary, so mean gives spikes per dt_spike
    # Assert that mean output rates are close to mean input rates (allowing 10% relative tolerance)
    np.testing.assert_allclose(
        mean_output_rates, mean_input_rates, rtol=0.1,
        err_msg="Mean output firing rates do not match input rates within tolerance"
    )

def test_st_to_fr_zero_rate_produces_no_spikes():
    duration = 100
    dt_spike = 0.001
    dt_rate = 0.1
    num_neurons = 5

    num_timesteps_spike = int(duration / dt_spike)
    zero_spikes = np.zeros((num_timesteps_spike, num_neurons))

    firing_rates = spike_train_to_firing_rate(zero_spikes, dt_spike=dt_spike, dt_rate=dt_rate)

    assert np.all(firing_rates == 0), "zero spike train should produce zero firing rates"

def test_st_to_fr_max_rate_produces_all_spikes():
    duration = 100
    dt_spike = 0.001
    dt_rate = 0.1
    num_neurons = 5

    num_timesteps_spike = int(duration / dt_spike)
    max_spikes = np.ones((num_timesteps_spike, num_neurons))

    firing_rates = spike_train_to_firing_rate(max_spikes, dt_spike=dt_spike, dt_rate=dt_rate)

    assert np.allclose(firing_rates, dt_rate / dt_spike, rtol=1.0), "max spike train should produce max firing rates"


def test_st_to_fr_random():
    duration = 100
    dt_spike = 0.001
    dt_rate = 0.1
    num_neurons = 5
    prob = 0.5

    num_timesteps_spike = int(duration / dt_spike)
    spikes = np.random.rand(num_timesteps_spike, num_neurons) < prob

    firing_rates = spike_train_to_firing_rate(spikes, dt_spike=dt_spike, dt_rate=dt_rate)

    assert firing_rates.shape == (int(duration / dt_rate), num_neurons), "firing rates shape mismatch"

    # Compute mean input rate per neuron (Hz)
    mean_input_rates = np.mean(spikes, axis=0) / dt_spike * dt_rate  # spikes are binary, so mean gives spikes per dt_spike
    # Compute mean output rate per neuron (Hz)
    mean_output_rates = np.mean(firing_rates, axis=0)  # firing rates are in spikes per dt_rate, so multiply by dt_rate
    # Assert that mean output rates are close to mean input rates (allowing 10% relative tolerance)
    np.testing.assert_allclose(
        mean_output_rates, mean_input_rates, rtol=0.1,
        err_msg="Mean output firing rates do not match input rates within tolerance"
    )