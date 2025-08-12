import numpy as np


def spike_train_to_firing_rate(
    spike_train: np.ndarray, dt_spike: float, dt_rate: float
) -> np.ndarray:
    """
    Converts a high-resolution spike train to a low-resolution firing rate signal.

    This function bins the spikes into larger time windows (`dt_rate`) and calculates
    the average firing rate for each bin.

    Args:
        spike_train (np.ndarray):
            2D array of shape (timesteps_spike, num_neurons) representing the high-res spike train.
        dt_spike (float):
            The time step of the input spike train in seconds (e.g., 0.001s).
        dt_rate (float):
            The desired time step of the output firing rate in dt_rate (e.g., 0.1s).

    Returns:
        np.ndarray:
            A 2D array of shape (timesteps_rate, num_neurons) with firing rates in Hz.
    """
    if spike_train.ndim != 2:
        raise ValueError("spike_train must be a 2D array.")
    if dt_rate < dt_spike:
        raise ValueError("dt_rate must be greater than or equal to dt_spike.")

    num_timesteps_spike, num_neurons = spike_train.shape
    duration_s = num_timesteps_spike * dt_spike

    # output timesteps based on the desired dt_rate
    num_timesteps_rate = int(np.floor(duration_s / dt_rate))
    output_rates = np.zeros((num_timesteps_rate, num_neurons))

    # find the indices of spikes and their corresponding times
    spike_indices, neuron_indices = np.where(spike_train)
    spike_times = spike_indices * dt_spike

    for n in range(num_neurons):
        # obtain spike times for the current neuron
        neuron_spike_times = spike_times[neuron_indices == n]

        # define bins for the histogram
        bins = np.arange(0, duration_s + dt_rate, dt_rate)

        # compute the histogram of spikes in the defined bins
        spike_counts_in_bins, _ = np.histogram(neuron_spike_times, bins=bins)

        output_rates[:, n] = spike_counts_in_bins

    return output_rates


def firing_rate_to_spike_train(
    firing_rates: np.ndarray, dt_rate: float, dt_spike: float
) -> np.ndarray:
    """
    Converts a low-resolution firing rate signal to a high-resolution binary spike train.

    This function generates spikes using a Bernoulli process in each high-resolution time bin.
    The probability of a spike in each bin is calculated as:
    P(spike in dt_spike) = rate (spikes/dt_rate) / dt_rate (sec) * dt_spike (sec)

    Note:
        A Bernoulli process is used, not a Poisson process. This means that in each time bin,
        at most one spike can occur. For high firing rates, the computed spike probability may
        exceed 1 and will be clipped to 1. This can lead to deviations from the expected
        Poisson statistics at high rates.

    Args:
        firing_rates (np.ndarray):
            2D array of shape (timesteps_rate, num_neurons) with firing rates in dt_rate.
        dt_rate (float):
            The time step of the input firing rate in seconds (e.g., 0.1s).
        dt_spike (float):
            The desired time step of the output spike train in seconds (e.g., 0.001s).

    Returns:
        np.ndarray:
            A 2D integer array of shape (timesteps_spike, num_neurons) with binary
            values (0 or 1) representing the high-resolution spike train.
    """
    if firing_rates.ndim != 2:
        raise ValueError("firing_rates must be a 2D array.")
    if dt_spike > dt_rate:
        raise ValueError("dt_spike must be smaller than or equal to dt_rate.")

    num_timesteps_rate, num_neurons = firing_rates.shape
    duration_s = num_timesteps_rate * dt_rate

    # Create high and low resolution time axes
    low_res_time = (np.arange(num_timesteps_rate) + 0.5) * dt_rate
    num_timesteps_spike = int(np.floor(duration_s / dt_spike))
    high_res_time = np.arange(num_timesteps_spike) * dt_spike

    # Interpolate firing rates to high resolution
    high_res_rates = np.zeros((num_timesteps_spike, num_neurons))
    for n in range(num_neurons):
        high_res_rates[:, n] = np.interp(high_res_time, low_res_time, firing_rates[:, n])

    # Ensure interpolated rates are non-negative
    high_res_rates[high_res_rates < 0] = 0

    # Generate spikes using a Bernoulli process
    spike_probabilities = high_res_rates * dt_spike / dt_rate

    # Probabilities must be <= 1. This can happen if rate > 1/dt_spike.
    # We clip to handle this physical constraint.
    np.clip(spike_probabilities, 0, 1, out=spike_probabilities)

    # Generate random numbers and compare with probabilities to create spikes.
    random_values = np.random.rand(*spike_probabilities.shape)
    spike_train = random_values < spike_probabilities

    return spike_train.astype(np.int8)


def normalize_firing_rates(
    firing_rates: np.ndarray,
    method: str = "min_max",
) -> np.ndarray:
    """
    Normalizes firing rates to a range of [0, 1] based on the maximum firing rate.

    Args:
        firing_rates (np.ndarray):
            2D array of shape (timesteps_rate, num_neurons) with firing rates in dt_rate.
        method (str):
            Normalization method, either 'min_max' or 'z_score'.
            - 'min_max': Normalizes to the range [0, 1].
            - 'z_score': Normalizes to have mean 0 and standard deviation 1.

    Returns:
        np.ndarray:
            A 2D array of shape (timesteps_rate, num_neurons) with normalized firing rates.
    """
    if firing_rates.ndim != 2:
        raise ValueError("firing_rates must be a 2D array.")

    if method not in ["min_max", "z_score"]:
        raise ValueError("Normalization method must be 'min_max' or 'z_score'.")

    match method:
        case "min_max":
            data_min = firing_rates.min()
            data_max = firing_rates.max()

            if data_max - data_min != 0:
                min_max_scaled_data = (firing_rates - data_min) / (data_max - data_min)
            else:
                min_max_scaled_data = np.zeros_like(firing_rates, dtype=float)
            return min_max_scaled_data
        case "z_score":
            data_mean = firing_rates.mean()
            data_std = firing_rates.std()

            if data_std != 0:
                z_score_scaled_data = (firing_rates - data_mean) / data_std
            else:
                z_score_scaled_data = np.zeros_like(firing_rates, dtype=float)
            return z_score_scaled_data
