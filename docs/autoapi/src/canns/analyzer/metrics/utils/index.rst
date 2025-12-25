src.canns.analyzer.metrics.utils
================================

.. py:module:: src.canns.analyzer.metrics.utils


Functions
---------

.. autoapisummary::

   src.canns.analyzer.metrics.utils.firing_rate_to_spike_train
   src.canns.analyzer.metrics.utils.normalize_firing_rates
   src.canns.analyzer.metrics.utils.spike_train_to_firing_rate


Module Contents
---------------

.. py:function:: firing_rate_to_spike_train(firing_rates, dt_rate, dt_spike)

   Converts a low-resolution firing rate signal to a high-resolution binary spike train.

   This function generates spikes using a Bernoulli process in each high-resolution time bin.
   The probability of a spike in each bin is calculated as:
   P(spike in dt_spike) = rate (spikes/dt_rate) / dt_rate (sec) * dt_spike (sec)

   .. note::

      A Bernoulli process is used, not a Poisson process. This means that in each time bin,
      at most one spike can occur. For high firing rates, the computed spike probability may
      exceed 1 and will be clipped to 1. This can lead to deviations from the expected
      Poisson statistics at high rates.

   :param firing_rates: 2D array of shape (timesteps_rate, num_neurons) with firing rates in dt_rate.
   :type firing_rates: np.ndarray
   :param dt_rate: The time step of the input firing rate in seconds (e.g., 0.1s).
   :type dt_rate: float
   :param dt_spike: The desired time step of the output spike train in seconds (e.g., 0.001s).
   :type dt_spike: float

   :returns:     A 2D integer array of shape (timesteps_spike, num_neurons) with binary
                 values (0 or 1) representing the high-resolution spike train.
   :rtype: np.ndarray


.. py:function:: normalize_firing_rates(firing_rates, method = 'min_max')

   Normalizes firing rates to a range of [0, 1] based on the maximum firing rate.

   :param firing_rates: 2D array of shape (timesteps_rate, num_neurons) with firing rates in dt_rate.
   :type firing_rates: np.ndarray
   :param method: Normalization method, either 'min_max' or 'z_score'.
                  - 'min_max': Normalizes to the range [0, 1].
                  - 'z_score': Normalizes to have mean 0 and standard deviation 1.
   :type method: str

   :returns:     A 2D array of shape (timesteps_rate, num_neurons) with normalized firing rates.
   :rtype: np.ndarray


.. py:function:: spike_train_to_firing_rate(spike_train, dt_spike, dt_rate)

   Converts a high-resolution spike train to a low-resolution firing rate signal.

   This function bins the spikes into larger time windows (`dt_rate`) and calculates
   the average firing rate for each bin.

   :param spike_train: 2D array of shape (timesteps_spike, num_neurons) representing the high-res spike train.
   :type spike_train: np.ndarray
   :param dt_spike: The time step of the input spike train in seconds (e.g., 0.001s).
   :type dt_spike: float
   :param dt_rate: The desired time step of the output firing rate in dt_rate (e.g., 0.1s).
   :type dt_rate: float

   :returns:     A 2D array of shape (timesteps_rate, num_neurons) with firing rates in Hz.
   :rtype: np.ndarray


