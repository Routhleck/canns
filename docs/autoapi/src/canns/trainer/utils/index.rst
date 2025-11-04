src.canns.trainer.utils
=======================

.. py:module:: src.canns.trainer.utils

.. autoapi-nested-parse::

   Shared utilities for brain-inspired trainers.



Attributes
----------

.. autoapisummary::

   src.canns.trainer.utils.stdp_kernel_vec


Functions
---------

.. autoapisummary::

   src.canns.trainer.utils.compute_running_average
   src.canns.trainer.utils.initialize_spike_buffer
   src.canns.trainer.utils.normalize_weight_rows
   src.canns.trainer.utils.stdp_kernel
   src.canns.trainer.utils.update_spike_buffer


Module Contents
---------------

.. py:function:: compute_running_average(current_avg, new_value, tau)

   Compute exponential running average for BCM sliding thresholds.

   :param current_avg: Current average value
   :param new_value: New value to incorporate
   :param tau: Time constant (higher = slower adaptation)

   :returns: Updated running average


.. py:function:: initialize_spike_buffer(num_neurons, buffer_size)

   Initialize spike time buffer for STDP learning.

   :param num_neurons: Number of neurons in the network
   :param buffer_size: Number of recent spike times to store per neuron

   :returns: Spike buffer of shape (num_neurons, buffer_size) initialized to -inf


.. py:function:: normalize_weight_rows(W)

   Normalize each row of weight matrix to unit norm (for Oja's rule).

   :param W: Weight matrix of shape (N, M)

   :returns: Normalized weight matrix with unit-norm rows


.. py:function:: stdp_kernel(dt, tau_plus = 20.0, tau_minus = 20.0)

   Compute STDP timing kernel for weight change.

   :param dt: Time difference (post_spike_time - pre_spike_time)
   :param tau_plus: Time constant for potentiation (dt > 0)
   :param tau_minus: Time constant for depression (dt < 0)

   :returns: Weight change magnitude (positive for potentiation, negative for depression)


.. py:function:: update_spike_buffer(buffer, neuron_idx, spike_time)

   Update spike buffer with new spike time (circular buffer).

   :param buffer: Current spike buffer of shape (num_neurons, buffer_size)
   :param neuron_idx: Index of neuron that spiked
   :param spike_time: Time of spike

   :returns: Updated spike buffer


.. py:data:: stdp_kernel_vec

