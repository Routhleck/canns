src.canns.models.brain_inspired.spiking
=======================================

.. py:module:: src.canns.models.brain_inspired.spiking

.. autoapi-nested-parse::

   Simple spiking neuron layer for STDP learning.



Classes
-------

.. autoapisummary::

   src.canns.models.brain_inspired.spiking.SpikingLayer


Module Contents
---------------

.. py:class:: SpikingLayer(input_size, output_size, threshold = 1.0, v_reset = 0.0, leak = 0.9, trace_decay = 0.95, dt = 1.0, **kwargs)

   Bases: :py:obj:`src.canns.models.brain_inspired._base.BrainInspiredModel`


   Simple Leaky Integrate-and-Fire (LIF) spiking neuron layer.

   This model provides a minimal spiking neuron implementation for demonstrating
   spike-timing-dependent plasticity (STDP). It features:
   - Leaky integration of input currents
   - Threshold-based spike generation
   - Reset mechanism after spiking
   - Exponential spike traces for STDP learning

   Dynamics:
       v[t+1] = leak * v[t] + W @ x[t]
       spike = 1 if v >= threshold else 0
       v = v_reset if spike else v
       trace = decay * trace + spike

   .. rubric:: References

   - Gerstner & Kistler (2002): Spiking Neuron Models
   - Morrison et al. (2008): Phenomenological models of synaptic plasticity

   Initialize the spiking layer.

   :param input_size: Number of input neurons
   :param output_size: Number of output neurons
   :param threshold: Spike threshold for membrane potential
   :param v_reset: Reset potential after spike
   :param leak: Membrane leak factor (0-1, closer to 1 = less leaky)
   :param trace_decay: Decay factor for spike traces (used in STDP)
   :param dt: Time step size
   :param \*\*kwargs: Additional arguments passed to parent class


   .. py:method:: forward(x)

      Forward pass through the spiking layer.

      :param x: Input spikes of shape (input_size,) with binary values (0 or 1)

      :returns: Output spikes of shape (output_size,) with binary values (0 or 1)



   .. py:method:: reset_state()

      Reset membrane potentials and spike traces.



   .. py:method:: update(prev_energy)

      Update method for trainer compatibility (no-op for spiking layer).



   .. py:attribute:: W


   .. py:attribute:: dt
      :value: 1.0



   .. py:property:: energy
      :type: float


      Energy for trainer compatibility (0 for spiking layer).


   .. py:attribute:: input_size


   .. py:attribute:: leak
      :value: 0.9



   .. py:attribute:: output_size


   .. py:property:: predict_state_attr
      :type: str


      Name of output state for prediction.


   .. py:attribute:: spike


   .. py:attribute:: threshold
      :value: 1.0



   .. py:attribute:: trace_decay
      :value: 0.95



   .. py:attribute:: trace_post


   .. py:attribute:: trace_pre


   .. py:attribute:: v


   .. py:attribute:: v_reset
      :value: 0.0



   .. py:property:: weight_attr
      :type: str


      Name of weight parameter for generic training.


   .. py:attribute:: x


