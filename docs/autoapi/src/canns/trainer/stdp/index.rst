src.canns.trainer.stdp
======================

.. py:module:: src.canns.trainer.stdp

.. autoapi-nested-parse::

   STDP (Spike-Timing-Dependent Plasticity) trainer.



Classes
-------

.. autoapisummary::

   src.canns.trainer.stdp.STDPTrainer


Module Contents
---------------

.. py:class:: STDPTrainer(model, learning_rate = 0.01, A_plus = 0.005, A_minus = 0.00525, weight_attr = 'W', w_min = 0.0, w_max = 1.0, compiled = True, **kwargs)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   STDP (Spike-Timing-Dependent Plasticity) trainer.

   STDP is a biologically-inspired learning rule that adjusts synaptic weights
   based on the precise timing of pre- and post-synaptic spikes. Synapses are
   strengthened when pre-synaptic spikes precede post-synaptic spikes (LTP),
   and weakened when the order is reversed (LTD).

   Trace-based Learning Rule:
       ΔW_ij = A_plus * trace_pre[j] * spike_post[i] - A_minus * trace_post[i] * spike_pre[j]

   where:
       - W_ij is the weight from input j to neuron i
       - spike_pre[j] is the presynaptic spike (0 or 1)
       - spike_post[i] is the postsynaptic spike (0 or 1)
       - trace_pre[j] is the exponential trace of presynaptic spikes
       - trace_post[i] is the exponential trace of postsynaptic spikes
       - A_plus controls LTP (long-term potentiation) magnitude
       - A_minus controls LTD (long-term depression) magnitude

   The spike traces evolve as:
       trace = decay * trace + spike

   This provides a temporal window for spike-timing correlations.

   .. rubric:: References

   - Gerstner & Kistler (2002): Spiking Neuron Models
   - Morrison et al. (2008): Phenomenological models of synaptic plasticity
   - Bi & Poo (1998): Synaptic modifications in cultured hippocampal neurons

   Initialize STDP trainer.

   :param model: The spiking model to train (typically SpikingLayer)
   :param learning_rate: Global learning rate multiplier (default: 0.01)
   :param A_plus: LTP magnitude (default: 0.005)
   :param A_minus: LTD magnitude (default: 0.00525, slightly > A_plus for stability)
   :param weight_attr: Name of model attribute holding the connection weights
   :param w_min: Minimum weight value (default: 0.0 for excitatory synapses)
   :param w_max: Maximum weight value (default: 1.0)
   :param compiled: Whether to use JIT-compiled training loop (default: True)
   :param \*\*kwargs: Additional arguments passed to parent Trainer


   .. py:method:: predict(pattern, *args, **kwargs)

      Predict output spikes for a single input spike pattern.

      :param pattern: Input spike pattern of shape (input_size,)

      :returns: Output spike pattern of shape (output_size,) with binary values (0 or 1)



   .. py:method:: train(train_data)

      Train the model using STDP rule.

      :param train_data: Iterable of input spike patterns (each of shape (input_size,))
                         Each pattern should contain binary values (0 or 1)



   .. py:attribute:: A_minus
      :value: 0.00525



   .. py:attribute:: A_plus
      :value: 0.005



   .. py:attribute:: compiled
      :value: True



   .. py:attribute:: learning_rate
      :value: 0.01



   .. py:attribute:: w_max
      :value: 1.0



   .. py:attribute:: w_min
      :value: 0.0



   .. py:attribute:: weight_attr
      :value: 'W'



