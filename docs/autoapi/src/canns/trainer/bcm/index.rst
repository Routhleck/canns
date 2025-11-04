src.canns.trainer.bcm
=====================

.. py:module:: src.canns.trainer.bcm

.. autoapi-nested-parse::

   BCM (Bienenstock-Cooper-Munro) sliding-threshold plasticity trainer.



Classes
-------

.. autoapisummary::

   src.canns.trainer.bcm.BCMTrainer


Module Contents
---------------

.. py:class:: BCMTrainer(model, learning_rate = 0.01, weight_attr = 'W', compiled = True, **kwargs)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   BCM (Bienenstock-Cooper-Munro) sliding-threshold plasticity trainer.

   The BCM rule uses a dynamic postsynaptic threshold to switch between
   potentiation and depression based on recent activity, yielding stable
   receptive-field development and experience-dependent refinement.

   Learning Rule:
       ΔW_ij = η * y_i * (y_i - θ_i) * x_j

   where:
       - W_ij is the weight from input j to neuron i
       - x_j is the presynaptic activity
       - y_i is the postsynaptic activity
       - θ_i is the modification threshold for neuron i

   The threshold θ evolves as a sliding average:
       θ_i = <y_i^2>

   This creates two regimes:
       - If y > θ: potentiation (LTP, strengthen synapses)
       - If y < θ: depression (LTD, weaken synapses)

   Reference:
       Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982).
       Theory for the development of neuron selectivity. Journal of Neuroscience, 2(1), 32-48.

   Initialize BCM trainer.

   :param model: The model to train (typically LinearLayer with use_bcm_threshold=True)
   :param learning_rate: Learning rate η for weight updates
   :param weight_attr: Name of model attribute holding the connection weights
   :param compiled: Whether to use JIT-compiled training loop (default: True)
   :param \*\*kwargs: Additional arguments passed to parent Trainer


   .. py:method:: predict(pattern, *args, **kwargs)

      Predict output for a single input pattern.

      :param pattern: Input pattern of shape (input_size,)

      :returns: Output pattern of shape (output_size,)



   .. py:method:: train(train_data)

      Train the model using BCM rule.

      :param train_data: Iterable of input patterns (each of shape (input_size,))



   .. py:attribute:: compiled
      :value: True



   .. py:attribute:: learning_rate
      :value: 0.01



   .. py:attribute:: weight_attr
      :value: 'W'



