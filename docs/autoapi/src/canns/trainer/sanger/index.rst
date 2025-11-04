src.canns.trainer.sanger
========================

.. py:module:: src.canns.trainer.sanger

.. autoapi-nested-parse::

   Sanger's rule for sequential principal component extraction.



Classes
-------

.. autoapisummary::

   src.canns.trainer.sanger.SangerTrainer


Module Contents
---------------

.. py:class:: SangerTrainer(model, learning_rate = 0.01, normalize_weights = True, weight_attr = 'W', compiled = True, **kwargs)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   Sanger's rule (Generalized Hebbian Algorithm) for multiple PC extraction.

   Extends Oja's rule with Gram-Schmidt orthogonalization to extract multiple
   principal components. Each neuron learns to be orthogonal to all previous ones.

   Learning Rule:
       ΔW_i = η * (y_i * x - y_i * Σ_{j≤i} y_j * W_j)

   where:
       - W_i is the i-th neuron's weight vector
       - y = W @ x is the output vector
       - The sum enforces orthogonality (Gram-Schmidt process)

   This allows sequential extraction of orthogonal principal components,
   with neuron i converging to the i-th principal component.

   Reference:
       Sanger, T. D. (1989). Optimal unsupervised learning in a single-layer
       linear feedforward neural network. Neural Networks, 2(6), 459-473.

   Initialize Sanger trainer.

   :param model: The model to train (typically LinearLayer)
   :param learning_rate: Learning rate η for weight updates
   :param normalize_weights: Whether to normalize weights to unit norm after each update
   :param weight_attr: Name of model attribute holding the connection weights
   :param compiled: Whether to use JIT-compiled training loop (default: True)
   :param \*\*kwargs: Additional arguments passed to parent Trainer


   .. py:method:: predict(pattern, *args, **kwargs)

      Predict output for a single input pattern.

      :param pattern: Input pattern of shape (input_size,)

      :returns: Output pattern of shape (output_size,)



   .. py:method:: train(train_data)

      Train the model using Sanger's rule.

      :param train_data: Iterable of input patterns (each of shape (input_size,))



   .. py:attribute:: compiled
      :value: True



   .. py:attribute:: learning_rate
      :value: 0.01



   .. py:attribute:: normalize_weights
      :value: True



   .. py:attribute:: weight_attr
      :value: 'W'



