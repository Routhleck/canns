src.canns.trainer.oja
=====================

.. py:module:: src.canns.trainer.oja

.. autoapi-nested-parse::

   Oja's normalized Hebbian learning trainer.



Classes
-------

.. autoapisummary::

   src.canns.trainer.oja.OjaTrainer


Module Contents
---------------

.. py:class:: OjaTrainer(model, learning_rate = 0.01, normalize_weights = True, weight_attr = 'W', compiled = True, **kwargs)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   Oja's normalized Hebbian learning trainer.

   Oja's rule stabilizes pure Hebbian growth by introducing a weight-dependent
   normalization term, enabling single-neuron principal component extraction
   without unbounded weight magnitudes.

   Learning Rule:
       ΔW_ij = η * (y_i * x_j - y_i^2 * W_ij)

   where:
       - W_ij is the weight from input j to output i
       - x_j is the input activity
       - y_i is the output activity (y = W @ x)
       - η is the learning rate

   The rule can be rewritten as:
       ΔW = η * (y @ x^T - diag(y^2) @ W)

   This naturally leads to weight normalization and PCA extraction.

   Reference:
       Oja, E. (1982). Simplified neuron model as a principal component analyzer.
       Journal of Mathematical Biology, 15(3), 267-273.

   Initialize Oja trainer.

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

      Train the model using Oja's rule.

      :param train_data: Iterable of input patterns (each of shape (input_size,))



   .. py:attribute:: compiled
      :value: True



   .. py:attribute:: learning_rate
      :value: 0.01



   .. py:attribute:: normalize_weights
      :value: True



   .. py:attribute:: weight_attr
      :value: 'W'



