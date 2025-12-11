src.canns.analyzer.slow_points.checkpoint
=========================================

.. py:module:: src.canns.analyzer.slow_points.checkpoint

.. autoapi-nested-parse::

   Checkpoint utilities for saving and loading trained RNN models using BrainPy's built-in checkpointing.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.slow_points.checkpoint.load_checkpoint
   src.canns.analyzer.slow_points.checkpoint.save_checkpoint


Module Contents
---------------

.. py:function:: load_checkpoint(model, filepath)

   Load model parameters from a checkpoint file using BrainPy checkpointing.

   :param model: BrainPy model to load parameters into.
   :param filepath: Path to the checkpoint file.

   :returns: True if checkpoint was loaded successfully, False otherwise.

   .. rubric:: Example

   >>> from canns.analyzer.slow_points import load_checkpoint
   >>> if load_checkpoint(rnn, "my_model.msgpack"):
   ...     print("Loaded successfully")
   ... else:
   ...     print("No checkpoint found")
   Loaded checkpoint from: my_model.msgpack
   Loaded successfully


.. py:function:: save_checkpoint(model, filepath)

   Save model parameters to a checkpoint file using BrainPy checkpointing.

   :param model: BrainPy model to save.
   :param filepath: Path to save the checkpoint file.

   .. rubric:: Example

   >>> from canns.analyzer.slow_points import save_checkpoint
   >>> save_checkpoint(rnn, "my_model.msgpack")
   Saved checkpoint to: my_model.msgpack


