canns.analyzer.slow_points.checkpoint
=====================================

.. py:module:: canns.analyzer.slow_points.checkpoint

.. autoapi-nested-parse::

   Checkpoint utilities for saving and loading trained RNN models using BrainPy's built-in checkpointing.



Functions
---------

.. autoapisummary::

   canns.analyzer.slow_points.checkpoint.load_checkpoint
   canns.analyzer.slow_points.checkpoint.save_checkpoint


Module Contents
---------------

.. py:function:: load_checkpoint(model, filepath)

   Load model parameters from a checkpoint file using BrainPy checkpointing.

   :param model: BrainPy model to load parameters into.
   :param filepath: Path to the checkpoint file.

   :returns: True if checkpoint was loaded successfully, False otherwise.

   .. rubric:: Example

   >>> import tempfile
   >>> import brainpy as bp
   >>> import brainpy.math as bm
   >>> from canns.analyzer.slow_points import save_checkpoint, load_checkpoint
   >>>
   >>> class DummyModel(bp.DynamicalSystem):
   ...     def __init__(self):
   ...         super().__init__()
   ...         self.w = bm.Variable(bm.ones(1))
   >>>
   >>> model = DummyModel()
   >>> with tempfile.TemporaryDirectory() as tmpdir:
   ...     path = f"{tmpdir}/model.msgpack"
   ...     save_checkpoint(model, path)
   ...     ok = load_checkpoint(model, path)
   ...     print(ok)
   True


.. py:function:: save_checkpoint(model, filepath)

   Save model parameters to a checkpoint file using BrainPy checkpointing.

   :param model: BrainPy model to save.
   :param filepath: Path to save the checkpoint file.

   .. rubric:: Example

   >>> import tempfile
   >>> import brainpy as bp
   >>> import brainpy.math as bm
   >>> from canns.analyzer.slow_points import save_checkpoint
   >>>
   >>> class DummyModel(bp.DynamicalSystem):
   ...     def __init__(self):
   ...         super().__init__()
   ...         self.w = bm.Variable(bm.ones(1))
   >>>
   >>> model = DummyModel()
   >>> with tempfile.TemporaryDirectory() as tmpdir:
   ...     path = f"{tmpdir}/model.msgpack"
   ...     save_checkpoint(model, path)
   ...     print(path.endswith(".msgpack"))
   True


