src.canns.pipeline.asa_gui.controllers.preprocess_controller
============================================================

.. py:module:: src.canns.pipeline.asa_gui.controllers.preprocess_controller

.. autoapi-nested-parse::

   Controller for preprocessing workflow.



Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.controllers.preprocess_controller.PreprocessController


Module Contents
---------------

.. py:class:: PreprocessController(state_manager, runner, parent=None)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   .. py:method:: finalize_preprocess()


   .. py:method:: mark_idle()


   .. py:method:: run_preprocess(*, worker_manager, on_log, on_progress, on_finished, on_error, on_cleanup = None)


   .. py:method:: update_inputs(*, input_mode, preset, asa_file, neuron_file, traj_file, preprocess_method, preprocess_params, preclass, preclass_params)


