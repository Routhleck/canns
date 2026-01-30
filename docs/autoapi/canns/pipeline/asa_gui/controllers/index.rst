canns.pipeline.asa_gui.controllers
==================================

.. py:module:: canns.pipeline.asa_gui.controllers

.. autoapi-nested-parse::

   Controllers for ASA GUI.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/canns/pipeline/asa_gui/controllers/analysis_controller/index
   /autoapi/canns/pipeline/asa_gui/controllers/preprocess_controller/index


Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.controllers.AnalysisController
   canns.pipeline.asa_gui.controllers.PreprocessController


Package Contents
----------------

.. py:class:: AnalysisController(state_manager, runner, parent=None)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   .. py:method:: finalize_analysis(artifacts)


   .. py:method:: get_state()


   .. py:method:: mark_idle()


   .. py:method:: run_analysis(*, worker_manager, on_log, on_progress, on_finished, on_error, on_cleanup = None)


   .. py:method:: update_analysis(*, analysis_mode, analysis_params)


.. py:class:: PreprocessController(state_manager, runner, parent=None)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   .. py:method:: finalize_preprocess()


   .. py:method:: mark_idle()


   .. py:method:: run_preprocess(*, worker_manager, on_log, on_progress, on_finished, on_error, on_cleanup = None)


   .. py:method:: update_inputs(*, input_mode, preset, asa_file, neuron_file, traj_file, preprocess_method, preprocess_params, preclass, preclass_params)


