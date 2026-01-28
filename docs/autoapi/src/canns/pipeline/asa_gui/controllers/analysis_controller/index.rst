src.canns.pipeline.asa_gui.controllers.analysis_controller
==========================================================

.. py:module:: src.canns.pipeline.asa_gui.controllers.analysis_controller

.. autoapi-nested-parse::

   Controller for analysis workflow.



Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.controllers.analysis_controller.AnalysisController


Module Contents
---------------

.. py:class:: AnalysisController(state_manager, runner, parent=None)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   .. py:method:: finalize_analysis(artifacts)


   .. py:method:: get_state()


   .. py:method:: mark_idle()


   .. py:method:: run_analysis(*, worker_manager, on_log, on_progress, on_finished, on_error, on_cleanup = None)


   .. py:method:: update_analysis(*, analysis_mode, analysis_params)


