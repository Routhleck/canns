canns.pipeline.asa_gui.core.runner
==================================

.. py:module:: canns.pipeline.asa_gui.core.runner

.. autoapi-nested-parse::

   Pipeline execution wrapper for ASA GUI.

   Provides synchronous pipeline execution that wraps canns.analyzer.data.asa APIs
   and mirrors the TUI runner behavior for caching and artifacts.



Exceptions
----------

.. autoapisummary::

   canns.pipeline.asa_gui.core.runner.ProcessingError


Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.core.runner.PipelineResult
   canns.pipeline.asa_gui.core.runner.PipelineRunner


Module Contents
---------------

.. py:exception:: ProcessingError

   Bases: :py:obj:`RuntimeError`


   Raised when a pipeline stage fails.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: PipelineResult

   Result from pipeline execution.


   .. py:attribute:: artifacts
      :type:  dict[str, pathlib.Path]


   .. py:attribute:: elapsed_time
      :type:  float
      :value: 0.0



   .. py:attribute:: error
      :type:  str | None
      :value: None



   .. py:attribute:: success
      :type:  bool


   .. py:attribute:: summary
      :type:  str


.. py:class:: PipelineRunner

   Synchronous pipeline execution wrapper.


   .. py:method:: has_preprocessed_data()


   .. py:method:: reset_input()


   .. py:method:: results_dir(state)


   .. py:method:: run_analysis(state, log_callback, progress_callback, cancel_check = None)


   .. py:method:: run_preprocessing(state, log_callback, progress_callback, cancel_check = None)


   .. py:property:: aligned_pos
      :type: dict[str, numpy.ndarray] | None



   .. py:property:: embed_data
      :type: numpy.ndarray | None



