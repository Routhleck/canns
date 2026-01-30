canns.pipeline.asa.runner
=========================

.. py:module:: canns.pipeline.asa.runner

.. autoapi-nested-parse::

   Pipeline execution wrapper for ASA TUI.

   This module provides async pipeline execution that integrates with the existing
   canns.analyzer.data.asa module. It wraps the analysis functions and provides
   progress callbacks for the TUI.



Exceptions
----------

.. autoapisummary::

   canns.pipeline.asa.runner.ProcessingError


Classes
-------

.. autoapisummary::

   canns.pipeline.asa.runner.PipelineResult
   canns.pipeline.asa.runner.PipelineRunner


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

   Async pipeline execution wrapper.

   Initialize pipeline runner.


   .. py:method:: has_preprocessed_data()

      Check if preprocessing has been completed.



   .. py:method:: reset_input()

      Clear cached input/preprocessing state when input files change.



   .. py:method:: results_dir(state)

      Public accessor for results directory.



   .. py:method:: run_analysis(state, log_callback, progress_callback)
      :async:


      Run analysis pipeline based on workflow state.

      :param state: Current workflow state
      :param log_callback: Callback for log messages
      :param progress_callback: Callback for progress updates (0-100)

      :returns: PipelineResult with success status and artifacts



   .. py:method:: run_preprocessing(state, log_callback, progress_callback)
      :async:


      Run preprocessing pipeline to generate embed_data.

      :param state: Current workflow state
      :param log_callback: Callback for log messages
      :param progress_callback: Callback for progress updates (0-100)

      :returns: PipelineResult with preprocessing status



