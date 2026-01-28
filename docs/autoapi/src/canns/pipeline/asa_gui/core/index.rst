src.canns.pipeline.asa_gui.core
===============================

.. py:module:: src.canns.pipeline.asa_gui.core

.. autoapi-nested-parse::

   Core infrastructure for ASA GUI.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/pipeline/asa_gui/core/cache/index
   /autoapi/src/canns/pipeline/asa_gui/core/runner/index
   /autoapi/src/canns/pipeline/asa_gui/core/state/index
   /autoapi/src/canns/pipeline/asa_gui/core/worker/index


Exceptions
----------

.. autoapisummary::

   src.canns.pipeline.asa_gui.core.ProcessingError


Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.core.AnalysisWorker
   src.canns.pipeline.asa_gui.core.PipelineResult
   src.canns.pipeline.asa_gui.core.PipelineRunner
   src.canns.pipeline.asa_gui.core.StateManager
   src.canns.pipeline.asa_gui.core.WorkerManager
   src.canns.pipeline.asa_gui.core.WorkflowState


Package Contents
----------------

.. py:exception:: ProcessingError

   Bases: :py:obj:`RuntimeError`


   Raised when a pipeline stage fails.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: AnalysisWorker(parent = None)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   Background worker for analysis execution.

   Runs analysis in a separate thread and emits signals for
   progress updates, logging, and completion.


   .. py:method:: request_cancel()

      Request cancellation of running task.



   .. py:method:: run()

      Execute the configured task.



   .. py:method:: setup(task, *args, **kwargs)

      Configure the task to run.

      :param task: Callable to execute
      :param \*args: Positional arguments for task
      :param \*\*kwargs: Keyword arguments for task



   .. py:attribute:: error


   .. py:attribute:: finished


   .. py:attribute:: log


   .. py:attribute:: progress


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



.. py:class:: StateManager(parent = None)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   Reactive state manager with Qt signals.

   Emits signals when state changes to enable reactive UI updates.
   Supports undo/redo through state history.


   .. py:method:: batch_update(**kwargs)

      Update multiple fields without emitting individual signals.

      Emits state_replaced at the end.



   .. py:method:: can_redo()

      Check if redo is available.



   .. py:method:: can_undo()

      Check if undo is available.



   .. py:method:: push_history()

      Save current state for undo.



   .. py:method:: redo()

      Restore next state.

      :returns: True if redo was successful



   .. py:method:: reset()

      Reset state to defaults.



   .. py:method:: undo()

      Restore previous state.

      :returns: True if undo was successful



   .. py:method:: update(**kwargs)

      Update state fields and emit signals.

      :param \*\*kwargs: Field names and their new values



   .. py:property:: state
      :type: WorkflowState


      Get current workflow state.


   .. py:attribute:: state_changed


   .. py:attribute:: state_replaced


.. py:class:: WorkerManager

   Manages worker thread lifecycle.

   Ensures only one worker runs at a time and handles
   proper cleanup on completion or cancellation.


   .. py:method:: is_running()

      Check if a worker is currently running.



   .. py:method:: request_cancel()

      Request cancellation of running task.



   .. py:method:: start(task, *args, on_log = None, on_progress = None, on_finished = None, on_error = None, on_cleanup = None, **kwargs)

      Start a task in a background thread.

      :param task: Callable to execute
      :param \*args: Positional arguments for task
      :param on_log: Callback for log messages
      :param on_progress: Callback for progress updates
      :param on_finished: Callback on successful completion
      :param on_error: Callback on error
      :param on_cleanup: Callback after thread cleanup
      :param \*\*kwargs: Keyword arguments for task

      :raises RuntimeError: If a task is already running



   .. py:method:: wait(timeout_ms = 5000)

      Wait for worker to finish.

      :param timeout_ms: Maximum time to wait in milliseconds

      :returns: True if worker finished, False if timeout



.. py:class:: WorkflowState

   Centralized state for ASA analysis workflow.

   All file paths are relative to workdir for portability.


   .. py:method:: copy()

      Create a shallow copy of the state (excluding large arrays).



   .. py:attribute:: aligned_pos
      :type:  dict[str, numpy.typing.NDArray[numpy.floating]] | None
      :value: None



   .. py:attribute:: analysis_mode
      :type:  str
      :value: 'tda'



   .. py:attribute:: analysis_params
      :type:  dict[str, Any]


   .. py:attribute:: artifacts
      :type:  dict[str, pathlib.Path]


   .. py:attribute:: asa_file
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: current_stage
      :type:  str
      :value: ''



   .. py:attribute:: embed_data
      :type:  numpy.typing.NDArray[numpy.floating] | None
      :value: None



   .. py:attribute:: input_mode
      :type:  str
      :value: 'asa'



   .. py:attribute:: is_running
      :type:  bool
      :value: False



   .. py:attribute:: neuron_file
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: preclass
      :type:  str
      :value: 'none'



   .. py:attribute:: preclass_params
      :type:  dict[str, Any]


   .. py:attribute:: preprocess_method
      :type:  str
      :value: 'none'



   .. py:attribute:: preprocess_params
      :type:  dict[str, Any]


   .. py:attribute:: preset
      :type:  str
      :value: 'grid'



   .. py:attribute:: progress
      :type:  int
      :value: 0



   .. py:attribute:: traj_file
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: workdir
      :type:  pathlib.Path


