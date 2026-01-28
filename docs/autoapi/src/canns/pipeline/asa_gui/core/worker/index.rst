src.canns.pipeline.asa_gui.core.worker
======================================

.. py:module:: src.canns.pipeline.asa_gui.core.worker

.. autoapi-nested-parse::

   Async worker infrastructure for ASA GUI.

   This module provides QThread-based workers for running analysis
   in the background without blocking the UI.



Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.core.worker.AnalysisWorker
   src.canns.pipeline.asa_gui.core.worker.WorkerManager


Module Contents
---------------

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



