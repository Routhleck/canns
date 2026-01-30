canns.pipeline.asa_gui.core.state
=================================

.. py:module:: canns.pipeline.asa_gui.core.state

.. autoapi-nested-parse::

   State management for ASA GUI.

   This module provides centralized workflow state management with Qt signals
   for reactive UI updates. All file paths are stored relative to the working
   directory for portability.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.core.state.StateManager
   canns.pipeline.asa_gui.core.state.WorkflowState


Functions
---------

.. autoapisummary::

   canns.pipeline.asa_gui.core.state.relative_path
   canns.pipeline.asa_gui.core.state.resolve_path
   canns.pipeline.asa_gui.core.state.validate_files
   canns.pipeline.asa_gui.core.state.validate_preprocessing


Module Contents
---------------

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


.. py:function:: relative_path(state, path)

   Convert absolute path to workdir-relative path.

   :param state: Current workflow state
   :param path: Absolute path to convert

   :returns: Path relative to workdir


.. py:function:: resolve_path(state, path)

   Convert relative path to absolute path.

   :param state: Current workflow state
   :param path: Relative path to convert

   :returns: Absolute path or None if path is None


.. py:function:: validate_files(state)

   Check if required files exist and are valid.

   :param state: Current workflow state

   :returns: Tuple of (is_valid, error_message)


.. py:function:: validate_preprocessing(state)

   Check if preprocessing is complete.

   :param state: Current workflow state

   :returns: Tuple of (is_valid, error_message)


