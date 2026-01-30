canns.pipeline.asa.state
========================

.. py:module:: canns.pipeline.asa.state

.. autoapi-nested-parse::

   State management for ASA TUI.

   This module provides centralized workflow state management with workdir-centric design.
   All file paths are stored relative to the working directory for portability.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa.state.WorkflowState


Functions
---------

.. autoapisummary::

   canns.pipeline.asa.state.check_cached_artifacts
   canns.pipeline.asa.state.get_preset_params
   canns.pipeline.asa.state.load_cached_result
   canns.pipeline.asa.state.relative_path
   canns.pipeline.asa.state.resolve_path
   canns.pipeline.asa.state.validate_files


Module Contents
---------------

.. py:class:: WorkflowState

   Centralized state for ASA analysis workflow.

   All file paths are relative to workdir for portability.


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



   .. py:attribute:: input_mode
      :type:  str
      :value: 'asa'



   .. py:attribute:: is_running
      :type:  bool
      :value: False



   .. py:attribute:: neuron_file
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: preprocess_method
      :type:  str
      :value: 'none'



   .. py:attribute:: preprocess_params
      :type:  dict[str, Any]


   .. py:attribute:: preset
      :type:  str
      :value: 'grid'



   .. py:attribute:: traj_file
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: workdir
      :type:  pathlib.Path


.. py:function:: check_cached_artifacts(state, stage)

   Check if stage artifacts exist and are valid.

   :param state: Current workflow state
   :param stage: Analysis stage name

   :returns: True if cached artifacts exist and are valid


.. py:function:: get_preset_params(preset)

   Load preset configurations for analysis.

   :param preset: Preset name ("grid", "hd", or "none")

   :returns: Dictionary of preset parameters


.. py:function:: load_cached_result(state, stage)

   Load cached results from previous run.

   :param state: Current workflow state
   :param stage: Analysis stage name

   :returns: Dictionary of cached data


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

   Check if required files exist.

   :param state: Current workflow state

   :returns: Tuple of (is_valid, error_message)


