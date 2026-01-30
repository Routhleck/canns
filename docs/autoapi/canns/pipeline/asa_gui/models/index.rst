canns.pipeline.asa_gui.models
=============================

.. py:module:: canns.pipeline.asa_gui.models

.. autoapi-nested-parse::

   Data models for ASA GUI.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/canns/pipeline/asa_gui/models/config/index
   /autoapi/canns/pipeline/asa_gui/models/job/index
   /autoapi/canns/pipeline/asa_gui/models/presets/index


Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.models.AnalysisConfig
   canns.pipeline.asa_gui.models.JobResult
   canns.pipeline.asa_gui.models.JobSpec


Functions
---------

.. autoapisummary::

   canns.pipeline.asa_gui.models.get_preset_params


Package Contents
----------------

.. py:class:: AnalysisConfig

   Simple container for analysis parameters.


   .. py:attribute:: mode
      :type:  str
      :value: 'tda'



   .. py:attribute:: params
      :type:  dict[str, Any]


.. py:class:: JobResult

   Result container for pipeline execution.


   .. py:attribute:: artifacts
      :type:  dict[str, pathlib.Path]


   .. py:attribute:: errors
      :type:  list[str]
      :value: []



   .. py:attribute:: ok
      :type:  bool


   .. py:attribute:: out_dir
      :type:  pathlib.Path


   .. py:attribute:: summary
      :type:  dict[str, Any]


.. py:class:: JobSpec

   Inputs and parameters for a single analysis run.


   .. py:attribute:: asa_file
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: input_mode
      :type:  str


   .. py:attribute:: neuron_file
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: out_dir
      :type:  pathlib.Path


   .. py:attribute:: params
      :type:  dict[str, Any]


   .. py:attribute:: preset
      :type:  str


   .. py:attribute:: traj_file
      :type:  pathlib.Path | None
      :value: None



.. py:function:: get_preset_params(preset)

   Return default parameter overrides for a preset.


