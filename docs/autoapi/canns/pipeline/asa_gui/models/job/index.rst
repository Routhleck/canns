canns.pipeline.asa_gui.models.job
=================================

.. py:module:: canns.pipeline.asa_gui.models.job

.. autoapi-nested-parse::

   Job specification and results for ASA GUI.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.models.job.JobResult
   canns.pipeline.asa_gui.models.job.JobSpec


Module Contents
---------------

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



