src.canns.pipeline.asa_gui.analysis_modes
=========================================

.. py:module:: src.canns.pipeline.asa_gui.analysis_modes

.. autoapi-nested-parse::

   Analysis modes for ASA GUI.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/base/index
   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/batch_mode/index
   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/cohomap_mode/index
   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/cohospace_mode/index
   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/decode_mode/index
   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/fr_mode/index
   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/frm_mode/index
   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/gridscore_mode/index
   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/pathcompare_mode/index
   /autoapi/src/canns/pipeline/asa_gui/analysis_modes/tda_mode/index


Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.analysis_modes.AbstractAnalysisMode
   src.canns.pipeline.asa_gui.analysis_modes.BatchMode
   src.canns.pipeline.asa_gui.analysis_modes.CohoMapMode
   src.canns.pipeline.asa_gui.analysis_modes.CohoSpaceMode
   src.canns.pipeline.asa_gui.analysis_modes.DecodeMode
   src.canns.pipeline.asa_gui.analysis_modes.FRMMode
   src.canns.pipeline.asa_gui.analysis_modes.FRMode
   src.canns.pipeline.asa_gui.analysis_modes.GridScoreInspectMode
   src.canns.pipeline.asa_gui.analysis_modes.GridScoreMode
   src.canns.pipeline.asa_gui.analysis_modes.PathCompareMode
   src.canns.pipeline.asa_gui.analysis_modes.TDAMode


Functions
---------

.. autoapisummary::

   src.canns.pipeline.asa_gui.analysis_modes.get_analysis_modes


Package Contents
----------------

.. py:class:: AbstractAnalysisMode

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_preset(preset)

      Apply preset hints (grid/hd) to parameters.



   .. py:method:: apply_ranges(neuron_count, total_steps)

      Apply neuron/time ranges based on loaded data.



   .. py:method:: collect_params()
      :abstractmethod:


      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()
      :abstractmethod:


      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :type:  str


   .. py:attribute:: name
      :type:  str


.. py:class:: BatchMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'Batch'



   .. py:attribute:: name
      :value: 'batch'



.. py:class:: CohoMapMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: apply_preset(preset)


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'CohoMap (TDA + decode)'



   .. py:attribute:: name
      :value: 'cohomap'



.. py:class:: CohoSpaceMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: apply_ranges(neuron_count, total_steps)


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'CohoSpace / CohoScore'



   .. py:attribute:: name
      :value: 'cohospace'



.. py:class:: DecodeMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: apply_preset(preset)


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'Decode'



   .. py:attribute:: name
      :value: 'decode'



.. py:class:: FRMMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: apply_ranges(neuron_count, total_steps)


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'FRM (single neuron)'



   .. py:attribute:: name
      :value: 'frm'



.. py:class:: FRMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: apply_ranges(neuron_count, total_steps)


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'FR Heatmap'



   .. py:attribute:: name
      :value: 'fr'



.. py:class:: GridScoreInspectMode

   Bases: :py:obj:`GridScoreMode`


   .. py:attribute:: display_name
      :value: 'GridScore Inspect'



   .. py:attribute:: name
      :value: 'gridscore_inspect'



.. py:class:: GridScoreMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: apply_meta(meta)


   .. py:method:: apply_ranges(neuron_count, total_steps)


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'Grid Score (classic)'



   .. py:attribute:: name
      :value: 'gridscore'



.. py:class:: PathCompareMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'Path Compare (CohoMap required)'



   .. py:attribute:: name
      :value: 'pathcompare'



.. py:class:: TDAMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: apply_preset(preset)


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'TDA'



   .. py:attribute:: name
      :value: 'tda'



.. py:function:: get_analysis_modes()

