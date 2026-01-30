canns.pipeline.asa_gui.analysis_modes
=====================================

.. py:module:: canns.pipeline.asa_gui.analysis_modes

.. autoapi-nested-parse::

   Analysis modes for ASA GUI.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/canns/pipeline/asa_gui/analysis_modes/base/index
   /autoapi/canns/pipeline/asa_gui/analysis_modes/batch_mode/index
   /autoapi/canns/pipeline/asa_gui/analysis_modes/cohomap_mode/index
   /autoapi/canns/pipeline/asa_gui/analysis_modes/cohospace_mode/index
   /autoapi/canns/pipeline/asa_gui/analysis_modes/decode_mode/index
   /autoapi/canns/pipeline/asa_gui/analysis_modes/fr_mode/index
   /autoapi/canns/pipeline/asa_gui/analysis_modes/frm_mode/index
   /autoapi/canns/pipeline/asa_gui/analysis_modes/gridscore_mode/index
   /autoapi/canns/pipeline/asa_gui/analysis_modes/pathcompare_mode/index
   /autoapi/canns/pipeline/asa_gui/analysis_modes/tda_mode/index


Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.analysis_modes.AbstractAnalysisMode
   canns.pipeline.asa_gui.analysis_modes.BatchMode
   canns.pipeline.asa_gui.analysis_modes.CohoMapMode
   canns.pipeline.asa_gui.analysis_modes.CohoSpaceMode
   canns.pipeline.asa_gui.analysis_modes.DecodeMode
   canns.pipeline.asa_gui.analysis_modes.FRMMode
   canns.pipeline.asa_gui.analysis_modes.FRMode
   canns.pipeline.asa_gui.analysis_modes.GridScoreInspectMode
   canns.pipeline.asa_gui.analysis_modes.GridScoreMode
   canns.pipeline.asa_gui.analysis_modes.PathCompareMode
   canns.pipeline.asa_gui.analysis_modes.TDAMode


Functions
---------

.. autoapisummary::

   canns.pipeline.asa_gui.analysis_modes.get_analysis_modes


Package Contents
----------------

.. py:class:: AbstractAnalysisMode

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_language(lang)

      Apply localized tooltips/text.



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

   Bases: :py:obj:`canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: collect_params()

      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()

      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :value: 'Batch'



   .. py:attribute:: name
      :value: 'batch'



.. py:class:: CohoMapMode

   Bases: :py:obj:`canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_language(lang)

      Apply localized tooltips/text.



   .. py:method:: apply_preset(preset)

      Apply preset hints (grid/hd) to parameters.



   .. py:method:: collect_params()

      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()

      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :value: 'CohoMap (TDA + decode)'



   .. py:attribute:: name
      :value: 'cohomap'



.. py:class:: CohoSpaceMode

   Bases: :py:obj:`canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_language(lang)

      Apply localized tooltips/text.



   .. py:method:: apply_ranges(neuron_count, total_steps)

      Apply neuron/time ranges based on loaded data.



   .. py:method:: collect_params()

      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()

      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :value: 'CohoSpace / CohoScore'



   .. py:attribute:: name
      :value: 'cohospace'



.. py:class:: DecodeMode

   Bases: :py:obj:`canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_language(lang)

      Apply localized tooltips/text.



   .. py:method:: apply_preset(preset)

      Apply preset hints (grid/hd) to parameters.



   .. py:method:: collect_params()

      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()

      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :value: 'Decode'



   .. py:attribute:: name
      :value: 'decode'



.. py:class:: FRMMode

   Bases: :py:obj:`canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_language(lang)

      Apply localized tooltips/text.



   .. py:method:: apply_ranges(neuron_count, total_steps)

      Apply neuron/time ranges based on loaded data.



   .. py:method:: collect_params()

      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()

      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :value: 'FRM (single neuron)'



   .. py:attribute:: name
      :value: 'frm'



.. py:class:: FRMode

   Bases: :py:obj:`canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_language(lang)

      Apply localized tooltips/text.



   .. py:method:: apply_ranges(neuron_count, total_steps)

      Apply neuron/time ranges based on loaded data.



   .. py:method:: collect_params()

      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()

      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :value: 'FR Heatmap'



   .. py:attribute:: name
      :value: 'fr'



.. py:class:: GridScoreInspectMode

   Bases: :py:obj:`GridScoreMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: display_name
      :value: 'GridScore Inspect'



   .. py:attribute:: name
      :value: 'gridscore_inspect'



.. py:class:: GridScoreMode

   Bases: :py:obj:`canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_language(lang)

      Apply localized tooltips/text.



   .. py:method:: apply_meta(meta)


   .. py:method:: apply_ranges(neuron_count, total_steps)

      Apply neuron/time ranges based on loaded data.



   .. py:method:: collect_params()

      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()

      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :value: 'Grid Score (classic)'



   .. py:attribute:: name
      :value: 'gridscore'



.. py:class:: PathCompareMode

   Bases: :py:obj:`canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_language(lang)

      Apply localized tooltips/text.



   .. py:method:: collect_params()

      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()

      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :value: 'Path Compare (CohoMap required)'



   .. py:attribute:: name
      :value: 'pathcompare'



.. py:class:: TDAMode

   Bases: :py:obj:`canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: apply_language(lang)

      Apply localized tooltips/text.



   .. py:method:: apply_preset(preset)

      Apply preset hints (grid/hd) to parameters.



   .. py:method:: collect_params()

      Collect parameters from the widget into a dict.



   .. py:method:: create_params_widget()

      Create and return the parameter editor widget.



   .. py:attribute:: display_name
      :value: 'TDA'



   .. py:attribute:: name
      :value: 'tda'



.. py:function:: get_analysis_modes()

