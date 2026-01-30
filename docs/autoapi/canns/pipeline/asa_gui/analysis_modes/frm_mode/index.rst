canns.pipeline.asa_gui.analysis_modes.frm_mode
==============================================

.. py:module:: canns.pipeline.asa_gui.analysis_modes.frm_mode

.. autoapi-nested-parse::

   FRM analysis mode.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.analysis_modes.frm_mode.FRMMode


Module Contents
---------------

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



