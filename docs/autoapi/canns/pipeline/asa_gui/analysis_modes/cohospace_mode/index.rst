canns.pipeline.asa_gui.analysis_modes.cohospace_mode
====================================================

.. py:module:: canns.pipeline.asa_gui.analysis_modes.cohospace_mode

.. autoapi-nested-parse::

   CohoSpace analysis mode.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.analysis_modes.cohospace_mode.CohoSpaceMode


Module Contents
---------------

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



