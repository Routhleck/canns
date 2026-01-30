canns.pipeline.asa_gui.analysis_modes.gridscore_mode
====================================================

.. py:module:: canns.pipeline.asa_gui.analysis_modes.gridscore_mode

.. autoapi-nested-parse::

   GridScore analysis modes.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.analysis_modes.gridscore_mode.GridScoreInspectMode
   canns.pipeline.asa_gui.analysis_modes.gridscore_mode.GridScoreMode


Module Contents
---------------

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



