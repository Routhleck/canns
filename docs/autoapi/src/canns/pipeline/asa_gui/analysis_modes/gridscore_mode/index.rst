src.canns.pipeline.asa_gui.analysis_modes.gridscore_mode
========================================================

.. py:module:: src.canns.pipeline.asa_gui.analysis_modes.gridscore_mode

.. autoapi-nested-parse::

   GridScore analysis modes.



Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.analysis_modes.gridscore_mode.GridScoreInspectMode
   src.canns.pipeline.asa_gui.analysis_modes.gridscore_mode.GridScoreMode


Module Contents
---------------

.. py:class:: GridScoreInspectMode

   Bases: :py:obj:`GridScoreMode`


   .. py:attribute:: display_name
      :value: 'GridScore Inspect'



   .. py:attribute:: name
      :value: 'gridscore_inspect'



.. py:class:: GridScoreMode

   Bases: :py:obj:`src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode`


   .. py:method:: apply_language(lang)


   .. py:method:: apply_meta(meta)


   .. py:method:: apply_ranges(neuron_count, total_steps)


   .. py:method:: collect_params()


   .. py:method:: create_params_widget()


   .. py:attribute:: display_name
      :value: 'Grid Score (classic)'



   .. py:attribute:: name
      :value: 'gridscore'



