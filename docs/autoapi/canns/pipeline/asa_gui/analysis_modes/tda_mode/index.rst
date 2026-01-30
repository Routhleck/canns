canns.pipeline.asa_gui.analysis_modes.tda_mode
==============================================

.. py:module:: canns.pipeline.asa_gui.analysis_modes.tda_mode

.. autoapi-nested-parse::

   TDA analysis mode.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.analysis_modes.tda_mode.TDAMode


Module Contents
---------------

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



