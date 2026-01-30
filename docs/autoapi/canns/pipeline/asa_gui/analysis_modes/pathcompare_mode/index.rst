canns.pipeline.asa_gui.analysis_modes.pathcompare_mode
======================================================

.. py:module:: canns.pipeline.asa_gui.analysis_modes.pathcompare_mode

.. autoapi-nested-parse::

   PathCompare analysis mode.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.analysis_modes.pathcompare_mode.PathCompareMode


Module Contents
---------------

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



