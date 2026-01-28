src.canns.pipeline.asa_gui.analysis_modes.base
==============================================

.. py:module:: src.canns.pipeline.asa_gui.analysis_modes.base

.. autoapi-nested-parse::

   Base classes for analysis modes.



Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.analysis_modes.base.AbstractAnalysisMode


Functions
---------

.. autoapisummary::

   src.canns.pipeline.asa_gui.analysis_modes.base.configure_form_layout


Module Contents
---------------

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


.. py:function:: configure_form_layout(form)

   Apply consistent spacing/alignment for analysis parameter forms.


