src.canns.pipeline.gallery.state
================================

.. py:module:: src.canns.pipeline.gallery.state

.. autoapi-nested-parse::

   State management for the model gallery TUI.



Attributes
----------

.. autoapisummary::

   src.canns.pipeline.gallery.state.MODEL_ANALYSIS_OPTIONS


Classes
-------

.. autoapisummary::

   src.canns.pipeline.gallery.state.GalleryState


Functions
---------

.. autoapisummary::

   src.canns.pipeline.gallery.state.get_analysis_options
   src.canns.pipeline.gallery.state.get_default_analysis


Module Contents
---------------

.. py:class:: GalleryState

   Centralized state for the model gallery TUI.


   .. py:attribute:: analysis
      :type:  str
      :value: 'connectivity'



   .. py:attribute:: artifacts
      :type:  dict[str, pathlib.Path]


   .. py:attribute:: model
      :type:  str
      :value: 'cann1d'



   .. py:attribute:: workdir
      :type:  pathlib.Path


.. py:function:: get_analysis_options(model)

   Return analysis options for the selected model.


.. py:function:: get_default_analysis(model)

   Return the default analysis key for the selected model.


.. py:data:: MODEL_ANALYSIS_OPTIONS
   :type:  dict[str, list[tuple[str, str]]]

