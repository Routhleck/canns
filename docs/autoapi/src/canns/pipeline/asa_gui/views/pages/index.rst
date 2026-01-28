src.canns.pipeline.asa_gui.views.pages
======================================

.. py:module:: src.canns.pipeline.asa_gui.views.pages

.. autoapi-nested-parse::

   Page views for ASA GUI.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/pipeline/asa_gui/views/pages/analysis_page/index
   /autoapi/src/canns/pipeline/asa_gui/views/pages/preprocess_page/index


Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.views.pages.AnalysisPage
   src.canns.pipeline.asa_gui.views.pages.PreprocessPage


Package Contents
----------------

.. py:class:: AnalysisPage(controller, worker_manager, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Page for running analyses and viewing results.


   .. py:method:: apply_language(lang)


   .. py:method:: load_state(state)


   .. py:attribute:: analysis_completed


.. py:class:: PreprocessPage(controller, worker_manager, parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Page for loading inputs and running preprocessing.


   .. py:method:: apply_language(lang)


   .. py:attribute:: preprocess_completed


