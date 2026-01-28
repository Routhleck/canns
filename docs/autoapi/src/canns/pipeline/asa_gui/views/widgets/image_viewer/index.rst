src.canns.pipeline.asa_gui.views.widgets.image_viewer
=====================================================

.. py:module:: src.canns.pipeline.asa_gui.views.widgets.image_viewer

.. autoapi-nested-parse::

   Image viewer widget with zoom and pan.



Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.views.widgets.image_viewer.ImageViewer


Module Contents
---------------

.. py:class:: ImageViewer(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsView`


   Image viewer with fit-to-view, zoom (wheel), and pan (drag).


   .. py:method:: mouseDoubleClickEvent(event)


   .. py:method:: resizeEvent(event)


   .. py:method:: set_image(path)


   .. py:method:: wheelEvent(event)


