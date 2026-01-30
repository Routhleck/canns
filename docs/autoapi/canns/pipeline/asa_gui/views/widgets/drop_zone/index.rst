canns.pipeline.asa_gui.views.widgets.drop_zone
==============================================

.. py:module:: canns.pipeline.asa_gui.views.widgets.drop_zone

.. autoapi-nested-parse::

   Drag-and-drop file input widget.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa_gui.views.widgets.drop_zone.DropZone


Module Contents
---------------

.. py:class:: DropZone(title, hint = '', parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QFrame`


   Simple drag-and-drop target for file paths.


   .. py:method:: dragEnterEvent(event)


   .. py:method:: dragLeaveEvent(event)


   .. py:method:: dropEvent(event)


   .. py:method:: path()


   .. py:method:: set_empty_text(text)


   .. py:method:: set_hint(hint)


   .. py:method:: set_path(path)


   .. py:method:: set_title(title)


   .. py:attribute:: fileDropped


