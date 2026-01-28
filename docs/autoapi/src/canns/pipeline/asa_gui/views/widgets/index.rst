src.canns.pipeline.asa_gui.views.widgets
========================================

.. py:module:: src.canns.pipeline.asa_gui.views.widgets

.. autoapi-nested-parse::

   Reusable widgets for ASA GUI.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/pipeline/asa_gui/views/widgets/artifacts_tab/index
   /autoapi/src/canns/pipeline/asa_gui/views/widgets/drop_zone/index
   /autoapi/src/canns/pipeline/asa_gui/views/widgets/file_list/index
   /autoapi/src/canns/pipeline/asa_gui/views/widgets/gridscore_tab/index
   /autoapi/src/canns/pipeline/asa_gui/views/widgets/help_dialog/index
   /autoapi/src/canns/pipeline/asa_gui/views/widgets/image_tab/index
   /autoapi/src/canns/pipeline/asa_gui/views/widgets/image_viewer/index
   /autoapi/src/canns/pipeline/asa_gui/views/widgets/log_box/index
   /autoapi/src/canns/pipeline/asa_gui/views/widgets/pathcompare_tab/index
   /autoapi/src/canns/pipeline/asa_gui/views/widgets/popup_combo/index


Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.views.widgets.ArtifactsTab
   src.canns.pipeline.asa_gui.views.widgets.DropZone
   src.canns.pipeline.asa_gui.views.widgets.FileList
   src.canns.pipeline.asa_gui.views.widgets.GridScoreTab
   src.canns.pipeline.asa_gui.views.widgets.ImageTab
   src.canns.pipeline.asa_gui.views.widgets.ImageViewer
   src.canns.pipeline.asa_gui.views.widgets.LogBox
   src.canns.pipeline.asa_gui.views.widgets.PathCompareTab


Package Contents
----------------

.. py:class:: ArtifactsTab(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   .. py:method:: set_artifacts(artifacts)


   .. py:attribute:: btn_open_folder


   .. py:attribute:: files_list


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


.. py:class:: FileList(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QListWidget`


.. py:class:: GridScoreTab(title = 'Grid Score')

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   GridScore viewer with distribution and neuron inspector.


   .. py:method:: clear()


   .. py:method:: get_meta_params()


   .. py:method:: has_scores()


   .. py:method:: load_gridscore_npz(path)


   .. py:method:: set_autocorr_image(path)


   .. py:method:: set_distribution_image(path)


   .. py:method:: set_enabled(enabled)


   .. py:method:: set_status(msg)


   .. py:attribute:: auto_viewer


   .. py:attribute:: btn_next


   .. py:attribute:: btn_prev


   .. py:attribute:: btn_show


   .. py:attribute:: dist_header


   .. py:attribute:: dist_viewer


   .. py:attribute:: insp_header


   .. py:attribute:: inspectRequested


   .. py:attribute:: lbl_metrics


   .. py:attribute:: lbl_status


   .. py:attribute:: neuron_id


   .. py:attribute:: sort_combo


   .. py:attribute:: title
      :value: 'Grid Score'



.. py:class:: ImageTab(title)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   .. py:method:: set_image(path)


   .. py:attribute:: viewer


.. py:class:: ImageViewer(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsView`


   Image viewer with fit-to-view, zoom (wheel), and pan (drag).


   .. py:method:: mouseDoubleClickEvent(event)


   .. py:method:: resizeEvent(event)


   .. py:method:: set_image(path)


   .. py:method:: wheelEvent(event)


.. py:class:: LogBox(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QTextEdit`


   .. py:method:: log(msg)


.. py:class:: PathCompareTab(title = 'Path Compare')

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   .. py:method:: set_animation(path)


   .. py:method:: set_animation_progress(pct)


   .. py:method:: set_artifacts(png_path, gif_path)


   .. py:attribute:: anim_progress


   .. py:attribute:: btn_open_anim


   .. py:attribute:: gif_label


   .. py:attribute:: gif_view


   .. py:attribute:: header


   .. py:attribute:: png_label


   .. py:attribute:: png_view


   .. py:attribute:: splitter


   .. py:attribute:: title
      :value: 'Path Compare'



