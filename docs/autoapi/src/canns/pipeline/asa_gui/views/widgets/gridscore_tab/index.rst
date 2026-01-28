src.canns.pipeline.asa_gui.views.widgets.gridscore_tab
======================================================

.. py:module:: src.canns.pipeline.asa_gui.views.widgets.gridscore_tab

.. autoapi-nested-parse::

   GridScore result viewer tab.



Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa_gui.views.widgets.gridscore_tab.GridScoreTab


Module Contents
---------------

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



