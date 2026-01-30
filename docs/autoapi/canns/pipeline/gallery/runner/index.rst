canns.pipeline.gallery.runner
=============================

.. py:module:: canns.pipeline.gallery.runner

.. autoapi-nested-parse::

   Execution helpers for the model gallery TUI.



Classes
-------

.. autoapisummary::

   canns.pipeline.gallery.runner.GalleryResult
   canns.pipeline.gallery.runner.GalleryRunner


Module Contents
---------------

.. py:class:: GalleryResult

   Result from running a gallery analysis.


   .. py:attribute:: artifacts
      :type:  dict[str, pathlib.Path]


   .. py:attribute:: elapsed_time
      :type:  float
      :value: 0.0



   .. py:attribute:: error
      :type:  str | None
      :value: None



   .. py:attribute:: success
      :type:  bool


   .. py:attribute:: summary
      :type:  str


.. py:class:: GalleryRunner

   Runner for gallery model analyses.


   .. py:method:: run(model, analysis, model_params, analysis_params, output_dir, log_callback, progress_callback)
      :async:



