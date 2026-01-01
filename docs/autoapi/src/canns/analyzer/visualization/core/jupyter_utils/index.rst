src.canns.analyzer.visualization.core.jupyter_utils
===================================================

.. py:module:: src.canns.analyzer.visualization.core.jupyter_utils

.. autoapi-nested-parse::

   Utilities for Jupyter notebook integration with matplotlib animations.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.core.jupyter_utils.display_animation_in_jupyter
   src.canns.analyzer.visualization.core.jupyter_utils.is_jupyter_environment


Module Contents
---------------

.. py:function:: display_animation_in_jupyter(animation, format = 'html5')

   Display a matplotlib animation in Jupyter notebook.

   Performance comparison (100 frames):
       - html5 (default): 1.3s, 134 KB - Fast encoding, small size, smooth playback
       - jshtml: 2.6s, 4837 KB - Slower, 36x larger, but works without FFmpeg

   :param animation: matplotlib.animation.FuncAnimation object
   :param format: Display format - 'html5' (default, MP4 video) or 'jshtml' (JS animation)

   :returns: IPython.display.HTML object if successful, None otherwise

   .. note::

      'html5' format requires FFmpeg to be installed. If FFmpeg is not available,
      it will automatically fall back to 'jshtml'.


.. py:function:: is_jupyter_environment()

   Detect if code is running in a Jupyter notebook environment.

   :returns: True if running in Jupyter/IPython notebook, False otherwise.
   :rtype: bool


