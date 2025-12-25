src.canns.analyzer.visualization.jupyter_utils
==============================================

.. py:module:: src.canns.analyzer.visualization.jupyter_utils

.. autoapi-nested-parse::

   Utilities for Jupyter notebook integration with matplotlib animations.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.jupyter_utils.display_animation_in_jupyter
   src.canns.analyzer.visualization.jupyter_utils.is_jupyter_environment


Module Contents
---------------

.. py:function:: display_animation_in_jupyter(animation, format = 'jshtml')

   Display a matplotlib animation in Jupyter notebook using HTML/JavaScript.

   :param animation: matplotlib.animation.FuncAnimation object
   :param format: Display format - 'jshtml' (default) or 'html5' (video tag)

   :returns: IPython.display.HTML object if successful, None otherwise


.. py:function:: is_jupyter_environment()

   Detect if code is running in a Jupyter notebook environment.

   :returns: True if running in Jupyter/IPython notebook, False otherwise.
   :rtype: bool


