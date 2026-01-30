canns.analyzer.visualization.core.jupyter_utils
===============================================

.. py:module:: canns.analyzer.visualization.core.jupyter_utils

.. autoapi-nested-parse::

   Utilities for Jupyter notebook integration with matplotlib animations.



Functions
---------

.. autoapisummary::

   canns.analyzer.visualization.core.jupyter_utils.display_animation_in_jupyter
   canns.analyzer.visualization.core.jupyter_utils.is_jupyter_environment


Module Contents
---------------

.. py:function:: display_animation_in_jupyter(animation, format = 'html5')

   Display a matplotlib animation in a Jupyter notebook.

   :param animation: ``matplotlib.animation.FuncAnimation`` instance.
   :param format: Display format - ``"html5"`` (default) or ``"jshtml"``.

   :returns: ``IPython.display.HTML`` object if successful, otherwise ``None``.

   .. rubric:: Examples

   >>> import numpy as np
   >>> from matplotlib import pyplot as plt
   >>> from matplotlib.animation import FuncAnimation
   >>> from canns.analyzer.visualization.core.jupyter_utils import (
   ...     display_animation_in_jupyter,
   ...     is_jupyter_environment,
   ... )
   >>>
   >>> x = np.linspace(0, 2 * np.pi, 50)
   >>> fig, ax = plt.subplots()
   >>> (line,) = ax.plot([], [])
   >>>
   >>> def update(i):
   ...     line.set_data(x[: i + 1], np.sin(x[: i + 1]))
   ...     return (line,)
   >>>
   >>> anim = FuncAnimation(fig, update, frames=5, interval=50, blit=True)
   >>> if is_jupyter_environment():
   ...     _ = display_animation_in_jupyter(anim, format="jshtml")
   ... print(anim is not None)
   True


.. py:function:: is_jupyter_environment()

   Detect if code is running in a Jupyter notebook environment.

   :returns: True if running in a Jupyter notebook, False otherwise.
   :rtype: bool

   .. rubric:: Examples

   >>> from canns.analyzer.visualization.core.jupyter_utils import is_jupyter_environment
   >>> print(is_jupyter_environment() in {True, False})
   True


