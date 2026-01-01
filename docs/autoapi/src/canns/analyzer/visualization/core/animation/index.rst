src.canns.analyzer.visualization.core.animation
===============================================

.. py:module:: src.canns.analyzer.visualization.core.animation

.. autoapi-nested-parse::

   High-performance animation framework with blitting and parallel rendering support.

   This module provides base classes and utilities for creating optimized matplotlib
   animations using blitting and optional parallel rendering for long animations.



Classes
-------

.. autoapisummary::

   src.canns.analyzer.visualization.core.animation.OptimizedAnimationBase


Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.core.animation.create_buffer
   src.canns.analyzer.visualization.core.animation.optimize_colormap
   src.canns.analyzer.visualization.core.animation.supports_blitting


Module Contents
---------------

.. py:class:: OptimizedAnimationBase(fig, ax, config = None)

   Bases: :py:obj:`abc.ABC`


   High-performance animation base class with blitting support.

   This abstract base class enforces best practices for matplotlib animations:
   - Artists are pre-created in create_artists()
   - Frame updates only modify data, never rebuild objects
   - Automatic blitting support detection
   - Optional parallel rendering for long animations

   Subclasses must implement:
   - create_artists(): Pre-create all artist objects with animated=True
   - update_frame(frame_idx): Update artist data and return modified artists

   Initialize the animation base.

   :param fig: Matplotlib figure
   :param ax: Matplotlib axes
   :param config: Animation configuration (uses defaults if None)


   .. py:method:: create_artists()
      :abstractmethod:


      Pre-create all artist objects for the animation.

      This method should:
      1. Create all plot objects (lines, scatter, images, etc.)
      2. Set animated=True for objects that will be updated
      3. Set initial data (can be empty with [], [])
      4. Return list of all animated artists

      :returns: List of artist objects that will be animated



   .. py:method:: init_func()

      Initialize animation (called by FuncAnimation).

      :returns: Tuple of all animated artists



   .. py:method:: render_animation(nframes, interval = None, repeat = True, save_path = None, **save_kwargs)

      Render the animation with automatic optimization selection.

      :param nframes: Total number of frames
      :param interval: Milliseconds between frames (computed from fps if None)
      :param repeat: Whether to loop the animation
      :param save_path: Path to save animation (None to skip saving)
      :param \*\*save_kwargs: Additional arguments for animation.save()

      :returns: FuncAnimation object



   .. py:method:: update_frame(frame_idx)
      :abstractmethod:


      Update artists for a specific frame.

      This method should:
      1. Compute data for the current frame
      2. Update artist data using set_data(), set_array(), etc.
      3. Return tuple of all modified artists

      Important: Do NOT call ax.clear() or recreate artists here!

      :param frame_idx: Index of the current frame

      :returns: Tuple of modified artist objects



   .. py:attribute:: artists
      :type:  list[matplotlib.artist.Artist]
      :value: []



   .. py:attribute:: ax


   .. py:attribute:: config


   .. py:attribute:: fig


.. py:function:: create_buffer(shape, dtype=np.float32)

   Pre-allocate a numpy buffer for efficient in-place updates.

   :param shape: Shape of the buffer array
   :param dtype: Data type (default: float32 for memory efficiency)

   :returns: Pre-allocated numpy array


.. py:function:: optimize_colormap(data, cmap_name = 'viridis', vmin = None, vmax = None)

   Pre-compute colormap normalization for efficient color mapping.

   :param data: Data array to normalize
   :param cmap_name: Name of the colormap
   :param vmin: Minimum value for normalization (computed if None)
   :param vmax: Maximum value for normalization (computed if None)

   :returns: Tuple of (colormap function, vmin, vmax)


.. py:function:: supports_blitting()

   Check if the current matplotlib backend supports blitting.

   :returns: True if blitting is supported, False otherwise


