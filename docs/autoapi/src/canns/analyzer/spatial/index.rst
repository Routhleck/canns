src.canns.analyzer.spatial
==========================

.. py:module:: src.canns.analyzer.spatial

.. autoapi-nested-parse::

   Spatial analysis utilities for neural activity data.

   This module provides functions for analyzing spatial patterns in neural data,
   particularly for computing firing fields and spatial smoothing operations.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.spatial.compute_firing_field
   src.canns.analyzer.spatial.gaussian_smooth_heatmaps


Module Contents
---------------

.. py:function:: compute_firing_field(A, positions, width, height, M, K)

   Compute spatial firing fields for neural population activity.

   This function bins neural activity into a 2D spatial grid based on the
   animal's position, creating a heatmap for each neuron showing where it
   fires most strongly. Uses Numba JIT compilation for high performance.

   :param A: Neural activity array of shape (T, N) where T is the
             number of time steps and N is the number of neurons.
   :type A: np.ndarray
   :param positions: Position data of shape (T, 2) containing
                     (x, y) coordinates at each time step.
   :type positions: np.ndarray
   :param width: Width of the spatial environment.
   :type width: float
   :param height: Height of the spatial environment.
   :type height: float
   :param M: Number of bins along the width dimension.
   :type M: int
   :param K: Number of bins along the height dimension.
   :type K: int

   :returns:

             Heatmaps array of shape (N, M, K) containing the average
                 firing rate of each neuron in each spatial bin.
   :rtype: np.ndarray

   .. rubric:: Example

   >>> activity = np.random.rand(1000, 30)  # 1000 timesteps, 30 neurons
   >>> positions = np.random.rand(1000, 2) * 5.0  # Random walk in 5x5 space
   >>> heatmaps = compute_firing_field(activity, positions, 5.0, 5.0, 50, 50)
   >>> heatmaps.shape
   (30, 50, 50)


.. py:function:: gaussian_smooth_heatmaps(heatmaps, sigma = 1.0)

   Apply Gaussian smoothing to spatial heatmaps without mixing channels.

   This function applies Gaussian filtering to each heatmap independently,
   preserving zero values (unvisited spatial bins) and only smoothing regions
   with activity.

   :param heatmaps: Array of shape (N, M, K) where N is the number
                    of neurons/channels and (M, K) is the spatial grid size.
   :type heatmaps: np.ndarray
   :param sigma: Standard deviation for Gaussian kernel.
                 Defaults to 1.0.
   :type sigma: float, optional

   :returns:

             Smoothed heatmaps with the same shape as input. Zero values
                 in the original heatmaps are preserved.
   :rtype: np.ndarray

   .. rubric:: Example

   >>> heatmaps = np.random.rand(30, 50, 50)
   >>> smoothed = gaussian_smooth_heatmaps(heatmaps, sigma=1.5)
   >>> smoothed.shape
   (30, 50, 50)


