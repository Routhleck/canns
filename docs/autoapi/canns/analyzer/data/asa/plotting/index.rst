canns.analyzer.data.asa.plotting
================================

.. py:module:: canns.analyzer.data.asa.plotting


Functions
---------

.. autoapisummary::

   canns.analyzer.data.asa.plotting.plot_2d_bump_on_manifold
   canns.analyzer.data.asa.plotting.plot_3d_bump_on_torus
   canns.analyzer.data.asa.plotting.plot_cohomap_scatter
   canns.analyzer.data.asa.plotting.plot_cohomap_scatter_multi
   canns.analyzer.data.asa.plotting.plot_path_compare_1d
   canns.analyzer.data.asa.plotting.plot_path_compare_2d
   canns.analyzer.data.asa.plotting.plot_projection


Module Contents
---------------

.. py:function:: plot_2d_bump_on_manifold(decoding_result, spike_data, save_path = None, fps = 20, show = True, mode = 'fast', window_size = 10, frame_step = 5, numangsint = 20, figsize = (8, 6), show_progress = False, config = None, render_backend = 'auto', output_dpi = 150, render_workers = None)

   Create 2D projection animation of CANN2D bump activity with full blitting support.

   This function provides a fast 2D heatmap visualization as an alternative to the
   3D torus animation. It achieves 10-20x speedup using matplotlib blitting
   optimization, making it ideal for rapid prototyping and daily analysis.

   :param decoding_result: Decoding results containing coords and times (dict or file path)
   :param spike_data: Dictionary containing spike train data
   :param save_path: Path to save animation (None to skip saving)
   :param fps: Frames per second
   :param show: Whether to display the animation
   :param mode: Visualization mode - 'fast' for 2D heatmap (default), '3d' falls back to 3D
   :param window_size: Time window for activity aggregation
   :param frame_step: Time step between frames
   :param numangsint: Number of angular bins for spatial discretization
   :param figsize: Figure size (width, height) in inches
   :param show_progress: Show progress bar during processing

   :returns: * *matplotlib.animation.FuncAnimation | None* -- Animation object (or None in Jupyter when showing).
             * *Raises* -- ProcessingError: If mode is invalid or animation generation fails

   .. rubric:: Examples

   >>> # Fast 2D visualization (recommended for daily use)
   >>> ani = plot_2d_bump_on_manifold(
   ...     decoding_result, spike_data,
   ...     save_path='bump_2d.mp4', mode='fast'
   ... )
   >>> # For publication-ready 3D visualization, use mode='3d'
   >>> ani = plot_2d_bump_on_manifold(
   ...     decoding_result, spike_data, mode='3d'
   ... )


.. py:function:: plot_3d_bump_on_torus(decoding_result, spike_data, config = None, save_path = None, numangsint = 51, r1 = 1.5, r2 = 1.0, window_size = 300, frame_step = 5, n_frames = 20, fps = 5, show_progress = True, show = True, figsize = (8, 8), render_backend = 'auto', output_dpi = 150, render_workers = None, **kwargs)

   Visualize the movement of the neural activity bump on a torus using matplotlib animation.

   This function follows the canns.analyzer.plotting patterns for animation generation
   with progress tracking and proper resource cleanup.

   :param decoding_result: dict or str
                           Dictionary containing decoding results with 'coordsbox' and 'times_box' keys,
                           or path to .npz file containing these results
   :param spike_data: dict, optional
                      Spike data dictionary containing spike information
   :param config: PlotConfig, optional
                  Configuration object for unified plotting parameters
   :param \*\*kwargs: backward compatibility parameters
   :param save_path: str, optional
                     Path to save the animation (e.g., 'animation.gif' or 'animation.mp4')
   :param numangsint: int
                      Grid resolution for the torus surface
   :param r1: float
              Major radius of the torus
   :param r2: float
              Minor radius of the torus
   :param window_size: int
                       Time window (in number of time points) for each frame
   :param frame_step: int
                      Step size to slide the time window between frames
   :param n_frames: int
                    Total number of frames in the animation
   :param fps: int
               Frames per second for the output animation
   :param show_progress: bool
                         Whether to show progress bar during generation
   :param show: bool
                Whether to display the animation
   :param figsize: tuple[int, int]
                   Figure size for the animation

   :returns: The animation object, or None when shown in Jupyter.
   :rtype: matplotlib.animation.FuncAnimation | None

   .. rubric:: Examples

   >>> ani = plot_3d_bump_on_torus(decoding, spike_data, show=False)  # doctest: +SKIP


.. py:function:: plot_cohomap_scatter(decoding_result, position_data, config = None, save_path = None, show = False, figsize = (10, 4), dpi = 300, subsample = 10)

   Visualize CohoMap 1.0: decoded circular coordinates mapped onto spatial trajectory.

   Creates a two-panel visualization showing how the two decoded circular coordinates
   vary across the animal's spatial trajectory. Each panel displays the spatial path
   colored by the cosine of one circular coordinate dimension.

   :param decoding_result: dict
                           Dictionary from decode_circular_coordinates() containing:
                           - 'coordsbox': decoded coordinates for box timepoints (n_times x n_dims)
                           - 'times_box': time indices for coordsbox
   :param position_data: dict
                         Position data containing 'x' and 'y' arrays for spatial coordinates
   :param save_path: str, optional
                     Path to save the visualization. If None, no save performed
   :param show: bool, default=False
                Whether to display the visualization
   :param figsize: tuple[int, int], default=(10, 4)
                   Figure size (width, height) in inches
   :param dpi: int, default=300
               Resolution for saved figure
   :param subsample: int, default=10
                     Subsampling interval for plotting (plot every Nth timepoint)

   :returns: * *matplotlib.figure.Figure* -- The matplotlib figure object.
             * *Raises* -- KeyError : If required keys are missing from input dictionaries
               ValueError : If data dimensions are inconsistent
               IndexError : If time indices are out of bounds

   .. rubric:: Examples

   >>> # Decode coordinates
   >>> decoding = decode_circular_coordinates(persistence_result, spike_data)
   >>> # Visualize with trajectory data
   >>> fig = plot_cohomap_scatter(
   ...     decoding,
   ...     position_data={'x': xx, 'y': yy},
   ...     save_path='cohomap.png',
   ...     show=True
   ... )


.. py:function:: plot_cohomap_scatter_multi(decoding_result, position_data, config = None, save_path = None, show = False, figsize = (10, 4), dpi = 300, subsample = 10)

   Visualize CohoMap with N-dimensional decoded coordinates.

   Each subplot shows the spatial trajectory colored by ``cos(coord_i)`` for a single
   circular coordinate.

   :param decoding_result: Dictionary containing ``coordsbox`` and ``times_box``.
   :type decoding_result: dict
   :param position_data: Position data containing ``x`` and ``y`` arrays.
   :type position_data: dict
   :param config: Plot configuration for styling, saving, and showing.
   :type config: PlotConfig, optional
   :param save_path: Path to save the figure.
   :type save_path: str, optional
   :param show: Whether to show the figure.
   :type show: bool
   :param figsize: Figure size in inches.
   :type figsize: tuple[int, int]
   :param dpi: Save DPI.
   :type dpi: int
   :param subsample: Subsample stride for plotting.
   :type subsample: int

   :returns: The created figure.
   :rtype: matplotlib.figure.Figure

   .. rubric:: Examples

   >>> fig = plot_cohomap_scatter_multi(decoding, {"x": xx, "y": yy}, show=False)  # doctest: +SKIP


.. py:function:: plot_path_compare_1d(x, y, coords, config = None, *, title = 'Path Compare (1D)', figsize = (12, 5), show = True, save_path = None)

   Plot physical path vs decoded coho-space path (1D) side-by-side.


.. py:function:: plot_path_compare_2d(x, y, coords, config = None, *, title = 'Path Compare', figsize = (12, 5), show = True, save_path = None)

   Plot physical path vs decoded coho-space path (2D) side-by-side.

   :param x: Physical position arrays of shape (T,).
   :type x: np.ndarray
   :param y: Physical position arrays of shape (T,).
   :type y: np.ndarray
   :param coords: Decoded circular coordinates, shape (T, 2) or (T, 2+).
   :type coords: np.ndarray
   :param config: Plot configuration. If None, a default config is created.
   :type config: PlotConfig, optional
   :param title: Backward-compatibility parameters.
   :type title: optional
   :param figsize: Backward-compatibility parameters.
   :type figsize: optional
   :param show: Backward-compatibility parameters.
   :type show: optional
   :param save_path: Backward-compatibility parameters.
   :type save_path: optional

   :returns: Figure and axes array.
   :rtype: (Figure, ndarray)

   .. rubric:: Examples

   >>> fig, axes = plot_path_compare_2d(x, y, coords, show=False)  # doctest: +SKIP


.. py:function:: plot_projection(reduce_func, embed_data, config = None, title='Projection (3D)', xlabel='Component 1', ylabel='Component 2', zlabel='Component 3', save_path=None, show=True, dpi=300, figsize=(10, 8), **kwargs)

   Plot a 3D projection of the embedded data.

   :param reduce_func (callable):
   :type reduce_func (callable): Function to reduce the dimensionality of the data.
   :param embed_data (ndarray):
   :type embed_data (ndarray): Data to be projected.
   :param config (PlotConfig:
   :type config (PlotConfig: Configuration object for unified plotting parameters
   :param optional):
   :type optional): Configuration object for unified plotting parameters
   :param \*\*kwargs:
   :type \*\*kwargs: backward compatibility parameters
   :param title (str):
   :type title (str): Title of the plot.
   :param xlabel (str):
   :type xlabel (str): Label for the x-axis.
   :param ylabel (str):
   :type ylabel (str): Label for the y-axis.
   :param zlabel (str):
   :type zlabel (str): Label for the z-axis.
   :param save_path (str:
   :type save_path (str: Path to save the plot. If None, plot will not be saved.
   :param optional):
   :type optional): Path to save the plot. If None, plot will not be saved.
   :param show (bool):
   :type show (bool): Whether to display the plot.
   :param dpi (int):
   :type dpi (int): Dots per inch for saving the figure.
   :param figsize (tuple):
   :type figsize (tuple): Size of the figure.

   :returns: The created figure.
   :rtype: matplotlib.figure.Figure

   .. rubric:: Examples

   >>> fig = plot_projection(reduce_func, embed_data, show=False)  # doctest: +SKIP


