src.canns.analyzer.visualization.energy_plots
=============================================

.. py:module:: src.canns.analyzer.visualization.energy_plots

.. autoapi-nested-parse::

   Energy landscape visualization utilities.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.energy_plots.energy_landscape_1d_animation
   src.canns.analyzer.visualization.energy_plots.energy_landscape_1d_static
   src.canns.analyzer.visualization.energy_plots.energy_landscape_2d_animation
   src.canns.analyzer.visualization.energy_plots.energy_landscape_2d_static


Module Contents
---------------

.. py:function:: energy_landscape_1d_animation(data_sets, time_steps_per_second = None, config = None, *, fps = 30, title = 'Evolving 1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', figsize = (10, 6), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create an animation of an evolving 1D energy landscape.

   The docstring intentionally preserves the guidance from the previous
   implementation so existing callers can rely on the same parameter
   explanations.

   :param data_sets: Dictionary whose keys are legend labels and values are
                     ``(x_data, y_data)`` tuples where ``y_data`` is shaped as
                     ``(time, state)``.
   :param time_steps_per_second: Number of simulation time steps per second of
                                 wall-clock time (e.g., ``1/dt``).
   :param config: Optional :class:`PlotConfig` with shared styling overrides.
   :param fps: Frames per second to render in the resulting animation.
   :param title: Title used when ``config`` is not provided.
   :param xlabel: X-axis label used when ``config`` is not provided.
   :param ylabel: Y-axis label used when ``config`` is not provided.
   :param figsize: Figure size passed to Matplotlib when building the canvas.
   :param grid: Whether to overlay a grid on the animation axes.
   :param repeat: Whether the animation should loop once it finishes.
   :param save_path: Optional path to persist the animation (``.gif`` / ``.mp4``).
   :param show: Whether to display the animation interactively.
   :param show_progress_bar: Whether to show a ``tqdm`` progress bar when saving.
   :param \*\*kwargs: Further keyword arguments passed through to ``ax.plot``.

   :returns: The constructed animation.
   :rtype: ``matplotlib.animation.FuncAnimation``


.. py:function:: energy_landscape_1d_static(data_sets, config = None, *, title = '1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', show_legend = True, figsize = (10, 6), grid = False, save_path = None, show = True, **kwargs)

   Plot a 1D static energy landscape using Matplotlib.

   This mirrors the long-form description from the pre-reorganisation module so
   existing documentation references stay accurate. The function accepts a
   dictionary of datasets, plotting each curve on the same set of axes while
   honouring the ``PlotConfig`` defaults callers relied on previously.

   :param data_sets: Mapping of series labels to ``(x, y)`` tuples representing
                     the energy curve to draw.
   :param config: Optional :class:`PlotConfig` carrying shared styling.
   :param title: Plot title when no config override is supplied.
   :param xlabel: X-axis label when no config override is supplied.
   :param ylabel: Y-axis label when no config override is supplied.
   :param show_legend: Whether to display the legend for labelled curves.
   :param figsize: Figure size forwarded to Matplotlib when creating the axes.
   :param grid: Whether to enable a grid background.
   :param save_path: Optional path for persisting the plot to disk.
   :param show: Whether to display the generated figure.
   :param \*\*kwargs: Additional keyword arguments forwarded to ``ax.plot``.

   :returns: The created figure and axes handles.
   :rtype: Tuple[plt.Figure, plt.Axes]


.. py:function:: energy_landscape_2d_animation(zs_data, config = None, *, time_steps_per_second = None, fps = 30, title = 'Evolving 2D Landscape', xlabel = 'X-Index', ylabel = 'Y-Index', clabel = 'Value', figsize = (8, 7), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create an animation of an evolving 2D landscape.

   The long-form description mirrors the previous implementation to maintain
   backwards-compatible documentation for downstream users.

   :param zs_data: Array of shape ``(timesteps, dim_y, dim_x)`` describing the
                   landscape at each simulation step.
   :param config: Optional :class:`PlotConfig` carrying display preferences.
   :param time_steps_per_second: Number of simulation steps per second of
                                 simulated time; required unless encoded in ``config``.
   :param fps: Frames per second in the generated animation.
   :param title: Title used when ``config`` is not provided.
   :param xlabel: X-axis label used when ``config`` is not provided.
   :param ylabel: Y-axis label used when ``config`` is not provided.
   :param clabel: Colorbar label used when ``config`` is not provided.
   :param figsize: Figure size passed to Matplotlib.
   :param grid: Whether to overlay a grid on the heatmap.
   :param repeat: Whether the animation should loop.
   :param save_path: Optional output path (``.gif`` / ``.mp4``).
   :param show: Whether to display the animation interactively.
   :param show_progress_bar: Whether to render a ``tqdm`` progress bar during save.
   :param \*\*kwargs: Additional keyword arguments forwarded to ``ax.imshow``.

   :returns: The constructed animation.
   :rtype: ``matplotlib.animation.FuncAnimation``


.. py:function:: energy_landscape_2d_static(z_data, config = None, *, title = '2D Static Landscape', xlabel = 'X-Index', ylabel = 'Y-Index', clabel = 'Value', figsize = (8, 7), grid = False, save_path = None, show = True, **kwargs)

   Plot a static 2D landscape from a 2D array as a heatmap.

   :param z_data: 2D array ``(dim_y, dim_x)`` representing the landscape.
   :param config: Optional :class:`PlotConfig` with pre-set styling.
   :param title: Plot title when ``config`` is not provided.
   :param xlabel: X-axis label when ``config`` is not provided.
   :param ylabel: Y-axis label when ``config`` is not provided.
   :param clabel: Colorbar label when ``config`` is not provided.
   :param figsize: Figure size forwarded to Matplotlib when allocating the canvas.
   :param grid: Whether to draw a grid overlay.
   :param save_path: Optional path that triggers saving the figure to disk.
   :param show: Whether to display the figure interactively.
   :param \*\*kwargs: Additional keyword arguments passed through to ``ax.imshow``.

   :returns: The Matplotlib figure and axes objects.
   :rtype: Tuple[plt.Figure, plt.Axes]


