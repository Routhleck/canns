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

.. py:function:: energy_landscape_1d_animation(data_sets, time_steps_per_second = None, config = None, *, fps = 30, title = 'Evolving 1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', figsize = (10, 6), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, render_backend = 'auto', output_dpi = 150, render_workers = None, render_start_method = None, **kwargs)

   Create an animation of an evolving 1D energy landscape.

   :param data_sets: Mapping ``label -> (x, y_series)``, where ``y_series`` is
                     shaped ``(timesteps, npoints)``.
   :param time_steps_per_second: Simulation steps per second (e.g., ``1/dt``).
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
   :param render_backend: Rendering backend ('imageio', 'matplotlib', or 'auto')
   :param output_dpi: DPI for rendered frames (affects file size and quality)
   :param render_workers: Number of parallel workers (None = auto-detect)
   :param render_start_method: Multiprocessing start method ('fork', 'spawn', or None)
   :param \*\*kwargs: Further keyword arguments passed through to ``ax.plot``.

   :returns: The constructed animation.
   :rtype: ``matplotlib.animation.FuncAnimation``

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.visualization import energy_landscape_1d_animation, PlotConfigs
   >>>
   >>> x = np.linspace(0, 1, 5)
   >>> y_series = np.stack([np.sin(x), np.cos(x)], axis=0)
   >>> data_sets = {"u": (x, y_series), "Iext": (x, y_series)}
   >>> config = PlotConfigs.energy_landscape_1d_animation(
   ...     time_steps_per_second=10,
   ...     fps=2,
   ...     show=False,
   ... )
   >>> anim = energy_landscape_1d_animation(data_sets, config=config)
   >>> print(anim is not None)
   True


.. py:function:: energy_landscape_1d_static(data_sets, config = None, *, title = '1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', show_legend = True, figsize = (10, 6), grid = False, save_path = None, show = True, **kwargs)

   Plot a 1D static energy landscape.

   :param data_sets: Mapping ``label -> (x, y)`` where ``x`` and ``y`` are 1D arrays.
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

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.visualization import energy_landscape_1d_static, PlotConfigs
   >>>
   >>> x = np.linspace(0, 1, 5)
   >>> data_sets = {"u": (x, np.sin(x)), "Iext": (x, np.cos(x))}
   >>> config = PlotConfigs.energy_landscape_1d_static(show=False)
   >>> fig, ax = energy_landscape_1d_static(data_sets, config=config)
   >>> print(fig is not None)
   True


.. py:function:: energy_landscape_2d_animation(zs_data, config = None, *, time_steps_per_second = None, fps = 30, title = 'Evolving 2D Landscape', xlabel = 'X-Index', ylabel = 'Y-Index', clabel = 'Value', figsize = (8, 7), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, render_backend = 'auto', output_dpi = 150, render_workers = None, render_start_method = None, **kwargs)

   Create an animation of an evolving 2D landscape.

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
   :param render_backend: Rendering backend ('imageio', 'matplotlib', or 'auto')
   :param output_dpi: DPI for rendered frames (affects file size and quality)
   :param render_workers: Number of parallel workers (None = auto-detect)
   :param render_start_method: Multiprocessing start method ('fork', 'spawn', or None)
   :param \*\*kwargs: Additional keyword arguments forwarded to ``ax.imshow``.

   :returns: The constructed animation.
   :rtype: ``matplotlib.animation.FuncAnimation``

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.visualization import energy_landscape_2d_animation, PlotConfigs
   >>>
   >>> zs = np.random.rand(3, 4, 4)
   >>> config = PlotConfigs.energy_landscape_2d_animation(
   ...     time_steps_per_second=10,
   ...     fps=2,
   ...     show=False,
   ... )
   >>> anim = energy_landscape_2d_animation(zs, config=config)
   >>> print(anim is not None)
   True


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

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.visualization import energy_landscape_2d_static, PlotConfigs
   >>>
   >>> z = np.random.rand(4, 4)
   >>> config = PlotConfigs.energy_landscape_2d_static(show=False)
   >>> fig, ax = energy_landscape_2d_static(z, config=config)
   >>> print(fig is not None)
   True


