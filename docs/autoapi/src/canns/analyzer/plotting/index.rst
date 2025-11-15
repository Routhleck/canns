src.canns.analyzer.plotting
===========================

.. py:module:: src.canns.analyzer.plotting

.. autoapi-nested-parse::

   High-level plotting helpers for analyzer functionality.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/analyzer/plotting/config/index
   /autoapi/src/canns/analyzer/plotting/energy/index
   /autoapi/src/canns/analyzer/plotting/jupyter_utils/index
   /autoapi/src/canns/analyzer/plotting/spatial/index
   /autoapi/src/canns/analyzer/plotting/spikes/index
   /autoapi/src/canns/analyzer/plotting/tuning/index


Classes
-------

.. autoapisummary::

   src.canns.analyzer.plotting.PlotConfig
   src.canns.analyzer.plotting.PlotConfigs


Functions
---------

.. autoapisummary::

   src.canns.analyzer.plotting.average_firing_rate_plot
   src.canns.analyzer.plotting.energy_landscape_1d_animation
   src.canns.analyzer.plotting.energy_landscape_1d_static
   src.canns.analyzer.plotting.energy_landscape_2d_animation
   src.canns.analyzer.plotting.energy_landscape_2d_static
   src.canns.analyzer.plotting.plot_firing_field_heatmap
   src.canns.analyzer.plotting.raster_plot
   src.canns.analyzer.plotting.tuning_curve


Package Contents
----------------

.. py:class:: PlotConfig

   Unified configuration class for all plotting helpers in ``canns.analyzer``.

   This mirrors the behaviour of the previous ``visualize`` module so that
   reorganising the files does not affect the public API. The attributes map
   directly to keyword arguments exposed by the high-level plotting functions,
   allowing users to keep existing configuration objects unchanged after the
   reorganisation.


   .. py:method:: __post_init__()


   .. py:method:: for_animation(time_steps_per_second, **kwargs)
      :classmethod:


      Return configuration tailored for animations.



   .. py:method:: for_static_plot(**kwargs)
      :classmethod:


      Return configuration tailored for static plots.



   .. py:method:: to_matplotlib_kwargs()

      Materialize matplotlib keyword arguments from the config.



   .. py:attribute:: clabel
      :type:  str
      :value: 'Value'



   .. py:attribute:: color
      :type:  str
      :value: 'black'



   .. py:attribute:: figsize
      :type:  tuple[int, int]
      :value: (10, 6)



   .. py:attribute:: fps
      :type:  int
      :value: 30



   .. py:attribute:: grid
      :type:  bool
      :value: False



   .. py:attribute:: kwargs
      :type:  dict[str, Any] | None
      :value: None



   .. py:attribute:: repeat
      :type:  bool
      :value: True



   .. py:attribute:: save_path
      :type:  str | None
      :value: None



   .. py:attribute:: show
      :type:  bool
      :value: True



   .. py:attribute:: show_legend
      :type:  bool
      :value: True



   .. py:attribute:: show_progress_bar
      :type:  bool
      :value: True



   .. py:attribute:: time_steps_per_second
      :type:  int | None
      :value: None



   .. py:attribute:: title
      :type:  str
      :value: ''



   .. py:attribute:: xlabel
      :type:  str
      :value: ''



   .. py:attribute:: ylabel
      :type:  str
      :value: ''



.. py:class:: PlotConfigs

   Collection of commonly used plot configurations.

   These helpers mirror the presets that existed in ``canns.analyzer.visualize``
   so that callers relying on them continue to receive the exact same defaults.


   .. py:method:: average_firing_rate_plot(mode = 'per_neuron', **kwargs)
      :staticmethod:



   .. py:method:: energy_landscape_1d_animation(**kwargs)
      :staticmethod:



   .. py:method:: energy_landscape_1d_static(**kwargs)
      :staticmethod:



   .. py:method:: energy_landscape_2d_animation(**kwargs)
      :staticmethod:



   .. py:method:: energy_landscape_2d_static(**kwargs)
      :staticmethod:



   .. py:method:: grid_cell_manifold_static(**kwargs)
      :staticmethod:



   .. py:method:: raster_plot(mode = 'block', **kwargs)
      :staticmethod:



   .. py:method:: theta_population_activity_static(**kwargs)
      :staticmethod:



   .. py:method:: theta_sweep_animation(**kwargs)
      :staticmethod:



   .. py:method:: tuning_curve(num_bins = 50, pref_stim = None, **kwargs)
      :staticmethod:



.. py:function:: average_firing_rate_plot(spike_train, dt, config = None, *, mode = 'population', weights = None, title = 'Average Firing Rate', figsize = (12, 5), save_path = None, show = True, **kwargs)

   Calculate and plot average neural activity from a spike train.

   :param spike_train: Boolean/integer array of shape ``(timesteps, neurons)``.
   :param dt: Simulation time step in seconds.
   :param config: Optional :class:`PlotConfig` with styling overrides.
   :param mode: One of ``"per_neuron"``, ``"population"`` or
                ``"weighted_average"``.
   :param weights: Neuron-wise weights required for ``"weighted_average"``.
   :param title: Plot title when ``config`` is not provided.
   :param figsize: Figure size forwarded to Matplotlib when creating the axes.
   :param save_path: Optional path used to persist the plot.
   :param show: Whether to display the plot interactively.
   :param \*\*kwargs: Additional keyword arguments forwarded to Matplotlib.


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


.. py:function:: plot_firing_field_heatmap(heatmap, config = None, figsize = (5, 5), cmap = 'jet', interpolation = 'nearest', origin = 'lower', show = True, save_path = None, **kwargs)

   Plot a single spatial firing field heatmap.

   This function creates a publication-quality heatmap visualization of neural
   spatial firing patterns. It supports both modern PlotConfig-based configuration
   and legacy keyword arguments for backward compatibility.

   :param heatmap: 2D array of shape (M, K) representing spatial
                   firing rates in each bin.
   :type heatmap: np.ndarray
   :param config: Unified configuration object. If None,
                  uses backward compatibility parameters.
   :type config: PlotConfig | None
   :param figsize: Figure size (width, height) in inches.
                   Defaults to (5, 5).
   :type figsize: tuple[int, int]
   :param cmap: Colormap name for the heatmap. Defaults to 'jet'.
   :type cmap: str
   :param interpolation: Interpolation method for imshow. Defaults to 'nearest'.
   :type interpolation: str
   :param origin: Origin position for imshow ('lower' or 'upper').
                  Defaults to 'lower'.
   :type origin: str
   :param show: Whether to display the plot. Defaults to True.
   :type show: bool
   :param save_path: Path to save the figure. If None, figure is not saved.
   :type save_path: str | None
   :param \*\*kwargs: Additional keyword arguments passed to plt.imshow().

   :returns: The figure and axis objects for further customization.
   :rtype: tuple[plt.Figure, plt.Axes]

   .. rubric:: Example

   >>> from canns.analyzer.spatial import compute_firing_field
   >>> from canns.analyzer.plotting import plot_firing_field_heatmap, PlotConfig
   >>> # Compute firing field
   >>> heatmaps = compute_firing_field(activity, positions, 5.0, 5.0, 50, 50)
   >>> # Plot single neuron with PlotConfig
   >>> config = PlotConfig(figsize=(6, 6), save_path='neuron_0.png', show=False)
   >>> fig, ax = plot_firing_field_heatmap(heatmaps[0], config=config)
   >>> # Plot with legacy parameters
   >>> fig, ax = plot_firing_field_heatmap(heatmaps[1], cmap='viridis', save_path='neuron_1.png')


.. py:function:: raster_plot(spike_train, config = None, *, mode = 'block', title = 'Raster Plot', xlabel = 'Time Step', ylabel = 'Neuron Index', figsize = (12, 6), color = 'black', save_path = None, show = True, **kwargs)

   Generate a raster plot from a spike train matrix.

   The explanatory text mirrors the former ``visualize`` module so callers see
   the same guidance after the reorganisation.

   :param spike_train: Boolean/integer array of shape ``(timesteps, neurons)``.
   :param config: Optional :class:`PlotConfig` with shared styling options.
   :param mode: Either ``"scatter"`` or ``"block"`` to pick the rendering style.
   :param title: Plot title when ``config`` is not provided.
   :param xlabel: X-axis label when ``config`` is not provided.
   :param ylabel: Y-axis label when ``config`` is not provided.
   :param figsize: Figure size forwarded to Matplotlib when creating the axes.
   :param color: Spike colour (or "on" colour for block mode).
   :param save_path: Optional path used to persist the plot.
   :param show: Whether to display the plot interactively.
   :param \*\*kwargs: Additional keyword arguments passed through to Matplotlib.


.. py:function:: tuning_curve(stimulus, firing_rates, neuron_indices, config = None, *, pref_stim = None, num_bins = 50, title = 'Tuning Curve', xlabel = 'Stimulus Value', ylabel = 'Average Firing Rate', figsize = (10, 6), save_path = None, show = True, **kwargs)

   Plot the tuning curve for one or more neurons.

   The wording mirrors the original ``visualize`` module to avoid API drift and
   to keep existing references valid.

   :param stimulus: 1D array with the stimulus value at each time step.
   :param firing_rates: 2D array of firing rates shaped ``(timesteps, neurons)``.
   :param neuron_indices: Integer or iterable of neuron indices to analyse.
   :param config: Optional :class:`PlotConfig` containing styling overrides.
   :param pref_stim: Optional 1D array of preferred stimuli used in legend text.
   :param num_bins: Number of bins when mapping stimulus to mean activity.
   :param title: Plot title when ``config`` is not provided.
   :param xlabel: X-axis label when ``config`` is not provided.
   :param ylabel: Y-axis label when ``config`` is not provided.
   :param figsize: Figure size forwarded to Matplotlib when creating the axes.
   :param save_path: Optional location where the figure should be stored.
   :param show: Whether to display the plot interactively.
   :param \*\*kwargs: Additional keyword arguments passed through to ``ax.plot``.


