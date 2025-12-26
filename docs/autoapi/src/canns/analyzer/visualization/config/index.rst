src.canns.analyzer.visualization.config
=======================================

.. py:module:: src.canns.analyzer.visualization.config

.. autoapi-nested-parse::

   Reusable plotting configuration utilities for analyzer visualizations.



Classes
-------

.. autoapisummary::

   src.canns.analyzer.visualization.config.PlotConfig
   src.canns.analyzer.visualization.config.PlotConfigs


Module Contents
---------------

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



   .. py:method:: grid_autocorrelation(**kwargs)
      :staticmethod:


      Configuration for spatial autocorrelation heatmap visualization.

      Used to visualize hexagonal periodicity patterns in grid cell firing fields.
      Applies diverging colormap (RdBu_r) suitable for correlation values [-1, 1].

      :param \*\*kwargs: Additional configuration parameters to override defaults.

      :returns: Configuration object for autocorrelation plots.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> from canns.analyzer.visualization import PlotConfigs
      >>> config = PlotConfigs.grid_autocorrelation(
      ...     title="Grid Cell Autocorrelation",
      ...     save_path="autocorr.png"
      ... )



   .. py:method:: grid_cell_manifold_static(**kwargs)
      :staticmethod:



   .. py:method:: grid_cell_tracking_animation(**kwargs)
      :staticmethod:


      Configuration for grid cell tracking animation.

      Creates 3-panel synchronized animation showing trajectory, activity time course,
      and rate map with position overlay for analyzing grid cell behavior.

      :param \*\*kwargs: Additional configuration parameters to override defaults.
                         Must include 'time_steps_per_second' if not using default.

      :returns: Configuration object for tracking animations.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> config = PlotConfigs.grid_cell_tracking_animation(
      ...     time_steps_per_second=1000,  # dt=1ms
      ...     fps=20,
      ...     save_path="tracking.gif"
      ... )



   .. py:method:: grid_score_plot(**kwargs)
      :staticmethod:


      Configuration for grid score bar chart visualization.

      Displays rotational correlations at different angles used to compute grid score.
      Highlights hexagonal angles (60°, 120°) versus non-hexagonal angles.

      :param \*\*kwargs: Additional configuration parameters to override defaults.

      :returns: Configuration object for grid score plots.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> config = PlotConfigs.grid_score_plot(
      ...     title="Grid Cell Quality Assessment",
      ...     save_path="grid_score.png"
      ... )



   .. py:method:: grid_spacing_plot(**kwargs)
      :staticmethod:


      Configuration for grid spacing radial profile visualization.

      Shows how autocorrelation decays with distance from center, revealing
      the periodic spacing of grid fields.

      :param \*\*kwargs: Additional configuration parameters to override defaults.

      :returns: Configuration object for spacing analysis plots.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> config = PlotConfigs.grid_spacing_plot(
      ...     title="Grid Field Spacing",
      ...     save_path="spacing.png"
      ... )



   .. py:method:: raster_plot(mode = 'block', **kwargs)
      :staticmethod:



   .. py:method:: theta_population_activity_static(**kwargs)
      :staticmethod:



   .. py:method:: theta_sweep_animation(**kwargs)
      :staticmethod:



   .. py:method:: tuning_curve(num_bins = 50, pref_stim = None, **kwargs)
      :staticmethod:



