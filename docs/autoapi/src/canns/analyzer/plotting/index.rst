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
   src.canns.analyzer.plotting.raster_plot
   src.canns.analyzer.plotting.tuning_curve


Package Contents
----------------

.. py:class:: PlotConfig

   Unified configuration class for analyzer plotting helpers.


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

   Collection of high-level presets for common analyzer plots.


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

   Plot different summaries of average activity derived from a spike train.


.. py:function:: energy_landscape_1d_animation(data_sets, time_steps_per_second = None, config = None, *, fps = 30, title = 'Evolving 1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', figsize = (10, 6), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create an animation for evolving 1D energy landscapes.


.. py:function:: energy_landscape_1d_static(data_sets, config = None, *, title = '1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', show_legend = True, figsize = (10, 6), grid = False, save_path = None, show = True, **kwargs)

   Plot static 1D energy landscapes for one or more datasets.


.. py:function:: energy_landscape_2d_animation(zs_data, config = None, *, time_steps_per_second = None, fps = 30, title = 'Evolving 2D Landscape', xlabel = 'X-Index', ylabel = 'Y-Index', clabel = 'Value', figsize = (8, 7), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create an animation for evolving 2D landscapes.


.. py:function:: energy_landscape_2d_static(z_data, config = None, *, title = '2D Static Landscape', xlabel = 'X-Index', ylabel = 'Y-Index', clabel = 'Value', figsize = (8, 7), grid = False, save_path = None, show = True, **kwargs)

   Plot a static 2D landscape using a heatmap.


.. py:function:: raster_plot(spike_train, config = None, *, mode = 'block', title = 'Raster Plot', xlabel = 'Time Step', ylabel = 'Neuron Index', figsize = (12, 6), color = 'black', save_path = None, show = True, **kwargs)

   Generate a raster plot for a spike train.


.. py:function:: tuning_curve(stimulus, firing_rates, neuron_indices, config = None, *, pref_stim = None, num_bins = 50, title = 'Tuning Curve', xlabel = 'Stimulus Value', ylabel = 'Average Firing Rate', figsize = (10, 6), save_path = None, show = True, **kwargs)

   Compute and plot tuning curves for one or more neurons.


