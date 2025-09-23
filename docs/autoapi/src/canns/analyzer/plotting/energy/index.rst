src.canns.analyzer.plotting.energy
==================================

.. py:module:: src.canns.analyzer.plotting.energy

.. autoapi-nested-parse::

   Energy landscape visualization utilities.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.plotting.energy.energy_landscape_1d_animation
   src.canns.analyzer.plotting.energy.energy_landscape_1d_static
   src.canns.analyzer.plotting.energy.energy_landscape_2d_animation
   src.canns.analyzer.plotting.energy.energy_landscape_2d_static


Module Contents
---------------

.. py:function:: energy_landscape_1d_animation(data_sets, time_steps_per_second = None, config = None, *, fps = 30, title = 'Evolving 1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', figsize = (10, 6), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create an animation for evolving 1D energy landscapes.


.. py:function:: energy_landscape_1d_static(data_sets, config = None, *, title = '1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', show_legend = True, figsize = (10, 6), grid = False, save_path = None, show = True, **kwargs)

   Plot static 1D energy landscapes for one or more datasets.


.. py:function:: energy_landscape_2d_animation(zs_data, config = None, *, time_steps_per_second = None, fps = 30, title = 'Evolving 2D Landscape', xlabel = 'X-Index', ylabel = 'Y-Index', clabel = 'Value', figsize = (8, 7), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create an animation for evolving 2D landscapes.


.. py:function:: energy_landscape_2d_static(z_data, config = None, *, title = '2D Static Landscape', xlabel = 'X-Index', ylabel = 'Y-Index', clabel = 'Value', figsize = (8, 7), grid = False, save_path = None, show = True, **kwargs)

   Plot a static 2D landscape using a heatmap.


