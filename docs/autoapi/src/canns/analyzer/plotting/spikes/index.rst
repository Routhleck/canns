src.canns.analyzer.plotting.spikes
==================================

.. py:module:: src.canns.analyzer.plotting.spikes

.. autoapi-nested-parse::

   Spike train visualization helpers.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.plotting.spikes.average_firing_rate_plot
   src.canns.analyzer.plotting.spikes.raster_plot


Module Contents
---------------

.. py:function:: average_firing_rate_plot(spike_train, dt, config = None, *, mode = 'population', weights = None, title = 'Average Firing Rate', figsize = (12, 5), save_path = None, show = True, **kwargs)

   Plot different summaries of average activity derived from a spike train.


.. py:function:: raster_plot(spike_train, config = None, *, mode = 'block', title = 'Raster Plot', xlabel = 'Time Step', ylabel = 'Neuron Index', figsize = (12, 6), color = 'black', save_path = None, show = True, **kwargs)

   Generate a raster plot for a spike train.


