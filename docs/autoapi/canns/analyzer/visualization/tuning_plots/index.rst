canns.analyzer.visualization.tuning_plots
=========================================

.. py:module:: canns.analyzer.visualization.tuning_plots

.. autoapi-nested-parse::

   Tuning curve visualization utilities.



Functions
---------

.. autoapisummary::

   canns.analyzer.visualization.tuning_plots.tuning_curve


Module Contents
---------------

.. py:function:: tuning_curve(stimulus, firing_rates, neuron_indices, config = None, *, pref_stim = None, num_bins = 50, title = 'Tuning Curve', xlabel = 'Stimulus Value', ylabel = 'Average Firing Rate', figsize = (10, 6), save_path = None, show = True, **kwargs)

   Plot the tuning curve for one or more neurons.

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

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.visualization import tuning_curve, PlotConfigs
   >>>
   >>> stimulus = np.linspace(0, 1, 10)
   >>> firing_rates = np.random.rand(10, 3)
   >>> config = PlotConfigs.tuning_curve(num_bins=5, pref_stim=np.array([0.2, 0.5, 0.8]), show=False)
   >>> fig, ax = tuning_curve(stimulus, firing_rates, neuron_indices=[0, 1], config=config)
   >>> print(fig is not None)
   True


