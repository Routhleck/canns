src.canns.analyzer.data.asa.fr
==============================

.. py:module:: src.canns.analyzer.data.asa.fr


Classes
-------

.. autoapisummary::

   src.canns.analyzer.data.asa.fr.FRMResult


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.asa.fr.compute_fr_heatmap_matrix
   src.canns.analyzer.data.asa.fr.compute_frm
   src.canns.analyzer.data.asa.fr.plot_frm
   src.canns.analyzer.data.asa.fr.save_fr_heatmap_png


Module Contents
---------------

.. py:class:: FRMResult

   Return object for firing-rate map computation.

   .. attribute:: frm

      Firing rate map (bins_x, bins_y).

      :type: np.ndarray

   .. attribute:: occupancy

      Occupancy counts per spatial bin.

      :type: np.ndarray

   .. attribute:: spike_sum

      Spike counts per spatial bin.

      :type: np.ndarray

   .. attribute:: x_edges, y_edges

      Bin edges used for the FRM computation.

      :type: np.ndarray

   .. rubric:: Examples

   >>> res = FRMResult(frm=None, occupancy=None, spike_sum=None, x_edges=None, y_edges=None)  # doctest: +SKIP


   .. py:attribute:: frm
      :type:  numpy.ndarray


   .. py:attribute:: occupancy
      :type:  numpy.ndarray


   .. py:attribute:: spike_sum
      :type:  numpy.ndarray


   .. py:attribute:: x_edges
      :type:  numpy.ndarray


   .. py:attribute:: y_edges
      :type:  numpy.ndarray


.. py:function:: compute_fr_heatmap_matrix(spike, neuron_range = None, time_range = None, *, transpose = True, normalize = None)

   Compute a matrix for FR heatmap display from spike-like data.

   :param spike: Shape (T, N). Can be continuous (float) or binned (int/float).
   :type spike: np.ndarray
   :param neuron_range: Neuron index range in [0, N]. End is exclusive.
   :type neuron_range: (start, end) or None
   :param time_range: Time index range in [0, T]. End is exclusive.
   :type time_range: (start, end) or None
   :param transpose: If True, returns (N_sel, T_sel) which is convenient for imshow with
                     neurons on Y and time on X (like your utils did with spike.T).
                     If False, returns (T_sel, N_sel).
   :type transpose: bool
   :param normalize: Optional display normalization along time for each neuron.
   :type normalize: {'zscore_per_neuron','minmax_per_neuron', None}

   :returns: **M** -- Heatmap matrix. Default shape (N_sel, T_sel) if transpose=True.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> M = compute_fr_heatmap_matrix(spikes, transpose=True)  # doctest: +SKIP
   >>> M.ndim
   2


.. py:function:: compute_frm(spike, x, y, neuron_id, *, bins = 50, x_range = None, y_range = None, min_occupancy = 1, smoothing = False, sigma = 1.0, nan_for_empty = True)

   Compute a single-neuron firing rate map (FRM) on 2D space.

   :param spike: Shape (T, N). Can be continuous (float) or binned counts (int/float).
   :type spike: np.ndarray
   :param x: Shape (T,). Position samples aligned with spike rows.
   :type x: np.ndarray
   :param y: Shape (T,). Position samples aligned with spike rows.
   :type y: np.ndarray
   :param neuron_id: Neuron index in [0, N).
   :type neuron_id: int
   :param bins: Number of spatial bins per dimension.
   :type bins: int
   :param x_range: Explicit ranges. If None, uses data min/max.
   :type x_range: (min, max) or None
   :param y_range: Explicit ranges. If None, uses data min/max.
   :type y_range: (min, max) or None
   :param min_occupancy: Bins with occupancy < min_occupancy are treated as empty.
   :type min_occupancy: int
   :param smoothing: If True, apply Gaussian smoothing to frm (and optionally to occupancy/spike_sum if you want later).
   :type smoothing: bool
   :param sigma: Gaussian sigma for smoothing (in bin units).
   :type sigma: float
   :param nan_for_empty: If True, empty bins become NaN; else 0.
   :type nan_for_empty: bool

   :returns: frm: 2D array (bins_x, bins_y) in Hz-like units per sample (relative scale).
   :rtype: FRMResult

   .. rubric:: Examples

   >>> res = compute_frm(spikes, x, y, neuron_id=0)  # doctest: +SKIP
   >>> res.frm.shape  # doctest: +SKIP


.. py:function:: plot_frm(frm, *, title = 'Firing Rate Map', dpi = 200, show = None, config = None, **kwargs)

   Save FRM as PNG. Expects frm as 2D array (bins,bins).

   :param frm: Firing rate map (2D).
   :type frm: np.ndarray
   :param title: Figure title (used when ``config`` is None or missing fields).
   :type title: str
   :param dpi: Save DPI.
   :type dpi: int
   :param show: Whether to show the plot (overrides ``config.show`` if not None).
   :type show: bool | None
   :param config: Plot configuration. Use ``config.save_path`` to specify output file.
   :type config: PlotConfig, optional
   :param \*\*kwargs: Additional ``imshow`` keyword arguments. ``save_path`` may be provided here
                      as a fallback if not set in ``config``.
   :type \*\*kwargs: Any

   .. rubric:: Examples

   >>> cfg = PlotConfig.for_static_plot(save_path="frm.png", show=False)  # doctest: +SKIP
   >>> plot_frm(frm, config=cfg)  # doctest: +SKIP


.. py:function:: save_fr_heatmap_png(M, *, title = 'Firing Rate Heatmap', xlabel = 'Time', ylabel = 'Neuron', cmap = None, interpolation = 'nearest', origin = 'lower', aspect = 'auto', clabel = None, colorbar = True, dpi = 200, show = None, config = None, **kwargs)

   Save a heatmap PNG from a matrix (typically output of compute_fr_heatmap_matrix).

   :param M: Heatmap matrix (2D).
   :type M: np.ndarray
   :param title: Plot labels (used when ``config`` is None or missing fields).
   :type title: str
   :param xlabel: Plot labels (used when ``config`` is None or missing fields).
   :type xlabel: str
   :param ylabel: Plot labels (used when ``config`` is None or missing fields).
   :type ylabel: str
   :param cmap: Matplotlib imshow options.
   :type cmap: str, optional
   :param interpolation: Matplotlib imshow options.
   :type interpolation: str, optional
   :param origin: Matplotlib imshow options.
   :type origin: str, optional
   :param aspect: Matplotlib imshow options.
   :type aspect: str, optional
   :param clabel: Colorbar label (defaults to ``config.clabel``).
   :type clabel: str, optional
   :param colorbar: Whether to draw a colorbar.
   :type colorbar: bool
   :param dpi: Save DPI.
   :type dpi: int
   :param show: Whether to show the plot (overrides ``config.show`` if not None).
   :type show: bool | None
   :param config: Plot configuration. Use ``config.save_path`` to specify output file.
   :type config: PlotConfig, optional
   :param \*\*kwargs: Additional ``imshow`` keyword arguments. ``save_path`` may be provided here
                      as a fallback if not set in ``config``.
   :type \*\*kwargs: Any

   .. rubric:: Notes

   - Does not reorder neurons.
   - Uses matplotlib only here (ASA core stays compute-friendly).

   .. rubric:: Examples

   >>> config = PlotConfig.for_static_plot(save_path="fr.png", show=False)  # doctest: +SKIP
   >>> save_fr_heatmap_png(M, config=config)  # doctest: +SKIP


