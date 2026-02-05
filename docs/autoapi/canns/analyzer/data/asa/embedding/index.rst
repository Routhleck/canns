canns.analyzer.data.asa.embedding
=================================

.. py:module:: canns.analyzer.data.asa.embedding


Functions
---------

.. autoapisummary::

   canns.analyzer.data.asa.embedding.embed_spike_trains


Module Contents
---------------

.. py:function:: embed_spike_trains(spike_trains, config = None, **kwargs)

   Load and preprocess spike train data from npz file.

   This function converts raw spike times into a time-binned spike matrix,
   optionally applying Gaussian smoothing and filtering based on animal movement speed.

   :param spike_trains: Dictionary containing ``'spike'`` and ``'t'``, and optionally ``'x'``/``'y'``.
                        ``'spike'`` can be a dict of neuron->spike_times, a list/array of arrays, or
                        a numpy object array from ``np.load``.
   :type spike_trains: dict
   :param config: Configuration object controlling binning, smoothing, and speed filtering.
   :type config: SpikeEmbeddingConfig, optional
   :param \*\*kwargs: Legacy keyword parameters (``res``, ``dt``, ``sigma``, ``smooth0``, ``speed0``,
                      ``min_speed``). Prefer ``config`` in new code.
   :type \*\*kwargs: Any

   :returns: ``(spikes_bin, xx, yy, tt)``. ``spikes_bin`` is a (T, N) binned spike matrix.
             ``xx``, ``yy``, ``tt`` are position/time arrays when ``speed_filter=True``,
             otherwise ``None``.
   :rtype: tuple

   .. rubric:: Examples

   >>> from canns.analyzer.data import SpikeEmbeddingConfig, embed_spike_trains
   >>> cfg = SpikeEmbeddingConfig(smooth=False, speed_filter=False)
   >>> spikes, xx, yy, tt = embed_spike_trains(mock_data, config=cfg)  # doctest: +SKIP
   >>> spikes.ndim
   2


