src.canns.analyzer.data.asa.tda
===============================

.. py:module:: src.canns.analyzer.data.asa.tda


Attributes
----------

.. autoapisummary::

   src.canns.analyzer.data.asa.tda.HAS_NUMBA


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.asa.tda.tda_vis


Module Contents
---------------

.. py:function:: tda_vis(embed_data, config = None, **kwargs)

   Topological Data Analysis visualization with optional shuffle testing.

   :param embed_data: Embedded spike train data of shape (T, N).
   :type embed_data: np.ndarray
   :param config: Configuration object with all TDA parameters. If None, legacy kwargs are used.
   :type config: TDAConfig, optional
   :param \*\*kwargs: Legacy keyword parameters (``dim``, ``num_times``, ``active_times``, ``k``,
                      ``n_points``, ``metric``, ``nbs``, ``maxdim``, ``coeff``, ``show``,
                      ``do_shuffle``, ``num_shuffles``, ``progress_bar``).
   :type \*\*kwargs: Any

   :returns: Dictionary containing:
             - ``persistence``: persistence diagrams from real data.
             - ``indstemp``: indices of sampled points.
             - ``movetimes``: selected time points.
             - ``n_points``: number of sampled points.
             - ``shuffle_max``: shuffle analysis results (if ``do_shuffle=True``), else ``None``.
   :rtype: dict

   .. rubric:: Examples

   >>> from canns.analyzer.data import TDAConfig, tda_vis
   >>> cfg = TDAConfig(maxdim=1, do_shuffle=False, show=False)
   >>> result = tda_vis(embed_data, config=cfg)  # doctest: +SKIP
   >>> sorted(result.keys())
   ['indstemp', 'movetimes', 'n_points', 'persistence', 'shuffle_max']


.. py:data:: HAS_NUMBA
   :value: True


