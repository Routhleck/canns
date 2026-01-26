src.canns.analyzer.data.asa.decode
==================================

.. py:module:: src.canns.analyzer.data.asa.decode


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.asa.decode.decode_circular_coordinates
   src.canns.analyzer.data.asa.decode.decode_circular_coordinates1
   src.canns.analyzer.data.asa.decode.decode_circular_coordinates_multi


Module Contents
---------------

.. py:function:: decode_circular_coordinates(persistence_result, spike_data, real_ground = True, real_of = True, save_path = None)

   Decode circular coordinates (bump positions) from cohomology.

   :param persistence_result: Output from :func:`canns.analyzer.data.tda_vis`, containing keys:
                              ``persistence``, ``indstemp``, ``movetimes``, ``n_points``.
   :type persistence_result: dict
   :param spike_data: Spike data dictionary containing ``'spike'``, ``'t'`` and optionally ``'x'``/``'y'``.
   :type spike_data: dict
   :param real_ground: Whether x/y/t ground-truth exists (controls whether speed filtering is applied).
   :type real_ground: bool
   :param real_of: Whether the experiment is open-field (controls box coordinate handling).
   :type real_of: bool
   :param save_path: Path to save decoding results. Defaults to ``Results/spikes_decoding.npz``.
   :type save_path: str, optional

   :returns: Dictionary containing:
             - ``coords``: decoded coordinates for all timepoints.
             - ``coordsbox``: decoded coordinates for box timepoints.
             - ``times``: time indices for ``coords``.
             - ``times_box``: time indices for ``coordsbox``.
             - ``centcosall`` / ``centsinall``: cosine/sine centroids.
   :rtype: dict

   .. rubric:: Examples

   >>> from canns.analyzer.data import tda_vis, decode_circular_coordinates
   >>> persistence = tda_vis(embed_spikes, config=tda_cfg)  # doctest: +SKIP
   >>> decoding = decode_circular_coordinates(persistence, spike_data)  # doctest: +SKIP
   >>> decoding["coords"].shape  # doctest: +SKIP


.. py:function:: decode_circular_coordinates1(persistence_result, spike_data, save_path = None)

   Legacy helper kept for backward compatibility.


.. py:function:: decode_circular_coordinates_multi(persistence_result, spike_data, save_path = None, num_circ = 2)

   Decode multiple circular coordinates from TDA persistence.

   :param persistence_result: Output from :func:`canns.analyzer.data.tda_vis`, containing keys:
                              ``persistence``, ``indstemp``, ``movetimes``, ``n_points``.
   :type persistence_result: dict
   :param spike_data: Spike data dictionary containing ``'spike'``, ``'t'`` and optionally ``'x'``/``'y'``.
   :type spike_data: dict
   :param save_path: Path to save decoding results. Defaults to ``Results/spikes_decoding.npz``.
   :type save_path: str, optional
   :param num_circ: Number of H1 cocycles/circular coordinates to decode.
   :type num_circ: int

   :returns: Dictionary with ``coords``, ``coordsbox``, ``times``, ``times_box`` and centroid terms.
   :rtype: dict

   .. rubric:: Examples

   >>> decoding = decode_circular_coordinates_multi(persistence, spike_data, num_circ=2)  # doctest: +SKIP
   >>> decoding["coords"].shape  # doctest: +SKIP


