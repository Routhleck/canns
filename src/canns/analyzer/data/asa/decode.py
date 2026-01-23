from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.sparse.linalg import lsmr
from sklearn import preprocessing

from .config import SpikeEmbeddingConfig
from .embedding import embed_spike_trains


def decode_circular_coordinates(
    persistence_result: dict[str, Any],
    spike_data: dict[str, Any],
    real_ground: bool = True,
    real_of: bool = True,
    save_path: str | None = None,
) -> dict[str, Any]:
    """
    Decode circular coordinates (bump positions) from cohomology.

    Parameters
    ----------
    persistence_result : dict
        Output from :func:`canns.analyzer.data.tda_vis`, containing keys:
        ``persistence``, ``indstemp``, ``movetimes``, ``n_points``.
    spike_data : dict
        Spike data dictionary containing ``'spike'``, ``'t'`` and optionally ``'x'``/``'y'``.
    real_ground : bool
        Whether x/y/t ground-truth exists (controls whether speed filtering is applied).
    real_of : bool
        Whether the experiment is open-field (controls box coordinate handling).
    save_path : str, optional
        Path to save decoding results. Defaults to ``Results/spikes_decoding.npz``.

    Returns
    -------
    dict
        Dictionary containing:
        - ``coords``: decoded coordinates for all timepoints.
        - ``coordsbox``: decoded coordinates for box timepoints.
        - ``times``: time indices for ``coords``.
        - ``times_box``: time indices for ``coordsbox``.
        - ``centcosall`` / ``centsinall``: cosine/sine centroids.

    Examples
    --------
    >>> from canns.analyzer.data import tda_vis, decode_circular_coordinates
    >>> persistence = tda_vis(embed_spikes, config=tda_cfg)  # doctest: +SKIP
    >>> decoding = decode_circular_coordinates(persistence, spike_data)  # doctest: +SKIP
    >>> decoding["coords"].shape  # doctest: +SKIP
    """
    ph_classes = [0, 1]  # Decode the ith most persistent cohomology class
    num_circ = len(ph_classes)
    dec_tresh = 0.99
    coeff = 47

    # Extract persistence analysis results
    persistence = persistence_result["persistence"]
    indstemp = persistence_result["indstemp"]
    movetimes = persistence_result["movetimes"]
    n_points = persistence_result["n_points"]

    diagrams = persistence["dgms"]  # the multiset describing the lives of the persistence classes
    cocycles = persistence["cocycles"][1]  # the cocycle representatives for the 1-dim classes
    dists_land = persistence["dperm2all"]  # the pairwise distance between the points
    births1 = diagrams[1][:, 0]  # the time of birth for the 1-dim classes
    deaths1 = diagrams[1][:, 1]  # the time of death for the 1-dim classes
    deaths1[np.isinf(deaths1)] = 0
    lives1 = deaths1 - births1  # the lifetime for the 1-dim classes
    iMax = np.argsort(lives1)
    coords1 = np.zeros((num_circ, len(indstemp)))
    threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]]) * dec_tresh

    for c in ph_classes:
        cocycle = cocycles[iMax[-(c + 1)]]
        coords1[c, :], _ = _get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

    # Whether the user-provided dataset has ground-truth x/y/t.
    if real_ground:
        sspikes, _, _, _ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=True)
        )
    else:
        sspikes, _, _, _ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=False)
        )

    num_neurons = sspikes.shape[1]
    centcosall = np.zeros((num_neurons, 2, n_points))
    centsinall = np.zeros((num_neurons, 2, n_points))
    dspk = preprocessing.scale(sspikes[movetimes[indstemp], :])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid, :, :] = np.multiply(np.cos(coords1[:, :] * 2 * np.pi), spktemp)
        centsinall[neurid, :, :] = np.multiply(np.sin(coords1[:, :] * 2 * np.pi), spktemp)

    # Whether the user-provided dataset has ground-truth x/y/t.
    if real_ground:
        sspikes, _, _, _ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=True)
        )
        spikes, __, __, __ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=False, speed_filter=True)
        )
    else:
        sspikes, _, _, _ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=False)
        )
        spikes, _, _, _ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=False, speed_filter=False)
        )

    times = np.where(np.sum(spikes > 0, 1) >= 1)[0]
    dspk = preprocessing.scale(sspikes)
    sspikes = sspikes[times, :]
    dspk = dspk[times, :]

    a = np.zeros((len(sspikes[:, 0]), 2, num_neurons))
    for n in range(num_neurons):
        a[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centcosall[n, :, :], 1))

    c = np.zeros((len(sspikes[:, 0]), 2, num_neurons))
    for n in range(num_neurons):
        c[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centsinall[n, :, :], 1))

    mtot2 = np.sum(c, 2)
    mtot1 = np.sum(a, 2)
    coords = np.arctan2(mtot2, mtot1) % (2 * np.pi)

    # Whether the dataset comes from a real open-field (OF) environment.
    if real_of:
        coordsbox = coords.copy()
        times_box = times.copy()
    else:
        sspikes, _, _, _ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=True)
        )
        spikes, __, __, __ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=False, speed_filter=True)
        )
        dspk = preprocessing.scale(sspikes)
        times_box = np.where(np.sum(spikes > 0, 1) >= 1)[0]
        dspk = dspk[times_box, :]

        a = np.zeros((len(times_box), 2, num_neurons))
        for n in range(num_neurons):
            a[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centcosall[n, :, :], 1))

        c = np.zeros((len(times_box), 2, num_neurons))
        for n in range(num_neurons):
            c[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centsinall[n, :, :], 1))

        mtot2 = np.sum(c, 2)
        mtot1 = np.sum(a, 2)
        coordsbox = np.arctan2(mtot2, mtot1) % (2 * np.pi)

    # Prepare results dictionary
    results = {
        "coords": coords,
        "coordsbox": coordsbox,
        "times": times,
        "times_box": times_box,
        "centcosall": centcosall,
        "centsinall": centsinall,
    }

    # Save results
    if save_path is None:
        os.makedirs("Results", exist_ok=True)
        save_path = "Results/spikes_decoding.npz"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **results)

    return results


def decode_circular_coordinates1(
    persistence_result: dict[str, Any],
    spike_data: dict[str, Any],
    save_path: str | None = None,
) -> dict[str, Any]:
    """Legacy helper kept for backward compatibility."""
    ph_classes = [0, 1]  # Decode the ith most persistent cohomology class
    num_circ = len(ph_classes)
    dec_tresh = 0.99
    coeff = 47

    # Extract persistence analysis results
    persistence = persistence_result["persistence"]
    indstemp = persistence_result["indstemp"]
    movetimes = persistence_result["movetimes"]
    n_points = persistence_result["n_points"]

    diagrams = persistence["dgms"]  # the multiset describing the lives of the persistence classes
    cocycles = persistence["cocycles"][1]  # the cocycle representatives for the 1-dim classes
    dists_land = persistence["dperm2all"]  # the pairwise distance between the points
    births1 = diagrams[1][:, 0]  # the time of birth for the 1-dim classes
    deaths1 = diagrams[1][:, 1]  # the time of death for the 1-dim classes
    deaths1[np.isinf(deaths1)] = 0
    lives1 = deaths1 - births1  # the lifetime for the 1-dim classes
    iMax = np.argsort(lives1)
    coords1 = np.zeros((num_circ, len(indstemp)))
    threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]]) * dec_tresh

    for c in ph_classes:
        cocycle = cocycles[iMax[-(c + 1)]]
        coords1[c, :], _ = _get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

    sspikes = spike_data["spike"]
    num_neurons = sspikes.shape[1]
    centcosall = np.zeros((num_neurons, 2, n_points))
    centsinall = np.zeros((num_neurons, 2, n_points))
    dspk = preprocessing.scale(sspikes[movetimes[indstemp], :])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid, :, :] = np.multiply(np.cos(coords1[:, :] * 2 * np.pi), spktemp)
        centsinall[neurid, :, :] = np.multiply(np.sin(coords1[:, :] * 2 * np.pi), spktemp)

    times = np.where(np.sum(sspikes > 0, 1) >= 1)[0]
    dspk = preprocessing.scale(sspikes)
    sspikes = sspikes[times, :]
    dspk = dspk[times, :]

    a = np.zeros((len(sspikes[:, 0]), 2, num_neurons))
    for n in range(num_neurons):
        a[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centcosall[n, :, :], 1))

    c = np.zeros((len(sspikes[:, 0]), 2, num_neurons))
    for n in range(num_neurons):
        c[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centsinall[n, :, :], 1))

    mtot2 = np.sum(c, 2)
    mtot1 = np.sum(a, 2)
    coords = np.arctan2(mtot2, mtot1) % (2 * np.pi)

    coordsbox = coords.copy()
    times_box = times.copy()

    # Prepare results dictionary
    results = {
        "coords": coords,
        "coordsbox": coordsbox,
        "times": times,
        "times_box": times_box,
        "centcosall": centcosall,
        "centsinall": centsinall,
    }

    # Save results
    if save_path is None:
        os.makedirs("Results", exist_ok=True)
        save_path = "Results/spikes_decoding.npz"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **results)

    return results


def decode_circular_coordinates_multi(
    persistence_result: dict,
    spike_data: dict,
    save_path: str | None = None,
    num_circ: int = 2,  # Number of H1 cocycles/circular coordinates to decode
) -> dict:
    """Decode multiple circular coordinates from TDA persistence.

    Parameters
    ----------
    persistence_result : dict
        Output from :func:`canns.analyzer.data.tda_vis`, containing keys:
        ``persistence``, ``indstemp``, ``movetimes``, ``n_points``.
    spike_data : dict
        Spike data dictionary containing ``'spike'``, ``'t'`` and optionally ``'x'``/``'y'``.
    save_path : str, optional
        Path to save decoding results. Defaults to ``Results/spikes_decoding.npz``.
    num_circ : int
        Number of H1 cocycles/circular coordinates to decode.

    Returns
    -------
    dict
        Dictionary with ``coords``, ``coordsbox``, ``times``, ``times_box`` and centroid terms.

    Examples
    --------
    >>> decoding = decode_circular_coordinates_multi(persistence, spike_data, num_circ=2)  # doctest: +SKIP
    >>> decoding["coords"].shape  # doctest: +SKIP
    """
    from sklearn import preprocessing

    dec_tresh = 0.99
    coeff = 47

    persistence = persistence_result["persistence"]
    indstemp = persistence_result["indstemp"]
    movetimes = persistence_result["movetimes"]
    n_points = persistence_result["n_points"]

    diagrams = persistence["dgms"]
    cocycles = persistence["cocycles"][1]
    dists_land = persistence["dperm2all"]

    births1 = diagrams[1][:, 0]
    deaths1 = diagrams[1][:, 1]
    deaths1[np.isinf(deaths1)] = 0
    lives1 = deaths1 - births1

    if lives1.size < num_circ or len(cocycles) < num_circ:
        raise ValueError(
            f"Requested num_circ={num_circ}, but only {lives1.size} H1 feature(s) are available. "
            "This usually means the chosen time window is too short, the data are too sparse, "
            "or the embedding parameters are not appropriate."
        )

    iMax = np.argsort(lives1)
    coords1 = np.zeros((num_circ, len(indstemp)))

    for i in range(num_circ):
        idx = iMax[-(i + 1)]
        threshold = births1[idx] + (deaths1[idx] - births1[idx]) * dec_tresh
        cocycle = cocycles[idx]
        coords1[i, :], _ = _get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

    sspikes = spike_data["spike"]
    num_neurons = sspikes.shape[1]

    centcosall = np.zeros((num_neurons, num_circ, n_points))
    centsinall = np.zeros((num_neurons, num_circ, n_points))
    dspk = preprocessing.scale(sspikes[movetimes[indstemp], :])

    for n in range(num_neurons):
        spktemp = dspk[:, n].copy()
        centcosall[n, :, :] = np.multiply(np.cos(coords1 * 2 * np.pi), spktemp)
        centsinall[n, :, :] = np.multiply(np.sin(coords1 * 2 * np.pi), spktemp)

    times = np.where(np.sum(sspikes > 0, 1) >= 1)[0]
    dspk = preprocessing.scale(sspikes)
    sspikes = sspikes[times, :]
    dspk = dspk[times, :]

    a = np.zeros((len(sspikes), num_circ, num_neurons))
    c = np.zeros((len(sspikes), num_circ, num_neurons))

    for n in range(num_neurons):
        a[:, :, n] = dspk[:, n : n + 1] * np.sum(centcosall[n, :, :], axis=1)
        c[:, :, n] = dspk[:, n : n + 1] * np.sum(centsinall[n, :, :], axis=1)

    mtot1 = np.sum(a, 2)
    mtot2 = np.sum(c, 2)
    coords = np.arctan2(mtot2, mtot1) % (2 * np.pi)

    results = {
        "coords": coords,
        "coordsbox": coords.copy(),
        "times": times,
        "times_box": times.copy(),
        "centcosall": centcosall,
        "centsinall": centsinall,
    }

    if save_path is None:
        os.makedirs("Results", exist_ok=True)
        save_path = "Results/spikes_decoding.npz"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **results)
    return results


def _get_coords(cocycle, threshold, num_sampled, dists, coeff):
    """
    Reconstruct circular coordinates from cocycle information.

    Parameters:
        cocycle (ndarray): Persistent cocycle representative.
        threshold (float): Maximum allowable edge distance.
        num_sampled (int): Number of sampled points.
        dists (ndarray): Pairwise distance matrix.
        coeff (int): Finite field modulus for cohomology.

    Returns:
        f (ndarray): Circular coordinate values (in [0,1]).
        verts (ndarray): Indices of used vertices.
    """
    zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
    cocycle[zint, 2] = cocycle[zint, 2] - coeff
    d = np.zeros((num_sampled, num_sampled))
    d[np.tril_indices(num_sampled)] = np.nan
    d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
    d[dists > threshold] = np.nan
    d[dists == 0] = np.nan
    edges = np.where(~np.isnan(d))
    verts = np.array(np.unique(edges))
    num_edges = np.shape(edges)[1]
    num_verts = np.size(verts)
    values = d[edges]
    A = np.zeros((num_edges, num_verts), dtype=int)
    v1 = np.zeros((num_edges, 2), dtype=int)
    v2 = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        # Extract scalar indices from np.where results
        idx1 = np.where(verts == edges[0][i])[0]
        idx2 = np.where(verts == edges[1][i])[0]

        # Handle case where np.where returns multiple matches (shouldn't happen in valid data)
        if len(idx1) > 0:
            v1[i, :] = [i, idx1[0]]
        else:
            raise ValueError(f"No vertex found for edge {edges[0][i]}")

        if len(idx2) > 0:
            v2[i, :] = [i, idx2[0]]
        else:
            raise ValueError(f"No vertex found for edge {edges[1][i]}")

    A[v1[:, 0], v1[:, 1]] = -1
    A[v2[:, 0], v2[:, 1]] = 1

    L = np.ones((num_edges,))
    Aw = A * np.sqrt(L[:, np.newaxis])
    Bw = values * np.sqrt(L)
    f = lsmr(Aw, Bw)[0] % 1
    return f, verts
