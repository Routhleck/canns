"""
BTN (Bursty/Theta/Non-bursty) classification.

Workflow:
1) Compute ISI autocorrelograms from spike times.
2) Normalize and smooth each autocorr curve.
3) Compute cosine distance between curves.
4) Cluster with Tomato using a manual kNN graph and density weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform

try:
    from gudhi.clustering.tomato import Tomato
except Exception as exc:  # pragma: no cover - optional dependency
    Tomato = None
    _TOMATO_IMPORT_ERROR = exc
else:
    _TOMATO_IMPORT_ERROR = None


@dataclass
class BTNConfig:
    """Configuration for BTN clustering."""

    maxt: float = 0.2
    res: float = 1e-3
    smooth_sigma: float = 4.0
    nbs: int = 80
    n_clusters: int = 4
    metric: str = "cosine"
    b_one: bool = True
    b_log: bool = False


@dataclass
class BTNResult:
    """Result container for BTN clustering."""

    labels: np.ndarray
    n_clusters: int
    cluster_sizes: np.ndarray
    btn_labels: np.ndarray | None
    mapping: dict[int, str] | None
    intermediates: dict[str, np.ndarray] | None


class BTNAnalyzer:
    """Analyzer that clusters neurons into BTN groups using Tomato."""

    def __init__(self, config: BTNConfig | None = None):
        self.config = config or BTNConfig()

    def classify_btn(
        self,
        spike_data: dict[str, Any],
        *,
        mapping: dict[int, str] | None = None,
        return_intermediates: bool = False,
        plot_diagram: bool = False,
    ) -> BTNResult:
        """Cluster neurons into BTN classes using ISI autocorr + Tomato.

        Parameters
        ----------
        spike_data : dict
            ASA-style dict with keys ``spike`` and ``t`` (and optionally x/y).
        mapping : dict, optional
            Optional mapping from cluster id to BTN label string.
        return_intermediates : bool
            If True, include intermediate arrays in the result.
        plot_diagram : bool
            If True, call Tomato.plot_diagram() for visual inspection.
        """
        _require_tomato()
        spikes = _extract_spike_times(spike_data)
        acorr, bin_times = _isi_autocorr(
            spikes,
            maxt=self.config.maxt,
            res=self.config.res,
            b_one=self.config.b_one,
            b_log=self.config.b_log,
        )
        acorr_norm = _normalize_autocorr(acorr)
        acorr_smooth = gaussian_filter1d(acorr_norm, sigma=self.config.smooth_sigma, axis=1)

        dist = squareform(pdist(acorr_smooth, metric=self.config.metric))
        num_nodes = dist.shape[0]
        order = np.argsort(dist, axis=1)
        if num_nodes > 1:
            nbs_max = num_nodes - 1
            nbs = max(1, min(int(self.config.nbs), nbs_max))
            knn_indices = order[:, 1 : nbs + 1]
        else:
            nbs = 1
            knn_indices = order[:, :1]
        knn_dists = dist[np.arange(dist.shape[0])[:, None], knn_indices]
        weights = np.sum(np.exp(-knn_dists), axis=1)

        t = Tomato(graph_type="manual", density_type="manual", metric="precomputed")
        t.fit(knn_indices, weights=weights)
        if plot_diagram:
            t.plot_diagram()
        if self.config.n_clusters is not None:
            t.n_clusters_ = int(self.config.n_clusters)
        labels = np.asarray(t.labels_, dtype=int)
        if labels.size:
            if np.any(labels < 0):
                valid = labels[labels >= 0]
                cluster_sizes = np.bincount(valid) if valid.size else np.array([])
            else:
                cluster_sizes = np.bincount(labels)
        else:
            cluster_sizes = np.array([])

        btn_labels = None
        if mapping is not None:
            btn_labels = np.array([mapping.get(int(c), "unknown") for c in labels], dtype=object)

        intermediates = None
        if return_intermediates:
            intermediates = {
                "acorr": acorr,
                "acorr_norm": acorr_norm,
                "acorr_smooth": acorr_smooth,
                "distance_matrix": dist,
                "knn_indices": knn_indices,
                "knn_dists": knn_dists,
                "density_weights": weights,
                "bin_times": bin_times,
            }

        if self.config.n_clusters is not None:
            n_clusters = int(self.config.n_clusters)
        else:
            if labels.size:
                valid_labels = labels[labels >= 0]
                n_clusters = int(np.unique(valid_labels).size)
            else:
                n_clusters = 0

        return BTNResult(
            labels=labels,
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            btn_labels=btn_labels,
            mapping=mapping,
            intermediates=intermediates,
        )


def _require_tomato() -> None:
    if Tomato is None:
        raise ImportError(
            "Tomato clustering requires gudhi. Install with: pip install gudhi"
        ) from _TOMATO_IMPORT_ERROR


def _extract_spike_times(spike_data: dict[str, Any]) -> dict[int, np.ndarray]:
    if not isinstance(spike_data, dict) or "spike" not in spike_data or "t" not in spike_data:
        raise ValueError("spike_data must be a dict with keys 'spike' and 't'")

    spike_raw = spike_data["spike"]
    t = np.asarray(spike_data["t"])
    if (
        isinstance(spike_raw, np.ndarray)
        and spike_raw.ndim == 2
        and spike_raw.shape[0] == t.shape[0]
    ):
        raise ValueError(
            "BTN expects spike times per neuron, but got a binned spike matrix (T x N)."
        )
    if (
        hasattr(spike_raw, "item")
        and callable(spike_raw.item)
        and np.asarray(spike_raw).shape == ()
    ):
        spikes_all = spike_raw[()]
    elif isinstance(spike_raw, dict):
        spikes_all = spike_raw
    elif isinstance(spike_raw, (list, np.ndarray)):
        spikes_all = spike_raw
    else:
        spikes_all = spike_raw

    min_time0 = np.min(t)
    max_time0 = np.max(t)

    spikes: dict[int, np.ndarray] = {}
    if isinstance(spikes_all, dict):
        for i, key in enumerate(spikes_all.keys()):
            s = np.asarray(spikes_all[key])
            spikes[i] = s[(s >= min_time0) & (s < max_time0)]
    else:
        cell_inds = np.arange(len(spikes_all))
        for i, m in enumerate(cell_inds):
            s = np.asarray(spikes_all[m]) if len(spikes_all[m]) > 0 else np.array([])
            if s.size > 0:
                spikes[i] = s[(s >= min_time0) & (s < max_time0)]
            else:
                spikes[i] = np.array([])

    return spikes


def _isi_autocorr(
    spk: dict[int, np.ndarray],
    *,
    maxt: float = 0.2,
    res: float = 1e-3,
    b_one: bool = True,
    b_log: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ISI autocorrelogram for each neuron.

    Returns
    -------
    acorr : ndarray
        Shape (N, num_bins).
    bin_times : ndarray
        Bin edges used for histogramming.
    """
    if b_log:
        num_bins = 100
        bin_times = np.ones(num_bins + 1) * 10
        bin_times = np.power(bin_times, np.linspace(np.log10(0.005), np.log10(maxt), num_bins + 1))
        bin_times = np.unique(np.concatenate((-bin_times, bin_times)))
        num_bins = len(bin_times)
    elif b_one:
        num_bins = int(maxt / res) + 1
        bin_times = np.linspace(0, maxt, num_bins)
    else:
        num_bins = int(2 * maxt / res) + 1
        bin_times = np.linspace(-maxt, maxt, num_bins)

    num_neurons = len(spk)
    acorr = np.zeros((num_neurons, len(bin_times) - 1), dtype=int)

    maxt = maxt - 1e-5
    mint = -maxt
    if b_one:
        mint = -1e-5

    for i, n in enumerate(spk):
        spike_times = np.asarray(spk[n])
        for ss in spike_times:
            stemp = spike_times[(spike_times < ss + maxt) & (spike_times > ss + mint)]
            dd = stemp - ss
            if b_one:
                dd = dd[dd >= 0]
            bins = np.digitize(dd, bin_times) - 1
            bins = bins[(bins >= 0) & (bins < num_bins - 1)]
            if bins.size:
                acorr[i, :] += np.bincount(bins, minlength=num_bins)[:-1]

    return acorr, bin_times


def _normalize_autocorr(acorr: np.ndarray) -> np.ndarray:
    acorr = acorr.astype(float, copy=True)
    denom = acorr[:, 0].copy()
    valid = denom > 0
    acorr[valid, :] = acorr[valid, :] / denom[valid, None]
    acorr[:, 0] = 0.0
    return acorr
