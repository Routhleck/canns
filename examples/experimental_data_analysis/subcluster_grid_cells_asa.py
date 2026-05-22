#!/usr/bin/env python3
"""
ASA-adapted subcluster workflow inspired by GridCellTorus/Subcluster grid cells.ipynb.

This script focuses on the most portable part of the notebook:
1. Compute temporal autocorr features for each neuron.
2. Cluster neurons with Tomato into BTN-like subclusters.
3. Save summary plots and per-subcluster ASA subsets for downstream TDA/decode/cohomap.

Two input formats are supported:
- spike-times per neuron (preferred, closest to the original notebook)
- dense/binned spike matrix with shape (T, N)

Example
-------
python examples/experimental_data_analysis/subcluster_grid_cells_asa.py \
  --data /path/to/asa_data.npz \
  --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform

root = Path(__file__).resolve().parents[2]
sys.path.append(str(root / "src"))

from canns.analyzer.data.asa.path import load_npz_any
from canns.analyzer.data.cell_classification import (
    BTNAnalyzer,
    BTNConfig,
    plot_btn_autocorr_summary,
    plot_btn_distance_matrix,
)
from canns.utils.example_outputs import get_example_output_dir

try:
    from gudhi.clustering.tomato import Tomato
except Exception as exc:  # pragma: no cover - optional dependency
    Tomato = None
    _TOMATO_IMPORT_ERROR = exc
else:
    _TOMATO_IMPORT_ERROR = None


DEFAULT_OUT_DIR = get_example_output_dir(__file__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASA-adapted BTN/subcluster analysis from GridCellTorus."
    )
    parser.add_argument("--data", required=True, help="Input ASA .npz file.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for plots, summary, and subset ASA files.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Target number of Tomato clusters.",
    )
    parser.add_argument("--nbs", type=int, default=40, help="Tomato kNN neighbors.")
    parser.add_argument(
        "--maxt",
        type=float,
        default=0.2,
        help="Max lag in seconds for temporal autocorr.",
    )
    parser.add_argument(
        "--res",
        type=float,
        default=1e-3,
        help="Lag bin size in seconds for temporal autocorr.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma for smoothing autocorr curves.",
    )
    parser.add_argument(
        "--min-spikes",
        type=int,
        default=0,
        help="Drop neurons with fewer than this many spikes/events before clustering.",
    )
    parser.add_argument(
        "--dense-binary",
        action="store_true",
        help="For dense spike matrices, binarize activity before autocorr.",
    )
    parser.add_argument(
        "--plot-diagram",
        action="store_true",
        help="Show Tomato persistence diagram when available.",
    )
    parser.add_argument(
        "--cluster-labels",
        default=None,
        help="Optional manual mapping like '0:B,1:T,2:N'.",
    )
    parser.add_argument(
        "--save-subsets",
        action="store_true",
        help="Save one ASA .npz file per discovered subcluster.",
    )
    return parser.parse_args()


def _require_tomato() -> None:
    if Tomato is None:
        raise ImportError(
            "Tomato clustering requires gudhi. Install with: pip install gudhi"
        ) from _TOMATO_IMPORT_ERROR


def _unwrap_spike_container(spike_raw: Any) -> Any:
    if isinstance(spike_raw, dict):
        return spike_raw
    arr = np.asarray(spike_raw)
    if arr.dtype == object and arr.shape == ():
        return spike_raw.item()
    return spike_raw


def _is_dense_spike_matrix(asa: dict[str, Any]) -> bool:
    if "spike" not in asa:
        return False
    spike_raw = _unwrap_spike_container(asa["spike"])
    if not isinstance(spike_raw, np.ndarray) or spike_raw.ndim != 2:
        return False
    if "t" not in asa:
        return True
    t = np.asarray(asa["t"]).ravel()
    return spike_raw.shape[0] == t.shape[0]


def _extract_spike_times_list(asa: dict[str, Any]) -> list[np.ndarray]:
    spike_raw = _unwrap_spike_container(asa["spike"])
    if isinstance(spike_raw, dict):
        return [np.asarray(spike_raw[k], dtype=float).ravel() for k in spike_raw.keys()]
    if isinstance(spike_raw, (list, tuple)):
        return [np.asarray(v, dtype=float).ravel() for v in spike_raw]
    if isinstance(spike_raw, np.ndarray) and spike_raw.dtype == object:
        return [np.asarray(v, dtype=float).ravel() for v in spike_raw]
    raise ValueError("Spike input is not spike-times per neuron.")


def _count_events_for_dense(spikes: np.ndarray, binary: bool) -> np.ndarray:
    if binary:
        return np.count_nonzero(spikes > 0, axis=0)
    return np.asarray(np.rint(spikes.sum(axis=0)), dtype=int)


def _subset_neuron_metadata(value: Any, keep_indices: np.ndarray, n_total: int) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return value
    if arr.shape[0] != n_total:
        return value
    return arr[keep_indices]


def _filter_asa_neurons(
    asa: dict[str, Any],
    keep_indices: np.ndarray,
    *,
    dense: bool,
) -> dict[str, Any]:
    out = dict(asa)
    keep_indices = np.asarray(keep_indices, dtype=int)
    spike_raw = _unwrap_spike_container(asa["spike"])
    n_total = spike_raw.shape[1] if dense else len(spike_raw)

    if dense:
        out["spike"] = np.asarray(spike_raw)[:, keep_indices]
    elif isinstance(spike_raw, dict):
        keys = list(spike_raw.keys())
        out["spike"] = {i: np.asarray(spike_raw[keys[idx]]) for i, idx in enumerate(keep_indices)}
    else:
        out["spike"] = np.array([np.asarray(spike_raw[idx]) for idx in keep_indices], dtype=object)

    for key, value in list(out.items()):
        if key in {"spike", "x", "y", "t"}:
            continue
        out[key] = _subset_neuron_metadata(value, keep_indices, n_total)

    out["original_neuron_indices"] = keep_indices
    return out


def _normalize_rows(acorr: np.ndarray) -> np.ndarray:
    acorr = np.asarray(acorr, dtype=float)
    denom = acorr.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return acorr / denom


def _binned_temporal_autocorr(
    spikes: np.ndarray,
    *,
    max_lag_bins: int,
    binary: bool,
) -> np.ndarray:
    spikes = np.asarray(spikes, dtype=float)
    if spikes.ndim != 2:
        raise ValueError(f"Dense spike input must be 2D, got {spikes.shape}")
    if binary:
        spikes = (spikes > 0).astype(float)

    n_time, n_neurons = spikes.shape
    acorr = np.zeros((n_neurons, max_lag_bins), dtype=float)
    for lag in range(max_lag_bins):
        if lag == 0:
            left = spikes
            right = spikes
        else:
            left = spikes[:-lag, :]
            right = spikes[lag:, :]
        acorr[:, lag] = np.sum(left * right, axis=0)
    return acorr


def _cluster_tomato_from_acorr(
    acorr: np.ndarray,
    *,
    smooth_sigma: float,
    nbs: int,
    n_clusters: int,
    plot_diagram: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _require_tomato()

    acorr_norm = _normalize_rows(acorr)
    acorr_smooth = gaussian_filter1d(acorr_norm, sigma=smooth_sigma, axis=1)

    dist = squareform(pdist(acorr_smooth, metric="cosine"))
    num_nodes = dist.shape[0]
    order = np.argsort(dist, axis=1)

    if num_nodes > 1:
        nbs_eff = max(1, min(int(nbs), num_nodes - 1))
        knn_indices = order[:, 1 : nbs_eff + 1]
    else:
        nbs_eff = 1
        knn_indices = order[:, :1]

    knn_dists = dist[np.arange(num_nodes)[:, None], knn_indices]
    weights = np.sum(np.exp(-knn_dists), axis=1)

    t = Tomato(graph_type="manual", density_type="manual", metric="precomputed")
    t.fit(knn_indices, weights=weights)
    if plot_diagram:
        t.plot_diagram()
    t.n_clusters_ = int(n_clusters)

    labels = np.asarray(t.labels_, dtype=int)
    intermediates = {
        "acorr": acorr,
        "acorr_norm": acorr_norm,
        "acorr_smooth": acorr_smooth,
        "distance_matrix": dist,
        "knn_indices": knn_indices,
        "knn_dists": knn_dists,
        "density_weights": weights,
    }
    return labels, intermediates


def _cluster_mean_features(
    acorr_smooth: np.ndarray,
    labels: np.ndarray,
    bin_times: np.ndarray,
) -> dict[int, dict[str, float]]:
    centers = 0.5 * (bin_times[:-1] + bin_times[1:])
    early_mask = centers <= 0.015
    theta_mask = (centers >= 0.08) & (centers <= 0.16)
    late_mask = centers >= 0.16

    features: dict[int, dict[str, float]] = {}
    for cid in np.unique(labels):
        rows = acorr_smooth[labels == cid]
        if rows.size == 0:
            continue
        mean_curve = rows.mean(axis=0)
        features[int(cid)] = {
            "early_mass": float(mean_curve[early_mask].sum()) if np.any(early_mask) else 0.0,
            "theta_mass": float(mean_curve[theta_mask].sum()) if np.any(theta_mask) else 0.0,
            "late_mass": float(mean_curve[late_mask].sum()) if np.any(late_mask) else 0.0,
            "peak_index": int(np.argmax(mean_curve)) if mean_curve.size else 0,
        }
    return features


def _guess_btn_mapping(
    labels: np.ndarray,
    acorr_smooth: np.ndarray,
    bin_times: np.ndarray,
) -> dict[int, str]:
    cids = [int(c) for c in np.unique(labels)]
    if not cids:
        return {}

    features = _cluster_mean_features(acorr_smooth, labels, bin_times)
    remaining = set(cids)
    mapping: dict[int, str] = {}

    bursty_cid = max(remaining, key=lambda c: features.get(c, {}).get("early_mass", -np.inf))
    mapping[bursty_cid] = "B"
    remaining.remove(bursty_cid)

    if remaining:
        theta_cid = max(remaining, key=lambda c: features.get(c, {}).get("theta_mass", -np.inf))
        mapping[theta_cid] = "T"
        remaining.remove(theta_cid)

    if remaining:
        ordered_remaining = sorted(
            remaining,
            key=lambda c: features.get(c, {}).get("late_mass", 0.0),
            reverse=True,
        )
        mapping[ordered_remaining[0]] = "N"
        for extra_i, cid in enumerate(ordered_remaining[1:], start=1):
            mapping[cid] = f"extra_{extra_i}"

    return mapping


def _parse_manual_mapping(text: str | None) -> dict[int, str] | None:
    if not text:
        return None
    mapping: dict[int, str] = {}
    for item in text.split(","):
        left, right = item.split(":")
        mapping[int(left.strip())] = right.strip()
    return mapping


def _save_cluster_subsets(
    asa: dict[str, Any],
    labels: np.ndarray,
    mapping: dict[int, str],
    out_dir: Path,
    *,
    dense: bool,
) -> None:
    for cid in np.unique(labels):
        keep = np.where(labels == cid)[0]
        subset = _filter_asa_neurons(asa, keep, dense=dense)
        label_name = mapping.get(int(cid), f"cluster_{int(cid)}").lower()
        subset["subcluster_cluster_id"] = np.array(int(cid))
        subset["subcluster_label"] = np.array(label_name)
        out_path = out_dir / f"asa_{label_name}.npz"
        np.savez(out_path, **subset)


def _run_spike_times_btn(
    asa: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    cfg = BTNConfig(
        n_clusters=args.n_clusters,
        nbs=args.nbs,
        maxt=args.maxt,
        res=args.res,
        smooth_sigma=args.smooth_sigma,
    )
    analyzer = BTNAnalyzer(cfg)
    result = analyzer.classify_btn(
        asa,
        return_intermediates=True,
        plot_diagram=args.plot_diagram,
    )
    labels = np.asarray(result.labels, dtype=int)
    bin_times = np.asarray(result.intermediates["bin_times"])
    return labels, result.intermediates, bin_times


def _run_dense_btn(
    asa: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    spikes = np.asarray(_unwrap_spike_container(asa["spike"]), dtype=float)
    max_lag_bins = max(2, int(np.floor(args.maxt / args.res)) + 1)
    acorr = _binned_temporal_autocorr(
        spikes,
        max_lag_bins=max_lag_bins,
        binary=args.dense_binary,
    )
    labels, intermediates = _cluster_tomato_from_acorr(
        acorr,
        smooth_sigma=args.smooth_sigma,
        nbs=args.nbs,
        n_clusters=args.n_clusters,
        plot_diagram=args.plot_diagram,
    )
    bin_times = np.linspace(0.0, args.maxt, acorr.shape[1] + 1)
    intermediates["bin_times"] = bin_times
    return labels, intermediates, bin_times


def main() -> int:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)

    asa = load_npz_any(args.data)
    if "spike" not in asa:
        raise ValueError("Input ASA file must contain 'spike'.")

    dense = _is_dense_spike_matrix(asa)
    manual_mapping = _parse_manual_mapping(args.cluster_labels)

    if dense:
        spike_matrix = np.asarray(_unwrap_spike_container(asa["spike"]), dtype=float)
        event_counts = _count_events_for_dense(spike_matrix, args.dense_binary)
    else:
        if "t" not in asa:
            raise ValueError("Spike-times BTN analysis requires ASA data to contain 't'.")
        spike_list = _extract_spike_times_list(asa)
        event_counts = np.array([len(s) for s in spike_list], dtype=int)

    keep = np.where(event_counts >= int(args.min_spikes))[0]
    if keep.size == 0:
        raise ValueError("No neurons remain after min-spikes filtering.")

    asa_filtered = _filter_asa_neurons(asa, keep, dense=dense)

    if dense:
        labels, intermediates, bin_times = _run_dense_btn(asa_filtered, args)
        mode = "dense-binned"
    else:
        labels, intermediates, bin_times = _run_spike_times_btn(asa_filtered, args)
        mode = "spike-times"

    mapping = manual_mapping or _guess_btn_mapping(
        labels,
        np.asarray(intermediates["acorr_smooth"]),
        bin_times,
    )

    unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
    print("input mode:", mode)
    print("kept neurons:", int(keep.size))
    print("cluster sizes:", {int(k): int(v) for k, v in zip(unique_labels, cluster_sizes, strict=False)})
    print("cluster mapping:", mapping)

    plot_btn_distance_matrix(
        dist=np.asarray(intermediates["distance_matrix"]),
        labels=labels,
        mapping=mapping,
        sort_by_label=True,
        save_path=str(out_dir / "btn_distance.png"),
        show=False,
    )
    plot_btn_autocorr_summary(
        acorr=np.asarray(intermediates["acorr"]),
        labels=labels,
        bin_times=bin_times,
        mapping=mapping,
        normalize="probability",
        smooth_sigma=args.smooth_sigma,
        save_path=str(out_dir / "btn_autocorr.png"),
        show=False,
    )

    summary = {
        "labels": labels,
        "original_neuron_indices": np.asarray(keep, dtype=int),
        "cluster_ids": unique_labels,
        "cluster_sizes": cluster_sizes,
        "bin_times": bin_times,
        "acorr": np.asarray(intermediates["acorr"]),
        "acorr_smooth": np.asarray(intermediates["acorr_smooth"]),
        "distance_matrix": np.asarray(intermediates["distance_matrix"]),
        "mapping_cluster_ids": np.array(sorted(mapping.keys()), dtype=int),
        "mapping_cluster_labels": np.array(
            [mapping[k] for k in sorted(mapping.keys())],
            dtype=object,
        ),
        "input_mode": np.array(mode),
    }
    np.savez(out_dir / "btn_summary.npz", **summary)

    if args.save_subsets:
        _save_cluster_subsets(asa_filtered, labels, mapping, out_dir, dense=dense)

    print(f"Saved summary: {out_dir / 'btn_summary.npz'}")
    print(f"Saved plots: {out_dir / 'btn_distance.png'}, {out_dir / 'btn_autocorr.png'}")
    if args.save_subsets:
        print("Saved subset ASA files per cluster.")
    else:
        print("Skipped subset ASA export. Use --save-subsets to enable it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
