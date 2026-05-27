from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from canns.analyzer.data.asa.cohospace_scatter import compute_cohoscore_scatter_2d
from canns.analyzer.data.asa.config import SpikeEmbeddingConfig, TDAConfig
from canns.analyzer.data.asa.decode import decode_circular_coordinates_multi
from canns.analyzer.data.asa.embedding import embed_spike_trains
from canns.analyzer.data.asa.tda import tda_vis


def _finite_lifetimes(diagram: np.ndarray) -> np.ndarray:
    if diagram is None:
        return np.zeros(0, dtype=float)
    dgm = np.asarray(diagram, dtype=float)
    if dgm.size == 0:
        return np.zeros(0, dtype=float)
    mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
    lives = dgm[mask, 1] - dgm[mask, 0]
    lives = lives[np.isfinite(lives) & (lives > 0)]
    if lives.size == 0:
        return np.zeros(0, dtype=float)
    return np.sort(lives)[::-1]


def _safe_stat(arr: np.ndarray, idx: int) -> float:
    return float(arr[idx]) if idx < arr.size else 0.0


def summarize_torus_topology(
    persistence_result: dict[str, Any],
    *,
    mode: str = "h1h2",
    min_h1_prominence_ratio: float = 1.2,
    min_h2_prominence_ratio: float = 1.2,
    min_h1_strength: float = 0.05,
    min_h2_strength: float = 0.03,
) -> dict[str, Any]:
    """
    Summarize whether a persistence result is plausibly torus-like (1,2,1).

    Notes
    -----
    This is a lightweight internal heuristic.
    Modes:
    - ``h1``: require two dominant H1 bars.
    - ``h1h2``: require two dominant H1 bars plus one dominant H2 bar.
    """
    persistence = persistence_result["persistence"]
    dgms = persistence["dgms"]

    h0 = _finite_lifetimes(dgms[0]) if len(dgms) > 0 else np.zeros(0, dtype=float)
    h1 = _finite_lifetimes(dgms[1]) if len(dgms) > 1 else np.zeros(0, dtype=float)
    h2 = _finite_lifetimes(dgms[2]) if len(dgms) > 2 else np.zeros(0, dtype=float)

    h1_top1 = _safe_stat(h1, 0)
    h1_top2 = _safe_stat(h1, 1)
    h1_top3 = _safe_stat(h1, 2)
    h2_top1 = _safe_stat(h2, 0)
    h2_top2 = _safe_stat(h2, 1)

    h1_ratio = h1_top2 / max(h1_top3, 1e-8)
    h2_ratio = h2_top1 / max(h2_top2, 1e-8)
    h1_strength = 0.5 * (h1_top1 + h1_top2)
    h2_strength = h2_top1

    # Blend bar prominence and absolute persistence magnitude.
    topology_score = float(
        h1_strength + h2_strength + 0.1 * np.log1p(h1_ratio) + 0.1 * np.log1p(h2_ratio)
    )

    h1_valid = bool(
        h1.size >= 2 and h1_top2 >= min_h1_strength and h1_ratio >= min_h1_prominence_ratio
    )
    h2_valid = bool(
        h2.size >= 1 and h2_top1 >= min_h2_strength and h2_ratio >= min_h2_prominence_ratio
    )
    if mode == "h1":
        valid = h1_valid
    elif mode == "h1h2":
        valid = h1_valid and h2_valid
    else:
        raise ValueError(f"Unknown topology mode: {mode}")

    return {
        "mode": mode,
        "valid": valid,
        "h1_valid": h1_valid,
        "h2_valid": h2_valid,
        "topology_score": topology_score,
        "h0_lifetimes": h0,
        "h1_lifetimes": h1,
        "h2_lifetimes": h2,
        "h1_top1": h1_top1,
        "h1_top2": h1_top2,
        "h1_top3": h1_top3,
        "h2_top1": h2_top1,
        "h2_top2": h2_top2,
        "h1_prominence_ratio": float(h1_ratio),
        "h2_prominence_ratio": float(h2_ratio),
        "h1_strength": float(h1_strength),
        "h2_strength": float(h2_strength),
    }


def summarize_shuffle_topology(
    persistence_result: dict[str, Any],
    *,
    mode: str = "h1",
    h1_primary_percentile: float = 99.0,
    h1_secondary_percentile: float = 95.0,
    h2_primary_percentile: float = 95.0,
    min_h1_primary_ratio: float = 1.0,
    min_h1_secondary_ratio: float = 1.0,
    min_h2_primary_ratio: float = 1.0,
) -> dict[str, Any]:
    """
    Summarize topology validity against shuffle-derived null thresholds.

    Notes
    -----
    For torus-like H1 structure we require both the strongest and the second
    strongest H1 bars to clear shuffle-derived thresholds, because a torus
    should contribute two meaningful H1 loops.
    """
    topo = summarize_torus_topology(persistence_result, mode=mode)
    shuffle_max = persistence_result.get("shuffle_max") or {}

    h1_null = np.asarray(shuffle_max.get(1, []), dtype=float)
    h2_null = np.asarray(shuffle_max.get(2, []), dtype=float)
    h1_null = h1_null[np.isfinite(h1_null)]
    h2_null = h2_null[np.isfinite(h2_null)]

    def _pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else float("nan")

    h1_primary_thr = _pct(h1_null, h1_primary_percentile)
    h1_secondary_thr = _pct(h1_null, h1_secondary_percentile)
    h2_primary_thr = _pct(h2_null, h2_primary_percentile)

    h1_top1 = float(topo["h1_top1"])
    h1_top2 = float(topo["h1_top2"])
    h2_top1 = float(topo["h2_top1"])

    h1_top1_ratio = (
        h1_top1 / max(h1_primary_thr, 1e-8) if np.isfinite(h1_primary_thr) else float("nan")
    )
    h1_top2_ratio = (
        h1_top2 / max(h1_secondary_thr, 1e-8) if np.isfinite(h1_secondary_thr) else float("nan")
    )
    h2_top1_ratio = (
        h2_top1 / max(h2_primary_thr, 1e-8) if np.isfinite(h2_primary_thr) else float("nan")
    )

    h1_valid = bool(
        h1_null.size > 0
        and np.isfinite(h1_primary_thr)
        and np.isfinite(h1_secondary_thr)
        and h1_top1_ratio >= min_h1_primary_ratio
        and h1_top2_ratio >= min_h1_secondary_ratio
    )
    h2_valid = bool(
        h2_null.size > 0 and np.isfinite(h2_primary_thr) and h2_top1_ratio >= min_h2_primary_ratio
    )

    if mode == "h1":
        valid = h1_valid
    elif mode == "h1h2":
        valid = h1_valid and h2_valid
    else:
        raise ValueError(f"Unknown shuffle topology mode: {mode}")

    return {
        "mode": mode,
        "valid": valid,
        "h1_valid": h1_valid,
        "h2_valid": h2_valid,
        "h1_null_count": int(h1_null.size),
        "h2_null_count": int(h2_null.size),
        "h1_primary_percentile": float(h1_primary_percentile),
        "h1_secondary_percentile": float(h1_secondary_percentile),
        "h2_primary_percentile": float(h2_primary_percentile),
        "h1_primary_threshold": h1_primary_thr,
        "h1_secondary_threshold": h1_secondary_thr,
        "h2_primary_threshold": h2_primary_thr,
        "h1_top1_ratio": float(h1_top1_ratio),
        "h1_top2_ratio": float(h1_top2_ratio),
        "h2_top1_ratio": float(h2_top1_ratio),
        "topology_score": float(
            np.nan_to_num(h1_top1_ratio, nan=0.0)
            + np.nan_to_num(h1_top2_ratio, nan=0.0)
            + 0.5 * np.nan_to_num(h2_top1_ratio, nan=0.0)
        ),
    }


def _corr_summary(coho_scores: np.ndarray, grid_scores: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(coho_scores) & np.isfinite(grid_scores)
    x = np.asarray(coho_scores[mask], dtype=float)
    y = np.asarray(grid_scores[mask], dtype=float)
    if x.size < 3:
        return {
            "n": int(x.size),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def _compute_grid_scores(
    x: np.ndarray,
    y: np.ndarray,
    spikes: np.ndarray,
    *,
    bins: int = 50,
    overlap: float = 0.8,
    min_occupancy: int = 1,
) -> np.ndarray:
    from canns.analyzer.data.cell_classification import GridnessAnalyzer, compute_2d_autocorrelation

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    sp = np.asarray(spikes, dtype=float)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    eps = 1e-12
    ix = np.floor((x - xmin) / (xmax - xmin + eps) * bins).astype(int)
    iy = np.floor((y - ymin) / (ymax - ymin + eps) * bins).astype(int)
    ix = np.clip(ix, 0, bins - 1)
    iy = np.clip(iy, 0, bins - 1)
    flat = (iy * bins + ix).astype(int)
    occ = np.bincount(flat, minlength=bins * bins).astype(float).reshape(bins, bins)
    occ_mask = occ >= float(min_occupancy)

    analyzer = GridnessAnalyzer()
    scores = np.full(sp.shape[1], np.nan, dtype=float)
    for nid in range(sp.shape[1]):
        spike_map = (
            np.bincount(flat, weights=sp[:, nid], minlength=bins * bins)
            .astype(float)
            .reshape(bins, bins)
        )
        rate_map = np.zeros_like(spike_map)
        rate_map[occ_mask] = spike_map[occ_mask] / occ[occ_mask]
        autocorr = compute_2d_autocorrelation(rate_map, overlap=overlap)
        result = analyzer.compute_gridness_score(autocorr)
        scores[nid] = float(result.score)
    return scores


def analyze_auto_grid_threshold(
    asa_data: dict[str, Any],
    *,
    embed_config: SpikeEmbeddingConfig | None = None,
    tda_config: TDAConfig | None = None,
    use_precomputed_spikes: bool = False,
    top_percent: float | None = 2.0,
    topk_values: list[int] | np.ndarray | None = None,
    min_topk: int = 30,
    max_topk: int | None = None,
    topk_step: int = 5,
    grid_bins: int = 50,
    grid_overlap: float = 0.8,
    min_corr_neurons: int = 15,
    topology_mode: str = "h1",
    min_h1_prominence_ratio: float = 1.2,
    min_h2_prominence_ratio: float = 1.2,
    min_h1_strength: float = 0.05,
    min_h2_strength: float = 0.03,
    use_shuffle_refinement: bool = False,
    shuffle_num_shuffles: int = 50,
    shuffle_shortlist_size: int = 5,
    shuffle_shortlist_neighbors: int = 1,
    shuffle_h1_primary_percentile: float = 99.0,
    shuffle_h1_secondary_percentile: float = 95.0,
    shuffle_h2_primary_percentile: float = 95.0,
    shuffle_min_h1_primary_ratio: float = 1.0,
    shuffle_min_h1_secondary_ratio: float = 1.0,
    shuffle_min_h2_primary_ratio: float = 1.0,
) -> dict[str, Any]:
    """
    Sweep grid-score thresholds using a top-k retained-neuron formulation.

    Workflow
    --------
    1. Compute embedded activity once on all neurons, or reuse precomputed
       dense spikes/activity when `use_precomputed_spikes=True`.
    2. Compute grid scores for all neurons.
    3. Sort neurons by grid score and evaluate top-k subsets.
    4. For each k:
       - run TDA,
       - summarize torus-likeness,
       - decode circular coordinates if valid enough,
       - compute cohoscore/gridscore correlation.
    5. Choose the best valid k by strongest negative Pearson correlation,
       breaking ties by larger topology score.
    """
    if embed_config is None:
        embed_config = SpikeEmbeddingConfig(smooth=True, speed_filter=True, min_speed=2.5)

    if tda_config is None:
        default_maxdim = 1 if topology_mode == "h1" else 2
        tda_config = TDAConfig(
            maxdim=default_maxdim, n_points=1200, show=False, do_shuffle=False, progress_bar=True
        )
    elif topology_mode == "h1h2" and tda_config.maxdim < 2:
        tda_config = TDAConfig(
            dim=tda_config.dim,
            num_times=tda_config.num_times,
            active_times=tda_config.active_times,
            k=tda_config.k,
            n_points=tda_config.n_points,
            metric=tda_config.metric,
            nbs=tda_config.nbs,
            maxdim=2,
            coeff=tda_config.coeff,
            show=tda_config.show,
            do_shuffle=tda_config.do_shuffle,
            num_shuffles=tda_config.num_shuffles,
            progress_bar=tda_config.progress_bar,
            standardize=tda_config.standardize,
        )

    if use_precomputed_spikes:
        spikes = np.asarray(asa_data["spike"], dtype=float)
        xx = np.asarray(asa_data["x"], dtype=float).ravel()
        yy = np.asarray(asa_data["y"], dtype=float).ravel()
        tt = np.asarray(asa_data["t"]) if "t" in asa_data else np.arange(spikes.shape[0])
        if spikes.ndim != 2:
            raise ValueError(f"Expected precomputed spikes to be 2D (T,N), got {spikes.shape}")
        m = min(spikes.shape[0], len(xx), len(yy), len(tt))
        spikes = spikes[:m, :]
        xx = xx[:m]
        yy = yy[:m]
        tt = np.asarray(tt)[:m]
    else:
        spikes, xx, yy, tt = embed_spike_trains(asa_data, config=embed_config)
    grid_scores_all = _compute_grid_scores(xx, yy, spikes, bins=grid_bins, overlap=grid_overlap)

    valid_grid_idx = np.where(np.isfinite(grid_scores_all))[0]
    if valid_grid_idx.size < 2:
        raise ValueError(
            f"Need at least 2 neurons with finite grid scores, got {valid_grid_idx.size}."
        )

    order = valid_grid_idx[np.argsort(grid_scores_all[valid_grid_idx])[::-1]]
    n_total = int(order.size)
    max_topk = n_total if max_topk is None else min(int(max_topk), n_total)
    min_topk = max(2, min(int(min_topk), max_topk))

    if topk_values is None:
        topk_values = list(range(min_topk, max_topk + 1, max(1, int(topk_step))))
        if not topk_values:
            raise ValueError(
                f"No top-k values to evaluate: min_topk={min_topk}, max_topk={max_topk}."
            )
        if topk_values[-1] != max_topk:
            topk_values.append(max_topk)
    else:
        topk_values = sorted(
            {int(k) for k in np.asarray(topk_values).ravel() if min_topk <= int(k) <= max_topk}
        )
        if not topk_values:
            raise ValueError(
                f"No top-k values to evaluate after filtering explicit topk_values "
                f"to [{min_topk}, {max_topk}]."
            )

    sweep_results: list[dict[str, Any]] = []

    def _score_candidate(candidate: dict[str, Any]) -> tuple[float, float, int]:
        corr = candidate.get("correlation") or {}
        topo = candidate.get("topology") or {}
        pearson = corr.get("pearson_r", np.nan)
        objective = -float(pearson) if np.isfinite(pearson) else float("-inf")
        topology_score = float(topo.get("topology_score", float("-inf")))
        topk_value = int(candidate.get("topk", 0))
        return (objective, topology_score, topk_value)

    for topk in topk_values:
        neuron_ids = order[:topk]
        threshold = float(grid_scores_all[neuron_ids[-1]])
        spikes_sub = spikes[:, neuron_ids]
        grid_scores_sub = grid_scores_all[neuron_ids]

        candidate: dict[str, Any] = {
            "topk": int(topk),
            "grid_threshold": threshold,
            "neuron_ids": neuron_ids,
            "grid_scores": grid_scores_sub,
            "status": "pending",
        }

        try:
            persistence_result = tda_vis(embed_data=spikes_sub, config=tda_config)
            topo = summarize_torus_topology(
                persistence_result,
                mode=topology_mode,
                min_h1_prominence_ratio=min_h1_prominence_ratio,
                min_h2_prominence_ratio=min_h2_prominence_ratio,
                min_h1_strength=min_h1_strength,
                min_h2_strength=min_h2_strength,
            )
            candidate["topology"] = topo

            # Decode even if topology is not formally valid, but mark invalid later.
            spike_data_sub = dict(asa_data)
            spike_data_sub["spike"] = spikes_sub
            decoding = decode_circular_coordinates_multi(
                persistence_result=persistence_result,
                spike_data=spike_data_sub,
                num_circ=2,
            )
            coho_scores = compute_cohoscore_scatter_2d(
                np.asarray(decoding["coords"], dtype=float),
                spikes_sub,
                top_percent=top_percent,
                times=np.asarray(decoding.get("times")),
            )
            corr = _corr_summary(coho_scores, grid_scores_sub)

            candidate["cohoscore"] = coho_scores
            candidate["correlation"] = corr
            candidate["status"] = (
                "valid" if (topo["valid"] and corr["n"] >= min_corr_neurons) else "invalid"
            )
            candidate["objective"] = (
                float(-corr["pearson_r"]) if candidate["status"] == "valid" else float("nan")
            )
            candidate["validation_stage"] = "coarse"
        except Exception as exc:
            candidate["status"] = "failed"
            candidate["error"] = str(exc)

        sweep_results.append(candidate)

    valid_candidates = [
        item
        for item in sweep_results
        if item["status"] == "valid" and np.isfinite(item.get("objective", np.nan))
    ]
    best = None
    if valid_candidates:
        best = max(valid_candidates, key=_score_candidate)

    refined_results: list[dict[str, Any]] = []
    if use_shuffle_refinement and sweep_results:
        candidate_map = {int(item["topk"]): item for item in sweep_results}
        ranked = sorted(valid_candidates, key=_score_candidate, reverse=True)
        if not ranked:
            ranked = sorted(
                [
                    item
                    for item in sweep_results
                    if item.get("status") != "failed"
                    and item.get("correlation", {}).get("n", 0) >= min_corr_neurons
                ],
                key=_score_candidate,
                reverse=True,
            )

        shortlist_topks: set[int] = set()
        for item in ranked[: max(1, int(shuffle_shortlist_size))]:
            base_topk = int(item["topk"])
            shortlist_topks.add(base_topk)
            for offset in range(1, max(0, int(shuffle_shortlist_neighbors)) + 1):
                shortlist_topks.add(base_topk - offset * max(1, int(topk_step)))
                shortlist_topks.add(base_topk + offset * max(1, int(topk_step)))

        shortlist_topks = {k for k in shortlist_topks if k in candidate_map}
        shuffle_tda_config = TDAConfig(
            dim=tda_config.dim,
            num_times=tda_config.num_times,
            active_times=tda_config.active_times,
            k=tda_config.k,
            n_points=tda_config.n_points,
            metric=tda_config.metric,
            nbs=tda_config.nbs,
            maxdim=tda_config.maxdim,
            coeff=tda_config.coeff,
            show=False,
            do_shuffle=True,
            num_shuffles=shuffle_num_shuffles,
            progress_bar=tda_config.progress_bar,
            standardize=tda_config.standardize,
        )

        for topk in sorted(shortlist_topks):
            coarse_candidate = candidate_map[topk]
            refined_candidate = dict(coarse_candidate)
            neuron_ids = np.asarray(coarse_candidate["neuron_ids"], dtype=int)
            spikes_sub = spikes[:, neuron_ids]
            try:
                persistence_result = tda_vis(embed_data=spikes_sub, config=shuffle_tda_config)
                shuffle_topology = summarize_shuffle_topology(
                    persistence_result,
                    mode=topology_mode,
                    h1_primary_percentile=shuffle_h1_primary_percentile,
                    h1_secondary_percentile=shuffle_h1_secondary_percentile,
                    h2_primary_percentile=shuffle_h2_primary_percentile,
                    min_h1_primary_ratio=shuffle_min_h1_primary_ratio,
                    min_h1_secondary_ratio=shuffle_min_h1_secondary_ratio,
                    min_h2_primary_ratio=shuffle_min_h2_primary_ratio,
                )
                refined_candidate["shuffle_topology"] = shuffle_topology
                refined_candidate["validation_stage"] = "shuffle"
                refined_candidate["status"] = (
                    "valid"
                    if shuffle_topology["valid"]
                    and refined_candidate.get("correlation", {}).get("n", 0) >= min_corr_neurons
                    else "invalid"
                )
                refined_candidate["objective"] = (
                    float(-refined_candidate["correlation"]["pearson_r"])
                    if refined_candidate["status"] == "valid"
                    else float("nan")
                )
            except Exception as exc:
                refined_candidate["status"] = "failed"
                refined_candidate["validation_stage"] = "shuffle"
                refined_candidate["error"] = str(exc)
            refined_results.append(refined_candidate)

        valid_refined = [
            item
            for item in refined_results
            if item["status"] == "valid" and np.isfinite(item.get("objective", np.nan))
        ]
        if valid_refined:
            best = max(
                valid_refined,
                key=lambda item: (
                    float(item["objective"]),
                    float(item.get("shuffle_topology", {}).get("topology_score", float("-inf"))),
                    int(item["topk"]),
                ),
            )

    return {
        "embed_config": {
            "smooth": embed_config.smooth,
            "speed_filter": embed_config.speed_filter,
            "min_speed": embed_config.min_speed,
        },
        "use_precomputed_spikes": use_precomputed_spikes,
        "tda_config": {
            "dim": tda_config.dim,
            "num_times": tda_config.num_times,
            "active_times": tda_config.active_times,
            "k": tda_config.k,
            "n_points": tda_config.n_points,
            "metric": tda_config.metric,
            "nbs": tda_config.nbs,
            "maxdim": tda_config.maxdim,
            "coeff": tda_config.coeff,
            "standardize": tda_config.standardize,
        },
        "top_percent": top_percent,
        "topology_mode": topology_mode,
        "use_shuffle_refinement": use_shuffle_refinement,
        "shuffle_refinement": {
            "num_shuffles": int(shuffle_num_shuffles),
            "shortlist_size": int(shuffle_shortlist_size),
            "shortlist_neighbors": int(shuffle_shortlist_neighbors),
            "h1_primary_percentile": float(shuffle_h1_primary_percentile),
            "h1_secondary_percentile": float(shuffle_h1_secondary_percentile),
            "h2_primary_percentile": float(shuffle_h2_primary_percentile),
            "min_h1_primary_ratio": float(shuffle_min_h1_primary_ratio),
            "min_h1_secondary_ratio": float(shuffle_min_h1_secondary_ratio),
            "min_h2_primary_ratio": float(shuffle_min_h2_primary_ratio),
        },
        "grid_scores_all": grid_scores_all,
        "sorted_neuron_ids": order,
        "topk_values": np.asarray(topk_values, dtype=int),
        "sweep_results": sweep_results,
        "refined_results": refined_results,
        "best_result": best,
    }


def analyze_auto_grid_threshold_workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the automatic grid-score threshold workflow."""
    return analyze_auto_grid_threshold(*args, **kwargs)


__all__ = [
    "analyze_auto_grid_threshold_workflow",
    "analyze_auto_grid_threshold",
    "summarize_shuffle_topology",
    "summarize_torus_topology",
]
