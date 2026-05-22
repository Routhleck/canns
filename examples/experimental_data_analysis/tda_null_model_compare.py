#!/usr/bin/env python3
"""
Compare persistent-homology null models on a grid dataset.

This script is designed for side-by-side evaluation of several commonly used
surrogate/null-model strategies for TDA barcode significance:

- ``local_circular_shift``: the current ASA-style mild shuffle
- ``full_circular_shift``: full-range independent circular shifts
- ``block_shuffle``: preserves local temporal structure within blocks
- ``time_permutation``: strong null by permuting time indices

The script keeps the persistent-homology pipeline aligned with
``canns.analyzer.data.asa.tda_vis`` by reusing the same internal PH routine.
It writes a JSON summary that is easy to compare across null models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from canns.analyzer.data.asa.config import SpikeEmbeddingConfig, TDAConfig
from canns.analyzer.data.asa.embedding import embed_spike_trains
from canns.analyzer.data.asa.tda import _compute_persistence
from canns.data.loaders import load_grid_data


def _finite_lifetimes(diagram: np.ndarray | None) -> np.ndarray:
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


def summarize_real_topology(persistence: dict[str, Any]) -> dict[str, float | bool]:
    dgms = persistence["dgms"]
    h0 = _finite_lifetimes(dgms[0]) if len(dgms) > 0 else np.zeros(0, dtype=float)
    h1 = _finite_lifetimes(dgms[1]) if len(dgms) > 1 else np.zeros(0, dtype=float)
    h2 = _finite_lifetimes(dgms[2]) if len(dgms) > 2 else np.zeros(0, dtype=float)

    h1_top1 = _safe_stat(h1, 0)
    h1_top2 = _safe_stat(h1, 1)
    h1_top3 = _safe_stat(h1, 2)
    h2_top1 = _safe_stat(h2, 0)
    h2_top2 = _safe_stat(h2, 1)

    return {
        "h0_count": int(h0.size),
        "h1_count": int(h1.size),
        "h2_count": int(h2.size),
        "h1_top1": h1_top1,
        "h1_top2": h1_top2,
        "h1_top3": h1_top3,
        "h2_top1": h2_top1,
        "h2_top2": h2_top2,
        "h1_prominence_ratio": h1_top2 / max(h1_top3, 1e-8) if h1.size >= 2 else 0.0,
        "h2_prominence_ratio": h2_top1 / max(h2_top2, 1e-8) if h2.size >= 1 else 0.0,
        "h1_strength": 0.5 * (h1_top1 + h1_top2),
        "h2_strength": h2_top1,
    }


def _extract_dim_maxima(persistence: dict[str, Any], maxdim: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for dim in range(maxdim + 1):
        if dim >= len(persistence["dgms"]):
            continue
        life = _finite_lifetimes(persistence["dgms"][dim])
        if life.size:
            out[dim] = float(life[0])
    return out


def _local_circular_shift(data: np.ndarray, rng: np.random.Generator, max_fraction: float) -> np.ndarray:
    shuffled = np.array(data, copy=True)
    t_steps, num_neurons = shuffled.shape
    max_shift = max(1, int(t_steps * max_fraction))
    for n in range(num_neurons):
        shift = int(rng.integers(0, max_shift))
        shuffled[:, n] = np.roll(shuffled[:, n], shift)
    return shuffled


def _full_circular_shift(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.array(data, copy=True)
    t_steps, num_neurons = shuffled.shape
    for n in range(num_neurons):
        shift = int(rng.integers(0, t_steps))
        shuffled[:, n] = np.roll(shuffled[:, n], shift)
    return shuffled


def _block_shuffle(data: np.ndarray, rng: np.random.Generator, block_size: int) -> np.ndarray:
    t_steps = data.shape[0]
    if block_size <= 1 or block_size >= t_steps:
        return np.array(data, copy=True)

    starts = list(range(0, t_steps, block_size))
    shuffled = np.empty_like(data)
    for n in range(data.shape[1]):
        blocks = [np.array(data[s : min(s + block_size, t_steps), n], copy=True) for s in starts]
        order = rng.permutation(len(blocks))
        shuffled[:, n] = np.concatenate([blocks[i] for i in order], axis=0)[:t_steps]
    return shuffled


def _time_permutation(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.empty_like(data)
    for n in range(data.shape[1]):
        shuffled[:, n] = data[rng.permutation(data.shape[0]), n]
    return shuffled


def _make_surrogate(
    data: np.ndarray,
    method: str,
    rng: np.random.Generator,
    *,
    local_shift_fraction: float,
    block_size: int,
) -> np.ndarray:
    if method == "local_circular_shift":
        return _local_circular_shift(data, rng, local_shift_fraction)
    if method == "full_circular_shift":
        return _full_circular_shift(data, rng)
    if method == "block_shuffle":
        return _block_shuffle(data, rng, block_size)
    if method == "time_permutation":
        return _time_permutation(data, rng)
    raise ValueError(f"Unknown null-model method: {method}")


def _summarize_null_model(
    name: str,
    maxima: dict[int, list[float]],
    real_topology: dict[str, float | bool],
    *,
    h1_primary_percentile: float,
    h1_secondary_percentile: float,
    h2_primary_percentile: float,
) -> dict[str, Any]:
    h1_null = np.asarray(maxima.get(1, []), dtype=float)
    h2_null = np.asarray(maxima.get(2, []), dtype=float)
    h1_null = h1_null[np.isfinite(h1_null)]
    h2_null = h2_null[np.isfinite(h2_null)]

    def _pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else float("nan")

    def _empirical_p(arr: np.ndarray, value: float) -> float:
        if arr.size == 0:
            return float("nan")
        return float((1.0 + np.sum(arr >= value)) / (arr.size + 1.0))

    h1_top1 = float(real_topology["h1_top1"])
    h1_top2 = float(real_topology["h1_top2"])
    h2_top1 = float(real_topology["h2_top1"])

    h1_primary_thr = _pct(h1_null, h1_primary_percentile)
    h1_secondary_thr = _pct(h1_null, h1_secondary_percentile)
    h2_primary_thr = _pct(h2_null, h2_primary_percentile)

    return {
        "method": name,
        "iterations": int(max(len(v) for v in maxima.values()) if maxima else 0),
        "h1_null_count": int(h1_null.size),
        "h2_null_count": int(h2_null.size),
        "h1_primary_threshold": h1_primary_thr,
        "h1_secondary_threshold": h1_secondary_thr,
        "h2_primary_threshold": h2_primary_thr,
        "h1_top1_ratio": h1_top1 / max(h1_primary_thr, 1e-8) if np.isfinite(h1_primary_thr) else float("nan"),
        "h1_top2_ratio": h1_top2 / max(h1_secondary_thr, 1e-8) if np.isfinite(h1_secondary_thr) else float("nan"),
        "h2_top1_ratio": h2_top1 / max(h2_primary_thr, 1e-8) if np.isfinite(h2_primary_thr) else float("nan"),
        "h1_top1_pvalue": _empirical_p(h1_null, h1_top1),
        "h1_top2_pvalue": _empirical_p(h1_null, h1_top2),
        "h2_top1_pvalue": _empirical_p(h2_null, h2_top1),
        "h1_null_mean": float(np.mean(h1_null)) if h1_null.size else float("nan"),
        "h1_null_std": float(np.std(h1_null)) if h1_null.size else float("nan"),
        "h2_null_mean": float(np.mean(h2_null)) if h2_null.size else float("nan"),
        "h2_null_std": float(np.std(h2_null)) if h2_null.size else float("nan"),
    }


def _thresholds_from_summary(summary: dict[str, Any], maxdim: int) -> dict[int, float]:
    thresholds = {0: 0.0}
    if maxdim >= 1:
        thresholds[1] = float(summary["h1_primary_threshold"])
    if maxdim >= 2:
        thresholds[2] = float(summary["h2_primary_threshold"])
    return thresholds


def _plot_multi_null_barcode(
    persistence: dict[str, Any],
    null_summaries: list[dict[str, Any]],
    output_path: Path,
    *,
    top_n: int = 30,
) -> None:
    maxdim = len(persistence["dgms"]) - 1
    dims = list(range(maxdim + 1))

    min_birth, max_death = 0.0, 0.0
    for dim in dims:
        valid = np.asarray([bar for bar in persistence["dgms"][dim] if not np.isinf(bar[1])])
        if valid.size:
            min_birth = min(min_birth, float(np.min(valid[:, 0])))
            max_death = max(max_death, float(np.max(valid[:, 1])))
    if max_death == 0 and min_birth == 0:
        max_death = 1.0
    infinity = max_death + (max_death - min_birth) * 0.1

    fig, axes = plt.subplots(
        len(dims),
        len(null_summaries),
        figsize=(4.8 * len(null_summaries), 2.8 * len(dims)),
        squeeze=False,
        sharex=True,
    )
    dim_colors = {
        0: (0.1, 0.45, 0.75),
        1: (0.0, 0.55, 0.2),
        2: (0.5, 0.25, 0.75),
    }

    for col, summary in enumerate(null_summaries):
        thresholds = _thresholds_from_summary(summary, maxdim)
        for row, dim in enumerate(dims):
            ax = axes[row][col]
            d = np.copy(np.asarray(persistence["dgms"][dim], dtype=float))
            if d.size == 0:
                d = np.zeros((0, 2), dtype=float)
            d[np.isinf(d[:, 1]), 1] = infinity
            lives = d[:, 1] - d[:, 0]

            threshold = thresholds.get(dim, 0.0)
            if np.isfinite(threshold) and threshold > 0:
                ax.axvspan(0, threshold, alpha=0.18, color="gray", zorder=-3)
                ax.axvline(threshold, color="gray", linestyle="--", alpha=0.75, linewidth=1.2)

            if lives.size:
                dinds = np.argsort(lives)[-top_n:]
                if dim > 0:
                    dinds = dinds[np.flip(np.argsort(d[dinds, 0]))]
                for idx, bar_idx in enumerate(dinds):
                    significant = lives[bar_idx] > threshold
                    ax.barh(
                        idx + 0.5,
                        lives[bar_idx],
                        left=d[bar_idx, 0],
                        height=0.76,
                        color="red" if significant else dim_colors.get(dim, (0.0, 0.55, 0.2)),
                        linewidth=0,
                    )
                y_max = len(dinds)
            else:
                y_max = 1

            ax.set_ylim(0, max(y_max, 1))
            ax.set_xlim(0, infinity)
            ax.set_yticks([])
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.8)
            if col == 0:
                ax.set_ylabel(f"H{dim}", rotation=0, labelpad=18, fontsize=12, fontweight="bold")
            if row == 0:
                ax.set_title(summary["method"], fontsize=11)
            if row == len(dims) - 1:
                ax.set_xlabel("lifetime")

    fig.suptitle(
        "Real barcode under different null-model thresholds\n"
        "gray = null threshold region, red = real bars above threshold",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_null_ratio_summary(null_summaries: list[dict[str, Any]], output_path: Path) -> None:
    methods = [item["method"] for item in null_summaries]
    metrics = [
        ("h1_top1_ratio", "H1 top1"),
        ("h1_top2_ratio", "H1 top2"),
        ("h2_top1_ratio", "H2 top1"),
    ]
    x = np.arange(len(methods))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#2f7bbd", "#15a36d", "#b24a3f"]
    for idx, (key, label) in enumerate(metrics):
        values = [float(item[key]) for item in null_summaries]
        ax.bar(x + (idx - 1) * width, values, width=width, label=label, color=colors[idx])

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel("real lifetime / null threshold")
    ax.set_title("Barcode strength relative to each null-model threshold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_null_model_comparison(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_grid_data(dataset_key=args.dataset)
    if dataset is None:
        raise RuntimeError(f"Failed to load dataset: {args.dataset}")

    spike_cfg = SpikeEmbeddingConfig(
        res=args.res,
        dt=args.dt,
        sigma=args.sigma,
        smooth=not args.no_smooth,
        speed_filter=args.speed_filter,
        min_speed=args.min_speed,
    )
    spikes, _, _, _ = embed_spike_trains(dataset, config=spike_cfg)

    tda_cfg = TDAConfig(
        dim=args.dim,
        num_times=args.num_times,
        active_times=args.active_times,
        k=args.k,
        n_points=args.n_points,
        metric=args.metric,
        nbs=args.nbs,
        maxdim=args.maxdim,
        coeff=args.coeff,
        show=False,
        do_shuffle=False,
        num_shuffles=args.num_shuffles,
        progress_bar=not args.no_progress_bar,
        standardize=not args.no_standardize,
    )

    ph_kwargs = {
        "dim": tda_cfg.dim,
        "num_times": tda_cfg.num_times,
        "active_times": tda_cfg.active_times,
        "k": tda_cfg.k,
        "n_points": tda_cfg.n_points,
        "metric": tda_cfg.metric,
        "nbs": tda_cfg.nbs,
        "maxdim": tda_cfg.maxdim,
        "coeff": tda_cfg.coeff,
        "progress_bar": tda_cfg.progress_bar,
    }

    print(
        "Loaded spikes for "
        f"{args.dataset}: shape={spikes.shape}, "
        f"smooth={spike_cfg.smooth}, speed_filter={spike_cfg.speed_filter}"
    )
    print(
        "Running real PH with "
        f"dim={tda_cfg.dim}, active_times={tda_cfg.active_times}, "
        f"n_points={tda_cfg.n_points}, maxdim={tda_cfg.maxdim}"
    )

    real_persistence = _compute_persistence(
        spikes,
        **ph_kwargs,
    )
    real_topology = summarize_real_topology(real_persistence)
    print(
        "Real topology summary: "
        f"H1 top1={real_topology['h1_top1']:.6f}, "
        f"H1 top2={real_topology['h1_top2']:.6f}, "
        f"H2 top1={real_topology['h2_top1']:.6f}"
    )

    rng = np.random.default_rng(args.seed)
    method_names = [
        "local_circular_shift",
        "full_circular_shift",
        "block_shuffle",
        "time_permutation",
    ]
    null_summaries: list[dict[str, Any]] = []

    for method in method_names:
        print(f"\n[{method}] starting {args.num_shuffles} surrogate iterations")
        maxima: dict[int, list[float]] = {dim: [] for dim in range(tda_cfg.maxdim + 1)}
        for idx in range(args.num_shuffles):
            surrogate = _make_surrogate(
                spikes,
                method,
                rng,
                local_shift_fraction=args.local_shift_fraction,
                block_size=args.block_size,
            )
            persistence = _compute_persistence(
                surrogate,
                **ph_kwargs,
            )
            dim_maxima = _extract_dim_maxima(persistence, tda_cfg.maxdim)
            for dim, value in dim_maxima.items():
                maxima[dim].append(value)
            print(
                f"[{method}] iteration {idx + 1}/{args.num_shuffles} "
                f"H1={dim_maxima.get(1, float('nan')):.6f} "
                f"H2={dim_maxima.get(2, float('nan')):.6f}"
            )

        summary = _summarize_null_model(
                method,
                maxima,
                real_topology,
                h1_primary_percentile=args.h1_primary_percentile,
                h1_secondary_percentile=args.h1_secondary_percentile,
                h2_primary_percentile=args.h2_primary_percentile,
            )
        print(
            f"[{method}] done: "
            f"H1 top1 ratio={summary['h1_top1_ratio']:.3f}, "
            f"H1 top2 ratio={summary['h1_top2_ratio']:.3f}, "
            f"H2 top1 ratio={summary['h2_top1_ratio']:.3f}"
        )
        null_summaries.append(summary)

    return {
        "dataset": args.dataset,
        "seed": args.seed,
        "spike_embedding_config": asdict(spike_cfg),
        "tda_config": asdict(tda_cfg),
        "spikes_shape": list(spikes.shape),
        "real_topology": real_topology,
        "null_model_results": null_summaries,
        "comparison_notes": {
            "local_circular_shift_fraction": args.local_shift_fraction,
            "block_size": args.block_size,
            "h1_primary_percentile": args.h1_primary_percentile,
            "h1_secondary_percentile": args.h1_secondary_percentile,
            "h2_primary_percentile": args.h2_primary_percentile,
        },
        "_real_persistence_for_plotting": real_persistence,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="grid_1", choices=["grid_1", "grid_2"])
    parser.add_argument("--output", type=Path, default=Path("tmp/tda_null_model_compare_grid_1.json"))
    parser.add_argument("--plot-output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shuffles", type=int, default=20)

    parser.add_argument("--res", type=int, default=100000)
    parser.add_argument("--dt", type=int, default=1000)
    parser.add_argument("--sigma", type=int, default=5000)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--speed-filter", action="store_true")
    parser.add_argument("--min-speed", type=float, default=2.5)

    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--num-times", type=int, default=5)
    parser.add_argument("--active-times", type=int, default=15000)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--n-points", type=int, default=1200)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--nbs", type=int, default=800)
    parser.add_argument("--maxdim", type=int, default=2)
    parser.add_argument("--coeff", type=int, default=47)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")

    parser.add_argument("--local-shift-fraction", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=250)
    parser.add_argument("--h1-primary-percentile", type=float, default=99.0)
    parser.add_argument("--h1-secondary-percentile", type=float, default=95.0)
    parser.add_argument("--h2-primary-percentile", type=float, default=95.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results = run_null_model_comparison(args)
    real_persistence = results.pop("_real_persistence_for_plotting")

    if args.plot_output_dir is not None:
        barcode_path = args.plot_output_dir / f"{args.dataset}_multi_null_barcode.png"
        ratio_path = args.plot_output_dir / f"{args.dataset}_null_ratio_summary.png"
        _plot_multi_null_barcode(real_persistence, results["null_model_results"], barcode_path)
        _plot_null_ratio_summary(results["null_model_results"], ratio_path)
        print(f"Saved barcode plot to {barcode_path}")
        print(f"Saved ratio summary plot to {ratio_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Saved comparison summary to {args.output}")
    print("Real topology:")
    for key in ["h1_top1", "h1_top2", "h1_top3", "h2_top1", "h2_top2"]:
        print(f"  {key}: {results['real_topology'][key]:.6f}")
    print("Null-model summary:")
    for item in results["null_model_results"]:
        print(
            f"  {item['method']}: "
            f"H1 top1 ratio={item['h1_top1_ratio']:.3f}, "
            f"H1 top2 ratio={item['h1_top2_ratio']:.3f}, "
            f"H2 top1 ratio={item['h2_top1_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
