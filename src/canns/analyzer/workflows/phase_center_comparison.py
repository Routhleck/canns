from __future__ import annotations

"""Workflow for comparing CohoSpace phase centers across two matched datasets."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from canns.analyzer.data.asa import cohospace, cohospace_phase_centers
from canns.analyzer.data.asa.path import draw_base_parallelogram, load_npz_any

ArrayLike = np.ndarray | Sequence[int] | Sequence[float]


def _load_dict_like(data: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, (str, Path)):
        return load_npz_any(str(data))
    return dict(data)


def _extract_spikes(spikes_data: dict[str, Any], spike_key: str) -> np.ndarray:
    if spike_key not in spikes_data:
        raise KeyError(f"spike_key '{spike_key}' not found. keys={list(spikes_data.keys())}")
    spikes = np.asarray(spikes_data[spike_key])
    if spikes.ndim != 2:
        raise ValueError(f"Expected 2D spike/rate matrix, got {spikes.shape}")
    return spikes


def _normalize_cell_ids(cell_ids: ArrayLike | None, n_neurons: int, label: str) -> np.ndarray:
    if cell_ids is None:
        return np.arange(n_neurons, dtype=int)
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or ids.shape[0] != n_neurons:
        raise ValueError(f"{label} must be 1D with length {n_neurons}, got shape {ids.shape}")
    if np.unique(ids).shape[0] != ids.shape[0]:
        raise ValueError(f"{label} contains duplicate cell ids.")
    return ids


def _build_phase_centers(
    decoding_data: str | Path | dict[str, Any],
    spikes_data: str | Path | dict[str, Any],
    *,
    spike_key: str = "spike",
    coords_key: str | None = None,
    bins: int = 51,
    smooth_sigma: float = 0.0,
) -> dict[str, Any]:
    decoding_dict = _load_dict_like(decoding_data)
    spikes_dict = _load_dict_like(spikes_data)
    spikes = _extract_spikes(spikes_dict, spike_key)
    cohospace_result = cohospace(
        decoding_dict,
        spikes,
        coords_key=coords_key,
        bins=bins,
        smooth_sigma=smooth_sigma,
    )
    phase_centers = cohospace_phase_centers(cohospace_result)
    return {
        "cohospace": cohospace_result,
        "phase_centers": phase_centers["centers"],
        "phase_centers_skew": phase_centers["centers_skew"],
    }


def _match_indices(
    cell_ids_a: np.ndarray,
    cell_ids_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_b = {cell_id: idx for idx, cell_id in enumerate(cell_ids_b.tolist())}
    common_ids: list[Any] = []
    idx_a: list[int] = []
    idx_b: list[int] = []
    for i, cell_id in enumerate(cell_ids_a.tolist()):
        if cell_id in index_b:
            common_ids.append(cell_id)
            idx_a.append(i)
            idx_b.append(index_b[cell_id])
    if not common_ids:
        raise ValueError("No overlapping cell ids were found between the two datasets.")
    return np.asarray(common_ids), np.asarray(idx_a, dtype=int), np.asarray(idx_b, dtype=int)


def _minimum_image_displacement_skew(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Return shortest skew-plane displacements on the torus lattice."""
    e1 = np.array([2 * np.pi, 0.0])
    e2 = np.array([np.pi, np.sqrt(3) * np.pi])
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    candidates = [delta + i * e1 + j * e2 for i in (-1, 0, 1) for j in (-1, 0, 1)]
    norms = np.linalg.norm(candidates, axis=1)
    return np.asarray(candidates[int(np.argmin(norms))])


def plot_phase_center_comparison(
    phase_centers_a_skew: np.ndarray,
    phase_centers_b_skew: np.ndarray,
    *,
    label_a: str = "A",
    label_b: str = "B",
    title: str = "CohoSpace phase-center comparison",
    color_a: str = "black",
    color_b: str = "red",
    line_color: str = "0.65",
    s: int = 16,
    line_width: float = 0.8,
    alpha_points: float = 0.9,
    alpha_lines: float = 0.8,
    figsize: tuple[int, int] = (6, 6),
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    draw_base_parallelogram(ax)

    for start, end in zip(phase_centers_a_skew, phase_centers_b_skew, strict=False):
        end_plot = start + _minimum_image_displacement_skew(start, end)
        ax.plot(
            [start[0], end_plot[0]],
            [start[1], end_plot[1]],
            color=line_color,
            lw=line_width,
            alpha=alpha_lines,
            zorder=1,
        )

    ax.scatter(
        phase_centers_a_skew[:, 0],
        phase_centers_a_skew[:, 1],
        s=s,
        color=color_a,
        alpha=alpha_points,
        label=label_a,
        zorder=2,
    )
    ax.scatter(
        phase_centers_b_skew[:, 0],
        phase_centers_b_skew[:, 1],
        s=s,
        color=color_b,
        alpha=alpha_points,
        label=label_b,
        zorder=3,
    )

    corners = np.vstack(
        [
            [0.0, 0.0],
            [2 * np.pi, 0.0],
            [np.pi, np.sqrt(3) * np.pi],
            [3 * np.pi, np.sqrt(3) * np.pi],
        ]
    )
    xmin, ymin = corners.min(axis=0)
    xmax, ymax = corners.max(axis=0)
    padx = 0.03 * (xmax - xmin)
    pady = 0.03 * (ymax - ymin)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def compare_phase_centers_workflow(
    decoding_a: str | Path | dict[str, Any],
    spikes_a: str | Path | dict[str, Any],
    decoding_b: str | Path | dict[str, Any],
    spikes_b: str | Path | dict[str, Any],
    *,
    cell_ids_a: ArrayLike | None = None,
    cell_ids_b: ArrayLike | None = None,
    spike_key: str = "spike",
    coords_key_a: str | None = None,
    coords_key_b: str | None = None,
    bins: int = 51,
    smooth_sigma: float = 0.0,
    label_a: str = "A",
    label_b: str = "B",
    title: str = "CohoSpace phase-center comparison",
    save_plot_path: str | Path | None = None,
    save_npz_path: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """
    Compare matched neurons across two datasets via CohoSpace phase centers.

    The workflow computes CohoSpace rate maps for both datasets, extracts each
    neuron's phase center, matches neurons either by supplied cell ids or by
    positional order, and produces the red/black point-plus-line comparison plot.
    """
    result_a = _build_phase_centers(
        decoding_a,
        spikes_a,
        spike_key=spike_key,
        coords_key=coords_key_a,
        bins=bins,
        smooth_sigma=smooth_sigma,
    )
    result_b = _build_phase_centers(
        decoding_b,
        spikes_b,
        spike_key=spike_key,
        coords_key=coords_key_b,
        bins=bins,
        smooth_sigma=smooth_sigma,
    )

    n_a = result_a["phase_centers"].shape[0]
    n_b = result_b["phase_centers"].shape[0]
    ids_a = _normalize_cell_ids(cell_ids_a, n_a, "cell_ids_a")
    ids_b = _normalize_cell_ids(cell_ids_b, n_b, "cell_ids_b")

    if cell_ids_a is None and cell_ids_b is None:
        if n_a != n_b:
            raise ValueError(
                "Datasets have different neuron counts. Provide cell_ids_a/cell_ids_b to align them."
            )
        common_ids = ids_a.copy()
        idx_a = np.arange(n_a, dtype=int)
        idx_b = np.arange(n_b, dtype=int)
    elif cell_ids_a is not None and cell_ids_b is not None:
        common_ids, idx_a, idx_b = _match_indices(ids_a, ids_b)
    else:
        raise ValueError("Provide both cell_ids_a and cell_ids_b, or omit both.")

    centers_a = result_a["phase_centers"][idx_a]
    centers_b = result_b["phase_centers"][idx_b]
    centers_a_skew = result_a["phase_centers_skew"][idx_a]
    centers_b_skew = result_b["phase_centers_skew"][idx_b]

    displacement_skew = np.vstack(
        [
            _minimum_image_displacement_skew(start, end)
            for start, end in zip(centers_a_skew, centers_b_skew, strict=False)
        ]
    )
    displacement_norm = np.linalg.norm(displacement_skew, axis=1)

    plot_phase_center_comparison(
        centers_a_skew,
        centers_b_skew,
        label_a=label_a,
        label_b=label_b,
        title=title,
        save_path=save_plot_path,
        show=show,
    )

    result = {
        "cell_ids": common_ids,
        "indices_a": idx_a,
        "indices_b": idx_b,
        "phase_centers_a": centers_a,
        "phase_centers_b": centers_b,
        "phase_centers_a_skew": centers_a_skew,
        "phase_centers_b_skew": centers_b_skew,
        "displacement_skew": displacement_skew,
        "displacement_norm": displacement_norm,
        "mean_displacement_norm": float(np.mean(displacement_norm)),
        "median_displacement_norm": float(np.median(displacement_norm)),
        "bins": int(bins),
        "smooth_sigma": float(smooth_sigma),
        "label_a": label_a,
        "label_b": label_b,
    }

    if save_npz_path is not None:
        save_npz_path = Path(save_npz_path)
        save_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_npz_path, **result)

    return result


__all__ = [
    "compare_phase_centers_workflow",
    "plot_phase_center_comparison",
]
