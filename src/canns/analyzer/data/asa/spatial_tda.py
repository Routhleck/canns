from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .embedding import embed_spike_trains
from .path import load_npz_any
from .tda import _plot_barcode_with_shuffle, ripser


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def crop_center(arr: np.ndarray, fraction: float) -> np.ndarray:
    """Crop a central square region from a 2D/3D array."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0,1], got {fraction}")
    if fraction == 1.0:
        return arr

    h, w = arr.shape[:2]
    ch = max(1, int(round(h * fraction)))
    cw = max(1, int(round(w * fraction)))
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return arr[y0 : y0 + ch, x0 : x0 + cw, ...]


def build_fr_tensor(
    spikes: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    smoothing: bool,
    sigma: float,
    min_occupancy: int,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    compute_frm_fn: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build single-neuron spatial firing-rate maps for all neurons."""
    spikes = np.asarray(spikes)
    _, num_neurons = spikes.shape

    fr_maps: list[np.ndarray] = []
    occupancy = None

    for neuron_id in tqdm(range(num_neurons), desc="Computing FR maps"):
        res = compute_frm_fn(
            spikes,
            x,
            y,
            neuron_id=neuron_id,
            bins=bins,
            x_range=x_range,
            y_range=y_range,
            min_occupancy=min_occupancy,
            smoothing=smoothing,
            sigma=sigma,
            nan_for_empty=True,
        )
        fr_maps.append(res.frm)
        if occupancy is None:
            occupancy = res.occupancy

    assert occupancy is not None
    return np.stack(fr_maps, axis=-1), occupancy


def fr_tensor_to_point_cloud(
    fr_tensor: np.ndarray,
    occupancy: np.ndarray,
    *,
    crop_fraction: float,
    min_occupancy: int,
    fill_nan: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a spatial FR tensor into a position-indexed population point cloud.

    Each retained spatial bin becomes one point whose coordinates are the firing
    rates of all neurons at that position.
    """
    fr_crop = crop_center(fr_tensor, crop_fraction)
    occ_crop = crop_center(occupancy, crop_fraction)

    valid = np.isfinite(fr_crop).all(axis=-1) & (occ_crop >= int(min_occupancy))
    points = np.where(np.isnan(fr_crop), fill_nan, fr_crop)[valid]
    if points.ndim != 2:
        points = points.reshape(-1, fr_crop.shape[-1])
    return points, valid


def pca_reduce(points: np.ndarray, dim: int, standardize: bool) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(points, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"points must be a 2D matrix, got shape={X.shape}")
    max_components = min(X.shape[0], X.shape[1])
    if max_components < 1:
        raise ValueError(
            f"points must contain at least one sample and feature, got shape={X.shape}"
        )
    dim_eff = min(int(dim), max_components)
    if standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pca = PCA(n_components=dim_eff, svd_solver="full")
    return pca.fit_transform(X), np.asarray(pca.explained_variance_ratio_, dtype=float)


def run_spatial_ph(points: np.ndarray, *, maxdim: int, coeff: int) -> dict[str, Any]:
    return ripser(
        points,
        maxdim=maxdim,
        coeff=coeff,
        do_cocycles=False,
        distance_matrix=False,
        metric="cosine",
        progress_bar=True,
    )


def finite_lifetimes(diagram: np.ndarray | None) -> np.ndarray:
    if diagram is None:
        return np.zeros(0, dtype=float)
    dgm = np.asarray(diagram, dtype=float)
    if dgm.size == 0:
        return np.zeros(0, dtype=float)
    mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
    lives = dgm[mask, 1] - dgm[mask, 0]
    lives = lives[np.isfinite(lives) & (lives > 0)]
    return np.sort(lives)[::-1]


def diagram_summary(persistence: dict[str, Any], maxdim: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    dgms = persistence["dgms"]
    for dim in range(maxdim + 1):
        life = finite_lifetimes(dgms[dim]) if dim < len(dgms) else np.zeros(0)
        out[f"h{dim}_count"] = int(life.size)
        out[f"h{dim}_top"] = life[:5].tolist()
    return out


def permute_each_neuron_map(
    fr_tensor: np.ndarray,
    rng: np.random.Generator,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Permute each neuron's rate map independently within valid spatial bins."""
    h, w, n = fr_tensor.shape
    out = np.array(fr_tensor, copy=True)
    if valid_mask is None:
        valid_flat = np.ones(h * w, dtype=bool)
    else:
        valid_flat = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid_flat.shape[0] != h * w:
            raise ValueError(
                f"valid_mask shape {np.asarray(valid_mask).shape} is incompatible with {(h, w)}"
            )
    for i in range(n):
        flat = np.asarray(out[:, :, i]).reshape(-1)
        values = flat[valid_flat]
        flat[valid_flat] = values[rng.permutation(values.size)]
        out[:, :, i] = flat.reshape(h, w)
    return out


def run_spatial_null_model(
    fr_tensor: np.ndarray,
    occupancy: np.ndarray,
    *,
    crop_fraction: float,
    min_occupancy: int,
    pca_dim: int,
    standardize: bool,
    maxdim: int,
    coeff: int,
    num_shuffles: int,
    seed: int,
) -> dict[int, list[float]]:
    rng = np.random.default_rng(seed)
    maxima = {dim: [] for dim in range(maxdim + 1)}
    _, valid_mask = fr_tensor_to_point_cloud(
        fr_tensor,
        occupancy,
        crop_fraction=crop_fraction,
        min_occupancy=min_occupancy,
    )

    for _ in tqdm(range(num_shuffles), desc="FR-map null shuffles"):
        fr_crop = crop_center(fr_tensor, crop_fraction)
        occ_crop = crop_center(occupancy, crop_fraction)
        fr_perm_crop = permute_each_neuron_map(fr_crop, rng, valid_mask=valid_mask)
        points, _ = fr_tensor_to_point_cloud(
            fr_perm_crop,
            occ_crop,
            crop_fraction=1.0,
            min_occupancy=min_occupancy,
        )
        if len(points) < 4:
            continue
        points_red, _ = pca_reduce(points, dim=pca_dim, standardize=standardize)
        ph = run_spatial_ph(points_red, maxdim=maxdim, coeff=coeff)
        for dim in range(maxdim + 1):
            life = finite_lifetimes(ph["dgms"][dim]) if dim < len(ph["dgms"]) else np.zeros(0)
            maxima[dim].append(float(life[0]) if life.size else 0.0)

    return maxima


def save_spatial_point_cloud_npz(
    path: Path,
    *,
    fr_tensor: np.ndarray,
    occupancy: np.ndarray,
    points: np.ndarray,
    valid_mask: np.ndarray,
    points_pca: np.ndarray,
    explained_variance_ratio: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        fr_tensor=fr_tensor,
        occupancy=occupancy,
        points=points,
        valid_mask=valid_mask,
        points_pca=points_pca,
        explained_variance_ratio=explained_variance_ratio,
    )


def run_spatial_tda_from_asa(
    asa: dict[str, Any],
    *,
    compute_frm_fn: Any,
    out_dir: str | Path,
    skip_embed: bool = False,
    embed_config: Any | None = None,
    bins: int = 50,
    fr_smooth: bool = False,
    fr_sigma: float = 1.0,
    crop_fraction: float = 1.0 / 3.0,
    min_occupancy: int = 1,
    pca_dim: int = 7,
    maxdim: int = 2,
    coeff: int = 47,
    standardize: bool = True,
    shuffle: bool = False,
    num_shuffles: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Run the Neuron 2023 Fig.4C-style spatial TDA pipeline on ASA data.

    This pipeline treats spatial position as the index variable. Each retained
    spatial bin becomes one point in neural activity space with coordinates
    `[r_1(x), ..., r_N(x)]`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "spike" not in asa or "x" not in asa or "y" not in asa:
        raise ValueError("ASA data must contain at least spike/x/y.")

    if skip_embed:
        spikes = ensure_2d(np.asarray(asa["spike"]))
        if spikes.ndim != 2 or not np.issubdtype(spikes.dtype, np.number):
            raise ValueError(
                "skip_embed=True requires asa['spike'] to be a dense numeric (T, N) matrix."
            )
        xx = np.asarray(asa["x"]).ravel()
        yy = np.asarray(asa["y"]).ravel()
    else:
        if embed_config is None:
            raise ValueError("embed_config is required when skip_embed=False")
        spikes, xx, yy, _ = embed_spike_trains(asa, config=embed_config)

    t_steps = min(spikes.shape[0], len(xx), len(yy))
    spikes = spikes[:t_steps]
    xx = xx[:t_steps]
    yy = yy[:t_steps]

    fr_tensor, occupancy = build_fr_tensor(
        spikes,
        xx,
        yy,
        bins=bins,
        smoothing=fr_smooth,
        sigma=fr_sigma,
        min_occupancy=min_occupancy,
        x_range=None,
        y_range=None,
        compute_frm_fn=compute_frm_fn,
    )
    points, valid_mask = fr_tensor_to_point_cloud(
        fr_tensor,
        occupancy,
        crop_fraction=crop_fraction,
        min_occupancy=min_occupancy,
    )
    if len(points) < 4:
        raise ValueError(
            "Too few valid spatial bins after cropping/occupancy filtering. "
            "Try larger bins, lower min_occupancy, or crop_fraction=1.0."
        )

    points_pca, evr = pca_reduce(points, dim=pca_dim, standardize=standardize)
    persistence = run_spatial_ph(points_pca, maxdim=maxdim, coeff=coeff)

    shuffle_max = None
    if shuffle:
        shuffle_max = run_spatial_null_model(
            fr_tensor,
            occupancy,
            crop_fraction=crop_fraction,
            min_occupancy=min_occupancy,
            pca_dim=pca_dim,
            standardize=standardize,
            maxdim=maxdim,
            coeff=coeff,
            num_shuffles=num_shuffles,
            seed=seed,
        )

    barcode_png = out_dir / "spatial_tda_barcode.png"
    fig = _plot_barcode_with_shuffle(persistence, shuffle_max)
    fig.savefig(barcode_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    cloud_npz = out_dir / "spatial_tda_point_cloud.npz"
    save_spatial_point_cloud_npz(
        cloud_npz,
        fr_tensor=fr_tensor,
        occupancy=occupancy,
        points=points,
        valid_mask=valid_mask,
        points_pca=points_pca,
        explained_variance_ratio=evr,
    )

    persistence_npz = out_dir / "spatial_tda_persistence.npz"
    np.savez_compressed(
        persistence_npz,
        dgms=np.array(persistence["dgms"], dtype=object),
        shuffle_max=np.array(shuffle_max, dtype=object) if shuffle_max is not None else None,
    )

    summary = {
        "data": os.path.abspath(asa.get("__source_path__", "")),
        "spikes_shape": list(spikes.shape),
        "fr_tensor_shape": list(fr_tensor.shape),
        "occupancy_shape": list(occupancy.shape),
        "points_shape": list(points.shape),
        "points_pca_shape": list(points_pca.shape),
        "explained_variance_ratio": evr.tolist(),
        "params": {
            "bins": bins,
            "fr_smooth": bool(fr_smooth),
            "fr_sigma": fr_sigma,
            "crop_fraction": crop_fraction,
            "min_occupancy": min_occupancy,
            "pca_dim": pca_dim,
            "pca_dim_effective": int(points_pca.shape[1]),
            "maxdim": maxdim,
            "coeff": coeff,
            "standardize": bool(standardize),
            "shuffle": bool(shuffle),
            "num_shuffles": num_shuffles,
        },
        "real_topology": diagram_summary(persistence, maxdim),
    }
    if shuffle_max is not None:
        summary["shuffle_thresholds"] = {
            f"h{dim}_p99_9": float(np.percentile(vals, 99.9)) if len(vals) else 0.0
            for dim, vals in shuffle_max.items()
        }

    summary_json = out_dir / "spatial_tda_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "barcode_png": barcode_png,
        "point_cloud_npz": cloud_npz,
        "persistence_npz": persistence_npz,
        "summary_json": summary_json,
        "summary": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASA spatial TDA from position-indexed activity point clouds."
    )
    parser.add_argument("--data", required=True, help="ASA .npz with spike/x/y/t.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Treat `spike` as already aligned/binned (T,N) and skip ASA embedding.",
    )
    parser.add_argument(
        "--no-smooth", action="store_true", help="Disable temporal embedding smoothing."
    )
    parser.add_argument(
        "--speed-filter", action="store_true", help="Enable speed filtering during embedding."
    )
    parser.add_argument(
        "--min-speed", type=float, default=2.5, help="Min speed for embedding filter."
    )
    parser.add_argument(
        "--bins", type=int, default=50, help="Spatial bins per dimension for FR maps."
    )
    parser.add_argument(
        "--fr-smooth", action="store_true", help="Apply Gaussian smoothing to each FR map."
    )
    parser.add_argument("--fr-sigma", type=float, default=1.0, help="Gaussian sigma for FR maps.")
    parser.add_argument(
        "--crop-fraction", type=float, default=1.0 / 3.0, help="Central crop fraction."
    )
    parser.add_argument(
        "--min-occupancy", type=int, default=1, help="Minimum occupancy per spatial bin."
    )
    parser.add_argument("--pca-dim", type=int, default=7, help="PCA dimension before PH.")
    parser.add_argument("--maxdim", type=int, default=2, help="Maximum homology dimension.")
    parser.add_argument("--coeff", type=int, default=47, help="Finite field coefficient for PH.")
    parser.add_argument(
        "--no-standardize", action="store_true", help="Disable z-scoring before PCA."
    )
    parser.add_argument("--shuffle", action="store_true", help="Run FR-map null model.")
    parser.add_argument("--num-shuffles", type=int, default=100, help="Number of null shuffles.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from .config import SpikeEmbeddingConfig
    from .fr import compute_frm

    args = build_arg_parser().parse_args(argv)
    asa = load_npz_any(args.data)
    asa["__source_path__"] = os.path.abspath(args.data)

    embed_config = None
    if not args.skip_embed:
        embed_config = SpikeEmbeddingConfig(
            smooth=not args.no_smooth,
            speed_filter=args.speed_filter,
            min_speed=args.min_speed,
        )

    result = run_spatial_tda_from_asa(
        asa,
        compute_frm_fn=compute_frm,
        out_dir=args.out_dir,
        skip_embed=args.skip_embed,
        embed_config=embed_config,
        bins=args.bins,
        fr_smooth=args.fr_smooth,
        fr_sigma=args.fr_sigma,
        crop_fraction=args.crop_fraction,
        min_occupancy=args.min_occupancy,
        pca_dim=args.pca_dim,
        maxdim=args.maxdim,
        coeff=args.coeff,
        standardize=not args.no_standardize,
        shuffle=args.shuffle,
        num_shuffles=args.num_shuffles,
        seed=args.seed,
    )
    print(f"Saved barcode: {result['barcode_png']}")
    print(f"Saved point cloud: {result['point_cloud_npz']}")
    print(f"Saved persistence: {result['persistence_npz']}")
    print(f"Saved summary: {result['summary_json']}")
    return 0
