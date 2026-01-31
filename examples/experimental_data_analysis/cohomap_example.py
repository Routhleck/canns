#!/usr/bin/env python3
"""
Full ASA pipeline (embed -> TDA -> decode) and run EcohoMap.

Example:
  python .\examples\experimental_data_analysis\cohomap_example.py ^
    --data C:\path\to\asa_data.npz ^
    --out-dir C:\path\to\out
"""

import argparse
import os

import numpy as np

from canns.analyzer import data
from canns.analyzer.data.asa.path import load_npz_any


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full ASA pipeline + EcohoMap (single data file)."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="ASA .npz with at least spike/x/y/t.",
    )
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument("--num-circ", type=int, default=2, help="Number of H1 coords to decode.")
    parser.add_argument("--maxdim", type=int, default=1, help="Max homology dim for TDA.")
    parser.add_argument("--n-points", type=int, default=1200, help="TDA n_points.")
    parser.add_argument("--show", action="store_true", help="Show barcode plot.")
    parser.add_argument("--shuffle", action="store_true", help="Run TDA shuffle.")
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Treat spike as already binned matrix and skip embedding.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable temporal smoothing in embedding.",
    )
    parser.add_argument(
        "--speed-filter",
        action="store_true",
        help="Enable speed filtering in embedding.",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=2.5,
        help="Min speed for filtering (cm/s).",
    )
    parser.add_argument(
        "--cohomap-bins",
        type=int,
        default=101,
        help="Spatial bins for EcohoMap.",
    )
    parser.add_argument(
        "--cohomap-smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for EcohoMap.",
    )

    args = parser.parse_args()

    # ==================== Load data ====================
    asa = load_npz_any(args.data)

    if "spike" not in asa:
        raise ValueError("ASA data must contain 'spike'.")

    # ==================== Embed spikes ====================
    if args.skip_embed:
        spikes = _ensure_2d(np.asarray(asa["spike"]))
        xx = np.asarray(asa.get("x"))
        yy = np.asarray(asa.get("y"))
        tt = np.asarray(asa.get("t")) if "t" in asa else None
    else:
        spike_cfg = data.SpikeEmbeddingConfig(
            smooth=not args.no_smooth,
            speed_filter=args.speed_filter,
            min_speed=args.min_speed,
        )
        spikes, xx, yy, tt = data.embed_spike_trains(asa, config=spike_cfg)

    if xx is None or yy is None:
        raise ValueError("ASA data must include x/y for cohomap.")

    spike_data = dict(asa)
    spike_data["spike"] = spikes
    spike_data["x"] = xx
    spike_data["y"] = yy
    if tt is not None:
        spike_data["t"] = tt

    # ==================== TDA + decode ====================
    tda_cfg = data.TDAConfig(
        maxdim=args.maxdim,
        n_points=args.n_points,
        show=args.show,
        do_shuffle=args.shuffle,
        progress_bar=True,
    )
    tda_result = data.tda_vis(embed_data=spikes, config=tda_cfg)

    decoding = data.decode_circular_coordinates_multi(
        persistence_result=tda_result,
        spike_data=spike_data,
        num_circ=args.num_circ,
    )

    # ==================== EcohoMap ====================
    os.makedirs(args.out_dir, exist_ok=True)
    coho_map = data.cohomap(
        decoding_result=decoding,
        position_data={"x": xx, "y": yy, **({"t": tt} if tt is not None else {})},
        bins=args.cohomap_bins,
        smooth_sigma=args.cohomap_smooth_sigma,
    )
    out_ecohomap = os.path.join(args.out_dir, "ecohomap.png")
    data.plot_cohomap(coho_map, save_path=out_ecohomap, show=False)
    print(f"Saved: {out_ecohomap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
