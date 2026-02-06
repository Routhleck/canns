#!/usr/bin/env python3
"""
Full ASA pipeline (embed -> TDA -> decode) and run EcohoSpace.

Example:
  python .\examples\experimental_data_analysis\cohospace_example.py ^
    --data C:\path\to\asa_data.npz ^
    --out-dir C:\path\to\out ^
    --skew
"""

import argparse

import numpy as np

from canns.analyzer import data
from canns.analyzer.data.asa.path import load_npz_any
from canns.utils.example_outputs import get_example_output_dir

DEFAULT_OUT_DIR = get_example_output_dir(__file__)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full ASA pipeline + EcohoSpace (single data file)."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="ASA .npz with at least spike/x/y/t.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--num-circ", type=int, default=2, help="Number of H1 coords to decode.")
    parser.add_argument("--maxdim", type=int, default=1, help="Max homology dim for TDA.")
    parser.add_argument("--n-points", type=int, default=1200, help="TDA n_points.")
    parser.add_argument("--show", action="store_true", help="Show barcode plot.")
    parser.add_argument("--shuffle", action="store_true", help="Run TDA shuffle.")
    parser.add_argument("--neuron-id", type=int, default=0, help="Neuron index for ecohospace.")
    parser.add_argument(
        "--skew",
        action="store_true",
        help="Also save skewed EcohoSpace.",
    )
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
        "--cohospace-smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for EcohoSpace (0 disables).",
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

    # ==================== EcohoSpace ====================
    os.makedirs(args.out_dir, exist_ok=True)
    coho_space = data.cohospace(
        decoding,
        spikes,
        smooth_sigma=args.cohospace_smooth_sigma,
    )
    out_ecohospace = os.path.join(args.out_dir, "ecohospace.png")
    data.plot_cohospace(
        coho_space,
        neuron_id=args.neuron_id,
        save_path=out_ecohospace,
        show=False,
    )

    if args.skew:
        out_skew = os.path.join(args.out_dir, "ecohospace_skewed.png")
        data.plot_cohospace_skewed(
            coho_space,
            neuron_id=args.neuron_id,
            save_path=out_skew,
            show=False,
        )

    print(f"Saved: {out_ecohospace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
