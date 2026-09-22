#!/usr/bin/env python3
"""Reproduce the published benchmark's grid_1 activity input, outside timing."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from benchmark import digest, write

RAW_SHA = "76bf469e0eb0c23f9c4bf2c296452470a92a9360ae904d8193c27eea6cf736ec"
ACTIVITY_SHA = "87f78f54beb3281d842a308260d477b1b3d76354a01759644339246c6e6a54ca"


def prepare(raw, output):
    from canns.analyzer.data import SpikeEmbeddingConfig, embed_spike_trains

    if output.exists():
        raise FileExistsError(output)
    if digest(raw) != RAW_SHA:
        raise ValueError("This is not the grid_1 archive used by the benchmark")
    parameters = dict(
        res=100000, dt=1000, sigma=5000, smooth=True, speed_filter=True, min_speed=2.5
    )
    with np.load(raw, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    activity, _, _, _ = embed_spike_trains(data, config=SpikeEmbeddingConfig(**parameters))
    sha = hashlib.sha256(np.ascontiguousarray(activity).tobytes()).hexdigest()
    if activity.shape != (126729, 172) or sha != ACTIVITY_SHA:
        raise ValueError(f"Embedding differs from the benchmark: shape={activity.shape}, sha={sha}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Open the path explicitly so NumPy cannot silently append another suffix.
    with output.open("xb") as stream:
        np.save(stream, activity)
    write(
        output.with_suffix(".json"),
        dict(
            raw_sha256=RAW_SHA,
            array_sha256=sha,
            activity_file_sha256=digest(output),
            embedding=parameters,
            shape=list(activity.shape),
            dtype=str(activity.dtype),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.raw, args.output)
