"""I/O adapters for loading ASA GUI inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_asa_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def pack_neuron_traj_to_asa(neuron_path: Path, traj_path: Path, out_path: Path) -> Path:
    """Pack neuron and trajectory arrays into ASA-style .npz.

    Minimal format support: neuron_path is .npy (T,N), traj_path is .npy (T,2).
    """

    neural = np.load(neuron_path)
    traj = np.load(traj_path)

    if neural.ndim != 2:
        raise ValueError(f"neural data must be (T,N). got {neural.shape}")
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError(f"traj must be (T,2). got {traj.shape}")
    if neural.shape[0] != traj.shape[0]:
        raise ValueError("T mismatch between neural and trajectory")

    t = np.arange(neural.shape[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        spike=neural.astype(np.float32),
        x=traj[:, 0],
        y=traj[:, 1],
        t=t,
    )
    return out_path
