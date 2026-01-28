"""Validation helpers for ASA GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def validate_asa_npz(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"File not found: {path}"
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        return False, f"Failed to load npz: {e}"

    required = {"spike", "t"}
    missing = [k for k in required if k not in data.files]
    if missing:
        return False, f"Missing keys in asa npz: {missing}"
    return True, ""


def validate_neuron_traj(neuron_path: Path, traj_path: Path) -> tuple[bool, str]:
    if not neuron_path.exists():
        return False, f"Neuron file not found: {neuron_path}"
    if not traj_path.exists():
        return False, f"Trajectory file not found: {traj_path}"
    try:
        neuron = np.load(neuron_path)
        traj = np.load(traj_path)
    except Exception as e:
        return False, f"Failed to load input arrays: {e}"
    if neuron.ndim != 2:
        return False, f"Neuron data must be (T,N), got {neuron.shape}"
    if traj.ndim != 2 or traj.shape[1] != 2:
        return False, f"Trajectory must be (T,2), got {traj.shape}"
    if neuron.shape[0] != traj.shape[0]:
        return False, "Neuron and trajectory length mismatch"
    return True, ""
