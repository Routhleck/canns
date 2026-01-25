"""State management for ASA TUI.

This module provides centralized workflow state management with workdir-centric design.
All file paths are stored relative to the working directory for portability.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class WorkflowState:
    """Centralized state for ASA analysis workflow.

    All file paths are relative to workdir for portability.
    """

    # Core paths
    workdir: Path = field(default_factory=lambda: Path("."))

    # Input configuration
    input_mode: str = "asa"  # "asa" | "neuron_traj" | "batch"
    preset: str = "grid"     # "grid" | "hd" | "none"

    # File paths (relative to workdir)
    asa_file: Optional[Path] = None
    neuron_file: Optional[Path] = None
    traj_file: Optional[Path] = None

    # Preprocessing
    preprocess_method: str = "none"  # "none" | "embed_spike_trains"
    preprocess_params: Dict[str, Any] = field(default_factory=dict)

    # Analysis configuration
    analysis_mode: str = "tda"  # "tda" | "cohomap" | "pathcompare" | "cohospace" | "fr" | "frm" | "gridscore"
    analysis_params: Dict[str, Any] = field(default_factory=dict)

    # Results
    artifacts: Dict[str, Path] = field(default_factory=dict)

    # Runtime state
    is_running: bool = False
    current_stage: str = ""


def relative_path(state: WorkflowState, path: Path) -> Path:
    """Convert absolute path to workdir-relative path.

    Args:
        state: Current workflow state
        path: Absolute path to convert

    Returns:
        Path relative to workdir
    """
    try:
        return path.relative_to(state.workdir)
    except ValueError:
        # Path is not relative to workdir, return as-is
        return path


def resolve_path(state: WorkflowState, path: Optional[Path]) -> Optional[Path]:
    """Convert relative path to absolute path.

    Args:
        state: Current workflow state
        path: Relative path to convert

    Returns:
        Absolute path or None if path is None
    """
    if path is None:
        return None

    if path.is_absolute():
        return path

    return state.workdir / path


def validate_files(state: WorkflowState) -> tuple[bool, str]:
    """Check if required files exist.

    Args:
        state: Current workflow state

    Returns:
        Tuple of (is_valid, error_message)
    """
    if state.input_mode == "asa":
        if state.asa_file is None:
            return False, "ASA file not selected"

        asa_path = resolve_path(state, state.asa_file)
        if not asa_path.exists():
            return False, f"ASA file not found: {asa_path}"

        # Validate .npz structure
        try:
            import numpy as np
            data = np.load(asa_path, allow_pickle=True)
            required_keys = ['spike', 't']
            missing = [k for k in required_keys if k not in data.files]
            if missing:
                return False, f"ASA file missing required keys: {missing}"
        except Exception as e:
            return False, f"Failed to load ASA file: {e}"

    elif state.input_mode == "neuron_traj":
        if state.neuron_file is None:
            return False, "Neuron file not selected"
        if state.traj_file is None:
            return False, "Trajectory file not selected"

        neuron_path = resolve_path(state, state.neuron_file)
        traj_path = resolve_path(state, state.traj_file)

        if not neuron_path.exists():
            return False, f"Neuron file not found: {neuron_path}"
        if not traj_path.exists():
            return False, f"Trajectory file not found: {traj_path}"

    return True, ""


def get_preset_params(preset: str) -> Dict[str, Any]:
    """Load preset configurations for analysis.

    Args:
        preset: Preset name ("grid", "hd", or "none")

    Returns:
        Dictionary of preset parameters
    """
    if preset == "grid":
        return {
            "preprocess": {
                "method": "embed_spike_trains",
                "dt": 0.02,
                "sigma": 0.1,
                "speed_filter": True,
                "min_speed": 2.5,
            },
            "tda": {
                "dim": 6,
                "num_times": 5,
                "active_times": 15000,
                "k": 1000,
                "n_points": 1200,
                "metric": "cosine",
                "nbs": 800,
                "maxdim": 1,
                "coeff": 47,
                "do_shuffle": False,
                "num_shuffles": 1000,
            },
            "gridscore": {
                "annulus_inner": 0.3,
                "annulus_outer": 0.7,
                "bin_size": 2.5,
                "smooth_sigma": 2.0,
            },
        }
    elif preset == "hd":
        return {
            "preprocess": {
                "method": "embed_spike_trains",
                "dt": 0.02,
                "sigma": 0.1,
                "speed_filter": True,
                "min_speed": 2.5,
            },
            "tda": {
                "dim": 4,
                "num_times": 5,
                "active_times": 15000,
                "k": 800,
                "n_points": 1000,
                "metric": "cosine",
                "nbs": 600,
                "maxdim": 1,
                "coeff": 47,
                "do_shuffle": False,
                "num_shuffles": 1000,
            },
        }
    else:  # "none"
        return {}


def check_cached_artifacts(state: WorkflowState, stage: str) -> bool:
    """Check if stage artifacts exist and are valid.

    Args:
        state: Current workflow state
        stage: Analysis stage name

    Returns:
        True if cached artifacts exist and are valid
    """
    stage_dir = state.workdir / stage
    if not stage_dir.exists():
        return False

    # Define required files for each stage
    stage_artifacts = {
        "TDA": ["barcode.png", "persistence.npz"],
        "CohoMap": ["decoding.npz", "cohomap.png"],
        "PathCompare": ["path_compare.png"],
        "CohoSpace": ["cohospace_trajectory.png"],
        "FR": ["fr_heatmap.png"],
        "FRM": [],  # Dynamic based on neuron_id
        "GridScore": ["gridscore_distribution.png", "gridscore.npz"],
    }

    required_files = stage_artifacts.get(stage, [])
    return all((stage_dir / f).exists() for f in required_files)


def load_cached_result(state: WorkflowState, stage: str) -> Dict[str, Any]:
    """Load cached results from previous run.

    Args:
        state: Current workflow state
        stage: Analysis stage name

    Returns:
        Dictionary of cached data
    """
    import numpy as np

    stage_dir = state.workdir / stage
    result = {}

    # Load .npz files
    for npz_file in stage_dir.glob("*.npz"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            result[npz_file.stem] = dict(data)
        except Exception:
            pass

    return result
