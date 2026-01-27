"""Job specification and results for ASA GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JobSpec:
    """Inputs and parameters for a single analysis run."""

    input_mode: str  # "asa" | "batch" | "neuron_traj"
    preset: str  # "grid" | "hd" | "none"
    asa_file: Path | None = None
    neuron_file: Path | None = None
    traj_file: Path | None = None
    out_dir: Path = Path("Results/asa_gui_job")
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Result container for pipeline execution."""

    ok: bool
    out_dir: Path
    artifacts: dict[str, Path] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
