"""State management for the model gallery TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MODEL_ANALYSIS_OPTIONS: dict[str, list[tuple[str, str]]] = {
    "cann1d": [
        ("Connectivity Matrix", "connectivity"),
        ("Energy Landscape", "energy"),
        ("Tuning Curves", "tuning"),
        ("Template Matching", "template"),
        ("Neural Manifold", "manifold"),
    ],
    "cann2d": [
        ("Connectivity Matrix", "connectivity"),
        ("Energy Landscape", "energy"),
        ("Firing Field", "firing_field"),
        ("Trajectory Comparison", "trajectory"),
        ("Neural Manifold", "manifold"),
    ],
    "gridcell": [
        ("Connectivity Matrix", "connectivity"),
        ("Energy Landscape", "energy"),
        ("Firing Field", "firing_field"),
        ("Path Integration", "path_integration"),
        ("Neural Manifold", "manifold"),
    ],
}


@dataclass
class GalleryState:
    """Centralized state for the model gallery TUI."""

    workdir: Path = field(default_factory=lambda: Path("."))
    model: str = "cann1d"
    analysis: str = "connectivity"
    artifacts: dict[str, Path] = field(default_factory=dict)


def get_analysis_options(model: str) -> list[tuple[str, str]]:
    """Return analysis options for the selected model."""
    return MODEL_ANALYSIS_OPTIONS.get(model, [("Connectivity Matrix", "connectivity")])


def get_default_analysis(model: str) -> str:
    """Return the default analysis key for the selected model."""
    options = get_analysis_options(model)
    return options[0][1] if options else "connectivity"
