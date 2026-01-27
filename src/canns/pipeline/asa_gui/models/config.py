"""Lightweight config helpers for ASA GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisConfig:
    """Simple container for analysis parameters."""

    mode: str = "tda"
    params: dict[str, Any] = field(default_factory=dict)
