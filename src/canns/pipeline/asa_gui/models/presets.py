"""Preset parameter hints for ASA GUI."""

from __future__ import annotations

from typing import Any


def get_preset_params(preset: str) -> dict[str, Any]:
    """Return default parameter overrides for a preset."""
    preset = (preset or "none").lower()
    if preset == "grid":
        return {
            "tda": {"maxdim": 2},
            "decode": {"num_circ": 2},
        }
    if preset == "hd":
        return {
            "tda": {"maxdim": 1},
            "decode": {"num_circ": 1},
        }
    return {}
