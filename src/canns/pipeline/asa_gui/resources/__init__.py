"""Static resources for ASA GUI."""

from __future__ import annotations

from pathlib import Path


def load_theme_qss(theme: str) -> str:
    """Load the QSS for a given theme name."""
    name = (theme or "Light").strip().lower()
    if name in {"dark", "dark mode", "darkmode"}:
        fname = "dark.qss"
    else:
        fname = "light.qss"
    path = Path(__file__).parent / fname
    return path.read_text(encoding="utf-8")


def resource_path(name: str) -> Path:
    """Return the package resource path for a bundled file."""
    return Path(__file__).parent / name
