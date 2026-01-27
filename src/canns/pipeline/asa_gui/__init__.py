"""ASA GUI - PySide6-based graphical interface for Attractor Structure Analyzer."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

__all__ = ["main", "ASAGuiApp"]


def main() -> int:
    """Entry point for canns-gui command."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(
            "PySide6 is not installed. Please install with: pip install canns[gui]",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    from .app import ASAGuiApp

    app = QApplication(sys.argv)
    window = ASAGuiApp()
    window.show()
    return app.exec()


# Lazy import for ASAGuiApp
def __getattr__(name: str):
    if name == "ASAGuiApp":
        from .app import ASAGuiApp

        return ASAGuiApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
