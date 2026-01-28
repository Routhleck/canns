"""ASA GUI - PySide6-based graphical interface for Attractor Structure Analyzer."""

from __future__ import annotations

import sys

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

    from pathlib import Path

    from PySide6.QtGui import QGuiApplication, QIcon

    from .app import ASAGuiApp

    app = QApplication(sys.argv)
    app.setOrganizationName("canns")
    app.setApplicationName("ASA GUI")
    app.setApplicationDisplayName("ASA GUI")
    QGuiApplication.setApplicationDisplayName("ASA GUI")

    base = Path(__file__).resolve().parents[4] / "images"
    logo_path = base / "logo_256.png"
    if not logo_path.exists():
        logo_path = base / "logo.svg"
    if not logo_path.exists():
        logo_path = base / "logo.ico"
    icon = QIcon(str(logo_path)) if logo_path.exists() else QIcon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    window = ASAGuiApp()
    if not icon.isNull():
        window.setWindowIcon(icon)
    window.show()
    return app.exec()


# Lazy import for ASAGuiApp
def __getattr__(name: str):
    if name == "ASAGuiApp":
        from .app import ASAGuiApp

        return ASAGuiApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
