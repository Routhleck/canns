"""Application bootstrap for ASA GUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


class ASAGuiApp(MainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._apply_styles()

    def _apply_styles(self) -> None:
        qss_path = Path(__file__).parent / "resources" / "styles.qss"
        if not qss_path.exists():
            return
        try:
            QApplication.instance().setStyleSheet(qss_path.read_text(encoding="utf-8"))
        except Exception:
            pass
