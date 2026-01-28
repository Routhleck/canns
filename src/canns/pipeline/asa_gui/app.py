"""Application bootstrap for ASA GUI."""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .resources import load_theme_qss


class ASAGuiApp(MainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._apply_styles()

    def _apply_styles(self) -> None:
        try:
            settings = QSettings("canns", "asa_gui")
            theme = settings.value("theme", "Light")
            qss = load_theme_qss(str(theme))
            app = QApplication.instance()
            app.setStyleSheet(qss)
            font = app.font()
            if font.pointSize() <= 0:
                font.setPointSize(10)
                app.setFont(font)
        except Exception:
            pass
