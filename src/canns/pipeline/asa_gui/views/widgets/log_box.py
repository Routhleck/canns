"""Log output widget."""

from __future__ import annotations

from PySide6.QtWidgets import QTextEdit


class LogBox(QTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(160)
        self.setPlaceholderText("Logs will appear here.")

    def log(self, msg: str) -> None:
        self.append(msg)
