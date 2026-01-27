"""Artifacts list widget with open actions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QListWidget


class FileList(QListWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.itemDoubleClicked.connect(self._open_item)

    def _open_item(self) -> None:
        item = self.currentItem()
        if item is None:
            return
        text = item.text()
        if ": " not in text:
            return
        _, path_str = text.split(": ", 1)
        path = Path(path_str)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
