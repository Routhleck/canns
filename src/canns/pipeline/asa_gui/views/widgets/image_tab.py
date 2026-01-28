"""Image tab widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .image_viewer import ImageViewer


class ImageTab(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self._path: Path | None = None

        layout = QVBoxLayout(self)
        header_row = QHBoxLayout()
        self._header = QLabel(title)
        self._header.setStyleSheet("font-weight: 600;")
        header_row.addWidget(self._header)
        header_row.addStretch(1)
        self._btn_open = QPushButton("Open Image")
        self._btn_open.setEnabled(False)
        self._btn_open.clicked.connect(self._open_image)
        header_row.addWidget(self._btn_open)
        layout.addLayout(header_row)

        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, 1)

    def set_image(self, path: Path | str | None) -> None:
        if path is None:
            self._header.setText(self._title)
            self._path = None
            self._btn_open.setEnabled(False)
        else:
            path = Path(path)
            self._header.setText(f"{self._title} — {path.name}")
            self._path = path
            self._btn_open.setEnabled(path.exists())
        self.viewer.set_image(path)

    def _open_image(self) -> None:
        if self._path is None or not self._path.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path)))
