"""Artifacts tab with file list and quick actions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .file_list import FileList


class ArtifactsTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._last_dir: Path | None = None

        root = QVBoxLayout(self)
        self.files_list = FileList()
        root.addWidget(self.files_list, 1)

        actions = QHBoxLayout()
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._open_folder)
        actions.addWidget(self.btn_open_folder)
        actions.addStretch(1)
        root.addLayout(actions)

    def set_artifacts(self, artifacts: dict) -> None:
        self.files_list.clear()
        self._last_dir = None
        for _, path in artifacts.items():
            self.files_list.addItem(f"{_}: {path}")
            if self._last_dir is None:
                self._last_dir = Path(path).parent
        self.btn_open_folder.setEnabled(self._last_dir is not None)

    def _open_folder(self) -> None:
        if self._last_dir is None:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_dir)))
