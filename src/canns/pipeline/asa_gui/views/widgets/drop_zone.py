"""Drag-and-drop file input widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class DropZone(QFrame):
    """Simple drag-and-drop target for file paths."""

    fileDropped = Signal(str)

    def __init__(self, title: str, hint: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("DropZone")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(110)

        self._title = QLabel(f"<b>{title}</b>")
        self._hint = QLabel(hint)
        self._hint.setStyleSheet("color: #666;")

        self._path_label = QLabel("No file")
        self._path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._current_path: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self._title)
        layout.addWidget(self._hint)
        layout.addWidget(self._path_label)

        self.setStyleSheet(
            """
            QFrame#DropZone {
                border: 2px dashed #b5b1a8;
                border-radius: 10px;
                padding: 8px;
                background: #fbfaf8;
            }
            QFrame#DropZone[drag="true"] {
                border-color: #1b8a6a;
                background: #eef7f4;
            }
            """
        )

    def set_path(self, path: str) -> None:
        self._path_label.setText(path)
        self._current_path = path

    def path(self) -> str | None:
        return self._current_path

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.mimeData().hasUrls():
            self.setProperty("drag", True)
            self.style().unpolish(self)
            self.style().polish(self)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.setProperty("drag", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.setProperty("drag", False)
        self.style().unpolish(self)
        self.style().polish(self)

        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        self.set_path(path)
        self.fileDropped.emit(path)
