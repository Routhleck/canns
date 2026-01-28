"""ComboBox with popup sizing to avoid clipping."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox


class PopupComboBox(QComboBox):
    def showPopup(self) -> None:  # pragma: no cover - UI callback
        view = self.view()
        if view is not None:
            try:
                extra = 24
                hint = view.sizeHintForColumn(0)
                width = max(self.width(), int(hint) + extra)
                view.setMinimumWidth(width)
                container = view.window()
                if container is not None:
                    flags = container.windowFlags()
                    container.setWindowFlags(flags | Qt.Popup | Qt.WindowStaysOnTopHint)
                    container.raise_()
            except Exception:
                pass
        super().showPopup()
