"""Image viewer widget with zoom and pan."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsTextItem, QGraphicsView


class ImageViewer(QGraphicsView):
    """Image viewer with fit-to-view, zoom (wheel), and pan (drag)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMinimumHeight(200)
        self.setStyleSheet(
            "QGraphicsView { background: #ffffff; border: 1px solid #d8d4cc; border-radius: 8px; }"
        )

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._placeholder = QGraphicsTextItem("No image")
        self._placeholder.setDefaultTextColor(QColor("#888"))
        self._scene.addItem(self._placeholder)

        self._has_image = False
        self._auto_fit = True

    def set_image(self, path: str | Path | None) -> None:
        if path is None:
            self._set_placeholder("No image")
            return
        path = Path(path)
        if not path.exists():
            self._set_placeholder(f"Missing: {path}")
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._set_placeholder("Failed to load image")
            return
        self._has_image = True
        self._auto_fit = True
        self._pixmap_item.setPixmap(pixmap)
        self._pixmap_item.setVisible(True)
        self._placeholder.setVisible(False)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._has_image:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else 1 / 1.2
        self._auto_fit = False
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._has_image:
            self._auto_fit = True
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self._has_image and self._auto_fit:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._center_placeholder()

    def _set_placeholder(self, text: str) -> None:
        self._has_image = False
        self._auto_fit = True
        self._pixmap_item.setPixmap(QPixmap())
        self._pixmap_item.setVisible(False)
        self._placeholder.setPlainText(text)
        self._placeholder.setVisible(True)
        self._scene.setSceneRect(QRectF(0, 0, self.viewport().width(), self.viewport().height()))
        self._center_placeholder()

    def _center_placeholder(self) -> None:
        if not self._placeholder.isVisible():
            return
        view_rect = self.viewport().rect()
        center_scene = self.mapToScene(view_rect.center())
        br = self._placeholder.boundingRect()
        self._placeholder.setPos(QPointF(center_scene.x() - br.width() / 2, center_scene.y() - br.height() / 2))
