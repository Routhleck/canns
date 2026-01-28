"""PathCompare result tab (PNG + optional GIF)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QMovie
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .image_viewer import ImageViewer


class _FitGifLabel(QLabel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._movie: QMovie | None = None
        self.setAlignment(Qt.AlignCenter)

    def setMovie(self, movie: QMovie) -> None:  # type: ignore[override]
        self._movie = movie
        super().setMovie(movie)
        movie.frameChanged.connect(self._on_frame)
        self._on_frame()

    def clearMovie(self) -> None:
        if self._movie is None:
            return
        try:
            self._movie.frameChanged.disconnect(self._on_frame)
        except Exception:
            pass
        self._movie = None
        self.clear()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._on_frame()

    def _on_frame(self, *_):
        if self._movie is None:
            return
        pix = self._movie.currentPixmap()
        if pix.isNull():
            return
        scaled = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class PathCompareTab(QWidget):
    def __init__(self, title: str = "Path Compare") -> None:
        super().__init__()
        self.title = title
        self._gif_movie: QMovie | None = None
        self._gif_retry_count = 0
        self._gif_last_path: Path | None = None

        root = QVBoxLayout(self)
        self.header = QLabel(title)
        self.header.setStyleSheet("font-weight: 600;")
        root.addWidget(self.header)

        actions = QHBoxLayout()
        self.btn_open_anim = QPushButton("Open Animation")
        self.btn_open_anim.setEnabled(False)
        self.btn_open_anim.clicked.connect(self._open_animation)
        actions.addWidget(self.btn_open_anim)
        actions.addStretch(1)
        root.addLayout(actions)

        self.splitter = QSplitter(Qt.Vertical)

        png_wrap = QWidget()
        png_layout = QVBoxLayout(png_wrap)
        png_layout.setContentsMargins(0, 0, 0, 0)
        self.png_label = QLabel("PNG")
        self.png_label.setStyleSheet("color: #666;")
        png_layout.addWidget(self.png_label)
        self.png_view = ImageViewer()
        png_layout.addWidget(self.png_view, 1)

        gif_wrap = QWidget()
        gif_layout = QVBoxLayout(gif_wrap)
        gif_layout.setContentsMargins(0, 0, 0, 0)
        self.gif_label = QLabel("Animation")
        self.gif_label.setStyleSheet("color: #666;")
        gif_layout.addWidget(self.gif_label)
        self.gif_view = _FitGifLabel("No animation yet")
        self.gif_view.setMinimumHeight(220)
        self.gif_view.setStyleSheet("border: 1px solid #ddd; background: #fafafa;")
        gif_layout.addWidget(self.gif_view, 1)

        self.anim_progress = QProgressBar()
        self.anim_progress.setRange(0, 100)
        self.anim_progress.setValue(0)
        self.anim_progress.setVisible(False)
        gif_layout.addWidget(self.anim_progress)

        self.splitter.addWidget(png_wrap)
        self.splitter.addWidget(gif_wrap)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        root.addWidget(self.splitter, 1)

        self.set_artifacts(None, None)

    def set_artifacts(self, png_path: Path | None, gif_path: Path | None) -> None:
        self._animation_path = None
        title = self.title
        if png_path:
            title = f"{self.title} — {png_path.name}"
        elif gif_path:
            title = f"{self.title} — {gif_path.name}"
        self.header.setText(title)

        if png_path and png_path.exists():
            self.png_label.setText(f"PNG: {png_path.name}")
            self.png_view.set_image(png_path)
        else:
            self.png_label.setText("PNG: (none)")
            self.png_view.set_image(None)

        if self._gif_movie is not None:
            try:
                self.gif_view.clearMovie()
            except Exception:
                pass
            self._gif_movie.stop()
            self._gif_movie.deleteLater()
            self._gif_movie = None

        if gif_path and gif_path.exists():
            self._gif_last_path = gif_path
            self._gif_retry_count = 0
            self._load_gif_movie(gif_path)
            self._animation_path = gif_path
        else:
            self.gif_label.setText("Animation: (none)")
            try:
                self.gif_view.clearMovie()
            except Exception:
                self.gif_view.clear()
            self.gif_view.setText("No animation yet")

        self.btn_open_anim.setEnabled(self._animation_path is not None)

    def set_animation(self, path: Path | None) -> None:
        if self._gif_movie is not None:
            try:
                self.gif_view.clearMovie()
            except Exception:
                pass
            self._gif_movie.stop()
            self._gif_movie.deleteLater()
            self._gif_movie = None

        if path and path.exists():
            self._animation_path = path
            self.gif_label.setText(f"Animation (MP4): {path.name}")
            self.gif_view.setText("MP4 preview not available. Use Open Animation.")
        else:
            self._animation_path = None
            self.gif_label.setText("Animation: (none)")
            self.gif_view.setText("No animation yet")
        self.btn_open_anim.setEnabled(self._animation_path is not None)

    def set_animation_progress(self, pct: int) -> None:
        pct = max(0, min(100, int(pct)))
        self.anim_progress.setVisible(True)
        self.anim_progress.setValue(pct)
        if pct >= 100:
            self.anim_progress.setVisible(False)

    def _open_animation(self) -> None:
        if self._animation_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._animation_path)))

    def _load_gif_movie(self, gif_path: Path) -> None:
        self.gif_label.setText(f"Animation (GIF): {gif_path.name}")
        self._gif_movie = QMovie(str(gif_path))
        if self._gif_movie.isValid():
            self.gif_view.setMovie(self._gif_movie)
            self._gif_movie.start()
            return

        self.gif_view.setText("GIF is not ready yet…")
        self._gif_retry_count += 1
        if self._gif_retry_count <= 10:
            QTimer.singleShot(200, lambda: self._load_gif_movie(gif_path))
        else:
            self.gif_view.setText("GIF loaded but QMovie is invalid")
