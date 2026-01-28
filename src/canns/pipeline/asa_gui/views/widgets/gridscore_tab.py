"""GridScore result viewer tab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .image_viewer import ImageViewer
from .popup_combo import PopupComboBox


class GridScoreTab(QWidget):
    """GridScore viewer with distribution and neuron inspector."""

    inspectRequested = Signal(int, dict)

    def __init__(self, title: str = "Grid Score") -> None:
        super().__init__()
        self.title = title

        self._npz_path: Path | None = None
        self._meta: dict[str, Any] = {}
        self._ids: np.ndarray | None = None
        self._scores: np.ndarray | None = None
        self._spacing: np.ndarray | None = None
        self._orientation: np.ndarray | None = None
        self._id_to_idx: dict[int, int] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, 1)

        # Distribution panel
        dist = QWidget()
        dist_layout = QVBoxLayout(dist)
        dist_layout.setContentsMargins(8, 8, 8, 8)
        self.dist_header = QLabel("Grid score distribution")
        self.dist_header.setStyleSheet("font-weight: 600;")
        dist_layout.addWidget(self.dist_header)

        self.dist_viewer = ImageViewer()
        dist_layout.addWidget(self.dist_viewer, 1)

        splitter.addWidget(dist)

        # Inspector panel
        insp = QWidget()
        insp_layout = QVBoxLayout(insp)
        insp_layout.setContentsMargins(8, 8, 8, 8)

        self.insp_header = QLabel("Neuron inspector")
        self.insp_header.setStyleSheet("font-weight: 600;")
        insp_layout.addWidget(self.insp_header)

        ctrl = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.neuron_id = QSpinBox()
        self.neuron_id.setRange(0, 0)
        self.neuron_id.setValue(0)
        self.sort_combo = PopupComboBox()
        self.sort_combo.addItems(
            [
                "neuron id (asc)",
                "neuron id (desc)",
                "grid score (desc)",
                "grid score (asc)",
            ]
        )
        self.btn_show = QPushButton("Show neuron")

        ctrl.addWidget(QLabel("neuron_id:"))
        ctrl.addWidget(self.btn_prev)
        ctrl.addWidget(self.neuron_id)
        ctrl.addWidget(self.btn_next)
        ctrl.addSpacing(8)
        ctrl.addWidget(QLabel("sort:"))
        ctrl.addWidget(self.sort_combo)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_show)
        insp_layout.addLayout(ctrl)

        self.lbl_metrics = QLabel("grid_score: —    spacing: —    orientation: —")
        self.lbl_metrics.setStyleSheet("font-family: Menlo, Consolas, monospace;")
        insp_layout.addWidget(self.lbl_metrics)

        self.auto_viewer = ImageViewer()
        insp_layout.addWidget(self.auto_viewer, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #666;")
        insp_layout.addWidget(self.lbl_status)

        splitter.addWidget(insp)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self.btn_prev.clicked.connect(lambda: self._shift(-1))
        self.btn_next.clicked.connect(lambda: self._shift(+1))
        self.neuron_id.valueChanged.connect(self._on_neuron_changed)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        self.btn_show.clicked.connect(self._emit_inspect)

        self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.neuron_id.setEnabled(enabled)
        self.sort_combo.setEnabled(enabled)
        self.btn_show.setEnabled(enabled)

    def clear(self) -> None:
        self._npz_path = None
        self._meta = {}
        self._ids = None
        self._scores = None
        self._spacing = None
        self._orientation = None
        self._id_to_idx = {}
        self._order_ids: list[int] = []
        self._order_index: dict[int, int] = {}
        self._current_order_pos = 0
        self.dist_header.setText("Grid score distribution")
        self.insp_header.setText("Neuron inspector")
        self.lbl_metrics.setText("grid_score: —    spacing: —    orientation: —")
        self.lbl_status.setText("")
        self.dist_viewer.set_image(None)
        self.auto_viewer.set_image(None)
        self.neuron_id.setRange(0, 0)
        self.neuron_id.setValue(0)
        self.set_enabled(False)

    def set_distribution_image(self, path: Path | None) -> None:
        if path is not None:
            self.dist_header.setText(f"{self.title} — {path.name}")
        else:
            self.dist_header.setText(f"{self.title} — (no figure)")
        self.dist_viewer.set_image(path)

    def load_gridscore_npz(self, path: Path) -> None:
        self._npz_path = path
        data = np.load(str(path), allow_pickle=True)

        self._meta = {}
        for k in (
            "neuron_start",
            "neuron_end",
            "bins",
            "min_occupancy",
            "smoothing",
            "sigma",
            "overlap",
            "mode",
        ):
            if k in data:
                v = data[k]
                try:
                    self._meta[k] = v.item() if hasattr(v, "item") else v
                except Exception:
                    self._meta[k] = v

        if "grid_score" in data:
            self._scores = np.asarray(data["grid_score"])
        elif "score" in data:
            self._scores = np.asarray(data["score"])
        else:
            self._scores = None

        self._spacing = np.asarray(data["spacing"]) if "spacing" in data else None
        self._orientation = np.asarray(data["orientation"]) if "orientation" in data else None

        if "neuron_ids" in data:
            self._ids = np.asarray(data["neuron_ids"]).astype(int)
        else:
            ns = int(self._meta.get("neuron_start", 0))
            ne = int(
                self._meta.get(
                    "neuron_end",
                    ns + (len(self._scores) if self._scores is not None else 1),
                )
            )
            self._ids = np.arange(ns, ne, dtype=int)

        self._id_to_idx = {int(nid): int(i) for i, nid in enumerate(self._ids.tolist())}

        if self._ids.size > 0:
            mn = int(self._ids.min())
            mx = int(self._ids.max())
            self.neuron_id.setRange(mn, mx)
            self.set_enabled(True)
            self._apply_sort(preserve_id=mn)
            self.lbl_status.setText(f"Loaded {len(self._ids)} neurons from {path.name}")
        else:
            self.set_enabled(False)
            self.lbl_status.setText("No neurons found in gridscore.npz")

    def has_scores(self) -> bool:
        return self._scores is not None and self._ids is not None and len(self._id_to_idx) > 0

    def get_meta_params(self) -> dict[str, Any]:
        return dict(self._meta) if isinstance(self._meta, dict) else {}

    def set_autocorr_image(self, path: Path | None) -> None:
        self.auto_viewer.set_image(path)

    def set_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _shift(self, step: int) -> None:
        if not self.neuron_id.isEnabled():
            return
        if not self._order_ids:
            self.neuron_id.setValue(self.neuron_id.value() + int(step))
            return
        new_pos = self._current_order_pos + int(step)
        new_pos = max(0, min(len(self._order_ids) - 1, new_pos))
        self._current_order_pos = new_pos
        self.neuron_id.setValue(int(self._order_ids[new_pos]))

    def _on_neuron_changed(self, nid: int) -> None:
        if not self.has_scores():
            self.lbl_metrics.setText("grid_score: —    spacing: —    orientation: —")
            return
        nid = int(nid)
        if self._order_index:
            self._current_order_pos = self._order_index.get(nid, self._current_order_pos)
        idx = self._id_to_idx.get(nid)
        if idx is None:
            self.lbl_metrics.setText("grid_score: —    spacing: —    orientation: —")
            return

        score = float(self._scores[idx]) if self._scores is not None else float("nan")

        spc_txt = "—"
        if self._spacing is not None and self._spacing.ndim >= 2:
            spc = self._spacing[idx][:3]
            spc_txt = f"{spc[0]:.2f}, {spc[1]:.2f}, {spc[2]:.2f}"

        ori_txt = "—"
        if self._orientation is not None and self._orientation.ndim >= 2:
            ori = self._orientation[idx][:3]
            ori_txt = f"{ori[0]:.1f}, {ori[1]:.1f}, {ori[2]:.1f}"

        self.lbl_metrics.setText(
            f"grid_score: {score:.3f}    spacing: {spc_txt}    orientation: {ori_txt}"
        )

    def _on_sort_changed(self) -> None:
        if not self.has_scores():
            return
        self._apply_sort(preserve_id=int(self.neuron_id.value()))

    def _apply_sort(self, preserve_id: int | None = None) -> None:
        if self._ids is None:
            return
        ids = [int(nid) for nid in self._ids.tolist()]
        scores = None
        if self._scores is not None:
            try:
                scores = [float(s) for s in self._scores.tolist()]
            except Exception:
                scores = None

        mode = self.sort_combo.currentText()
        if mode == "neuron id (desc)":
            order_ids = sorted(ids, reverse=True)
        elif mode == "grid score (asc)" and scores is not None:
            order_ids = [i for i, _ in sorted(zip(ids, scores, strict=False), key=lambda t: t[1])]
        elif mode == "grid score (desc)" and scores is not None:
            order_ids = [
                i
                for i, _ in sorted(zip(ids, scores, strict=False), key=lambda t: t[1], reverse=True)
            ]
        else:
            order_ids = sorted(ids)

        self._order_ids = order_ids
        self._order_index = {nid: idx for idx, nid in enumerate(order_ids)}

        if preserve_id is None or preserve_id not in self._order_index:
            preserve_id = order_ids[0] if order_ids else None

        if preserve_id is not None:
            self._current_order_pos = self._order_index.get(preserve_id, 0)
            self.neuron_id.setValue(int(preserve_id))

    def _emit_inspect(self) -> None:
        if not self.has_scores():
            return
        nid = int(self.neuron_id.value())
        meta = self.get_meta_params()
        meta["neuron_id"] = nid
        self.inspectRequested.emit(nid, meta)
