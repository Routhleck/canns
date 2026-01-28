"""GridScore analysis modes."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QSpinBox

from ..views.widgets.popup_combo import PopupComboBox
from .base import AbstractAnalysisMode, configure_form_layout


class GridScoreMode(AbstractAnalysisMode):
    name = "gridscore"
    display_name = "Grid Score (classic)"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("Grid Score (classic)")
        form = QFormLayout(box)
        configure_form_layout(form)

        self.neuron_start = QSpinBox()
        self.neuron_start.setRange(0, 10_000_000)
        self.neuron_start.setValue(0)

        self.neuron_end = QSpinBox()
        self.neuron_end.setRange(0, 10_000_000)
        self.neuron_end.setValue(0)

        self.bins = QSpinBox()
        self.bins.setRange(5, 500)
        self.bins.setValue(50)

        self.min_occupancy = QSpinBox()
        self.min_occupancy.setRange(1, 10_000)
        self.min_occupancy.setValue(1)

        self.smoothing = QCheckBox()
        self.smoothing.setChecked(False)

        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 50.0)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.0)

        self.overlap = QDoubleSpinBox()
        self.overlap.setRange(0.1, 1.0)
        self.overlap.setSingleStep(0.05)
        self.overlap.setValue(0.8)

        self.mode = PopupComboBox()
        self.mode.addItems(["fr", "spike"])
        self.mode.setToolTip("Use 'fr' for firing-rate maps (requires preprocessing).")

        self.score_thr = QDoubleSpinBox()
        self.score_thr.setRange(-2.0, 2.0)
        self.score_thr.setSingleStep(0.05)
        self.score_thr.setValue(0.3)

        self.neuron_id = QSpinBox()
        self.neuron_id.setRange(0, 10_000_000)
        self.neuron_id.setValue(0)

        form.addRow("neuron_start", self.neuron_start)
        form.addRow("neuron_end", self.neuron_end)
        form.addRow("bins", self.bins)
        form.addRow("min_occupancy", self.min_occupancy)
        form.addRow("smoothing", self.smoothing)
        form.addRow("sigma", self.sigma)
        form.addRow("autocorr overlap", self.overlap)
        form.addRow("mode", self.mode)
        form.addRow("score threshold (viz)", self.score_thr)

        return box

    def collect_params(self) -> dict:
        return {
            "gridscore": {
                "neuron_start": int(self.neuron_start.value()),
                "neuron_end": int(self.neuron_end.value()),
                "bins": int(self.bins.value()),
                "min_occupancy": int(self.min_occupancy.value()),
                "smoothing": bool(self.smoothing.isChecked()),
                "sigma": float(self.sigma.value()),
                "overlap": float(self.overlap.value()),
                "mode": str(self.mode.currentText()),
                "score_thr": float(self.score_thr.value()),
                "neuron_id": int(self.neuron_id.value()),
            }
        }

    def apply_ranges(self, neuron_count: int | None, total_steps: int | None) -> None:
        if neuron_count is None:
            return
        max_neuron = max(0, neuron_count - 1)
        self.neuron_start.setRange(0, max_neuron)
        self.neuron_end.setRange(0, neuron_count)
        if self.neuron_end.value() == 0:
            self.neuron_end.setValue(neuron_count)
        self.neuron_id.setRange(0, max_neuron)

    def apply_meta(self, meta: dict[str, Any]) -> None:
        if not isinstance(meta, dict):
            return
        if "bins" in meta:
            self.bins.setValue(int(meta["bins"]))
        if "min_occupancy" in meta:
            self.min_occupancy.setValue(int(meta["min_occupancy"]))
        if "smoothing" in meta:
            self.smoothing.setChecked(bool(meta["smoothing"]))
        if "sigma" in meta:
            self.sigma.setValue(float(meta["sigma"]))
        if "overlap" in meta:
            self.overlap.setValue(float(meta["overlap"]))
        if "mode" in meta:
            self.mode.setCurrentText(str(meta["mode"]))
        if "neuron_id" in meta:
            self.neuron_id.setValue(int(meta["neuron_id"]))

    def apply_language(self, lang: str) -> None:
        is_zh = str(lang).lower().startswith("zh")
        if is_zh:
            self.neuron_start.setToolTip("神经元起始索引。")
            self.neuron_end.setToolTip("神经元结束索引（不包含）。")
            self.bins.setToolTip("空间分箱数。")
            self.min_occupancy.setToolTip("最小占据数。")
            self.smoothing.setToolTip("是否启用平滑（需 scipy）。")
            self.sigma.setToolTip("平滑强度（sigma）。")
            self.overlap.setToolTip("自相关重叠比例。")
            self.mode.setToolTip("fr 需要预处理；spike 直接用事件。")
            self.score_thr.setToolTip("可视化阈值（仅显示）。")
        else:
            self.neuron_start.setToolTip("Start neuron index.")
            self.neuron_end.setToolTip("End neuron index (exclusive).")
            self.bins.setToolTip("Spatial bin count.")
            self.min_occupancy.setToolTip("Minimum occupancy.")
            self.smoothing.setToolTip("Enable smoothing (requires scipy).")
            self.sigma.setToolTip("Smoothing strength (sigma).")
            self.overlap.setToolTip("Autocorrelation overlap ratio.")
            self.mode.setToolTip("fr requires preprocess; spike uses events directly.")
            self.score_thr.setToolTip("Visualization threshold only.")


class GridScoreInspectMode(GridScoreMode):
    name = "gridscore_inspect"
    display_name = "GridScore Inspect"
