"""FRM analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ..views.widgets.popup_combo import PopupComboBox
from .base import AbstractAnalysisMode, configure_form_layout


class FRMMode(AbstractAnalysisMode):
    name = "frm"
    display_name = "FRM (single neuron)"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("FRM (single neuron)")
        form = QFormLayout(box)
        configure_form_layout(form)

        self.neuron_id = QSpinBox()
        self.neuron_id.setRange(0, 1_000_000)
        self.neuron_id.setValue(0)

        self.bin_size = QSpinBox()
        self.bin_size.setRange(5, 500)
        self.bin_size.setValue(50)

        self.min_occupancy = QSpinBox()
        self.min_occupancy.setRange(1, 10_000)
        self.min_occupancy.setValue(1)

        self.smoothing = QCheckBox()
        self.smoothing.setChecked(False)

        self.smooth_sigma = QDoubleSpinBox()
        self.smooth_sigma.setRange(0.1, 50.0)
        self.smooth_sigma.setSingleStep(0.1)
        self.smooth_sigma.setValue(1.0)

        self.mode = PopupComboBox()
        self.mode.addItems(["fr", "spike"])
        self.mode.setToolTip("Use 'fr' for firing-rate matrix (requires preprocessing).")

        self.btn_prev = QPushButton("←")
        self.btn_next = QPushButton("→")
        neuron_row = QWidget()
        neuron_row_layout = QHBoxLayout(neuron_row)
        neuron_row_layout.setContentsMargins(0, 0, 0, 0)
        neuron_row_layout.addWidget(self.btn_prev)
        neuron_row_layout.addWidget(self.neuron_id, 1)
        neuron_row_layout.addWidget(self.btn_next)

        self.btn_prev.clicked.connect(lambda: self._shift(-1))
        self.btn_next.clicked.connect(lambda: self._shift(+1))

        form.addRow("FRM neuron_id", neuron_row)
        form.addRow("FRM bins", self.bin_size)
        form.addRow("FRM min_occupancy", self.min_occupancy)
        form.addRow("FRM smoothing", self.smoothing)
        form.addRow("FRM sigma", self.smooth_sigma)
        form.addRow("FRM mode", self.mode)

        return box

    def collect_params(self) -> dict:
        return {
            "neuron_id": int(self.neuron_id.value()),
            "bin_size": int(self.bin_size.value()),
            "min_occupancy": int(self.min_occupancy.value()),
            "smoothing": bool(self.smoothing.isChecked()),
            "smooth_sigma": float(self.smooth_sigma.value()),
            "mode": str(self.mode.currentText()),
        }

    def apply_ranges(self, neuron_count: int | None, total_steps: int | None) -> None:
        if neuron_count is None:
            return
        self.neuron_id.setRange(0, max(0, neuron_count - 1))

    def apply_language(self, lang: str) -> None:
        is_zh = str(lang).lower().startswith("zh")
        if is_zh:
            self.neuron_id.setToolTip("要查看的神经元编号。")
            self.bin_size.setToolTip("空间分箱大小。")
            self.min_occupancy.setToolTip("最小占据数。")
            self.smoothing.setToolTip("是否启用平滑。")
            self.smooth_sigma.setToolTip("平滑强度（sigma）。")
            self.mode.setToolTip("fr 需要预处理；spike 直接用事件。")
        else:
            self.neuron_id.setToolTip("Neuron index to inspect.")
            self.bin_size.setToolTip("Spatial bin size.")
            self.min_occupancy.setToolTip("Minimum occupancy.")
            self.smoothing.setToolTip("Enable smoothing.")
            self.smooth_sigma.setToolTip("Smoothing strength (sigma).")
            self.mode.setToolTip("fr requires preprocess; spike uses events directly.")

    def _shift(self, delta: int) -> None:
        val = self.neuron_id.value() + int(delta)
        val = max(self.neuron_id.minimum(), min(self.neuron_id.maximum(), val))
        self.neuron_id.setValue(val)
