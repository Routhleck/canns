"""FRM analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QSpinBox

from .base import AbstractAnalysisMode


class FRMMode(AbstractAnalysisMode):
    name = "frm"
    display_name = "FRM"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("FRM Parameters")
        form = QFormLayout(box)

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

        self.mode = QComboBox()
        self.mode.addItems(["fr", "spike"])
        self.mode.setToolTip("Use 'fr' for firing-rate matrix (requires preprocessing).")

        form.addRow("neuron_id", self.neuron_id)
        form.addRow("bin_size", self.bin_size)
        form.addRow("min_occupancy", self.min_occupancy)
        form.addRow("smoothing", self.smoothing)
        form.addRow("smooth_sigma", self.smooth_sigma)
        form.addRow("mode", self.mode)

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
