"""CohoSpace analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox

from .base import AbstractAnalysisMode


class CohoSpaceMode(AbstractAnalysisMode):
    name = "cohospace"
    display_name = "CohoSpace"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("CohoSpace Parameters")
        form = QFormLayout(box)

        self.dim_mode = QComboBox()
        self.dim_mode.addItems(["2d", "1d"])
        self.dim_mode.setCurrentText("2d")

        self.dim = QSpinBox()
        self.dim.setRange(1, 10)
        self.dim.setValue(1)

        self.dim1 = QSpinBox()
        self.dim1.setRange(1, 10)
        self.dim1.setValue(1)

        self.dim2 = QSpinBox()
        self.dim2.setRange(1, 10)
        self.dim2.setValue(2)

        self.mode = QComboBox()
        self.mode.addItems(["spike", "fr"])
        self.mode.setCurrentText("spike")

        self.top_percent = QDoubleSpinBox()
        self.top_percent.setRange(0.1, 50.0)
        self.top_percent.setSingleStep(0.5)
        self.top_percent.setValue(2.0)

        self.view = QComboBox()
        self.view.addItem("single neuron", userData="single")
        self.view.addItem("all neurons (aggregate)", userData="population")
        self.view.setCurrentIndex(0)

        self.subsample = QSpinBox()
        self.subsample.setRange(1, 100)
        self.subsample.setValue(2)

        self.unfold = QComboBox()
        self.unfold.addItems(["square", "skew"])

        self.skew_show_grid = QCheckBox()
        self.skew_show_grid.setChecked(True)

        self.skew_tiles = QSpinBox()
        self.skew_tiles.setRange(0, 10)
        self.skew_tiles.setValue(0)

        self.neuron_id = QSpinBox()
        self.neuron_id.setRange(0, 1_000_000)
        self.neuron_id.setValue(0)

        form.addRow("dim_mode", self.dim_mode)
        form.addRow("dim", self.dim)
        form.addRow("dim1", self.dim1)
        form.addRow("dim2", self.dim2)
        form.addRow("mode", self.mode)
        form.addRow("top_percent", self.top_percent)
        form.addRow("view", self.view)
        form.addRow("subsample", self.subsample)
        form.addRow("unfold", self.unfold)
        form.addRow("skew_show_grid", self.skew_show_grid)
        form.addRow("skew_tiles", self.skew_tiles)
        form.addRow("neuron_id", self.neuron_id)

        return box

    def collect_params(self) -> dict:
        return {
            "dim_mode": str(self.dim_mode.currentText()),
            "dim": int(self.dim.value()),
            "dim1": int(self.dim1.value()),
            "dim2": int(self.dim2.value()),
            "mode": str(self.mode.currentText()),
            "top_percent": float(self.top_percent.value()),
            "view": str(self.view.currentData() or "single"),
            "subsample": int(self.subsample.value()),
            "unfold": str(self.unfold.currentText()),
            "skew_show_grid": bool(self.skew_show_grid.isChecked()),
            "skew_tiles": int(self.skew_tiles.value()),
            "neuron_id": int(self.neuron_id.value()),
        }

    def apply_ranges(self, neuron_count: int | None, total_steps: int | None) -> None:
        if neuron_count is None:
            return
        self.neuron_id.setRange(0, max(0, neuron_count - 1))
