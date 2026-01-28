"""FR analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QSpinBox

from ..views.widgets.popup_combo import PopupComboBox
from .base import AbstractAnalysisMode, configure_form_layout


class FRMode(AbstractAnalysisMode):
    name = "fr"
    display_name = "FR Heatmap"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("FR (population heatmap)")
        form = QFormLayout(box)
        configure_form_layout(form)

        self.neuron_start = QSpinBox()
        self.neuron_start.setRange(0, 1_000_000)
        self.neuron_start.setValue(0)

        self.neuron_end = QSpinBox()
        self.neuron_end.setRange(0, 1_000_000)
        self.neuron_end.setValue(0)

        self.time_start = QSpinBox()
        self.time_start.setRange(0, 10_000_000)
        self.time_start.setValue(0)

        self.time_end = QSpinBox()
        self.time_end.setRange(0, 10_000_000)
        self.time_end.setValue(0)

        self.normalize = PopupComboBox()
        self.normalize.addItems(["none", "zscore_per_neuron", "minmax_per_neuron"])
        self.normalize.setCurrentText("none")

        self.mode = PopupComboBox()
        self.mode.addItems(["fr", "spike"])
        self.mode.setToolTip("Use 'fr' for firing-rate matrix (requires preprocessing).")

        form.addRow("FR neuron_start", self.neuron_start)
        form.addRow("FR neuron_end", self.neuron_end)
        form.addRow("FR t_start", self.time_start)
        form.addRow("FR t_end", self.time_end)
        form.addRow("FR mode", self.mode)
        form.addRow("FR normalize", self.normalize)

        return box

    def collect_params(self) -> dict:
        neuron_start = int(self.neuron_start.value())
        neuron_end = int(self.neuron_end.value())
        time_start = int(self.time_start.value())
        time_end = int(self.time_end.value())
        neuron_range = None
        time_range = None
        if neuron_end > neuron_start:
            neuron_range = (neuron_start, neuron_end)
        if time_end > time_start:
            time_range = (time_start, time_end)
        return {
            "neuron_range": neuron_range,
            "time_range": time_range,
            "normalize": str(self.normalize.currentText()),
            "mode": str(self.mode.currentText()),
        }

    def apply_ranges(self, neuron_count: int | None, total_steps: int | None) -> None:
        if neuron_count is not None:
            self.neuron_start.setRange(0, max(0, neuron_count - 1))
            self.neuron_end.setRange(0, neuron_count)
            if self.neuron_end.value() == 0:
                self.neuron_end.setValue(neuron_count)
        if total_steps is not None:
            self.time_start.setRange(0, max(0, total_steps - 1))
            self.time_end.setRange(0, total_steps)
            if self.time_end.value() == 0:
                self.time_end.setValue(total_steps)
