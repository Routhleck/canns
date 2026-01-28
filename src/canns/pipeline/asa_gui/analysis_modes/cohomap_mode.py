"""CohoMap analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QSpinBox

from .base import AbstractAnalysisMode, configure_form_layout
from ..views.widgets.popup_combo import PopupComboBox


class CohoMapMode(AbstractAnalysisMode):
    name = "cohomap"
    display_name = "CohoMap (TDA + decode)"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("CohoMap Parameters")
        form = QFormLayout(box)
        configure_form_layout(form)

        self.decode_version = PopupComboBox()
        self.decode_version.addItems(["v2", "v0"])

        self.num_circ = QSpinBox()
        self.num_circ.setRange(1, 50)
        self.num_circ.setValue(2)

        self.real_ground = QCheckBox()
        self.real_ground.setChecked(True)

        self.real_of = QCheckBox()
        self.real_of.setChecked(True)

        self.subsample = QSpinBox()
        self.subsample.setRange(1, 5000)
        self.subsample.setValue(10)

        form.addRow("Decode version", self.decode_version)
        form.addRow("Decode num_circ", self.num_circ)
        form.addRow("CohoMap subsample", self.subsample)

        return box

    def collect_params(self) -> dict:
        return {
            "decode_version": str(self.decode_version.currentText()),
            "num_circ": int(self.num_circ.value()),
            "real_ground": bool(self.real_ground.isChecked()),
            "real_of": bool(self.real_of.isChecked()),
            "cohomap_subsample": int(self.subsample.value()),
        }

    def apply_preset(self, preset: str) -> None:
        if preset == "grid":
            self.num_circ.setValue(2)
        elif preset == "hd":
            self.num_circ.setValue(1)
