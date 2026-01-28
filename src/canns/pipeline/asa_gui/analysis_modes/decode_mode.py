"""Decode analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QSpinBox

from ..views.widgets.popup_combo import PopupComboBox
from .base import AbstractAnalysisMode, configure_form_layout


class DecodeMode(AbstractAnalysisMode):
    name = "decode"
    display_name = "Decode"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("Decode Parameters")
        form = QFormLayout(box)
        configure_form_layout(form)

        self.decode_version = PopupComboBox()
        self.decode_version.addItems(["v2", "v0"])

        self.num_circ = QSpinBox()
        self.num_circ.setRange(1, 10)
        self.num_circ.setValue(2)

        self.real_ground = QCheckBox()
        self.real_ground.setChecked(True)

        self.real_of = QCheckBox()
        self.real_of.setChecked(True)

        form.addRow("decode_version", self.decode_version)
        form.addRow("num_circ", self.num_circ)
        form.addRow("real_ground", self.real_ground)
        form.addRow("real_of", self.real_of)

        return box

    def collect_params(self) -> dict:
        return {
            "decode_version": str(self.decode_version.currentText()),
            "num_circ": int(self.num_circ.value()),
            "real_ground": bool(self.real_ground.isChecked()),
            "real_of": bool(self.real_of.isChecked()),
        }

    def apply_preset(self, preset: str) -> None:
        if preset == "grid":
            self.num_circ.setValue(2)
        elif preset == "hd":
            self.num_circ.setValue(1)

    def apply_language(self, lang: str) -> None:
        is_zh = str(lang).lower().startswith("zh")
        if is_zh:
            self.decode_version.setToolTip("解码版本（推荐 v2）。")
            self.num_circ.setToolTip("解码圆数（grid=2，hd=1）。")
            self.real_ground.setToolTip("是否使用 real_ground。")
            self.real_of.setToolTip("是否使用 real_of。")
        else:
            self.decode_version.setToolTip("Decode version (recommend v2).")
            self.num_circ.setToolTip("Number of circles to decode (grid=2, hd=1).")
            self.real_ground.setToolTip("Use real_ground if available.")
            self.real_of.setToolTip("Use real_of if available.")
