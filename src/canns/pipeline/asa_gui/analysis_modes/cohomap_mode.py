"""CohoMap analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QSpinBox

from ..views.widgets.popup_combo import PopupComboBox
from .base import AbstractAnalysisMode, configure_form_layout


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

        self.bins = QSpinBox()
        self.bins.setRange(20, 501)
        self.bins.setValue(101)
        self.bins.setSingleStep(10)

        self.smooth_sigma = QDoubleSpinBox()
        self.smooth_sigma.setRange(0.0, 20.0)
        self.smooth_sigma.setSingleStep(0.5)
        self.smooth_sigma.setValue(1.0)

        self.ecohomap_mode = PopupComboBox()
        self.ecohomap_mode.addItems(["cos", "phase", "sin"])

        self.align_torus = QCheckBox("Align torus coordinates")
        self.align_torus.setChecked(True)

        form.addRow("Decode version", self.decode_version)
        form.addRow("Decode num_circ", self.num_circ)
        form.addRow("EcohoMap bins", self.bins)
        form.addRow("EcohoMap smooth σ", self.smooth_sigma)
        form.addRow("EcohoMap display", self.ecohomap_mode)
        form.addRow("", self.align_torus)
        form.addRow("Scatter subsample", self.subsample)

        return box

    def collect_params(self) -> dict:
        return {
            "decode_version": str(self.decode_version.currentText()),
            "num_circ": int(self.num_circ.value()),
            "real_ground": bool(self.real_ground.isChecked()),
            "real_of": bool(self.real_of.isChecked()),
            "cohomap_subsample": int(self.subsample.value()),
            "cohomap_bins": int(self.bins.value()),
            "cohomap_smooth_sigma": float(self.smooth_sigma.value()),
            "cohomap_mode": str(self.ecohomap_mode.currentText()),
            "cohomap_align_torus": bool(self.align_torus.isChecked()),
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
            self.num_circ.setToolTip("解码圆数（grid 常用 2，hd 常用 1）。")
            self.bins.setToolTip("EcohoMap 空间分箱数量。")
            self.smooth_sigma.setToolTip("EcohoMap 相位图平滑强度。")
            self.ecohomap_mode.setToolTip("显示 cos(phase)、phase 或 sin(phase)。")
            self.align_torus.setToolTip("通过条纹拟合对齐两个 cohomology 坐标。")
            self.subsample.setToolTip("旧 scatter 辅助图的下采样步长。")
        else:
            self.decode_version.setToolTip("Decode version (recommend v2).")
            self.num_circ.setToolTip("Number of circles to decode (grid=2, hd=1).")
            self.bins.setToolTip("Spatial bin count for EcohoMap.")
            self.smooth_sigma.setToolTip("Smoothing sigma for EcohoMap phase maps.")
            self.ecohomap_mode.setToolTip("Display cos(phase), phase, or sin(phase).")
            self.align_torus.setToolTip("Align decoded cohomology coordinates by stripe fitting.")
            self.subsample.setToolTip("Subsample step for the legacy scatter helper plot.")
