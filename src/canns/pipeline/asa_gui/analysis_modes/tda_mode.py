"""TDA analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QSpinBox

from ..views.widgets.popup_combo import PopupComboBox
from .base import AbstractAnalysisMode, configure_form_layout


class TDAMode(AbstractAnalysisMode):
    name = "tda"
    display_name = "TDA"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("TDA parameters")
        form = QFormLayout(box)
        configure_form_layout(form)

        self.dim = QSpinBox()
        self.dim.setRange(1, 50)
        self.dim.setValue(6)
        self.dim.setToolTip("PCA 维度（常见起点 6–12）。")

        self.num_times = QSpinBox()
        self.num_times.setRange(1, 50)
        self.num_times.setValue(5)
        self.num_times.setToolTip("时间下采样步长；越大越快但可能丢细节。")

        self.active_times = QSpinBox()
        self.active_times.setRange(1, 10_000_000)
        self.active_times.setValue(15000)
        self.active_times.setToolTip("选取最活跃时间点数；过小不稳，过大更慢。")

        self.k = QSpinBox()
        self.k.setRange(1, 200_000)
        self.k.setValue(1000)
        self.k.setToolTip("采样/去噪相关参数，影响速度与稳定性。")

        self.n_points = QSpinBox()
        self.n_points.setRange(10, 500_000)
        self.n_points.setValue(1200)
        self.n_points.setToolTip("点云代表点数量，越大越慢。")

        self.metric = PopupComboBox()
        self.metric.addItems(["cosine", "euclidean", "manhattan"])
        self.metric.setToolTip("距离度量；推荐 cosine。")

        self.nbs = QSpinBox()
        self.nbs.setRange(1, 200_000)
        self.nbs.setValue(800)
        self.nbs.setToolTip("邻域规模参数，影响稳定性与速度。")

        self.maxdim = QSpinBox()
        self.maxdim.setRange(0, 3)
        self.maxdim.setValue(2)
        self.maxdim.setToolTip("最大同调维度；先 1 再 2。")

        self.coeff = QSpinBox()
        self.coeff.setRange(2, 997)
        self.coeff.setValue(47)
        self.coeff.setToolTip("有限域系数（默认 47）。")

        self.standardize = QCheckBox()
        self.standardize.setChecked(True)

        self.do_shuffle = QCheckBox()
        self.do_shuffle.setChecked(False)
        self.do_shuffle.setToolTip("显著性检验；代价高，建议少量。")

        self.num_shuffles = QSpinBox()
        self.num_shuffles.setRange(0, 5000)
        self.num_shuffles.setValue(100)
        self.num_shuffles.setEnabled(False)
        self.do_shuffle.toggled.connect(self.num_shuffles.setEnabled)
        self.num_shuffles.setToolTip("Shuffle 次数（越多越慢）。")

        form.addRow("TDA dim", self.dim)
        form.addRow("num_times", self.num_times)
        form.addRow("active_times", self.active_times)
        form.addRow("k (sampling)", self.k)
        form.addRow("n_points", self.n_points)
        form.addRow("metric", self.metric)
        form.addRow("nbs", self.nbs)
        form.addRow("maxdim", self.maxdim)
        form.addRow("coeff", self.coeff)
        form.addRow("Shuffle: Enable", self.do_shuffle)
        form.addRow("num_shuffles", self.num_shuffles)

        return box

    def collect_params(self) -> dict:
        return {
            "dim": int(self.dim.value()),
            "num_times": int(self.num_times.value()),
            "active_times": int(self.active_times.value()),
            "k": int(self.k.value()),
            "n_points": int(self.n_points.value()),
            "metric": str(self.metric.currentText()),
            "nbs": int(self.nbs.value()),
            "maxdim": int(self.maxdim.value()),
            "coeff": int(self.coeff.value()),
            "standardize": bool(self.standardize.isChecked()),
            "do_shuffle": bool(self.do_shuffle.isChecked()),
            "num_shuffles": int(self.num_shuffles.value()),
        }

    def apply_preset(self, preset: str) -> None:
        if preset == "grid":
            self.maxdim.setValue(2)
        elif preset == "hd":
            self.maxdim.setValue(1)

    def apply_language(self, lang: str) -> None:
        is_zh = str(lang).lower().startswith("zh")
        if is_zh:
            self.dim.setToolTip("PCA 维度（常见起点 6–12）。")
            self.num_times.setToolTip("时间下采样步长；越大越快但可能丢细节。")
            self.active_times.setToolTip("选取最活跃时间点数；过小不稳，过大更慢。")
            self.k.setToolTip("采样/去噪相关参数，影响速度与稳定性。")
            self.n_points.setToolTip("点云代表点数量，越大越慢。")
            self.metric.setToolTip("距离度量；推荐 cosine。")
            self.nbs.setToolTip("邻域规模参数，影响稳定性与速度。")
            self.maxdim.setToolTip("最大同调维度；先 1 再 2。")
            self.coeff.setToolTip("有限域系数（默认 47）。")
            self.do_shuffle.setToolTip("显著性检验；代价高，建议少量。")
            self.num_shuffles.setToolTip("Shuffle 次数（越多越慢）。")
        else:
            self.dim.setToolTip("PCA dimension (typical 6–12).")
            self.num_times.setToolTip("Time downsampling step; larger is faster but less detail.")
            self.active_times.setToolTip("Number of most active points; too small is unstable.")
            self.k.setToolTip("Sampling/denoising parameter; affects speed/stability.")
            self.n_points.setToolTip("Number of representative points; larger is slower.")
            self.metric.setToolTip("Distance metric; recommend cosine.")
            self.nbs.setToolTip("Neighborhood parameter; affects stability and speed.")
            self.maxdim.setToolTip("Max homology dimension; start with 1, then 2.")
            self.coeff.setToolTip("Finite field coefficient (default 47).")
            self.do_shuffle.setToolTip("Significance test; expensive, keep small.")
            self.num_shuffles.setToolTip("Number of shuffles (more is slower).")
