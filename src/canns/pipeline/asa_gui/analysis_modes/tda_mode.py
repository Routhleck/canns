"""TDA analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QSpinBox

from .base import AbstractAnalysisMode


class TDAMode(AbstractAnalysisMode):
    name = "tda"
    display_name = "TDA"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("TDA Parameters")
        form = QFormLayout(box)

        self.dim = QSpinBox()
        self.dim.setRange(1, 50)
        self.dim.setValue(6)

        self.num_times = QSpinBox()
        self.num_times.setRange(1, 50)
        self.num_times.setValue(5)

        self.active_times = QSpinBox()
        self.active_times.setRange(1, 10_000_000)
        self.active_times.setValue(15000)

        self.k = QSpinBox()
        self.k.setRange(1, 200_000)
        self.k.setValue(1000)

        self.n_points = QSpinBox()
        self.n_points.setRange(10, 500_000)
        self.n_points.setValue(1200)

        self.metric = QComboBox()
        self.metric.addItems(["cosine", "euclidean", "manhattan"])

        self.nbs = QSpinBox()
        self.nbs.setRange(1, 200_000)
        self.nbs.setValue(800)

        self.maxdim = QSpinBox()
        self.maxdim.setRange(0, 3)
        self.maxdim.setValue(1)

        self.coeff = QSpinBox()
        self.coeff.setRange(2, 997)
        self.coeff.setValue(47)

        self.standardize = QCheckBox()
        self.standardize.setChecked(False)

        self.do_shuffle = QCheckBox()
        self.do_shuffle.setChecked(False)

        self.num_shuffles = QSpinBox()
        self.num_shuffles.setRange(0, 5000)
        self.num_shuffles.setValue(100)
        self.num_shuffles.setEnabled(False)
        self.do_shuffle.toggled.connect(self.num_shuffles.setEnabled)

        form.addRow("dim", self.dim)
        form.addRow("num_times", self.num_times)
        form.addRow("active_times", self.active_times)
        form.addRow("k", self.k)
        form.addRow("n_points", self.n_points)
        form.addRow("metric", self.metric)
        form.addRow("nbs", self.nbs)
        form.addRow("maxdim", self.maxdim)
        form.addRow("coeff", self.coeff)
        form.addRow("standardize", self.standardize)
        form.addRow("do_shuffle", self.do_shuffle)
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
