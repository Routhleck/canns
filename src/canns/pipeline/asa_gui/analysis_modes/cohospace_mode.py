"""CohoSpace analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ..views.widgets.popup_combo import PopupComboBox
from .base import AbstractAnalysisMode, configure_form_layout


class CohoSpaceMode(AbstractAnalysisMode):
    name = "cohospace"
    display_name = "CohoSpace / CohoScore"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("CohoSpace / CohoScore")
        form = QFormLayout(box)
        configure_form_layout(form)

        self.dim_mode = PopupComboBox()
        self.dim_mode.addItem("2D", userData="2d")
        self.dim_mode.addItem("1D", userData="1d")
        self.dim_mode.setCurrentIndex(0)

        self.dim = QSpinBox()
        self.dim.setRange(1, 10)
        self.dim.setValue(1)

        self.dim1 = QSpinBox()
        self.dim1.setRange(1, 10)
        self.dim1.setValue(1)

        self.dim2 = QSpinBox()
        self.dim2.setRange(1, 10)
        self.dim2.setValue(2)

        self.mode = PopupComboBox()
        self.mode.addItems(["spike", "fr"])
        self.mode.setCurrentText("spike")

        self.top_percent = QDoubleSpinBox()
        self.top_percent.setRange(0.1, 50.0)
        self.top_percent.setSingleStep(0.5)
        self.top_percent.setValue(2.0)

        self.view = PopupComboBox()
        self.view.addItem("single neuron", userData="single")
        self.view.addItem("all neurons (aggregate)", userData="population")
        self.view.setCurrentIndex(0)

        self.subsample = QSpinBox()
        self.subsample.setRange(1, 100)
        self.subsample.setValue(2)

        self.unfold = PopupComboBox()
        self.unfold.addItems(["square", "skew"])

        self.skew_show_grid = QCheckBox("Skew: show grid")
        self.skew_show_grid.setChecked(True)

        self.skew_tiles = QSpinBox()
        self.skew_tiles.setRange(0, 10)
        self.skew_tiles.setValue(0)

        self.top_k = QSpinBox()
        self.top_k.setRange(1, 10_000_000)
        self.top_k.setValue(10)

        self.neuron_id = QSpinBox()
        self.neuron_id.setRange(0, 1_000_000)
        self.neuron_id.setValue(0)

        self.btn_prev = QPushButton("←")
        self.btn_next = QPushButton("→")
        self.btn_show = QPushButton("Show")
        self._neuron_row = QWidget()
        neuron_row_layout = QHBoxLayout(self._neuron_row)
        neuron_row_layout.setContentsMargins(0, 0, 0, 0)
        neuron_row_layout.addWidget(self.btn_prev)
        neuron_row_layout.addWidget(self.neuron_id, 1)
        neuron_row_layout.addWidget(self.btn_next)
        neuron_row_layout.addSpacing(8)
        neuron_row_layout.addWidget(self.btn_show)

        self.enable_score = QCheckBox("Compute CohoScore & top-K")
        self.enable_score.setChecked(True)

        self.use_best = QCheckBox("Use best neuron by CohoScore")
        self.use_best.setChecked(True)

        form.addRow("Dim mode", self.dim_mode)

        self._dims1d_label = QLabel("Dim")
        self._dims1d_wrap = QWidget()
        dims_1d_layout = QHBoxLayout(self._dims1d_wrap)
        dims_1d_layout.setContentsMargins(0, 0, 0, 0)
        dims_1d_layout.addWidget(self.dim)
        dims_1d_layout.addStretch(1)

        self._dims2d_label = QLabel("Dim X / Dim Y")
        self._dims2d_wrap = QWidget()
        dims_2d_layout = QHBoxLayout(self._dims2d_wrap)
        dims_2d_layout.setContentsMargins(0, 0, 0, 0)
        dims_2d_layout.addWidget(QLabel("dim1"))
        dims_2d_layout.addWidget(self.dim1)
        dims_2d_layout.addSpacing(8)
        dims_2d_layout.addWidget(QLabel("dim2"))
        dims_2d_layout.addWidget(self.dim2)
        dims_2d_layout.addStretch(1)

        form.addRow(self._dims1d_label, self._dims1d_wrap)
        form.addRow(self._dims2d_label, self._dims2d_wrap)
        form.addRow("Unfold", self.unfold)
        form.addRow("", self.skew_show_grid)
        form.addRow("Skew tiles", self.skew_tiles)
        form.addRow("Mode (spike / fr)", self.mode)
        form.addRow("Top % (threshold)", self.top_percent)
        form.addRow(self.enable_score)
        form.addRow("View", self.view)
        form.addRow("Top-K neurons", self.top_k)
        form.addRow("Neuron id", self._neuron_row)
        form.addRow("", self.use_best)

        self.dim_mode.currentIndexChanged.connect(self._refresh_dim_mode)
        self.view.currentIndexChanged.connect(self._refresh_view)
        self.enable_score.toggled.connect(self._refresh_view)
        self.btn_prev.clicked.connect(lambda: self._shift(-1))
        self.btn_next.clicked.connect(lambda: self._shift(+1))
        self._refresh_dim_mode()
        self._refresh_view()

        return box

    def collect_params(self) -> dict:
        return {
            "dim_mode": str(self.dim_mode.currentData() or "2d"),
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
            "enable_score": bool(self.enable_score.isChecked()),
            "top_k": int(self.top_k.value()),
            "use_best": bool(self.use_best.isChecked()),
        }

    def apply_ranges(self, neuron_count: int | None, total_steps: int | None) -> None:
        if neuron_count is None:
            return
        self.neuron_id.setRange(0, max(0, neuron_count - 1))
        self.top_k.setMaximum(max(1, neuron_count))
        if self.top_k.value() <= 0:
            self.top_k.setValue(min(10, neuron_count))

    def _refresh_dim_mode(self) -> None:
        mode = str(self.dim_mode.currentData() or "2d")
        is_1d = mode == "1d"
        self._dims1d_label.setVisible(is_1d)
        self._dims1d_wrap.setVisible(is_1d)
        self._dims2d_label.setVisible(not is_1d)
        self._dims2d_wrap.setVisible(not is_1d)
        self.unfold.setEnabled(not is_1d)
        self.skew_show_grid.setEnabled(not is_1d)
        self.skew_tiles.setEnabled(not is_1d)

    def _refresh_view(self) -> None:
        view_single = str(self.view.currentData() or "single") == "single"
        self.neuron_id.setEnabled(view_single)
        self.btn_prev.setEnabled(view_single)
        self.btn_next.setEnabled(view_single)
        self.btn_show.setEnabled(view_single)
        self.use_best.setEnabled(bool(self.enable_score.isChecked()))
        self.top_k.setEnabled(bool(self.enable_score.isChecked()))

    def _shift(self, delta: int) -> None:
        val = self.neuron_id.value() + int(delta)
        val = max(self.neuron_id.minimum(), min(self.neuron_id.maximum(), val))
        self.neuron_id.setValue(val)
