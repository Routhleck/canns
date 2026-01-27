"""PathCompare analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
)

from .base import AbstractAnalysisMode


class PathCompareMode(AbstractAnalysisMode):
    name = "pathcompare"
    display_name = "PathCompare"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("PathCompare Parameters")
        form = QFormLayout(box)

        self.angle_scale = QComboBox()
        self.angle_scale.addItems(["auto", "rad", "deg", "unit"])
        self.angle_scale.setCurrentText("auto")

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

        self.use_box = QCheckBox()
        self.use_box.setChecked(False)

        self.interp_full = QCheckBox()
        self.interp_full.setChecked(True)
        self.interp_full.setEnabled(False)

        self.coords_key = QLineEdit()
        self.coords_key.setPlaceholderText("coords / coordsbox (optional)")
        self.times_key = QLineEdit()
        self.times_key.setPlaceholderText("times_box (optional)")

        self.slice_mode = QComboBox()
        self.slice_mode.addItems(["time", "index"])
        self.slice_mode.setToolTip("Slice by time values or by index range.")

        self.tmin = QDoubleSpinBox()
        self.tmin.setRange(-1e9, 1e9)
        self.tmin.setDecimals(4)
        self.tmin.setValue(-1.0)

        self.tmax = QDoubleSpinBox()
        self.tmax.setRange(-1e9, 1e9)
        self.tmax.setDecimals(4)
        self.tmax.setValue(-1.0)

        self.imin = QSpinBox()
        self.imin.setRange(-1, 1_000_000_000)
        self.imin.setValue(-1)

        self.imax = QSpinBox()
        self.imax.setRange(-1, 1_000_000_000)
        self.imax.setValue(-1)

        self.stride = QSpinBox()
        self.stride.setRange(1, 100000)
        self.stride.setValue(1)

        self.tail = QSpinBox()
        self.tail.setRange(0, 100000)
        self.tail.setValue(200)

        self.fps = QSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setValue(30)

        self.no_wrap = QCheckBox()
        self.no_wrap.setChecked(False)

        self.animation_format = QComboBox()
        self.animation_format.addItems(["none", "gif", "mp4"])
        self.animation_format.setCurrentText("gif")
        self.animation_format.setToolTip("GIF matches the old GUI behavior; MP4 is faster if needed.")

        form.addRow("angle_scale", self.angle_scale)
        form.addRow("dim_mode", self.dim_mode)
        form.addRow("dim", self.dim)
        form.addRow("dim1", self.dim1)
        form.addRow("dim2", self.dim2)
        form.addRow("use_box", self.use_box)
        form.addRow("interp_full", self.interp_full)
        form.addRow("coords_key", self.coords_key)
        form.addRow("times_key", self.times_key)
        form.addRow("slice_mode", self.slice_mode)
        form.addRow("tmin", self.tmin)
        form.addRow("tmax", self.tmax)
        form.addRow("imin", self.imin)
        form.addRow("imax", self.imax)
        form.addRow("stride", self.stride)
        form.addRow("tail", self.tail)
        form.addRow("fps", self.fps)
        form.addRow("no_wrap", self.no_wrap)
        form.addRow("animation", self.animation_format)

        def _refresh_enabled() -> None:
            use_box = bool(self.use_box.isChecked())
            self.interp_full.setEnabled(use_box)

        def _refresh_slice_mode() -> None:
            is_time = (self.slice_mode.currentText() == "time")
            self.tmin.setEnabled(is_time)
            self.tmax.setEnabled(is_time)
            self.imin.setEnabled(not is_time)
            self.imax.setEnabled(not is_time)

        self.use_box.toggled.connect(_refresh_enabled)
        self.slice_mode.currentIndexChanged.connect(_refresh_slice_mode)
        _refresh_enabled()
        _refresh_slice_mode()

        return box

    def collect_params(self) -> dict:
        tmin = float(self.tmin.value())
        tmax = float(self.tmax.value())
        imin = int(self.imin.value())
        imax = int(self.imax.value())
        tmin = None if tmin < 0 else tmin
        tmax = None if tmax < 0 else tmax
        imin = None if imin < 0 else imin
        imax = None if imax < 0 else imax
        return {
            "angle_scale": str(self.angle_scale.currentText()),
            "dim_mode": str(self.dim_mode.currentText()),
            "dim": int(self.dim.value()),
            "dim1": int(self.dim1.value()),
            "dim2": int(self.dim2.value()),
            "use_box": bool(self.use_box.isChecked()),
            "interp_full": bool(self.interp_full.isChecked()),
            "coords_key": self.coords_key.text().strip() or None,
            "times_key": self.times_key.text().strip() or None,
            "slice_mode": str(self.slice_mode.currentText()),
            "tmin": tmin,
            "tmax": tmax,
            "imin": imin,
            "imax": imax,
            "stride": int(self.stride.value()),
            "tail": int(self.tail.value()),
            "fps": int(self.fps.value()),
            "no_wrap": bool(self.no_wrap.isChecked()),
            "animation_format": str(self.animation_format.currentText()),
        }
