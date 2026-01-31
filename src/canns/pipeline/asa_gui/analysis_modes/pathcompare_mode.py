"""PathCompare analysis mode."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ..views.widgets.popup_combo import PopupComboBox
from .base import AbstractAnalysisMode, configure_form_layout


class PathCompareMode(AbstractAnalysisMode):
    name = "pathcompare"
    display_name = "Path Compare (CohoMap required)"

    def create_params_widget(self) -> QGroupBox:
        box = QGroupBox("PathCompare Parameters")
        form = QFormLayout(box)
        configure_form_layout(form)

        self.angle_scale = PopupComboBox()
        self.angle_scale.addItems(["auto", "rad", "deg", "unit"])
        self.angle_scale.setCurrentText("auto")

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

        self.use_box = QCheckBox("Use coordsbox / times_box")
        self.use_box.setChecked(True)

        self.interp_full = QCheckBox("Interpolate to full trajectory")
        self.interp_full.setChecked(True)
        self.interp_full.setEnabled(False)

        self.coords_key = QLineEdit()
        self.coords_key.setPlaceholderText("coords / coordsbox (optional)")
        self.btn_coordsbox = QPushButton("coordsbox")
        self.btn_coordsbox.clicked.connect(lambda: self.coords_key.setText("coordsbox"))

        self.times_key = QLineEdit()
        self.times_key.setPlaceholderText("times_box (optional)")
        self.btn_times_box = QPushButton("times_box")
        self.btn_times_box.clicked.connect(lambda: self.times_key.setText("times_box"))

        self.slice_mode = PopupComboBox()
        self.slice_mode.addItem("Time (tmin/tmax)", userData="time")
        self.slice_mode.addItem("Index (imin/imax)", userData="index")
        self.slice_mode.setCurrentIndex(0)

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

        self.no_wrap = QCheckBox("Disable angle wrap")
        self.no_wrap.setChecked(False)

        self.save_gif = QCheckBox("Save GIF")
        self.save_gif.setChecked(False)

        form.addRow("Dim mode", self.dim_mode)

        self._dims1d_label = QLabel("Dim")
        dims_1d = QWidget()
        dims_1d_layout = QHBoxLayout(dims_1d)
        dims_1d_layout.setContentsMargins(0, 0, 0, 0)
        dims_1d_layout.addWidget(self.dim)
        dims_1d_layout.addStretch(1)

        self._dims2d_label = QLabel("Dim X / Dim Y")
        dims_2d = QWidget()
        dims_2d_layout = QHBoxLayout(dims_2d)
        dims_2d_layout.setContentsMargins(0, 0, 0, 0)
        dims_2d_layout.addWidget(QLabel("dim1"))
        dims_2d_layout.addWidget(self.dim1)
        dims_2d_layout.addSpacing(8)
        dims_2d_layout.addWidget(QLabel("dim2"))
        dims_2d_layout.addWidget(self.dim2)
        dims_2d_layout.addStretch(1)

        form.addRow(self._dims1d_label, dims_1d)
        form.addRow(self._dims2d_label, dims_2d)
        form.addRow(self.use_box)
        form.addRow(self.interp_full)
        coords_row = QWidget()
        coords_layout = QHBoxLayout(coords_row)
        coords_layout.setContentsMargins(0, 0, 0, 0)
        coords_layout.addWidget(self.coords_key, 1)
        coords_layout.addWidget(self.btn_coordsbox)

        times_row = QWidget()
        times_layout = QHBoxLayout(times_row)
        times_layout.setContentsMargins(0, 0, 0, 0)
        times_layout.addWidget(self.times_key, 1)
        times_layout.addWidget(self.btn_times_box)

        form.addRow("coords key", coords_row)
        form.addRow("times key", times_row)
        form.addRow("Slice mode", self.slice_mode)
        form.addRow("tmin (sec, -1=auto)", self.tmin)
        form.addRow("tmax (sec, -1=auto)", self.tmax)
        form.addRow("imin (-1=auto)", self.imin)
        form.addRow("imax (-1=auto)", self.imax)
        form.addRow("stride", self.stride)
        form.addRow("tail (frames)", self.tail)
        form.addRow("fps", self.fps)
        form.addRow("theta scale", self.angle_scale)
        form.addRow(self.no_wrap)
        form.addRow(self.save_gif)

        def _refresh_dim_mode() -> None:
            mode = str(self.dim_mode.currentData() or "2d")
            is_1d = mode == "1d"
            self._dims1d_label.setVisible(is_1d)
            dims_1d.setVisible(is_1d)
            self._dims2d_label.setVisible(not is_1d)
            dims_2d.setVisible(not is_1d)

        def _refresh_enabled() -> None:
            use_box = bool(self.use_box.isChecked())
            self.interp_full.setEnabled(use_box)
            self.coords_key.setEnabled(use_box)
            self.times_key.setEnabled(use_box)
            self.btn_coordsbox.setEnabled(use_box)
            self.btn_times_box.setEnabled(use_box)

        def _refresh_slice_mode() -> None:
            is_time = self.slice_mode.currentData() == "time"
            self.tmin.setEnabled(is_time)
            self.tmax.setEnabled(is_time)
            self.imin.setEnabled(not is_time)
            self.imax.setEnabled(not is_time)

        self.dim_mode.currentIndexChanged.connect(_refresh_dim_mode)
        self.use_box.toggled.connect(_refresh_enabled)
        self.slice_mode.currentIndexChanged.connect(_refresh_slice_mode)
        _refresh_dim_mode()
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
            "dim_mode": str(self.dim_mode.currentData() or "2d"),
            "dim": int(self.dim.value()),
            "dim1": int(self.dim1.value()),
            "dim2": int(self.dim2.value()),
            "use_box": bool(self.use_box.isChecked()),
            "interp_full": bool(self.interp_full.isChecked()),
            "coords_key": self.coords_key.text().strip() or None,
            "times_key": self.times_key.text().strip() or None,
            "slice_mode": str(self.slice_mode.currentData() or "time"),
            "tmin": tmin,
            "tmax": tmax,
            "imin": imin,
            "imax": imax,
            "stride": int(self.stride.value()),
            "tail": int(self.tail.value()),
            "fps": int(self.fps.value()),
            "no_wrap": bool(self.no_wrap.isChecked()),
            "animation_format": "gif" if self.save_gif.isChecked() else "none",
        }

    def apply_language(self, lang: str) -> None:
        is_zh = str(lang).lower().startswith("zh")
        if is_zh:
            self.angle_scale.setToolTip("角度尺度：auto / rad / deg / unit。")
            self.dim_mode.setToolTip("解码维度模式（1D/2D）。")
            self.dim.setToolTip("1D 解码维度索引。")
            self.dim1.setToolTip("2D 解码维度 1。")
            self.dim2.setToolTip("2D 解码维度 2。")
            self.use_box.setToolTip("使用 coordsbox / times_box 对齐（推荐速度过滤后开启）。")
            self.interp_full.setToolTip("插值回完整轨迹。")
            self.coords_key.setToolTip("可选：解码坐标键（默认 coords/coordsbox）。")
            self.times_key.setToolTip("可选：times_box 键名。")
            self.btn_coordsbox.setToolTip("填入 coordsbox。")
            self.btn_times_box.setToolTip("填入 times_box。")
            self.slice_mode.setToolTip("按时间或索引裁剪。")
            self.tmin.setToolTip("起始时间（秒），-1 自动。")
            self.tmax.setToolTip("结束时间（秒），-1 自动。")
            self.imin.setToolTip("起始索引，-1 自动。")
            self.imax.setToolTip("结束索引，-1 自动。")
            self.stride.setToolTip("采样步长。")
            self.tail.setToolTip("尾迹长度（帧）。")
            self.fps.setToolTip("动画帧率。")
            self.no_wrap.setToolTip("禁用角度环绕。")
            self.save_gif.setToolTip("保存 GIF 动画。")
        else:
            self.angle_scale.setToolTip("Angle scale: auto / rad / deg / unit.")
            self.dim_mode.setToolTip("Decode dimension mode (1D/2D).")
            self.dim.setToolTip("1D decoded dimension index.")
            self.dim1.setToolTip("2D decoded dimension 1.")
            self.dim2.setToolTip("2D decoded dimension 2.")
            self.use_box.setToolTip(
                "Use coordsbox/times_box alignment (recommended with speed_filter)."
            )
            self.interp_full.setToolTip("Interpolate back to full trajectory.")
            self.coords_key.setToolTip("Optional decode coords key (default coords/coordsbox).")
            self.times_key.setToolTip("Optional times_box key.")
            self.btn_coordsbox.setToolTip("Fill coordsbox.")
            self.btn_times_box.setToolTip("Fill times_box.")
            self.slice_mode.setToolTip("Slice by time or index.")
            self.tmin.setToolTip("Start time (sec), -1 = auto.")
            self.tmax.setToolTip("End time (sec), -1 = auto.")
            self.imin.setToolTip("Start index, -1 = auto.")
            self.imax.setToolTip("End index, -1 = auto.")
            self.stride.setToolTip("Sampling stride.")
            self.tail.setToolTip("Trail length (frames).")
            self.fps.setToolTip("Animation FPS.")
            self.no_wrap.setToolTip("Disable angle wrap.")
            self.save_gif.setToolTip("Save GIF animation.")
