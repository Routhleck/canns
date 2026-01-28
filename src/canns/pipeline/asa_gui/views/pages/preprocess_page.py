"""Preprocess page for ASA GUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...controllers import PreprocessController
from ...core import WorkerManager
from ..help_content import preprocess_help_markdown
from ..widgets.drop_zone import DropZone
from ..widgets.help_dialog import show_help_dialog
from ..widgets.log_box import LogBox
from ..widgets.popup_combo import PopupComboBox


class PreprocessPage(QWidget):
    """Page for loading inputs and running preprocessing."""

    preprocess_completed = Signal()

    def __init__(
        self,
        controller: PreprocessController,
        worker_manager: WorkerManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._workers = worker_manager
        self._lang = "en"
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        title_row = QHBoxLayout()
        self.title_label = QLabel("Preprocess")
        self.title_label.setAlignment(Qt.AlignLeft)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        self.help_btn = QPushButton("Help")
        self.help_btn.setToolTip("Show preprocess parameter guide.")
        self.help_btn.clicked.connect(self._show_help)
        title_row.addWidget(self.help_btn)
        root.addLayout(title_row)

        content_split = QSplitter(Qt.Vertical)
        root.addWidget(content_split, 1)

        top_wrap = QWidget()
        top_layout = QVBoxLayout(top_wrap)

        # Input group
        input_group = QGroupBox("Input")
        input_group.setObjectName("card")
        input_layout = QVBoxLayout(input_group)
        self.input_group = input_group

        top_row = QHBoxLayout()
        self.input_mode = PopupComboBox()
        self.input_mode.addItem("ASA (.npz)", userData="asa")
        self.input_mode.setEnabled(False)
        self.input_mode.setToolTip("Only ASA .npz input is supported in this GUI.")

        self.preset = PopupComboBox()
        self.preset.addItems(["grid", "hd", "none"])
        self.preset.setToolTip("Preset hints apply to analysis mode defaults.")

        self.label_mode = QLabel("Mode")
        top_row.addWidget(self.label_mode)
        top_row.addWidget(self.input_mode)
        top_row.addSpacing(16)
        self.label_preset = QLabel("Preset")
        top_row.addWidget(self.label_preset)
        top_row.addWidget(self.preset)
        top_row.addStretch(1)
        input_layout.addLayout(top_row)

        # ASA input
        self.asa_zone = DropZone("ASA file", "Drop a .npz with spike/x/y/t")
        self.asa_browse = QPushButton("Browse")
        self.asa_browse.clicked.connect(self._browse_asa)
        asa_row = QHBoxLayout()
        asa_row.addWidget(self.asa_zone, 1)
        asa_row.addWidget(self.asa_browse)
        input_layout.addLayout(asa_row)
        self.asa_hint = QLabel("Expected keys: spike, x, y, t")
        self.asa_hint.setObjectName("muted")
        input_layout.addWidget(self.asa_hint)

        # Neuron + Trajectory input
        self.neuron_zone = DropZone("Neuron file", "Drop neuron .npy or .npz")
        self.neuron_browse = QPushButton("Browse")
        self.neuron_browse.clicked.connect(self._browse_neuron)
        neuron_row = QHBoxLayout()
        neuron_row.addWidget(self.neuron_zone, 1)
        neuron_row.addWidget(self.neuron_browse)

        self.traj_zone = DropZone("Trajectory file", "Drop trajectory .npy or .npz")
        self.traj_browse = QPushButton("Browse")
        self.traj_browse.clicked.connect(self._browse_traj)
        traj_row = QHBoxLayout()
        traj_row.addWidget(self.traj_zone, 1)
        traj_row.addWidget(self.traj_browse)

        input_layout.addLayout(neuron_row)
        input_layout.addLayout(traj_row)

        top_layout.addWidget(input_group)

        # Preprocess group
        preprocess_group = QGroupBox("Preprocess")
        preprocess_group.setObjectName("card")
        preprocess_layout = QFormLayout(preprocess_group)
        self.preprocess_group = preprocess_group

        self.preprocess_method = PopupComboBox()
        self.preprocess_method.addItem("None", userData="none")
        self.preprocess_method.addItem("Embed spike trains", userData="embed_spike_trains")
        self.preprocess_method.setToolTip("Embedding builds a dense spike matrix for TDA/FR.")
        self.preprocess_method.currentIndexChanged.connect(self._toggle_embed_params)

        preprocess_layout.addRow("Method", self.preprocess_method)
        self.label_method = preprocess_layout.labelForField(self.preprocess_method)

        self.embed_params = QWidget()
        embed_form = QFormLayout(self.embed_params)

        defaults = self._embedding_defaults()

        self.embed_res = QSpinBox()
        self.embed_res.setRange(1, 1_000_000)
        self.embed_res.setValue(int(defaults["res"]))
        self.embed_res.setToolTip("时间分箱分辨率（与 t 的单位一致）。")

        self.embed_dt = QSpinBox()
        self.embed_dt.setRange(1, 1_000_000)
        self.embed_dt.setValue(int(defaults["dt"]))
        self.embed_dt.setToolTip("时间步长（与 t 的单位一致）。")

        self.embed_sigma = QSpinBox()
        self.embed_sigma.setRange(1, 1_000_000)
        self.embed_sigma.setValue(int(defaults["sigma"]))
        self.embed_sigma.setToolTip("高斯平滑尺度，越大越平滑。")

        self.embed_smooth = QCheckBox()
        self.embed_smooth.setChecked(bool(defaults["smooth"]))
        self.embed_smooth.setToolTip("是否对嵌入后的矩阵做平滑。")

        self.embed_speed_filter = QCheckBox()
        self.embed_speed_filter.setChecked(bool(defaults["speed_filter"]))
        self.embed_speed_filter.setToolTip("过滤低速时间点（常见于 grid 数据）。")

        self.embed_min_speed = QDoubleSpinBox()
        self.embed_min_speed.setRange(0.0, 1000.0)
        self.embed_min_speed.setDecimals(2)
        self.embed_min_speed.setValue(float(defaults["min_speed"]))
        self.embed_min_speed.setToolTip("速度阈值（与 t/x/y 的单位一致）。")

        embed_form.addRow("res", self.embed_res)
        self.label_res = embed_form.labelForField(self.embed_res)
        embed_form.addRow("dt", self.embed_dt)
        self.label_dt = embed_form.labelForField(self.embed_dt)
        embed_form.addRow("sigma", self.embed_sigma)
        self.label_sigma = embed_form.labelForField(self.embed_sigma)
        embed_form.addRow("smooth", self.embed_smooth)
        self.label_smooth = embed_form.labelForField(self.embed_smooth)
        embed_form.addRow("speed_filter", self.embed_speed_filter)
        self.label_speed = embed_form.labelForField(self.embed_speed_filter)
        embed_form.addRow("min_speed", self.embed_min_speed)
        self.label_min_speed = embed_form.labelForField(self.embed_min_speed)

        preprocess_layout.addRow(self.embed_params)

        top_layout.addWidget(preprocess_group)

        # Pre-classification (placeholder)
        preclass_group = QGroupBox("Pre-classification")
        preclass_group.setObjectName("card")
        preclass_form = QFormLayout(preclass_group)
        self.preclass_group = preclass_group
        self.preclass = PopupComboBox()
        self.preclass.addItems(["none", "grid", "hd"])
        self.preclass.setCurrentText("none")
        preclass_form.addRow("Preclass", self.preclass)
        self.label_preclass = preclass_form.labelForField(self.preclass)
        top_layout.addWidget(preclass_group)

        # Controls
        control_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Preprocess")
        self.run_btn.setObjectName("btn_run")
        self.run_btn.clicked.connect(self._run_preprocess)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("btn_stop")
        self.stop_btn.clicked.connect(self._stop_preprocess)
        self.stop_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        control_row.addWidget(self.run_btn)
        control_row.addWidget(self.stop_btn)
        control_row.addWidget(self.progress, 1)
        top_layout.addLayout(control_row)

        log_wrap = QWidget()
        log_layout = QVBoxLayout(log_wrap)
        self.logs_label = QLabel("Logs")
        log_layout.addWidget(self.logs_label)
        self.log_box = LogBox()
        log_layout.addWidget(self.log_box, 1)

        content_split.addWidget(top_wrap)
        content_split.addWidget(log_wrap)
        content_split.setStretchFactor(0, 3)
        content_split.setStretchFactor(1, 1)

        self.input_mode.currentIndexChanged.connect(self._toggle_input_mode)
        self.asa_zone.fileDropped.connect(lambda path: self.asa_zone.set_path(path))
        self.neuron_zone.fileDropped.connect(lambda path: self.neuron_zone.set_path(path))
        self.traj_zone.fileDropped.connect(lambda path: self.traj_zone.set_path(path))
        self.asa_zone.fileDropped.connect(lambda _: self._update_run_enabled())
        self.asa_browse.clicked.connect(self._update_run_enabled)

        self._toggle_input_mode()
        self._toggle_embed_params()
        self._update_run_enabled()
        self._apply_card_effects([input_group, preprocess_group, preclass_group])
        self.apply_language(str(QSettings("canns", "asa_gui").value("lang", "en")))

    def _apply_card_effects(self, widgets: list[QWidget]) -> None:
        for widget in widgets:
            effect = QGraphicsDropShadowEffect(self)
            effect.setBlurRadius(18)
            effect.setOffset(0, 3)
            effect.setColor(QColor(0, 0, 0, 40))
            widget.setGraphicsEffect(effect)

    def apply_language(self, lang: str) -> None:
        self._lang = str(lang or "en")
        is_zh = self._lang.lower().startswith("zh")
        self.title_label.setText("预处理" if is_zh else "Preprocess")
        self.help_btn.setText("帮助" if is_zh else "Help")
        self.help_btn.setToolTip("查看参数说明" if is_zh else "Show preprocess parameter guide.")

        self.input_group.setTitle("输入" if is_zh else "Input")
        self.preprocess_group.setTitle("预处理" if is_zh else "Preprocess")
        self.preclass_group.setTitle("预分类" if is_zh else "Pre-classification")

        self.label_mode.setText("模式" if is_zh else "Mode")
        self.label_preset.setText("预设" if is_zh else "Preset")
        if self.label_method is not None:
            self.label_method.setText("方法" if is_zh else "Method")
        if self.label_preclass is not None:
            self.label_preclass.setText("预分类" if is_zh else "Preclass")

        if self.label_res is not None:
            self.label_res.setText("res")
        if self.label_dt is not None:
            self.label_dt.setText("dt")
        if self.label_sigma is not None:
            self.label_sigma.setText("sigma")
        if self.label_smooth is not None:
            self.label_smooth.setText("smooth")
        if self.label_speed is not None:
            self.label_speed.setText("speed_filter")
        if self.label_min_speed is not None:
            self.label_min_speed.setText("min_speed")

        self.asa_zone.set_title("ASA 文件" if is_zh else "ASA file")
        self.asa_zone.set_hint(
            "拖入含 spike/x/y/t 的 .npz" if is_zh else "Drop a .npz with spike/x/y/t"
        )
        self.asa_zone.set_empty_text("未选择文件" if is_zh else "No file")
        self.asa_hint.setText(
            "需要字段：spike, x, y, t" if is_zh else "Expected keys: spike, x, y, t"
        )

        self.neuron_zone.set_title("Neuron 文件" if is_zh else "Neuron file")
        self.neuron_zone.set_hint(
            "拖入 neuron .npy 或 .npz" if is_zh else "Drop neuron .npy or .npz"
        )
        self.neuron_zone.set_empty_text("未选择文件" if is_zh else "No file")

        self.traj_zone.set_title("Trajectory 文件" if is_zh else "Trajectory file")
        self.traj_zone.set_hint(
            "拖入 trajectory .npy 或 .npz" if is_zh else "Drop trajectory .npy or .npz"
        )
        self.traj_zone.set_empty_text("未选择文件" if is_zh else "No file")

        self.asa_browse.setText("浏览" if is_zh else "Browse")
        self.neuron_browse.setText("浏览" if is_zh else "Browse")
        self.traj_browse.setText("浏览" if is_zh else "Browse")

        self.run_btn.setText("运行预处理" if is_zh else "Run Preprocess")
        self.stop_btn.setText("停止" if is_zh else "Stop")
        self.logs_label.setText("日志" if is_zh else "Logs")

        self.input_mode.setToolTip(
            "仅支持 ASA .npz 输入" if is_zh else "Only ASA .npz input is supported in this GUI."
        )
        self.preprocess_method.setToolTip(
            "嵌入会生成稠密矩阵供 TDA/FR 使用"
            if is_zh
            else "Embedding builds a dense spike matrix for TDA/FR."
        )
        self.embed_res.setToolTip(
            "时间分箱分辨率（与 t 单位一致）。" if is_zh else "Time bin resolution (same unit as t)."
        )
        self.embed_dt.setToolTip(
            "时间步长（与 t 单位一致）。" if is_zh else "Time step (same unit as t)."
        )
        self.embed_sigma.setToolTip(
            "高斯平滑尺度，越大越平滑。" if is_zh else "Gaussian smoothing scale."
        )
        self.embed_smooth.setToolTip("是否启用平滑。" if is_zh else "Enable smoothing.")
        self.embed_speed_filter.setToolTip(
            "过滤低速时间点（常见于 grid 数据）。"
            if is_zh
            else "Remove low-speed samples (common for grid data)."
        )
        self.embed_min_speed.setToolTip(
            "速度阈值（与 t/x/y 单位一致）。"
            if is_zh
            else "Speed threshold (same unit as t/x/y)."
        )

    def _show_help(self) -> None:
        lang = str(QSettings("canns", "asa_gui").value("lang", "en"))
        title = (
            "Preprocess Guide" if not str(lang).lower().startswith("zh") else "Preprocess 参数说明"
        )
        show_help_dialog(self, title, preprocess_help_markdown(lang=lang))

    def _embedding_defaults(self) -> dict:
        try:
            from canns.analyzer.data.asa import SpikeEmbeddingConfig

            cfg = SpikeEmbeddingConfig()
            return {
                "res": cfg.res,
                "dt": cfg.dt,
                "sigma": cfg.sigma,
                "smooth": cfg.smooth,
                "speed_filter": cfg.speed_filter,
                "min_speed": cfg.min_speed,
            }
        except Exception:
            return {
                "res": 100000,
                "dt": 1000,
                "sigma": 5000,
                "smooth": True,
                "speed_filter": True,
                "min_speed": 2.5,
            }

    def _toggle_input_mode(self) -> None:
        mode = self.input_mode.currentData() or "asa"
        use_asa = mode == "asa"
        self.asa_zone.setVisible(use_asa)
        self.asa_browse.setVisible(use_asa)
        self.neuron_zone.setVisible(not use_asa)
        self.neuron_browse.setVisible(not use_asa)
        self.traj_zone.setVisible(not use_asa)
        self.traj_browse.setVisible(not use_asa)

    def _toggle_embed_params(self) -> None:
        method = self.preprocess_method.currentData() or "none"
        self.embed_params.setVisible(method == "embed_spike_trains")

    def _browse_asa(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select ASA file", str(Path.cwd()))
        if path:
            self.asa_zone.set_path(path)
            self._update_run_enabled()

    def _browse_neuron(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select neuron file", str(Path.cwd()))
        if path:
            self.neuron_zone.set_path(path)

    def _browse_traj(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select trajectory file", str(Path.cwd()))
        if path:
            self.traj_zone.set_path(path)

    def _collect_params(self) -> dict:
        preprocess_method = self.preprocess_method.currentData() or "none"
        params = {}
        if preprocess_method == "embed_spike_trains":
            params = {
                "res": int(self.embed_res.value()),
                "dt": int(self.embed_dt.value()),
                "sigma": int(self.embed_sigma.value()),
                "smooth": bool(self.embed_smooth.isChecked()),
                "speed_filter": bool(self.embed_speed_filter.isChecked()),
                "min_speed": float(self.embed_min_speed.value()),
            }
        return params

    def _run_preprocess(self) -> None:
        if self._workers.is_running():
            self.log_box.log("A task is already running.")
            return

        input_mode = self.input_mode.currentData() or "asa"
        preset = self.preset.currentText()
        preprocess_method = self.preprocess_method.currentData() or "none"

        asa_file = self.asa_zone.path() if input_mode == "asa" else None
        neuron_file = self.neuron_zone.path() if input_mode != "asa" else None
        traj_file = self.traj_zone.path() if input_mode != "asa" else None

        if not self._validate_inputs(asa_file):
            return

        self._controller.update_inputs(
            input_mode=input_mode,
            preset=preset,
            asa_file=asa_file,
            neuron_file=neuron_file,
            traj_file=traj_file,
            preprocess_method=preprocess_method,
            preprocess_params=self._collect_params(),
            preclass=str(self.preclass.currentText()),
            preclass_params={},
        )

        self.progress.setValue(0)
        self.stop_btn.setEnabled(True)
        self.run_btn.setEnabled(False)
        self.log_box.log("Starting preprocessing...")

        def _on_log(msg: str) -> None:
            self.log_box.log(msg)

        def _on_progress(pct: int) -> None:
            self.progress.setValue(pct)

        def _on_finished(result) -> None:
            if hasattr(result, "success") and not result.success:
                self._controller.mark_idle()
                self.log_box.log(result.error or "Preprocessing failed")
                self.run_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
            self._controller.finalize_preprocess()
            self.log_box.log(result.summary)
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.preprocess_completed.emit()

        def _on_error(msg: str) -> None:
            self._controller.mark_idle()
            self.log_box.log(f"Error: {msg}")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

        def _on_cleanup() -> None:
            self._controller.mark_idle()

        self._controller.run_preprocess(
            worker_manager=self._workers,
            on_log=_on_log,
            on_progress=_on_progress,
            on_finished=_on_finished,
            on_error=_on_error,
            on_cleanup=_on_cleanup,
        )

    def _stop_preprocess(self) -> None:
        if self._workers.is_running():
            self._workers.request_cancel()
            self.log_box.log("Cancel requested.")

    def _validate_inputs(self, asa_file: str | None) -> bool:
        if not asa_file:
            self.log_box.log("Please select an ASA .npz file before running.")
            return False
        path = Path(asa_file)
        if not path.exists():
            self.log_box.log(f"ASA file not found: {path}")
            return False
        if path.suffix.lower() != ".npz":
            self.log_box.log("ASA input must be a .npz file.")
            return False
        return True

    def _update_run_enabled(self) -> None:
        asa_file = self.asa_zone.path()
        valid = False
        if asa_file:
            path = Path(asa_file)
            valid = path.exists() and path.suffix.lower() == ".npz"
        self.run_btn.setEnabled(True)
        if valid:
            self.run_btn.setToolTip("")
        else:
            self.run_btn.setToolTip("Select a valid ASA .npz file to run preprocessing.")
