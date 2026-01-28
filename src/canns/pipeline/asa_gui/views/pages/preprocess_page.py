"""Preprocess page for ASA GUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QGraphicsDropShadowEffect,
)
from PySide6.QtGui import QColor

from ...controllers import PreprocessController
from ...core import WorkerManager
from ..widgets.drop_zone import DropZone
from ..widgets.log_box import LogBox


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
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        title = QLabel("Preprocess")
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(title)

        content_split = QSplitter(Qt.Vertical)
        root.addWidget(content_split, 1)

        top_wrap = QWidget()
        top_layout = QVBoxLayout(top_wrap)

        # Input group
        input_group = QGroupBox("Input")
        input_group.setObjectName("card")
        input_layout = QVBoxLayout(input_group)

        top_row = QHBoxLayout()
        self.input_mode = QComboBox()
        self.input_mode.addItem("ASA (.npz)", userData="asa")
        self.input_mode.setEnabled(False)
        self.input_mode.setToolTip("Only ASA .npz input is supported in this GUI.")

        self.preset = QComboBox()
        self.preset.addItems(["grid", "hd", "none"])
        self.preset.setToolTip("Preset hints apply to analysis mode defaults.")

        top_row.addWidget(QLabel("Mode"))
        top_row.addWidget(self.input_mode)
        top_row.addSpacing(16)
        top_row.addWidget(QLabel("Preset"))
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

        self.preprocess_method = QComboBox()
        self.preprocess_method.addItem("None", userData="none")
        self.preprocess_method.addItem("Embed spike trains", userData="embed_spike_trains")
        self.preprocess_method.setToolTip("Embedding builds a dense spike matrix for TDA/FR.")
        self.preprocess_method.currentIndexChanged.connect(self._toggle_embed_params)

        preprocess_layout.addRow("Method", self.preprocess_method)

        self.embed_params = QWidget()
        embed_form = QFormLayout(self.embed_params)

        defaults = self._embedding_defaults()

        self.embed_res = QSpinBox()
        self.embed_res.setRange(1, 1_000_000)
        self.embed_res.setValue(int(defaults["res"]))

        self.embed_dt = QSpinBox()
        self.embed_dt.setRange(1, 1_000_000)
        self.embed_dt.setValue(int(defaults["dt"]))

        self.embed_sigma = QSpinBox()
        self.embed_sigma.setRange(1, 1_000_000)
        self.embed_sigma.setValue(int(defaults["sigma"]))

        self.embed_smooth = QCheckBox()
        self.embed_smooth.setChecked(bool(defaults["smooth"]))

        self.embed_speed_filter = QCheckBox()
        self.embed_speed_filter.setChecked(bool(defaults["speed_filter"]))

        self.embed_min_speed = QDoubleSpinBox()
        self.embed_min_speed.setRange(0.0, 1000.0)
        self.embed_min_speed.setDecimals(2)
        self.embed_min_speed.setValue(float(defaults["min_speed"]))

        embed_form.addRow("res", self.embed_res)
        embed_form.addRow("dt", self.embed_dt)
        embed_form.addRow("sigma", self.embed_sigma)
        embed_form.addRow("smooth", self.embed_smooth)
        embed_form.addRow("speed_filter", self.embed_speed_filter)
        embed_form.addRow("min_speed", self.embed_min_speed)

        preprocess_layout.addRow(self.embed_params)

        top_layout.addWidget(preprocess_group)

        # Pre-classification (placeholder)
        preclass_group = QGroupBox("Pre-classification")
        preclass_group.setObjectName("card")
        preclass_form = QFormLayout(preclass_group)
        self.preclass = QComboBox()
        self.preclass.addItems(["none", "grid", "hd"])
        self.preclass.setCurrentText("none")
        preclass_form.addRow("Preclass", self.preclass)
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
        log_layout.addWidget(QLabel("Logs"))
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

    def _apply_card_effects(self, widgets: list[QWidget]) -> None:
        for widget in widgets:
            effect = QGraphicsDropShadowEffect(self)
            effect.setBlurRadius(18)
            effect.setOffset(0, 3)
            effect.setColor(QColor(0, 0, 0, 40))
            widget.setGraphicsEffect(effect)

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
