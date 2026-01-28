"""Analysis page for ASA GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QPushButton,
    QSplitter,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QGraphicsDropShadowEffect,
)

from ...controllers import AnalysisController
from ...core import WorkerManager
from ...analysis_modes import AbstractAnalysisMode, get_analysis_modes
from ..help_content import analysis_help_markdown
from ..widgets.artifacts_tab import ArtifactsTab
from ..widgets.gridscore_tab import GridScoreTab
from ..widgets.help_dialog import show_help_dialog
from ..widgets.image_tab import ImageTab
from ..widgets.log_box import LogBox
from ..widgets.popup_combo import PopupComboBox
from ..widgets.pathcompare_tab import PathCompareTab


class AnalysisPage(QWidget):
    """Page for running analyses and viewing results."""

    analysis_completed = Signal()

    def __init__(
        self,
        controller: AnalysisController,
        worker_manager: WorkerManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._workers = worker_manager
        self._last_state = None
        self._lang = "en"
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        info_row = QHBoxLayout()
        self.info_label = QLabel("Mode=— | preset=— | preprocess=— | spike_main_shape=—")
        self.info_label.setObjectName("muted")
        info_row.addWidget(self.info_label, 1)
        root.addLayout(info_row)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left_wrap = QWidget()
        right_wrap = QWidget()
        left = QVBoxLayout(left_wrap)
        right = QVBoxLayout(right_wrap)

        self.param_container = QGroupBox("Analysis Parameters")
        self.param_container.setObjectName("card")
        self.param_layout = QVBoxLayout(self.param_container)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_layout.setSpacing(12)

        mode_row = QHBoxLayout()
        self.analysis_mode = PopupComboBox()
        self.analysis_mode.setToolTip("Select an analysis mode to run.")
        self._modes: dict[str, AbstractAnalysisMode] = {}
        hidden_modes = {"decode", "gridscore_inspect", "batch"}
        for mode in get_analysis_modes():
            self._modes[mode.name] = mode
            if mode.name not in hidden_modes:
                self.analysis_mode.addItem(mode.display_name, userData=mode.name)

        self.label_analysis_module = QLabel("Analysis module:")
        mode_row.addWidget(self.label_analysis_module)
        mode_row.addWidget(self.analysis_mode, 1)
        self.help_btn = QPushButton("Help")
        self.help_btn.setToolTip("Show parameter guide for the selected mode.")
        self.help_btn.clicked.connect(self._show_help)
        mode_row.addWidget(self.help_btn)
        self.param_layout.addLayout(mode_row)

        self.grp_standardize = QGroupBox("Preprocess (Standardization)")
        std_layout = QHBoxLayout(self.grp_standardize)
        self.chk_standardize = QCheckBox("StandardScaler")
        std_layout.addWidget(self.chk_standardize)
        std_layout.addStretch(1)
        self.param_layout.addWidget(self.grp_standardize)

        self.param_widgets: dict[str, QWidget] = {}
        for mode in self._modes.values():
            widget = mode.create_params_widget()
            widget.setObjectName("card")
            btn_show = getattr(mode, "btn_show", None)
            if btn_show is not None:
                btn_show.clicked.connect(
                    lambda _=False, mode_name=mode.name: self._run_analysis(mode_override=mode_name)
                )
            self.param_widgets[mode.name] = widget
            self.param_layout.addWidget(widget)
        self.param_layout.addStretch(1)
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setFrameShape(QFrame.NoFrame)
        self.param_scroll.setWidget(self.param_container)

        controls = QHBoxLayout()
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setObjectName("btn_run")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("btn_stop")
        self.stop_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        controls.addWidget(self.run_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.progress, 1)

        self.log_box = LogBox()
        log_wrap = QWidget()
        log_layout = QVBoxLayout(log_wrap)
        self.logs_label = QLabel("Logs")
        log_layout.addWidget(self.logs_label)
        log_layout.addWidget(self.log_box, 1)

        left.addWidget(self.param_scroll, 2)
        left.addLayout(controls)
        left.addWidget(log_wrap, 1)

        # Results
        self.tabs = QTabWidget()
        self.tab_barcode = ImageTab("TDA Barcode")
        self.tab_cohomap = ImageTab("CohoMap")
        self.tab_pathcompare = PathCompareTab("Path Compare")
        self.tab_cohospace = ImageTab("CohoSpace")
        self.tab_fr = ImageTab("FR Heatmap")
        self.tab_frm = ImageTab("FRM")
        self.tab_gridscore = GridScoreTab("Grid Score")

        self.tabs.addTab(self.tab_barcode, "Barcode")
        self.tabs.addTab(self.tab_cohomap, "CohoMap")
        self.tabs.addTab(self.tab_pathcompare, "Path Compare")
        self.tabs.addTab(self.tab_cohospace, "CohoSpace")
        self.tabs.addTab(self.tab_fr, "FR")
        self.tabs.addTab(self.tab_frm, "FRM")
        self.tabs.addTab(self.tab_gridscore, "GridScore")

        self.tab_files = ArtifactsTab()
        self.tabs.addTab(self.tab_files, "Files")

        right.addWidget(self.tabs, 1)

        splitter.addWidget(left_wrap)
        splitter.addWidget(right_wrap)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self.analysis_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.run_btn.clicked.connect(self._run_analysis)
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.tab_gridscore.inspectRequested.connect(self._run_gridscore_inspect)

        self._on_mode_changed()
        self._apply_card_effects([self.param_container] + list(self.param_widgets.values()))
        self._sync_standardize()
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
        self.param_container.setTitle("分析参数" if is_zh else "Analysis Parameters")
        self.label_analysis_module.setText("分析模块:" if is_zh else "Analysis module:")
        self.help_btn.setText("帮助" if is_zh else "Help")
        self.help_btn.setToolTip("查看参数说明" if is_zh else "Show parameter guide for the selected mode.")
        self.grp_standardize.setTitle("预处理（标准化）" if is_zh else "Preprocess (Standardization)")
        self.chk_standardize.setText("StandardScaler")
        self.run_btn.setText("运行分析" if is_zh else "Run Analysis")
        self.stop_btn.setText("停止" if is_zh else "Stop")
        self.logs_label.setText("日志" if is_zh else "Logs")

        if self._last_state is not None:
            self._update_info(self._last_state)
        else:
            self.info_label.setText(
                "模式=— | 预设=— | 预处理=— | spike_main_shape=—"
                if is_zh
                else "Mode=— | preset=— | preprocess=— | spike_main_shape=—"
            )

    def load_state(self, state) -> None:
        self._last_state = state
        self._update_info(state)
        preset = getattr(state, "preset", None)
        if preset:
            for mode in self._modes.values():
                mode.apply_preset(preset)

        total_steps = None
        neuron_count = None

        embed_data = getattr(state, "embed_data", None)
        if isinstance(embed_data, np.ndarray) and embed_data.ndim == 2:
            total_steps, neuron_count = embed_data.shape
        else:
            aligned_pos = getattr(state, "aligned_pos", None)
            if isinstance(aligned_pos, dict) and "t" in aligned_pos:
                try:
                    total_steps = len(aligned_pos["t"])
                except Exception:
                    total_steps = None
            inferred = self._infer_counts_from_state(state)
            if inferred is not None:
                inferred_steps, inferred_neurons = inferred
                total_steps = total_steps or inferred_steps
                neuron_count = neuron_count or inferred_neurons

        for mode in self._modes.values():
            mode.apply_ranges(neuron_count, total_steps)

    def _update_info(self, state) -> None:
        mode = getattr(state, "input_mode", "—")
        preset = getattr(state, "preset", "—")
        preprocess = getattr(state, "preprocess_method", "—")
        preclass = getattr(state, "preclass", None)
        shape = "None"
        embed = getattr(state, "embed_data", None)
        if isinstance(embed, np.ndarray) and embed.ndim == 2:
            shape = f"{embed.shape}"
        is_zh = str(self._lang).lower().startswith("zh")
        if is_zh:
            parts = [f"模式={mode}", f"预设={preset}", f"预处理={preprocess}"]
        else:
            parts = [f"Mode={mode}", f"preset={preset}", f"preprocess={preprocess}"]
        if preclass is not None:
            parts.append(("预分类" if is_zh else "preclass") + f"={preclass}")
        parts.append(f"spike_main_shape={shape}")
        self.info_label.setText(" | ".join(parts))

    def _infer_counts_from_state(self, state) -> tuple[int | None, int | None] | None:
        try:
            from ...core.state import resolve_path
        except Exception:
            return None

        def _infer_from_spike(spike_obj) -> tuple[int | None, int | None]:
            if spike_obj is None:
                return None, None
            if isinstance(spike_obj, np.ndarray):
                if spike_obj.ndim == 2:
                    return int(spike_obj.shape[0]), int(spike_obj.shape[1])
                if spike_obj.dtype == object:
                    if spike_obj.size == 1:
                        spike_obj = spike_obj.item()
                    elif spike_obj.ndim == 1:
                        return None, int(spike_obj.shape[0])
            if isinstance(spike_obj, dict):
                return None, int(len(spike_obj))
            if isinstance(spike_obj, (list, tuple)):
                return None, int(len(spike_obj))
            return None, None

        total_steps = None
        neuron_count = None

        if getattr(state, "input_mode", None) == "asa":
            path = resolve_path(state, state.asa_file)
            if path is None:
                return None
            data = np.load(path, allow_pickle=True)
            if "t" in data:
                try:
                    total_steps = len(data["t"])
                except Exception:
                    total_steps = None
            if "spike" in data:
                t_guess, n_guess = _infer_from_spike(data["spike"])
                total_steps = total_steps or t_guess
                neuron_count = neuron_count or n_guess
        elif getattr(state, "input_mode", None) == "neuron_traj":
            neuron_path = resolve_path(state, state.neuron_file)
            traj_path = resolve_path(state, state.traj_file)
            if neuron_path is not None and neuron_path.exists():
                neuron_data = np.load(neuron_path, allow_pickle=True)
                if isinstance(neuron_data, np.lib.npyio.NpzFile):
                    if "spike" in neuron_data:
                        spike_obj = neuron_data["spike"]
                    elif neuron_data.files:
                        spike_obj = neuron_data[neuron_data.files[0]]
                    else:
                        spike_obj = None
                    t_guess, n_guess = _infer_from_spike(spike_obj)
                else:
                    t_guess, n_guess = _infer_from_spike(neuron_data)
                total_steps = total_steps or t_guess
                neuron_count = neuron_count or n_guess
            if traj_path is not None and traj_path.exists():
                traj_data = np.load(traj_path, allow_pickle=True)
                if isinstance(traj_data, np.lib.npyio.NpzFile):
                    for key in ("t", "x", "y"):
                        if key in traj_data:
                            total_steps = total_steps or len(traj_data[key])
                            break
                else:
                    if hasattr(traj_data, "shape") and len(traj_data.shape) > 0:
                        total_steps = total_steps or int(traj_data.shape[0])

        if total_steps is None and neuron_count is None:
            return None
        return total_steps, neuron_count

    def _on_mode_changed(self) -> None:
        mode = self.analysis_mode.currentData() or "tda"
        self._sync_standardize()
        visible = {
            "tda": {"tda"},
            "cohomap": {"cohomap"},
            "pathcompare": {"pathcompare"},
            "cohospace": {"cohospace"},
            "fr": {"fr"},
            "frm": {"frm"},
            "gridscore": {"gridscore"},
            "gridscore_inspect": {"gridscore"},
            "decode": {"decode"},
        }.get(mode, {mode})

        for name, widget in self.param_widgets.items():
            widget.setVisible(name in visible)

        self.grp_standardize.setVisible(mode in {"tda", "cohomap"})

    def _sync_standardize(self) -> None:
        tda_mode = self._modes.get("tda")
        if tda_mode is None or not hasattr(tda_mode, "standardize"):
            return
        try:
            checkbox = tda_mode.standardize
            if checkbox.isChecked() != self.chk_standardize.isChecked():
                self.chk_standardize.setChecked(bool(checkbox.isChecked()))
        except Exception:
            return

        def _on_toggle(val: bool) -> None:
            try:
                checkbox.setChecked(bool(val))
            except Exception:
                pass

        try:
            self.chk_standardize.toggled.disconnect()
        except Exception:
            pass
        self.chk_standardize.toggled.connect(_on_toggle)

    def _run_analysis(self, mode_override: str | None = None) -> None:
        if self._workers.is_running():
            self.log_box.log("A task is already running.")
            return

        mode = mode_override or (self.analysis_mode.currentData() or "tda")
        params = self._collect_params(mode)

        state = self._controller.get_state()
        from ...core.state import validate_files, validate_preprocessing

        ok, msg = validate_files(state)
        if not ok:
            self.log_box.log(f"Input error: {msg}")
            return

        if mode == "tda":
            ok, msg = validate_preprocessing(state)
            if not ok:
                self.log_box.log(f"Preprocess required for TDA: {msg}")
                return

        if mode in {"fr", "frm", "gridscore", "gridscore_inspect", "cohospace"}:
            mode_flag = None
            if mode in {"fr", "frm"}:
                mode_flag = params.get("mode")
            elif mode in {"gridscore", "gridscore_inspect"}:
                mode_flag = params.get("gridscore", {}).get("mode")
            elif mode == "cohospace":
                mode_flag = params.get("mode")
            if mode_flag == "fr" and state.embed_data is None:
                self.log_box.log("Preprocess required for FR-mode. Use spike-mode or run preprocess.")
                return

        self._controller.update_analysis(analysis_mode=mode, analysis_params=params)

        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_box.log(f"Starting analysis: {mode}")
        if mode == "gridscore_inspect":
            self.tab_gridscore.set_status("Computing gridscore inspect…")

        def _on_log(msg: str) -> None:
            if msg.startswith("__PCANIM__"):
                parts = msg.split()
                if len(parts) >= 2:
                    try:
                        pct = int(parts[1])
                        self.tab_pathcompare.set_animation_progress(pct)
                        return
                    except Exception:
                        pass
            self.log_box.log(msg)

        def _on_progress(pct: int) -> None:
            self.progress.setValue(pct)

        def _on_finished(result) -> None:
            if hasattr(result, "success") and not result.success:
                self._controller.mark_idle()
                self.log_box.log(result.error or "Analysis failed")
                self.run_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
            artifacts = result.artifacts if hasattr(result, "artifacts") else {}
            self._controller.finalize_analysis(artifacts)
            self._populate_artifacts(artifacts)
            self._select_result_tab(mode, artifacts)
            self.log_box.log(result.summary)
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.analysis_completed.emit()

        def _on_error(msg: str) -> None:
            self._controller.mark_idle()
            self.log_box.log(f"Error: {msg}")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

        def _on_cleanup() -> None:
            self._controller.mark_idle()

        self._controller.run_analysis(
            worker_manager=self._workers,
            on_log=_on_log,
            on_progress=_on_progress,
            on_finished=_on_finished,
            on_error=_on_error,
            on_cleanup=_on_cleanup,
        )

    def _show_help(self) -> None:
        mode = self.analysis_mode.currentData()
        lang = str(QSettings("canns", "asa_gui").value("lang", "en"))
        markdown = analysis_help_markdown(str(mode) if mode is not None else "", lang=lang)
        title = "ASA Help" if not str(lang).lower().startswith("zh") else "ASA 参数说明"
        show_help_dialog(self, title, markdown)

    def _stop_analysis(self) -> None:
        if self._workers.is_running():
            self._workers.request_cancel()
            self.log_box.log("Cancel requested.")

    def _populate_artifacts(self, artifacts: dict) -> None:
        if "barcode" in artifacts:
            self.tab_barcode.set_image(artifacts.get("barcode"))
        if "cohomap" in artifacts:
            self.tab_cohomap.set_image(artifacts.get("cohomap"))
        if "path_compare" in artifacts or "path_compare_gif" in artifacts:
            self.tab_pathcompare.set_artifacts(
                artifacts.get("path_compare"),
                artifacts.get("path_compare_gif"),
            )
        if "path_compare_mp4" in artifacts:
            self.tab_pathcompare.set_animation(Path(artifacts["path_compare_mp4"]))
        if "neuron" in artifacts:
            self.tab_cohospace.set_image(artifacts.get("neuron"))
        elif "population" in artifacts:
            self.tab_cohospace.set_image(artifacts.get("population"))
        elif "trajectory" in artifacts:
            self.tab_cohospace.set_image(artifacts.get("trajectory"))
        if "fr_heatmap" in artifacts:
            self.tab_fr.set_image(artifacts.get("fr_heatmap"))
        if "frm" in artifacts:
            self.tab_frm.set_image(artifacts.get("frm"))

        if "gridscore_png" in artifacts:
            self.tab_gridscore.set_distribution_image(Path(artifacts["gridscore_png"]))
        if "gridscore_npz" in artifacts:
            try:
                self.tab_gridscore.load_gridscore_npz(Path(artifacts["gridscore_npz"]))
            except Exception as e:
                self.log_box.log(f"GridScore: failed to load gridscore.npz: {e}")
        if "gridscore_neuron_png" in artifacts:
            self.tab_gridscore.set_autocorr_image(Path(artifacts["gridscore_neuron_png"]))
            self.tab_gridscore.set_status("")

        self.tab_files.set_artifacts(artifacts)

    def _select_result_tab(self, mode: str, artifacts: dict) -> None:
        mapping = {
            "tda": self.tab_barcode,
            "decode": self.tab_files,
            "cohomap": self.tab_cohomap,
            "pathcompare": self.tab_pathcompare,
            "cohospace": self.tab_cohospace,
            "fr": self.tab_fr,
            "frm": self.tab_frm,
            "gridscore": self.tab_gridscore,
            "gridscore_inspect": self.tab_gridscore,
        }
        target = mapping.get(mode)
        if target is None:
            return
        idx = self.tabs.indexOf(target)
        if idx < 0:
            return
        if mode == "decode" and not artifacts:
            return
        self.tabs.setCurrentIndex(idx)

    def _run_gridscore_inspect(self, neuron_id: int, meta: dict) -> None:
        if self._workers.is_running():
            self.log_box.log("A task is already running.")
            return
        mode_obj = self._modes.get("gridscore_inspect")
        if mode_obj is None:
            self.log_box.log("GridScore Inspect is not available.")
            return
        if hasattr(mode_obj, "apply_meta"):
            meta = dict(meta or {})
            meta["neuron_id"] = int(neuron_id)
            mode_obj.apply_meta(meta)
        self._run_analysis(mode_override="gridscore_inspect")

    def _collect_params(self, mode: str) -> dict:
        mode_obj = self._modes.get(mode)
        if mode_obj is None:
            return {}
        return mode_obj.collect_params()
