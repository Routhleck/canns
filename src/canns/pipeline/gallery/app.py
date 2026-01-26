"""Model gallery TUI for quick CANN visualizations."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, ProgressBar, Select, Static
from textual.worker import Worker

from canns.pipeline.asa.screens import ErrorScreen, TerminalSizeWarning, WorkdirScreen
from canns.pipeline.asa.widgets import ImagePreview, LogViewer, ParamGroup

from .runner import GalleryRunner
from .state import GalleryState, get_analysis_options, get_default_analysis


CANN1D_DEFAULTS = {
    "seed": "42",
    "num": "256",
    "tau": "1.0",
    "k": "8.1",
    "a": "0.5",
    "A": "10.0",
    "J0": "4.0",
    "dt": "0.1",
    "energy_pos": "1.0",
    "energy_duration": "10.0",
    "tuning_start": "0.0",
    "tuning_mid": "3.1416",
    "tuning_end": "6.2832",
    "tuning_duration": "40.0",
    "tuning_bins": "50",
    "tuning_neurons": "64,128,192",
    "template_pos": "1.0",
    "template_duration": "10.0",
    "manifold_segment": "40.0",
    "manifold_warmup": "5.0",
}

CANN2D_DEFAULTS = {
    "seed": "42",
    "length": "64",
    "tau": "1.0",
    "k": "8.1",
    "a": "0.5",
    "A": "10.0",
    "J0": "4.0",
    "dt": "0.1",
    "energy_x": "1.0",
    "energy_y": "1.0",
    "energy_duration": "10.0",
    "field_duration": "8000.0",
    "field_box": "6.2832",
    "field_resolution": "80",
    "field_sigma": "2.0",
    "field_speed": "0.3",
    "field_speed_std": "0.1",
    "traj_segment": "40.0",
    "traj_warmup": "5.0",
    "manifold_duration": "22000.0",
    "manifold_warmup": "2000.0",
    "manifold_speed": "0.05",
    "manifold_speed_std": "0.02",
    "manifold_downsample": "10",
}

GRID_DEFAULTS = {
    "seed": "74",
    "length": "40",
    "tau": "0.01",
    "alpha": "0.1",
    "W_l": "3.0",
    "lambda_net": "17.0",
    "dt": "0.0005",
    "box_size": "2.2",
    "energy_duration": "10.0",
    "energy_speed": "0.2",
    "energy_speed_std": "0.05",
    "energy_heal_steps": "10000",
    "field_resolution": "100",
    "field_sigma": "2.0",
    "field_speed": "0.3",
    "field_batches": "10",
    "path_duration": "10.0",
    "path_dt": "0.01",
    "path_speed": "0.5",
    "path_speed_std": "0.05",
    "path_heal_steps": "5000",
}


class GalleryApp(App):
    """Main TUI application for the model gallery."""

    CSS_PATH = "styles.tcss"
    TITLE = "CANNs Model Gallery"

    MIN_WIDTH = 100
    RECOMMENDED_WIDTH = 120
    MIN_HEIGHT = 28
    RECOMMENDED_HEIGHT = 36

    BINDINGS = [
        Binding("ctrl+w", "change_workdir", "Workdir"),
        Binding("ctrl+r", "run", "Run"),
        Binding("f5", "refresh", "Refresh"),
        Binding("escape", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = GalleryState()
        self.runner = GalleryRunner()
        self.current_worker: Worker | None = None
        self._size_warning_shown = False

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                yield Label(f"Workdir: {self.state.workdir}", id="workdir-label")
                yield Button("Change Workdir", id="change-workdir-btn")

                yield Label("Model")
                yield Select(
                    [
                        ("CANN 1D", "cann1d"),
                        ("CANN 2D", "cann2d"),
                        ("Grid Cell", "gridcell"),
                    ],
                    value=self.state.model,
                    id="model-select",
                )

                yield Label("Analysis")
                yield Select(
                    get_analysis_options(self.state.model),
                    value=self.state.analysis,
                    id="analysis-select",
                )

                yield Button("Run", variant="primary", id="run-btn")
                yield ProgressBar(id="progress-bar")
                yield Static("Status: Idle", id="run-status")

            with Vertical(id="middle-panel"):
                with Vertical(id="params-panel"):
                    yield Static("Parameters", id="params-header")
                    with VerticalScroll(id="params-scroll"):
                        with Vertical(id="params-cann1d"):
                            with ParamGroup("CANN1D Model"):
                                yield Label("seed")
                                yield Input(value=CANN1D_DEFAULTS["seed"], id="c1-seed")
                                yield Label("num")
                                yield Input(value=CANN1D_DEFAULTS["num"], id="c1-num")
                                yield Label("tau")
                                yield Input(value=CANN1D_DEFAULTS["tau"], id="c1-tau")
                                yield Label("k")
                                yield Input(value=CANN1D_DEFAULTS["k"], id="c1-k")
                                yield Label("a")
                                yield Input(value=CANN1D_DEFAULTS["a"], id="c1-a")
                                yield Label("A")
                                yield Input(value=CANN1D_DEFAULTS["A"], id="c1-A")
                                yield Label("J0")
                                yield Input(value=CANN1D_DEFAULTS["J0"], id="c1-J0")
                                yield Label("dt")
                                yield Input(value=CANN1D_DEFAULTS["dt"], id="c1-dt")

                            with ParamGroup("Energy Landscape", id="c1-analysis-energy"):
                                yield Label("stimulus_pos")
                                yield Input(value=CANN1D_DEFAULTS["energy_pos"], id="c1-energy-pos")
                                yield Label("duration")
                                yield Input(
                                    value=CANN1D_DEFAULTS["energy_duration"],
                                    id="c1-energy-duration",
                                )

                            with ParamGroup("Tuning Curve", id="c1-analysis-tuning"):
                                yield Label("start_pos")
                                yield Input(
                                    value=CANN1D_DEFAULTS["tuning_start"],
                                    id="c1-tuning-start",
                                )
                                yield Label("mid_pos")
                                yield Input(
                                    value=CANN1D_DEFAULTS["tuning_mid"],
                                    id="c1-tuning-mid",
                                )
                                yield Label("end_pos")
                                yield Input(
                                    value=CANN1D_DEFAULTS["tuning_end"],
                                    id="c1-tuning-end",
                                )
                                yield Label("segment_duration")
                                yield Input(
                                    value=CANN1D_DEFAULTS["tuning_duration"],
                                    id="c1-tuning-duration",
                                )
                                yield Label("num_bins")
                                yield Input(
                                    value=CANN1D_DEFAULTS["tuning_bins"],
                                    id="c1-tuning-bins",
                                )
                                yield Label("neuron_indices")
                                yield Input(
                                    value=CANN1D_DEFAULTS["tuning_neurons"],
                                    id="c1-tuning-neurons",
                                )

                            with ParamGroup("Template Matching", id="c1-analysis-template"):
                                yield Label("stimulus_pos")
                                yield Input(
                                    value=CANN1D_DEFAULTS["template_pos"],
                                    id="c1-template-pos",
                                )
                                yield Label("duration")
                                yield Input(
                                    value=CANN1D_DEFAULTS["template_duration"],
                                    id="c1-template-duration",
                                )

                            with ParamGroup("Neural Manifold", id="c1-analysis-manifold"):
                                yield Label("segment_duration")
                                yield Input(
                                    value=CANN1D_DEFAULTS["manifold_segment"],
                                    id="c1-manifold-segment",
                                )
                                yield Label("warmup")
                                yield Input(
                                    value=CANN1D_DEFAULTS["manifold_warmup"],
                                    id="c1-manifold-warmup",
                                )

                            with ParamGroup("Connectivity", id="c1-analysis-connectivity"):
                                yield Static("No extra parameters")

                        with Vertical(id="params-cann2d", classes="hidden"):
                            with ParamGroup("CANN2D Model"):
                                yield Label("seed")
                                yield Input(value=CANN2D_DEFAULTS["seed"], id="c2-seed")
                                yield Label("length")
                                yield Input(value=CANN2D_DEFAULTS["length"], id="c2-length")
                                yield Label("tau")
                                yield Input(value=CANN2D_DEFAULTS["tau"], id="c2-tau")
                                yield Label("k")
                                yield Input(value=CANN2D_DEFAULTS["k"], id="c2-k")
                                yield Label("a")
                                yield Input(value=CANN2D_DEFAULTS["a"], id="c2-a")
                                yield Label("A")
                                yield Input(value=CANN2D_DEFAULTS["A"], id="c2-A")
                                yield Label("J0")
                                yield Input(value=CANN2D_DEFAULTS["J0"], id="c2-J0")
                                yield Label("dt")
                                yield Input(value=CANN2D_DEFAULTS["dt"], id="c2-dt")

                            with ParamGroup("Energy Landscape", id="c2-analysis-energy"):
                                yield Label("stimulus_x")
                                yield Input(value=CANN2D_DEFAULTS["energy_x"], id="c2-energy-x")
                                yield Label("stimulus_y")
                                yield Input(value=CANN2D_DEFAULTS["energy_y"], id="c2-energy-y")
                                yield Label("duration")
                                yield Input(
                                    value=CANN2D_DEFAULTS["energy_duration"],
                                    id="c2-energy-duration",
                                )

                            with ParamGroup("Firing Field", id="c2-analysis-firing"):
                                yield Label("duration")
                                yield Input(
                                    value=CANN2D_DEFAULTS["field_duration"], id="c2-field-duration"
                                )
                                yield Label("box_size")
                                yield Input(value=CANN2D_DEFAULTS["field_box"], id="c2-field-box")
                                yield Label("resolution")
                                yield Input(
                                    value=CANN2D_DEFAULTS["field_resolution"],
                                    id="c2-field-resolution",
                                )
                                yield Label("smooth_sigma")
                                yield Input(
                                    value=CANN2D_DEFAULTS["field_sigma"], id="c2-field-sigma"
                                )
                                yield Label("speed_mean")
                                yield Input(
                                    value=CANN2D_DEFAULTS["field_speed"], id="c2-field-speed"
                                )
                                yield Label("speed_std")
                                yield Input(
                                    value=CANN2D_DEFAULTS["field_speed_std"],
                                    id="c2-field-speed-std",
                                )

                            with ParamGroup("Trajectory Comparison", id="c2-analysis-trajectory"):
                                yield Label("segment_duration")
                                yield Input(
                                    value=CANN2D_DEFAULTS["traj_segment"],
                                    id="c2-traj-segment",
                                )
                                yield Label("warmup")
                                yield Input(
                                    value=CANN2D_DEFAULTS["traj_warmup"],
                                    id="c2-traj-warmup",
                                )

                            with ParamGroup("Neural Manifold", id="c2-analysis-manifold"):
                                yield Label("duration")
                                yield Input(
                                    value=CANN2D_DEFAULTS["manifold_duration"],
                                    id="c2-manifold-duration",
                                )
                                yield Label("warmup")
                                yield Input(
                                    value=CANN2D_DEFAULTS["manifold_warmup"],
                                    id="c2-manifold-warmup",
                                )
                                yield Label("speed_mean")
                                yield Input(
                                    value=CANN2D_DEFAULTS["manifold_speed"],
                                    id="c2-manifold-speed",
                                )
                                yield Label("speed_std")
                                yield Input(
                                    value=CANN2D_DEFAULTS["manifold_speed_std"],
                                    id="c2-manifold-speed-std",
                                )
                                yield Label("downsample")
                                yield Input(
                                    value=CANN2D_DEFAULTS["manifold_downsample"],
                                    id="c2-manifold-downsample",
                                )

                            with ParamGroup("Connectivity", id="c2-analysis-connectivity"):
                                yield Static("No extra parameters")

                        with Vertical(id="params-gridcell", classes="hidden"):
                            with ParamGroup("Grid Cell Model"):
                                yield Label("seed")
                                yield Input(value=GRID_DEFAULTS["seed"], id="g-seed")
                                yield Label("length")
                                yield Input(value=GRID_DEFAULTS["length"], id="g-length")
                                yield Label("tau")
                                yield Input(value=GRID_DEFAULTS["tau"], id="g-tau")
                                yield Label("alpha")
                                yield Input(value=GRID_DEFAULTS["alpha"], id="g-alpha")
                                yield Label("W_l")
                                yield Input(value=GRID_DEFAULTS["W_l"], id="g-Wl")
                                yield Label("lambda_net")
                                yield Input(value=GRID_DEFAULTS["lambda_net"], id="g-lambda")
                                yield Label("dt")
                                yield Input(value=GRID_DEFAULTS["dt"], id="g-dt")

                            with ParamGroup("Energy Landscape", id="g-analysis-energy"):
                                yield Label("duration")
                                yield Input(
                                    value=GRID_DEFAULTS["energy_duration"], id="g-energy-duration"
                                )
                                yield Label("speed_mean")
                                yield Input(
                                    value=GRID_DEFAULTS["energy_speed"], id="g-energy-speed"
                                )
                                yield Label("speed_std")
                                yield Input(
                                    value=GRID_DEFAULTS["energy_speed_std"],
                                    id="g-energy-speed-std",
                                )
                                yield Label("heal_steps")
                                yield Input(
                                    value=GRID_DEFAULTS["energy_heal_steps"],
                                    id="g-energy-heal",
                                )

                            with ParamGroup("Firing Field", id="g-analysis-firing"):
                                yield Label("box_size")
                                yield Input(value=GRID_DEFAULTS["box_size"], id="g-box-size")
                                yield Label("resolution")
                                yield Input(
                                    value=GRID_DEFAULTS["field_resolution"],
                                    id="g-field-resolution",
                                )
                                yield Label("smooth_sigma")
                                yield Input(value=GRID_DEFAULTS["field_sigma"], id="g-field-sigma")
                                yield Label("speed")
                                yield Input(value=GRID_DEFAULTS["field_speed"], id="g-field-speed")
                                yield Label("num_batches")
                                yield Input(
                                    value=GRID_DEFAULTS["field_batches"],
                                    id="g-field-batches",
                                )

                            with ParamGroup("Path Integration", id="g-analysis-path"):
                                yield Label("duration")
                                yield Input(value=GRID_DEFAULTS["path_duration"], id="g-path-duration")
                                yield Label("dt")
                                yield Input(value=GRID_DEFAULTS["path_dt"], id="g-path-dt")
                                yield Label("speed_mean")
                                yield Input(value=GRID_DEFAULTS["path_speed"], id="g-path-speed")
                                yield Label("speed_std")
                                yield Input(
                                    value=GRID_DEFAULTS["path_speed_std"], id="g-path-speed-std"
                                )
                                yield Label("heal_steps")
                                yield Input(
                                    value=GRID_DEFAULTS["path_heal_steps"], id="g-path-heal"
                                )

                            with ParamGroup("Connectivity", id="g-analysis-connectivity"):
                                yield Static("No extra parameters")

            with Vertical(id="right-panel"):
                yield ImagePreview(id="result-preview")
                yield LogViewer(id="log-viewer")

        yield Footer()

    def on_mount(self) -> None:
        self.check_terminal_size()
        self._update_analysis_options()
        self._update_param_visibility()

    def on_resize(self, event) -> None:
        self.check_terminal_size()

    def check_terminal_size(self) -> None:
        width, height = self.size
        if not self._size_warning_shown and (
            width < self.MIN_WIDTH or height < self.MIN_HEIGHT
        ):
            self._size_warning_shown = True
            self.push_screen(TerminalSizeWarning(width, height))

    def action_change_workdir(self) -> None:
        self.push_screen(WorkdirScreen(), self.on_workdir_selected)

    def on_workdir_selected(self, path: Path | None) -> None:
        if path:
            self.state.workdir = path
            self.update_workdir_label()

    def update_workdir_label(self) -> None:
        label = self.query_one("#workdir-label", Label)
        label.update(f"Workdir: {self.state.workdir}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "change-workdir-btn":
            self.action_change_workdir()
        elif event.button.id == "run-btn":
            self.action_run()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select":
            self.state.model = str(event.value)
            self.state.analysis = get_default_analysis(self.state.model)
            self._update_analysis_options()
            self._update_param_visibility()
        elif event.select.id == "analysis-select":
            self.state.analysis = str(event.value)
            self._update_param_visibility()

    def action_run(self) -> None:
        if self.current_worker and not self.current_worker.is_finished:
            self.log_message("A task is already running.")
            return

        try:
            model_params, analysis_params = self.collect_params()
        except ValueError as exc:
            self.push_screen(ErrorScreen("Parameter Error", str(exc)))
            return

        output_dir = self.state.workdir / "Results" / "gallery" / self.state.model
        self.set_run_status("Status: Running...", "running")
        self.update_progress(0)

        self.current_worker = self.run_worker(
            self.runner.run(
                self.state.model,
                self.state.analysis,
                model_params,
                analysis_params,
                output_dir,
                log_callback=self.log_message,
                progress_callback=self.update_progress,
            ),
            name="gallery_worker",
            thread=True,
        )

    def action_refresh(self) -> None:
        preview_path = self.state.artifacts.get("output")
        if preview_path:
            preview = self.query_one("#result-preview", ImagePreview)
            preview.update_image(preview_path)
            self.log_message(f"Refreshed preview: {preview_path}")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "gallery_worker" or not event.worker.is_finished:
            return
        result = event.worker.result
        if result.success:
            self.state.artifacts = result.artifacts
            self.set_run_status("Status: Complete", "success")
            self.log_message(result.summary)
            output_path = result.artifacts.get("output")
            if output_path:
                preview = self.query_one("#result-preview", ImagePreview)
                preview.update_image(output_path)
        else:
            self.set_run_status("Status: Failed", "error")
            self.push_screen(ErrorScreen("Gallery Error", result.error or "Unknown error"))

    def set_run_status(self, message: str, status_class: str | None = None) -> None:
        status = self.query_one("#run-status", Static)
        status.update(message)
        status.remove_class("running", "success", "error")
        if status_class:
            status.add_class(status_class)

    def update_progress(self, percent: int) -> None:
        progress = self.query_one("#progress-bar", ProgressBar)
        progress.update(total=100, progress=percent)

    def log_message(self, message: str) -> None:
        log_viewer = self.query_one("#log-viewer", LogViewer)
        log_viewer.add_log(message)

    def collect_params(self) -> tuple[dict[str, float | int], dict[str, float | int]]:
        if self.state.model == "cann1d":
            model_params = {
                "seed": self._int("#c1-seed"),
                "num": self._int("#c1-num"),
                "tau": self._float("#c1-tau"),
                "k": self._float("#c1-k"),
                "a": self._float("#c1-a"),
                "A": self._float("#c1-A"),
                "J0": self._float("#c1-J0"),
                "dt": self._float("#c1-dt"),
            }
            analysis_params = {
                "energy_pos": self._float("#c1-energy-pos"),
                "energy_duration": self._float("#c1-energy-duration"),
                "tuning_start": self._float("#c1-tuning-start"),
                "tuning_mid": self._float("#c1-tuning-mid"),
                "tuning_end": self._float("#c1-tuning-end"),
                "tuning_duration": self._float("#c1-tuning-duration"),
                "tuning_bins": self._int("#c1-tuning-bins"),
                "tuning_neurons": self._str("#c1-tuning-neurons"),
                "template_pos": self._float("#c1-template-pos"),
                "template_duration": self._float("#c1-template-duration"),
                "manifold_segment": self._float("#c1-manifold-segment"),
                "manifold_warmup": self._float("#c1-manifold-warmup"),
            }
            return model_params, analysis_params

        if self.state.model == "cann2d":
            model_params = {
                "seed": self._int("#c2-seed"),
                "length": self._int("#c2-length"),
                "tau": self._float("#c2-tau"),
                "k": self._float("#c2-k"),
                "a": self._float("#c2-a"),
                "A": self._float("#c2-A"),
                "J0": self._float("#c2-J0"),
                "dt": self._float("#c2-dt"),
            }
            analysis_params = {
                "energy_x": self._float("#c2-energy-x"),
                "energy_y": self._float("#c2-energy-y"),
                "energy_duration": self._float("#c2-energy-duration"),
                "field_duration": self._float("#c2-field-duration"),
                "field_box": self._float("#c2-field-box"),
                "field_resolution": self._int("#c2-field-resolution"),
                "field_sigma": self._float("#c2-field-sigma"),
                "field_speed": self._float("#c2-field-speed"),
                "field_speed_std": self._float("#c2-field-speed-std"),
                "traj_segment": self._float("#c2-traj-segment"),
                "traj_warmup": self._float("#c2-traj-warmup"),
                "manifold_duration": self._float("#c2-manifold-duration"),
                "manifold_warmup": self._float("#c2-manifold-warmup"),
                "manifold_speed": self._float("#c2-manifold-speed"),
                "manifold_speed_std": self._float("#c2-manifold-speed-std"),
                "manifold_downsample": self._int("#c2-manifold-downsample"),
            }
            return model_params, analysis_params

        if self.state.model == "gridcell":
            model_params = {
                "seed": self._int("#g-seed"),
                "length": self._int("#g-length"),
                "tau": self._float("#g-tau"),
                "alpha": self._float("#g-alpha"),
                "W_l": self._float("#g-Wl"),
                "lambda_net": self._float("#g-lambda"),
                "dt": self._float("#g-dt"),
            }
            analysis_params = {
                "box_size": self._float("#g-box-size"),
                "energy_duration": self._float("#g-energy-duration"),
                "energy_speed": self._float("#g-energy-speed"),
                "energy_speed_std": self._float("#g-energy-speed-std"),
                "energy_heal_steps": self._int("#g-energy-heal"),
                "field_resolution": self._int("#g-field-resolution"),
                "field_sigma": self._float("#g-field-sigma"),
                "field_speed": self._float("#g-field-speed"),
                "field_batches": self._int("#g-field-batches"),
                "path_duration": self._float("#g-path-duration"),
                "path_dt": self._float("#g-path-dt"),
                "path_speed": self._float("#g-path-speed"),
                "path_speed_std": self._float("#g-path-speed-std"),
                "path_heal_steps": self._int("#g-path-heal"),
            }
            return model_params, analysis_params

        raise ValueError(f"Unknown model: {self.state.model}")

    def _int(self, selector: str) -> int:
        return int(self.query_one(selector, Input).value)

    def _float(self, selector: str) -> float:
        return float(self.query_one(selector, Input).value)

    def _str(self, selector: str) -> str:
        return self.query_one(selector, Input).value

    def _update_analysis_options(self) -> None:
        select = self.query_one("#analysis-select", Select)
        options = get_analysis_options(self.state.model)
        select.set_options(options)
        select.value = self.state.analysis

    def _update_param_visibility(self) -> None:
        self._set_visible("#params-cann1d", self.state.model == "cann1d")
        self._set_visible("#params-cann2d", self.state.model == "cann2d")
        self._set_visible("#params-gridcell", self.state.model == "gridcell")

        # CANN1D analysis groups
        self._set_visible(
            "#c1-analysis-connectivity",
            self.state.model == "cann1d" and self.state.analysis == "connectivity",
        )
        self._set_visible(
            "#c1-analysis-energy",
            self.state.model == "cann1d" and self.state.analysis == "energy",
        )
        self._set_visible(
            "#c1-analysis-tuning",
            self.state.model == "cann1d" and self.state.analysis == "tuning",
        )
        self._set_visible(
            "#c1-analysis-template",
            self.state.model == "cann1d" and self.state.analysis == "template",
        )
        self._set_visible(
            "#c1-analysis-manifold",
            self.state.model == "cann1d" and self.state.analysis == "manifold",
        )

        # CANN2D analysis groups
        self._set_visible(
            "#c2-analysis-connectivity",
            self.state.model == "cann2d" and self.state.analysis == "connectivity",
        )
        self._set_visible(
            "#c2-analysis-energy",
            self.state.model == "cann2d" and self.state.analysis == "energy",
        )
        self._set_visible(
            "#c2-analysis-firing",
            self.state.model == "cann2d" and self.state.analysis == "firing_field",
        )
        self._set_visible(
            "#c2-analysis-trajectory",
            self.state.model == "cann2d" and self.state.analysis == "trajectory",
        )
        self._set_visible(
            "#c2-analysis-manifold",
            self.state.model == "cann2d" and self.state.analysis == "manifold",
        )

        # Grid cell analysis groups
        self._set_visible(
            "#g-analysis-connectivity",
            self.state.model == "gridcell" and self.state.analysis == "connectivity",
        )
        self._set_visible(
            "#g-analysis-energy",
            self.state.model == "gridcell" and self.state.analysis == "energy",
        )
        self._set_visible(
            "#g-analysis-firing",
            self.state.model == "gridcell"
            and self.state.analysis in {"firing_field", "manifold"},
        )
        self._set_visible(
            "#g-analysis-path",
            self.state.model == "gridcell" and self.state.analysis == "path_integration",
        )

    def _set_visible(self, selector: str, visible: bool) -> None:
        widget = self.query_one(selector)
        if visible:
            widget.remove_class("hidden")
        else:
            widget.add_class("hidden")
