"""Main ASA TUI application with two-page workflow.

This module provides the main Textual application for ASA analysis,
following the original GUI's two-page structure:
1. PreprocessPage - File selection and preprocessing
2. AnalysisPage - Analysis mode selection and execution
"""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker

from .runner import PipelineRunner
from .screens import ErrorScreen, HelpScreen, TerminalSizeWarning, WorkdirScreen
from .state import WorkflowState, get_preset_params, relative_path, validate_files
from .widgets import ImagePreview, LogViewer, ParamGroup


class ASAApp(App):
    """Main TUI application for ASA analysis."""

    CSS_PATH = "styles.tcss"
    TITLE = "Attractor Structure Analyzer (ASA)"

    # Terminal size requirements
    MIN_WIDTH = 100
    RECOMMENDED_WIDTH = 120
    MIN_HEIGHT = 30
    RECOMMENDED_HEIGHT = 40

    BINDINGS = [
        Binding("ctrl+w", "change_workdir", "Workdir"),
        Binding("ctrl+r", "run_action", "Run"),
        Binding("f5", "refresh", "Refresh"),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.state = WorkflowState()
        self.runner = PipelineRunner()
        self.current_worker: Worker = None
        self._size_warning_shown = False
        self.current_page = "preprocess"  # "preprocess" or "analysis"

    def compose(self) -> ComposeResult:
        """Compose the main UI layout."""
        yield Header()

        with Horizontal(id="main-container"):
            # Left panel (controls)
            with Vertical(id="left-panel"):
                yield Label(f"Workdir: {self.state.workdir}", id="workdir-label")
                yield Button("Change Workdir", id="change-workdir-btn")

                # Page indicator
                yield Label("Page: Preprocess", id="page-indicator")

                # Action buttons and progress (OUTSIDE scroll area)
                with Vertical(id="actions-bar"):
                    yield Button("Continue →", variant="primary", id="continue-btn")
                    yield Button("← Back", variant="primary", id="back-btn", classes="hidden")
                    yield Button(
                        "Run Analysis", variant="primary", id="run-analysis-btn", classes="hidden"
                    )
                    yield Button(
                        "Stop", variant="error", id="stop-btn", classes="hidden", disabled=True
                    )
                yield ProgressBar(id="progress-bar")
                yield Static("Status: Idle", id="run-status")

            # Middle panel (parameters + file browser)
            with Horizontal(id="middle-panel"):
                with Vertical(id="params-panel"):
                    yield Label("Parameters", id="params-header")
                    with VerticalScroll(id="controls-scroll"):
                        # Preprocess controls - single param group
                        with Vertical(id="preprocess-controls"):
                            with ParamGroup("Input & Preprocess"):
                                # Input section
                                yield Label("Input Mode:")
                                yield Select(
                                    [("ASA File", "asa"), ("Neuron + Traj", "neuron_traj")],
                                    value="asa",
                                    id="input-mode-select",
                                )

                                yield Label("Preset:")
                                yield Select(
                                    [("Grid", "grid"), ("HD", "hd"), ("None", "none")],
                                    value="grid",
                                    id="preset-select",
                                )

                                # Preprocess section
                                yield Label("Method:")
                                yield Select(
                                    [
                                        ("None", "none"),
                                        ("Embed Spike Trains", "embed_spike_trains"),
                                    ],
                                    value="none",
                                    id="preprocess-method-select",
                                )

                                # Preprocessing parameters (enabled when method is embed_spike_trains)
                                with Vertical(id="emb-params"):
                                    yield Label("res:", id="emb-res-label")
                                    yield Input(value="100000", id="emb-res", disabled=True)

                                    yield Label("dt:", id="emb-dt-label")
                                    yield Input(value="1000", id="emb-dt", disabled=True)

                                    yield Label("sigma:", id="emb-sigma-label")
                                    yield Input(value="5000", id="emb-sigma", disabled=True)

                                    yield Checkbox(
                                        "smooth", id="emb-smooth", value=True, disabled=True
                                    )
                                    yield Checkbox(
                                        "speed_filter",
                                        id="emb-speed-filter",
                                        value=False,
                                        disabled=True,
                                    )

                                    yield Label("min_speed:", id="emb-min-speed-label")
                                    yield Input(value="2.5", id="emb-min-speed", disabled=True)

                        # Analysis controls (initially hidden)
                        with Vertical(id="analysis-controls", classes="hidden"):
                            preset_params = get_preset_params(self.state.preset)
                            tda_defaults = preset_params.get("tda", {})
                            grid_defaults = preset_params.get("gridscore", {})

                            with ParamGroup("Analysis Mode"):
                                yield Label("Mode:")
                                yield Select(
                                    [
                                        ("TDA", "tda"),
                                        ("CohoMap", "cohomap"),
                                        ("PathCompare", "pathcompare"),
                                        ("CohoSpace", "cohospace"),
                                        ("FR", "fr"),
                                        ("FRM", "frm"),
                                        ("GridScore", "gridscore"),
                                    ],
                                    value=self.state.analysis_mode,
                                    id="analysis-mode-select",
                                )

                            with ParamGroup("TDA Parameters", id="analysis-params-tda"):
                                yield Label("dim:")
                                yield Input(value=str(tda_defaults.get("dim", 6)), id="tda-dim")
                                yield Label("num_times:")
                                yield Input(
                                    value=str(tda_defaults.get("num_times", 5)), id="tda-num-times"
                                )
                                yield Label("active_times:")
                                yield Input(
                                    value=str(tda_defaults.get("active_times", 15000)),
                                    id="tda-active-times",
                                )
                                yield Label("k:")
                                yield Input(value=str(tda_defaults.get("k", 1000)), id="tda-k")
                                yield Label("n_points:")
                                yield Input(
                                    value=str(tda_defaults.get("n_points", 1200)), id="tda-n-points"
                                )
                                yield Label("metric:")
                                yield Select(
                                    [
                                        ("cosine", "cosine"),
                                        ("euclidean", "euclidean"),
                                        ("correlation", "correlation"),
                                    ],
                                    value=str(tda_defaults.get("metric", "cosine")),
                                    id="tda-metric",
                                )
                                yield Label("nbs:")
                                yield Input(value=str(tda_defaults.get("nbs", 800)), id="tda-nbs")
                                yield Label("maxdim:")
                                yield Input(
                                    value=str(tda_defaults.get("maxdim", 1)), id="tda-maxdim"
                                )
                                yield Label("coeff:")
                                yield Input(
                                    value=str(tda_defaults.get("coeff", 47)), id="tda-coeff"
                                )
                                yield Checkbox(
                                    "do_shuffle",
                                    id="tda-do-shuffle",
                                    value=tda_defaults.get("do_shuffle", False),
                                )
                                yield Label("num_shuffles:")
                                yield Input(
                                    value=str(tda_defaults.get("num_shuffles", 1000)),
                                    id="tda-num-shuffles",
                                )
                                yield Checkbox(
                                    "standardize (StandardScaler)", id="tda-standardize", value=True
                                )

                            with ParamGroup(
                                "Decode / CohoMap", id="analysis-params-decode", classes="hidden"
                            ):
                                yield Label("decode_version:")
                                yield Select(
                                    [
                                        ("v2 (multi)", "v2"),
                                        ("v0 (legacy)", "v0"),
                                    ],
                                    value="v2",
                                    id="decode-version",
                                )
                                yield Label("num_circ:")
                                yield Input(value="2", id="decode-num-circ")
                                yield Label("cohomap_subsample:")
                                yield Input(value="10", id="cohomap-subsample")
                                yield Checkbox(
                                    "real_ground (v0)", id="decode-real-ground", value=True
                                )
                                yield Checkbox("real_of (v0)", id="decode-real-of", value=True)

                            with ParamGroup(
                                "PathCompare Parameters",
                                id="analysis-params-pathcompare",
                                classes="hidden",
                            ):
                                yield Checkbox(
                                    "use_box (coordsbox/times_box)", id="pc-use-box", value=True
                                )
                                yield Checkbox(
                                    "interp_to_full (use_box)", id="pc-interp-full", value=True
                                )
                                yield Label("dim_mode:")
                                yield Select(
                                    [("2d", "2d"), ("1d", "1d")],
                                    value="2d",
                                    id="pc-dim-mode",
                                )
                                yield Label("dim (1d):")
                                yield Input(value="1", id="pc-dim")
                                yield Label("dim1 (2d):")
                                yield Input(value="1", id="pc-dim1")
                                yield Label("dim2 (2d):")
                                yield Input(value="2", id="pc-dim2")
                                yield Label("coords_key (optional):")
                                yield Input(value="", id="pc-coords-key")
                                yield Label("times_box_key (optional):")
                                yield Input(value="", id="pc-times-key")
                                yield Label("slice_mode:")
                                yield Select(
                                    [("time", "time"), ("index", "index")],
                                    value="time",
                                    id="pc-slice-mode",
                                )
                                yield Label("tmin (sec, -1=auto):")
                                yield Input(value="-1", id="pc-tmin")
                                yield Label("tmax (sec, -1=auto):")
                                yield Input(value="-1", id="pc-tmax")
                                yield Label("imin (-1=auto):")
                                yield Input(value="-1", id="pc-imin")
                                yield Label("imax (-1=auto):")
                                yield Input(value="-1", id="pc-imax")
                                yield Label("stride:")
                                yield Input(value="1", id="pc-stride")
                                yield Label("theta_scale (rad/deg/unit/auto):")
                                yield Input(value="rad", id="pathcompare-angle-scale")

                            with ParamGroup(
                                "CohoSpace Parameters",
                                id="analysis-params-cohospace",
                                classes="hidden",
                            ):
                                yield Label("dim_mode:")
                                yield Select(
                                    [("2d", "2d"), ("1d", "1d")],
                                    value="2d",
                                    id="coho-dim-mode",
                                )
                                yield Label("dim (1d):")
                                yield Input(value="1", id="coho-dim")
                                yield Label("dim1 (2d):")
                                yield Input(value="1", id="coho-dim1")
                                yield Label("dim2 (2d):")
                                yield Input(value="2", id="coho-dim2")
                                yield Label("mode (fr/spike):")
                                yield Select(
                                    [("fr", "fr"), ("spike", "spike")],
                                    value="fr",
                                    id="coho-mode",
                                )
                                yield Label("top_percent (fr):")
                                yield Input(value="5.0", id="coho-top-percent")
                                yield Label("view:")
                                yield Select(
                                    [
                                        ("both", "both"),
                                        ("single", "single"),
                                        ("population", "population"),
                                    ],
                                    value="both",
                                    id="coho-view",
                                )
                                yield Label("neuron_id (optional):")
                                yield Input(value="", id="cohospace-neuron-id")
                                yield Label("subsample (trajectory):")
                                yield Input(value="2", id="coho-subsample")
                                yield Label("unfold:")
                                yield Select(
                                    [("square", "square"), ("skew", "skew")],
                                    value="square",
                                    id="coho-unfold",
                                )
                                yield Checkbox(
                                    "skew_show_grid", id="coho-skew-show-grid", value=True
                                )
                                yield Label("skew_tiles:")
                                yield Input(value="0", id="coho-skew-tiles")

                            with ParamGroup(
                                "FR Parameters", id="analysis-params-fr", classes="hidden"
                            ):
                                yield Label("neuron_start:")
                                yield Input(value="", id="fr-neuron-start")
                                yield Label("neuron_end:")
                                yield Input(value="", id="fr-neuron-end")
                                yield Label("time_start:")
                                yield Input(value="", id="fr-time-start")
                                yield Label("time_end:")
                                yield Input(value="", id="fr-time-end")
                                yield Label("mode (fr/spike):")
                                yield Select(
                                    [("fr", "fr"), ("spike", "spike")],
                                    value="fr",
                                    id="fr-mode",
                                )
                                yield Label("normalize:")
                                yield Select(
                                    [
                                        ("zscore_per_neuron", "zscore_per_neuron"),
                                        ("minmax_per_neuron", "minmax_per_neuron"),
                                        ("none", "none"),
                                    ],
                                    value="none",
                                    id="fr-normalize",
                                )

                            with ParamGroup(
                                "FRM Parameters", id="analysis-params-frm", classes="hidden"
                            ):
                                yield Label("neuron_id:")
                                yield Input(value="0", id="frm-neuron-id")
                                yield Label("bins:")
                                yield Input(value="50", id="frm-bins")
                                yield Label("min_occupancy:")
                                yield Input(value="1", id="frm-min-occupancy")
                                yield Checkbox("smoothing", id="frm-smoothing", value=False)
                                yield Label("smooth_sigma:")
                                yield Input(value="2.0", id="frm-smooth-sigma")
                                yield Label("mode (fr/spike):")
                                yield Select(
                                    [("fr", "fr"), ("spike", "spike")],
                                    value="fr",
                                    id="frm-mode",
                                )

                            with ParamGroup(
                                "GridScore Parameters",
                                id="analysis-params-gridscore",
                                classes="hidden",
                            ):
                                yield Label("annulus_inner:")
                                yield Input(
                                    value=str(grid_defaults.get("annulus_inner", 0.3)),
                                    id="gridscore-annulus-inner",
                                )
                                yield Label("annulus_outer:")
                                yield Input(
                                    value=str(grid_defaults.get("annulus_outer", 0.7)),
                                    id="gridscore-annulus-outer",
                                )
                                yield Label("bin_size:")
                                yield Input(
                                    value=str(grid_defaults.get("bin_size", 2.5)),
                                    id="gridscore-bin-size",
                                )
                                yield Label("smooth_sigma:")
                                yield Input(
                                    value=str(grid_defaults.get("smooth_sigma", 2.0)),
                                    id="gridscore-smooth-sigma",
                                )

                with Vertical(id="file-tree-panel"):
                    yield Label("Files in Workdir", id="files-header")
                    yield DirectoryTree(self.state.workdir, id="file-tree")

            # Right panel (results + log at bottom)
            with Vertical(id="right-panel"):
                with TabbedContent(id="results-tabs"):
                    with TabPane("Setup", id="setup-tab"):
                        yield Static(
                            "1. Select working directory (Ctrl-W)\n"
                            "2. Choose input mode and files\n"
                            "3. Configure preprocessing\n"
                            "4. Click 'Continue' to proceed to analysis",
                            id="setup-content",
                        )

                    with TabPane("Results", id="results-tab"):
                        yield ImagePreview(id="result-preview")
                        yield Static(
                            "No results yet. Complete preprocessing and run analysis.",
                            id="result-status",
                        )

                # Log viewer at bottom (25% height)
                yield LogViewer(id="log-viewer")

        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount event."""
        self.update_workdir_label()
        self.check_terminal_size()
        self.apply_preset_params()
        self.update_analysis_params_visibility()
        self.update_decode_controls()
        self.update_pathcompare_controls()
        self.update_cohospace_controls()

    def check_terminal_size(self) -> None:
        """Check terminal size and show warning if too small."""
        size = self.size
        width = size.width
        height = size.height

        # Adjust layout based on terminal size
        left_panel = self.query_one("#left-panel")

        if width < self.RECOMMENDED_WIDTH:
            if width < self.MIN_WIDTH:
                # Very small terminal
                left_panel.styles.width = 20
            else:
                # Small terminal
                left_panel.styles.width = 22
        else:
            # Normal/large terminal
            left_panel.styles.width = 22

        # Show warning if terminal is too small (only once)
        if not self._size_warning_shown and (width < self.MIN_WIDTH or height < self.MIN_HEIGHT):
            self._size_warning_shown = True
            self.push_screen(TerminalSizeWarning(width, height))

    def on_resize(self, event) -> None:
        """Handle terminal resize events."""
        self.check_terminal_size()

    def action_change_workdir(self) -> None:
        """Change working directory."""
        self.push_screen(WorkdirScreen(), self.on_workdir_selected)

    def on_workdir_selected(self, path: Path | None) -> None:
        """Handle workdir selection."""
        if path:
            self.state.workdir = path
            self.runner.reset_input()
            self.update_workdir_label()

            # Update file tree in middle panel
            tree = self.query_one("#file-tree", DirectoryTree)
            tree.path = path
            tree.reload()

    def update_workdir_label(self) -> None:
        """Update the workdir label."""
        label = self.query_one("#workdir-label", Label)
        label.update(f"Workdir: {self.state.workdir}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "change-workdir-btn":
            self.action_change_workdir()
        elif event.button.id == "continue-btn":
            self.action_continue_to_analysis()
        elif event.button.id == "back-btn":
            self.action_back_to_preprocess()
        elif event.button.id == "run-analysis-btn":
            self.action_run_analysis()
        elif event.button.id == "stop-btn":
            self.action_stop()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "input-mode-select":
            self.state.input_mode = str(event.value)
            self.runner.reset_input()
        elif event.select.id == "preset-select":
            self.state.preset = str(event.value)
            self.apply_preset_params()
        elif event.select.id == "preprocess-method-select":
            self.state.preprocess_method = str(event.value)
            # Enable/disable preprocessing parameters
            is_embed = event.value == "embed_spike_trains"
            self.query_one("#emb-dt", Input).disabled = not is_embed
            self.query_one("#emb-sigma", Input).disabled = not is_embed
            self.query_one("#emb-smooth", Checkbox).disabled = not is_embed
            self.query_one("#emb-speed-filter", Checkbox).disabled = not is_embed
            self.query_one("#emb-min-speed", Input).disabled = not is_embed
            self.query_one("#emb-res", Input).disabled = not is_embed
        elif event.select.id == "analysis-mode-select":
            self.state.analysis_mode = str(event.value)
            self.update_analysis_params_visibility()
        elif event.select.id == "decode-version":
            self.update_decode_controls()
        elif event.select.id == "pc-dim-mode" or event.select.id == "pc-slice-mode":
            self.update_pathcompare_controls()
        elif event.select.id == "coho-dim-mode" or event.select.id == "coho-unfold":
            self.update_cohospace_controls()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        if event.checkbox.id == "tda-do-shuffle":
            self.query_one("#tda-num-shuffles", Input).disabled = not event.value
        elif event.checkbox.id == "pc-use-box":
            self.update_pathcompare_controls()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection from tree."""
        selected_path = event.path

        if self.state.input_mode == "asa" and selected_path.suffix == ".npz":
            self.state.asa_file = relative_path(self.state, selected_path)
            self.runner.reset_input()
            self.log_message(f"Selected ASA file: {self.state.asa_file}")
            return

        if selected_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            preview = self.query_one("#result-preview", ImagePreview)
            preview.update_image(selected_path)
            self.log_message(f"Previewing image: {selected_path}")

    def action_continue_to_analysis(self) -> None:
        """Continue from preprocessing to analysis page."""
        if self.current_worker and not self.current_worker.is_finished:
            self.log_message("Preprocessing already running. Please wait.")
            return

        # Validate files
        is_valid, error = validate_files(self.state)
        if not is_valid:
            self.push_screen(ErrorScreen("Validation Error", error))
            return

        # Collect preprocessing parameters
        if self.state.preprocess_method == "embed_spike_trains":
            try:
                res_val = int(self.query_one("#emb-res", Input).value)
                dt_val = int(self.query_one("#emb-dt", Input).value)
                sigma_val = int(self.query_one("#emb-sigma", Input).value)
                smooth_val = self.query_one("#emb-smooth", Checkbox).value
                speed_filter_val = self.query_one("#emb-speed-filter", Checkbox).value
                min_speed_val = float(self.query_one("#emb-min-speed", Input).value)

                self.state.preprocess_params = {
                    "res": res_val,
                    "dt": dt_val,
                    "sigma": sigma_val,
                    "smooth": smooth_val,
                    "speed_filter": speed_filter_val,
                    "min_speed": min_speed_val,
                }
            except ValueError as e:
                self.push_screen(ErrorScreen("Parameter Error", f"Invalid parameter value: {e}"))
                return

        self.log_message("Loading and preprocessing data...")
        self.set_run_status("Status: Preprocessing...", "running")
        self.query_one("#continue-btn", Button).disabled = True

        # Run preprocessing in worker
        self.current_worker = self.run_worker(
            self.runner.run_preprocessing(
                self.state,
                log_callback=self.log_message,
                progress_callback=self.update_progress,
            ),
            name="preprocessing_worker",
            thread=True,
        )

    def action_back_to_preprocess(self) -> None:
        """Go back to preprocessing page."""
        self.current_page = "preprocess"
        self.query_one("#page-indicator", Label).update("Page: Preprocess")
        self.query_one("#preprocess-controls").remove_class("hidden")
        self.query_one("#analysis-controls").add_class("hidden")

        # Show/hide appropriate buttons
        self.query_one("#continue-btn").remove_class("hidden")
        self.query_one("#back-btn").add_class("hidden")
        self.query_one("#run-analysis-btn").add_class("hidden")
        self.query_one("#stop-btn").add_class("hidden")
        self.query_one("#stop-btn", Button).disabled = True

        self.log_message("Returned to preprocessing page")

    def action_run_analysis(self) -> None:
        """Run analysis on preprocessed data."""
        if self.current_worker and not self.current_worker.is_finished:
            self.log_message("Another task is already running. Please wait.")
            return

        if not self.runner.has_preprocessed_data():
            self.push_screen(
                ErrorScreen("Error", "No preprocessed data. Please complete preprocessing first.")
            )
            return

        try:
            self.collect_analysis_params()
        except ValueError as e:
            self.push_screen(ErrorScreen("Parameter Error", f"Invalid analysis parameter: {e}"))
            return

        self.log_message(f"Starting {self.state.analysis_mode} analysis...")
        self.set_run_status(f"Status: Running {self.state.analysis_mode}...", "running")
        self.query_one("#run-analysis-btn", Button).disabled = True
        self.query_one("#stop-btn", Button).disabled = False

        # Run analysis in worker
        self.current_worker = self.run_worker(
            self.runner.run_analysis(
                self.state,
                log_callback=self.log_message,
                progress_callback=self.update_progress,
            ),
            name="analysis_worker",
            thread=True,
        )

    def action_run_action(self) -> None:
        """Run current page action (Continue or Run Analysis)."""
        if self.current_page == "preprocess":
            self.action_continue_to_analysis()
        else:
            self.action_run_analysis()

    def action_stop(self) -> None:
        """Request cancellation of the running worker."""
        if not self.current_worker or self.current_worker.is_finished:
            self.log_message("No running task to stop.")
            return
        self.current_worker.cancel()
        self.set_run_status("Status: Cancel requested.", "error")
        self.log_message("Stop requested. Waiting for worker to exit...")
        self.query_one("#stop-btn", Button).disabled = True

    def log_message(self, message: str) -> None:
        """Add log message."""
        log_viewer = self.query_one("#log-viewer", LogViewer)
        log_viewer.add_log(message)
        self.append_log_file(message)

    def _log_file_path(self) -> Path:
        try:
            return self.runner.results_dir(self.state) / "asa_tui.log"
        except Exception:
            return self.state.workdir / "Results" / "asa_tui.log"

    def append_log_file(self, message: str) -> None:
        """Append log message to file for easy copying."""
        try:
            path = self._log_file_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("a", encoding="utf-8").write(f"{message}\n")
        except Exception:
            pass

    def update_progress(self, percent: int) -> None:
        """Update progress bar."""
        progress = self.query_one("#progress-bar", ProgressBar)
        progress.update(total=100, progress=percent)

    def apply_preset_params(self) -> None:
        """Apply preset defaults to analysis inputs."""
        preset_params = get_preset_params(self.state.preset)
        tda = preset_params.get("tda", {})
        grid = preset_params.get("gridscore", {})

        if tda:
            self.query_one("#tda-dim", Input).value = str(tda.get("dim", 6))
            self.query_one("#tda-num-times", Input).value = str(tda.get("num_times", 5))
            self.query_one("#tda-active-times", Input).value = str(tda.get("active_times", 15000))
            self.query_one("#tda-k", Input).value = str(tda.get("k", 1000))
            self.query_one("#tda-n-points", Input).value = str(tda.get("n_points", 1200))
            self.query_one("#tda-metric", Select).value = str(tda.get("metric", "cosine"))
            self.query_one("#tda-nbs", Input).value = str(tda.get("nbs", 800))
            self.query_one("#tda-maxdim", Input).value = str(tda.get("maxdim", 1))
            self.query_one("#tda-coeff", Input).value = str(tda.get("coeff", 47))
            self.query_one("#tda-do-shuffle", Checkbox).value = bool(tda.get("do_shuffle", False))
            self.query_one("#tda-num-shuffles", Input).value = str(tda.get("num_shuffles", 1000))
            self.query_one("#tda-num-shuffles", Input).disabled = not self.query_one(
                "#tda-do-shuffle", Checkbox
            ).value

        if grid:
            self.query_one("#gridscore-annulus-inner", Input).value = str(
                grid.get("annulus_inner", 0.3)
            )
            self.query_one("#gridscore-annulus-outer", Input).value = str(
                grid.get("annulus_outer", 0.7)
            )
            self.query_one("#gridscore-bin-size", Input).value = str(grid.get("bin_size", 2.5))
            self.query_one("#gridscore-smooth-sigma", Input).value = str(
                grid.get("smooth_sigma", 2.0)
            )

    def update_analysis_params_visibility(self) -> None:
        """Show params for the selected analysis mode."""
        mode = self.state.analysis_mode

        groups = {
            "analysis-params-tda": mode == "tda",
            "analysis-params-decode": mode in {"cohomap", "pathcompare", "cohospace"},
            "analysis-params-pathcompare": mode == "pathcompare",
            "analysis-params-cohospace": mode == "cohospace",
            "analysis-params-fr": mode == "fr",
            "analysis-params-frm": mode == "frm",
            "analysis-params-gridscore": mode == "gridscore",
        }

        for group_id, should_show in groups.items():
            group = self.query_one(f"#{group_id}")
            if should_show:
                group.remove_class("hidden")
            else:
                group.add_class("hidden")

    def update_decode_controls(self) -> None:
        """Enable/disable decode controls based on decode version."""
        version = str(self.query_one("#decode-version", Select).value)
        is_v0 = version == "v0"
        self.query_one("#decode-real-ground", Checkbox).disabled = not is_v0
        self.query_one("#decode-real-of", Checkbox).disabled = not is_v0

    def update_pathcompare_controls(self) -> None:
        """Enable/disable PathCompare controls based on mode."""
        dim_mode = str(self.query_one("#pc-dim-mode", Select).value)
        is_1d = dim_mode == "1d"
        self.query_one("#pc-dim", Input).disabled = not is_1d
        self.query_one("#pc-dim1", Input).disabled = is_1d
        self.query_one("#pc-dim2", Input).disabled = is_1d

        slice_mode = str(self.query_one("#pc-slice-mode", Select).value)
        is_time = slice_mode == "time"
        self.query_one("#pc-tmin", Input).disabled = not is_time
        self.query_one("#pc-tmax", Input).disabled = not is_time
        self.query_one("#pc-imin", Input).disabled = is_time
        self.query_one("#pc-imax", Input).disabled = is_time

        use_box = self.query_one("#pc-use-box", Checkbox).value
        self.query_one("#pc-interp-full", Checkbox).disabled = not use_box

    def update_cohospace_controls(self) -> None:
        """Enable/disable CohoSpace controls based on mode."""
        dim_mode = str(self.query_one("#coho-dim-mode", Select).value)
        is_1d = dim_mode == "1d"
        self.query_one("#coho-dim", Input).disabled = not is_1d
        self.query_one("#coho-dim1", Input).disabled = is_1d
        self.query_one("#coho-dim2", Input).disabled = is_1d

        unfold = str(self.query_one("#coho-unfold", Select).value)
        is_skew = unfold == "skew"
        self.query_one("#coho-skew-show-grid", Checkbox).disabled = not is_skew
        self.query_one("#coho-skew-tiles", Input).disabled = not is_skew

    def _parse_optional_number(self, raw: str, cast) -> int | float | None:
        text = raw.strip()
        if text == "" or text == "-1":
            return None
        return cast(text)

    def _parse_optional_int(self, raw: str) -> int | None:
        return self._parse_optional_number(raw, int)

    def _parse_optional_float(self, raw: str) -> float | None:
        return self._parse_optional_number(raw, float)

    def collect_analysis_params(self) -> None:
        """Collect analysis parameters from UI into state."""
        params: dict[str, object] = {}
        mode = self.state.analysis_mode

        if mode == "tda":
            params["dim"] = int(self.query_one("#tda-dim", Input).value)
            params["num_times"] = int(self.query_one("#tda-num-times", Input).value)
            params["active_times"] = int(self.query_one("#tda-active-times", Input).value)
            params["k"] = int(self.query_one("#tda-k", Input).value)
            params["n_points"] = int(self.query_one("#tda-n-points", Input).value)
            params["metric"] = str(self.query_one("#tda-metric", Select).value)
            params["nbs"] = int(self.query_one("#tda-nbs", Input).value)
            params["maxdim"] = int(self.query_one("#tda-maxdim", Input).value)
            params["coeff"] = int(self.query_one("#tda-coeff", Input).value)
            params["do_shuffle"] = self.query_one("#tda-do-shuffle", Checkbox).value
            params["num_shuffles"] = int(self.query_one("#tda-num-shuffles", Input).value)
            params["standardize"] = self.query_one("#tda-standardize", Checkbox).value
        elif mode == "cohomap":
            params["decode_version"] = str(self.query_one("#decode-version", Select).value)
            params["num_circ"] = int(self.query_one("#decode-num-circ", Input).value)
            params["cohomap_subsample"] = int(self.query_one("#cohomap-subsample", Input).value)
            params["real_ground"] = self.query_one("#decode-real-ground", Checkbox).value
            params["real_of"] = self.query_one("#decode-real-of", Checkbox).value
        elif mode == "pathcompare":
            params["decode_version"] = str(self.query_one("#decode-version", Select).value)
            params["num_circ"] = int(self.query_one("#decode-num-circ", Input).value)
            params["real_ground"] = self.query_one("#decode-real-ground", Checkbox).value
            params["real_of"] = self.query_one("#decode-real-of", Checkbox).value
            params["use_box"] = self.query_one("#pc-use-box", Checkbox).value
            params["interp_full"] = self.query_one("#pc-interp-full", Checkbox).value
            params["dim_mode"] = str(self.query_one("#pc-dim-mode", Select).value)
            params["dim"] = int(self.query_one("#pc-dim", Input).value)
            params["dim1"] = int(self.query_one("#pc-dim1", Input).value)
            params["dim2"] = int(self.query_one("#pc-dim2", Input).value)
            params["coords_key"] = self.query_one("#pc-coords-key", Input).value.strip() or None
            params["times_key"] = self.query_one("#pc-times-key", Input).value.strip() or None
            params["slice_mode"] = str(self.query_one("#pc-slice-mode", Select).value)
            params["tmin"] = self._parse_optional_float(self.query_one("#pc-tmin", Input).value)
            params["tmax"] = self._parse_optional_float(self.query_one("#pc-tmax", Input).value)
            params["imin"] = self._parse_optional_int(self.query_one("#pc-imin", Input).value)
            params["imax"] = self._parse_optional_int(self.query_one("#pc-imax", Input).value)
            params["stride"] = int(self.query_one("#pc-stride", Input).value)
            params["angle_scale"] = (
                self.query_one("#pathcompare-angle-scale", Input).value.strip() or "rad"
            )
        elif mode == "cohospace":
            params["decode_version"] = str(self.query_one("#decode-version", Select).value)
            params["num_circ"] = int(self.query_one("#decode-num-circ", Input).value)
            params["real_ground"] = self.query_one("#decode-real-ground", Checkbox).value
            params["real_of"] = self.query_one("#decode-real-of", Checkbox).value
            params["dim_mode"] = str(self.query_one("#coho-dim-mode", Select).value)
            params["dim"] = int(self.query_one("#coho-dim", Input).value)
            params["dim1"] = int(self.query_one("#coho-dim1", Input).value)
            params["dim2"] = int(self.query_one("#coho-dim2", Input).value)
            params["mode"] = str(self.query_one("#coho-mode", Select).value)
            params["top_percent"] = float(self.query_one("#coho-top-percent", Input).value)
            params["view"] = str(self.query_one("#coho-view", Select).value)
            neuron_id_raw = self.query_one("#cohospace-neuron-id", Input).value.strip()
            if neuron_id_raw:
                params["neuron_id"] = int(neuron_id_raw)
            params["subsample"] = int(self.query_one("#coho-subsample", Input).value)
            params["unfold"] = str(self.query_one("#coho-unfold", Select).value)
            params["skew_show_grid"] = self.query_one("#coho-skew-show-grid", Checkbox).value
            params["skew_tiles"] = int(self.query_one("#coho-skew-tiles", Input).value)
        elif mode == "fr":
            n_start = self._parse_optional_int(self.query_one("#fr-neuron-start", Input).value)
            n_end = self._parse_optional_int(self.query_one("#fr-neuron-end", Input).value)
            t_start = self._parse_optional_int(self.query_one("#fr-time-start", Input).value)
            t_end = self._parse_optional_int(self.query_one("#fr-time-end", Input).value)
            params["neuron_range"] = None if n_start is None and n_end is None else (n_start, n_end)
            params["time_range"] = None if t_start is None and t_end is None else (t_start, t_end)
            params["mode"] = str(self.query_one("#fr-mode", Select).value)
            normalize = str(self.query_one("#fr-normalize", Select).value)
            params["normalize"] = None if normalize == "none" else normalize
        elif mode == "frm":
            params["neuron_id"] = int(self.query_one("#frm-neuron-id", Input).value)
            params["bin_size"] = int(self.query_one("#frm-bins", Input).value)
            params["min_occupancy"] = int(self.query_one("#frm-min-occupancy", Input).value)
            params["smoothing"] = self.query_one("#frm-smoothing", Checkbox).value
            params["smooth_sigma"] = float(self.query_one("#frm-smooth-sigma", Input).value)
            params["mode"] = str(self.query_one("#frm-mode", Select).value)
        elif mode == "gridscore":
            params["annulus_inner"] = float(self.query_one("#gridscore-annulus-inner", Input).value)
            params["annulus_outer"] = float(self.query_one("#gridscore-annulus-outer", Input).value)
            params["bin_size"] = float(self.query_one("#gridscore-bin-size", Input).value)
            params["smooth_sigma"] = float(self.query_one("#gridscore-smooth-sigma", Input).value)

        self.state.analysis_params = params

    def set_run_status(self, message: str, status_class: str | None = None) -> None:
        """Update run status label and styling."""
        status = self.query_one("#run-status", Static)
        status.update(message)
        status.remove_class("running", "success", "error")
        if status_class:
            status.add_class(status_class)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes."""
        if event.worker.name == "preprocessing_worker" and event.worker.is_finished:
            result = event.worker.result
            self.query_one("#continue-btn", Button).disabled = False
            self.query_one("#stop-btn", Button).disabled = True

            if result.success:
                self.log_message(result.summary)
                self.set_run_status("Status: Preprocessing complete.", "success")
                # Switch to analysis page
                self.current_page = "analysis"
                self.query_one("#page-indicator", Label).update("Page: Analysis")
                self.query_one("#preprocess-controls").add_class("hidden")
                self.query_one("#analysis-controls").remove_class("hidden")

                # Show/hide appropriate buttons
                self.query_one("#continue-btn").add_class("hidden")
                self.query_one("#back-btn").remove_class("hidden")
                self.query_one("#run-analysis-btn").remove_class("hidden")
                self.query_one("#stop-btn").remove_class("hidden")
                self.query_one("#run-analysis-btn", Button).disabled = False

                self.log_message("Preprocessing complete. Ready for analysis.")
                tree = self.query_one("#file-tree", DirectoryTree)
                tree.reload()
            else:
                self.set_run_status("Status: Preprocessing failed.", "error")
                self.push_screen(
                    ErrorScreen("Preprocessing Error", result.error or "Unknown error")
                )

        elif event.worker.name == "analysis_worker" and event.worker.is_finished:
            result = event.worker.result
            self.query_one("#run-analysis-btn", Button).disabled = False
            self.query_one("#stop-btn", Button).disabled = True

            if result.success:
                self.log_message(result.summary)
                self.set_run_status("Status: Analysis complete.", "success")
                self.log_message(f"Artifacts: {list(result.artifacts.keys())}")

                # Update results tab + preview
                tabs = self.query_one("#results-tabs", TabbedContent)
                tabs.active = "results-tab"
                preview_path = self._select_result_image(result.artifacts)
                if preview_path is not None:
                    preview = self.query_one("#result-preview", ImagePreview)
                    preview.update_image(preview_path)

                status = self.query_one("#result-status", Static)
                status.update(f"Analysis completed: {result.summary}")
                tree = self.query_one("#file-tree", DirectoryTree)
                tree.reload()
            else:
                self.set_run_status("Status: Analysis failed.", "error")
                self.push_screen(ErrorScreen("Analysis Error", result.error or "Unknown error"))

    def action_refresh(self) -> None:
        """Refresh the UI."""
        self.update_workdir_label()
        tree = self.query_one("#file-tree", DirectoryTree)
        tree.reload()

    def _select_result_image(self, artifacts: dict[str, Path]) -> Path | None:
        """Pick a result image to preview, preferring known keys."""
        priority_keys = [
            "barcode",
            "cohomap",
            "path_compare",
            "trajectory",
            "cohospace_trajectory",
            "fr_heatmap",
            "frm",
            "distribution",
        ]
        for key in priority_keys:
            path = artifacts.get(key)
            if path and path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                return path
        for path in artifacts.values():
            if path and path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                return path
        return None

    def action_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
