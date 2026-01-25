"""Main ASA TUI application with two-page workflow.

This module provides the main Textual application for ASA analysis,
following the original GUI's two-page structure:
1. PreprocessPage - File selection and preprocessing
2. AnalysisPage - Analysis mode selection and execution
"""

from pathlib import Path

import numpy as np
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Header,
    Footer,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
    DirectoryTree,
    ProgressBar,
    Input,
    Checkbox,
)
from textual.worker import Worker

from .state import WorkflowState, validate_files, get_preset_params, relative_path
from .runner import PipelineRunner
from .screens import WorkdirScreen, HelpScreen, ErrorScreen, TerminalSizeWarning
from .widgets import ImagePreview, ParamGroup, LogViewer


class ASAApp(App):
    """Main TUI application for ASA analysis."""

    CSS_PATH = "styles.tcss"

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

                # Scrollable area for parameter groups only
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
                                [("None", "none"), ("Embed Spike Trains", "embed_spike_trains")],
                                value="none",
                                id="preprocess-method-select",
                            )

                            # Preprocessing parameters (enabled when method is embed_spike_trains)
                            with Vertical(id="emb-params"):
                                yield Label("dt:", id="emb-dt-label")
                                yield Input(value="1000", id="emb-dt", disabled=True)

                                yield Label("sigma:", id="emb-sigma-label")
                                yield Input(value="5000", id="emb-sigma", disabled=True)

                                yield Checkbox("smooth", id="emb-smooth", value=True, disabled=True)
                                yield Checkbox("speed_filter", id="emb-speed-filter", value=True, disabled=True)

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
                            yield Input(value=str(tda_defaults.get("num_times", 5)), id="tda-num-times")
                            yield Label("active_times:")
                            yield Input(value=str(tda_defaults.get("active_times", 15000)), id="tda-active-times")
                            yield Label("k:")
                            yield Input(value=str(tda_defaults.get("k", 1000)), id="tda-k")
                            yield Label("n_points:")
                            yield Input(value=str(tda_defaults.get("n_points", 1200)), id="tda-n-points")
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
                            yield Input(value=str(tda_defaults.get("maxdim", 1)), id="tda-maxdim")
                            yield Label("coeff:")
                            yield Input(value=str(tda_defaults.get("coeff", 47)), id="tda-coeff")
                            yield Checkbox("do_shuffle", id="tda-do-shuffle", value=tda_defaults.get("do_shuffle", False))
                            yield Label("num_shuffles:")
                            yield Input(value=str(tda_defaults.get("num_shuffles", 1000)), id="tda-num-shuffles")

                        with ParamGroup("CohoMap Parameters", id="analysis-params-cohomap", classes="hidden"):
                            yield Checkbox("real_ground", id="cohomap-real-ground", value=True)
                            yield Checkbox("real_of", id="cohomap-real-of", value=True)

                        with ParamGroup("PathCompare Parameters", id="analysis-params-pathcompare", classes="hidden"):
                            yield Label("theta_scale (rad/deg/unit/auto):")
                            yield Input(value="rad", id="pathcompare-angle-scale")

                        with ParamGroup("CohoSpace Parameters", id="analysis-params-cohospace", classes="hidden"):
                            yield Label("neuron_id (optional):")
                            yield Input(value="", id="cohospace-neuron-id")

                        with ParamGroup("FR Parameters", id="analysis-params-fr", classes="hidden"):
                            yield Label("neuron_range (start,end):")
                            yield Input(value="", id="fr-neuron-range")
                            yield Label("time_range (start,end):")
                            yield Input(value="", id="fr-time-range")
                            yield Label("normalize:")
                            yield Select(
                                [
                                    ("zscore_per_neuron", "zscore_per_neuron"),
                                    ("minmax_per_neuron", "minmax_per_neuron"),
                                    ("none", "none"),
                                ],
                                value="zscore_per_neuron",
                                id="fr-normalize",
                            )

                        with ParamGroup("FRM Parameters", id="analysis-params-frm", classes="hidden"):
                            yield Label("neuron_id:")
                            yield Input(value="0", id="frm-neuron-id")
                            yield Label("bins:")
                            yield Input(value="50", id="frm-bins")
                            yield Label("smooth_sigma:")
                            yield Input(value="2.0", id="frm-smooth-sigma")

                        with ParamGroup("GridScore Parameters", id="analysis-params-gridscore", classes="hidden"):
                            yield Label("annulus_inner:")
                            yield Input(value=str(grid_defaults.get("annulus_inner", 0.3)), id="gridscore-annulus-inner")
                            yield Label("annulus_outer:")
                            yield Input(value=str(grid_defaults.get("annulus_outer", 0.7)), id="gridscore-annulus-outer")
                            yield Label("bin_size:")
                            yield Input(value=str(grid_defaults.get("bin_size", 2.5)), id="gridscore-bin-size")
                            yield Label("smooth_sigma:")
                            yield Input(value=str(grid_defaults.get("smooth_sigma", 2.0)), id="gridscore-smooth-sigma")

                # Action buttons and progress (OUTSIDE scroll area)
                yield Button("Continue →", variant="primary", id="continue-btn")
                yield Button("← Back", id="back-btn", classes="hidden")
                yield Button("Run Analysis", variant="primary", id="run-analysis-btn", classes="hidden")
                yield ProgressBar(id="progress-bar")
                yield Static("Status: Idle", id="run-status")

            # Middle panel (file browser)
            with Vertical(id="middle-panel"):
                yield Label("Files in Workdir", id="files-header")
                yield DirectoryTree(self.state.workdir, id="file-tree")

            # Right panel (results + log at bottom)
            with Vertical(id="right-panel"):
                with TabbedContent(id="results-tabs"):
                    with TabPane("Setup"):
                        yield Static(
                            "1. Select working directory (Ctrl-W)\n"
                            "2. Choose input mode and files\n"
                            "3. Configure preprocessing\n"
                            "4. Click 'Continue' to proceed to analysis",
                            id="setup-content"
                        )

                    with TabPane("Results"):
                        yield ImagePreview(id="result-preview")
                        yield Static("No results yet. Complete preprocessing and run analysis.", id="result-status")

                # Log viewer at bottom (25% height)
                yield LogViewer(id="log-viewer")

        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount event."""
        self.update_workdir_label()
        self.check_terminal_size()
        self.apply_preset_params()
        self.update_analysis_params_visibility()

    def check_terminal_size(self) -> None:
        """Check terminal size and show warning if too small."""
        size = self.size
        width = size.width
        height = size.height

        # Adjust layout based on terminal size
        left_panel = self.query_one("#left-panel")
        middle_panel = self.query_one("#middle-panel")

        if width < self.RECOMMENDED_WIDTH:
            if width < self.MIN_WIDTH:
                # Very small terminal
                left_panel.styles.width = 30
                middle_panel.styles.width = 30
            else:
                # Small terminal
                left_panel.styles.width = 32
                middle_panel.styles.width = 32
        else:
            # Normal/large terminal
            left_panel.styles.width = 35
            middle_panel.styles.width = 35

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
            self.update_workdir_label()

            # Update file tree in middle panel
            tree = self.query_one("#file-tree", DirectoryTree)
            tree.path = path

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

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "input-mode-select":
            self.state.input_mode = str(event.value)
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
        elif event.select.id == "analysis-mode-select":
            self.state.analysis_mode = str(event.value)
            self.update_analysis_params_visibility()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        if event.checkbox.id == "tda-do-shuffle":
            self.query_one("#tda-num-shuffles", Input).disabled = not event.value

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection from tree."""
        selected_path = event.path

        if self.state.input_mode == "asa" and selected_path.suffix == ".npz":
            self.state.asa_file = relative_path(self.state, selected_path)
            self.log_message(f"Selected ASA file: {self.state.asa_file}")

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
                dt_val = int(self.query_one("#emb-dt", Input).value)
                sigma_val = int(self.query_one("#emb-sigma", Input).value)
                smooth_val = self.query_one("#emb-smooth", Checkbox).value
                speed_filter_val = self.query_one("#emb-speed-filter", Checkbox).value
                min_speed_val = float(self.query_one("#emb-min-speed", Input).value)

                self.state.preprocess_params = {
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

        self.log_message("Returned to preprocessing page")

    def action_run_analysis(self) -> None:
        """Run analysis on preprocessed data."""
        if self.current_worker and not self.current_worker.is_finished:
            self.log_message("Another task is already running. Please wait.")
            return

        if not self.runner.has_preprocessed_data():
            self.push_screen(ErrorScreen("Error", "No preprocessed data. Please complete preprocessing first."))
            return

        try:
            self.collect_analysis_params()
        except ValueError as e:
            self.push_screen(ErrorScreen("Parameter Error", f"Invalid analysis parameter: {e}"))
            return

        self.log_message(f"Starting {self.state.analysis_mode} analysis...")
        self.set_run_status(f"Status: Running {self.state.analysis_mode}...", "running")

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

    def log_message(self, message: str) -> None:
        """Add log message."""
        log_viewer = self.query_one("#log-viewer", LogViewer)
        log_viewer.add_log(message)

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
            self.query_one("#gridscore-annulus-inner", Input).value = str(grid.get("annulus_inner", 0.3))
            self.query_one("#gridscore-annulus-outer", Input).value = str(grid.get("annulus_outer", 0.7))
            self.query_one("#gridscore-bin-size", Input).value = str(grid.get("bin_size", 2.5))
            self.query_one("#gridscore-smooth-sigma", Input).value = str(grid.get("smooth_sigma", 2.0))

    def update_analysis_params_visibility(self) -> None:
        """Show params for the selected analysis mode."""
        modes = [
            "tda",
            "cohomap",
            "pathcompare",
            "cohospace",
            "fr",
            "frm",
            "gridscore",
        ]
        for mode in modes:
            group = self.query_one(f"#analysis-params-{mode}")
            if mode == self.state.analysis_mode:
                group.remove_class("hidden")
            else:
                group.add_class("hidden")

    def _parse_range(self, raw: str) -> tuple[int | None, int | None] | None:
        """Parse a 'start,end' or 'start:end' range string."""
        text = raw.strip()
        if not text:
            return None
        parts = [p.strip() for p in text.replace(":", ",").split(",")]
        if len(parts) != 2:
            raise ValueError("range must be 'start,end' or 'start:end'")

        def parse_part(value: str) -> int | None:
            return None if value == "" else int(value)

        return (parse_part(parts[0]), parse_part(parts[1]))

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
        elif mode == "cohomap":
            params["real_ground"] = self.query_one("#cohomap-real-ground", Checkbox).value
            params["real_of"] = self.query_one("#cohomap-real-of", Checkbox).value
        elif mode == "pathcompare":
            params["angle_scale"] = self.query_one("#pathcompare-angle-scale", Input).value.strip() or "rad"
        elif mode == "cohospace":
            neuron_id_raw = self.query_one("#cohospace-neuron-id", Input).value.strip()
            if neuron_id_raw:
                params["neuron_id"] = int(neuron_id_raw)
        elif mode == "fr":
            params["neuron_range"] = self._parse_range(self.query_one("#fr-neuron-range", Input).value)
            params["time_range"] = self._parse_range(self.query_one("#fr-time-range", Input).value)
            normalize = str(self.query_one("#fr-normalize", Select).value)
            params["normalize"] = None if normalize == "none" else normalize
        elif mode == "frm":
            params["neuron_id"] = int(self.query_one("#frm-neuron-id", Input).value)
            params["bin_size"] = int(self.query_one("#frm-bins", Input).value)
            params["smooth_sigma"] = float(self.query_one("#frm-smooth-sigma", Input).value)
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

                self.log_message("Preprocessing complete. Ready for analysis.")
            else:
                self.set_run_status("Status: Preprocessing failed.", "error")
                self.push_screen(ErrorScreen("Preprocessing Error", result.error or "Unknown error"))

        elif event.worker.name == "analysis_worker" and event.worker.is_finished:
            result = event.worker.result

            if result.success:
                self.log_message(result.summary)
                self.set_run_status("Status: Analysis complete.", "success")
                self.log_message(f"Artifacts: {list(result.artifacts.keys())}")

                # Update results tab
                if "barcode" in result.artifacts:
                    preview = self.query_one("#result-preview", ImagePreview)
                    preview.update_image(result.artifacts["barcode"])

                status = self.query_one("#result-status", Static)
                status.update(f"Analysis completed: {result.summary}")
            else:
                self.set_run_status("Status: Analysis failed.", "error")
                self.push_screen(ErrorScreen("Analysis Error", result.error or "Unknown error"))

    def action_refresh(self) -> None:
        """Refresh the UI."""
        self.update_workdir_label()

    def action_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
