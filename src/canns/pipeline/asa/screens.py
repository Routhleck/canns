"""Modal screens for ASA TUI.

This module provides modal overlays for directory selection, help, and error display.
"""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Static, Label


class WorkdirScreen(ModalScreen[Path]):
    """Modal screen for selecting working directory."""

    DEFAULT_CSS = """
    WorkdirScreen {
        align: center middle;
    }

    WorkdirScreen > Container {
        width: 80;
        height: 30;
        border: thick $accent;
        background: $surface;
    }

    #workdir-tree {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Select Working Directory")
            yield DirectoryTree(Path.home(), id="workdir-tree")
            with Container(id="button-container"):
                yield Button("Select", variant="primary", id="select-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "select-btn":
            tree = self.query_one("#workdir-tree", DirectoryTree)
            if tree.cursor_node:
                selected_path = tree.cursor_node.data.path
                self.dismiss(selected_path)
        elif event.button.id == "cancel-btn":
            self.dismiss(None)


class HelpScreen(ModalScreen):
    """Modal screen showing help and key bindings."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen > Container {
        width: 70;
        height: 25;
        border: thick $primary;
        background: $surface;
        padding: 2;
    }

    #help-content {
        overflow-y: scroll;
    }
    """

    HELP_TEXT = """
ASA TUI - Terminal User Interface for ASA Analysis

KEY BINDINGS:
  Ctrl-W    Change working directory
  Ctrl-R    Run analysis
  F5        Refresh previews
  ?         Show this help
  Esc       Quit application
  Tab       Navigate between panels

ANALYSIS MODULES:
  TDA           Topological Data Analysis
  CohoMap       Cohomology Map (requires TDA)
  PathCompare   Trajectory Comparison (requires CohoMap)
  CohoSpace     Cohomology Space Visualization (requires CohoMap)
  FR            Firing Rate Heatmap
  FRM           Single Neuron Firing Rate Map
  GridScore     Grid Cell Analysis

WORKFLOW:
  1. Select working directory (Ctrl-W)
  2. Choose input mode (ASA or Neuron+Traj)
  3. Load files
  4. Configure preprocessing
  5. Select analysis mode
  6. Set parameters
  7. Run analysis (Ctrl-R)
  8. View results

TERMINAL REQUIREMENTS:
  Minimum size: 100 cols × 30 rows
  Recommended: 120 cols × 40 rows
  Tip: Use smaller font size for better display

  If display is incomplete, try:
  - Reduce terminal font size
  - Maximize terminal window
  - Use fullscreen mode

Press any key to close...
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Help", id="help-title")
            yield Static(self.HELP_TEXT, id="help-content")
            yield Button("Close", variant="primary", id="close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.dismiss()

    def on_key(self, event) -> None:
        self.dismiss()


class ErrorScreen(ModalScreen):
    """Modal screen for displaying errors."""

    DEFAULT_CSS = """
    ErrorScreen {
        align: center middle;
    }

    ErrorScreen > Container {
        width: 60;
        height: 20;
        border: thick $error;
        background: $surface;
        padding: 2;
    }

    #error-message {
        color: $error;
        overflow-y: scroll;
    }
    """

    def __init__(self, title: str, message: str, **kwargs):
        super().__init__(**kwargs)
        self.error_title = title
        self.error_message = message

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self.error_title, id="error-title")
            yield Static(self.error_message, id="error-message")
            yield Button("Close", variant="error", id="close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.dismiss()


class TerminalSizeWarning(ModalScreen):
    """Warning screen for insufficient terminal size."""

    DEFAULT_CSS = """
    TerminalSizeWarning {
        align: center middle;
    }

    TerminalSizeWarning > Container {
        width: 50;
        height: 15;
        border: thick $warning;
        background: $surface;
        padding: 2;
    }

    #warning-message {
        color: $warning;
        text-align: center;
    }
    """

    def __init__(self, current_width: int, current_height: int, **kwargs):
        super().__init__(**kwargs)
        self.current_width = current_width
        self.current_height = current_height

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("⚠ Terminal Size Warning", id="warning-title")
            yield Static(
                f"Current terminal size: {self.current_width} cols × {self.current_height} rows\n\n"
                f"Recommended minimum:\n"
                f"• Width: 100 columns (recommended 120+)\n"
                f"• Height: 30 rows (recommended 40+)\n\n"
                f"Please resize terminal or reduce font size.\n"
                f"Press Esc to continue (may not display properly)",
                id="warning-message"
            )
            yield Button("Continue", variant="warning", id="continue-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "continue-btn":
            self.dismiss()

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss()
