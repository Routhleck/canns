"""Custom Textual widgets for ASA TUI.

This module provides reusable UI components for the ASA analysis interface.
"""

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Button, Static, Input, Select, Label


class ImagePreview(Container):
    """Widget for previewing images in the terminal using climage."""

    DEFAULT_CSS = """
    ImagePreview {
        height: 20;
        border: solid $accent;
        padding: 1;
    }
    """

    def __init__(self, image_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self.image_path = image_path

    def compose(self) -> ComposeResult:
        yield Label("Image Preview", id="preview-label")
        yield Static("No image loaded", id="preview-content")
        yield Button("Open Externally", id="preview-open-btn")

    def update_image(self, path: Optional[Path]):
        """Update the previewed image."""
        self.image_path = path
        content = self.query_one("#preview-content", Static)

        if path is None or not path.exists():
            content.update("No image loaded")
            return

        # Try to use climage for terminal preview
        try:
            import climage

            img_output = climage.convert(str(path), width=60, is_unicode=True)
            content.update(img_output)
        except ImportError:
            content.update(f"Image: {path.name}\n(Install climage for preview)")
        except Exception as e:
            content.update(f"Error loading image: {e}")


class ParamGroup(Vertical):
    """Widget for grouping related parameters."""

    DEFAULT_CSS = """
    ParamGroup {
        border: round $secondary;
        padding: 1;
        margin: 1 0;
        height: auto;
        width: 100%;
    }
    """

    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self.title = title

    def compose(self) -> ComposeResult:
        yield Label(self.title, classes="param-group-title")


class LogViewer(Vertical):
    """Widget for displaying log messages."""

    DEFAULT_CSS = """
    LogViewer {
        height: 10;
        border: solid $primary;
        padding: 1;
        overflow-y: scroll;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_lines = []

    def compose(self) -> ComposeResult:
        yield Static("", id="log-content")

    def add_log(self, message: str):
        """Add a log message."""
        self.log_lines.append(message)
        if len(self.log_lines) > 100:
            self.log_lines = self.log_lines[-100:]

        content = self.query_one("#log-content", Static)
        content.update("\n".join(self.log_lines))

    def clear(self):
        """Clear all log messages."""
        self.log_lines = []
        content = self.query_one("#log-content", Static)
        content.update("")
