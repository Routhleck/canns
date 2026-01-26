"""Custom Textual widgets for ASA TUI.

This module provides reusable UI components for the ASA analysis interface.
"""

from pathlib import Path

from rich.ansi import AnsiDecoder
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Input, Label, Static


class ImagePreview(Vertical):
    """Widget for previewing images in the terminal using climage."""

    DEFAULT_CSS = """
    ImagePreview {
        height: auto;
        min-height: 20;
        border: solid $accent;
        padding: 1;
    }
    #preview-content {
        width: 100%;
        height: auto;
    }
    #preview-scroll {
        height: 1fr;
    }
    #preview-controls Button {
        margin: 0 1 0 0;
    }
    #preview-arrows Button {
        margin: 0 1 0 0;
    }
    #preview-controls, #preview-arrows {
        height: auto;
    }
    """

    def __init__(self, image_path: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.image_path = image_path
        self._zoom_step = 0

    def compose(self) -> ComposeResult:
        yield Label("Image Preview", id="preview-label")
        yield Static("Path: (none)", id="preview-path")
        yield Input(value="", id="preview-path-input")
        with Horizontal(id="preview-controls"):
            yield Button("Load", id="preview-load-btn")
            yield Button("Open", id="preview-open-btn")
            yield Button("Zoom +", id="preview-zoom-in-btn")
            yield Button("Zoom -", id="preview-zoom-out-btn")
            yield Button("Fit", id="preview-zoom-fit-btn")
        with Horizontal(id="preview-arrows"):
            yield Button("←", id="preview-pan-left-btn")
            yield Button("→", id="preview-pan-right-btn")
            yield Button("↑", id="preview-pan-up-btn")
            yield Button("↓", id="preview-pan-down-btn")
        with ScrollableContainer(id="preview-scroll"):
            yield Static("No image loaded", id="preview-content")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "preview-load-btn":
            raw = self.query_one("#preview-path-input", Input).value.strip()
            if not raw:
                self.update_image(None)
                return
            path = self._resolve_path(raw)
            self.update_image(path)
        elif event.button.id == "preview-zoom-in-btn":
            self._zoom_step += 1
            self.update_image(self.image_path)
        elif event.button.id == "preview-zoom-out-btn":
            self._zoom_step -= 1
            self.update_image(self.image_path)
        elif event.button.id == "preview-zoom-fit-btn":
            self._zoom_step = 0
            self.update_image(self.image_path)
        elif event.button.id in {
            "preview-pan-left-btn",
            "preview-pan-right-btn",
            "preview-pan-up-btn",
            "preview-pan-down-btn",
        }:
            scroll = self.query_one("#preview-scroll", ScrollableContainer)
            dx = 0
            dy = 0
            step = 5
            if event.button.id == "preview-pan-left-btn":
                dx = -step
            elif event.button.id == "preview-pan-right-btn":
                dx = step
            elif event.button.id == "preview-pan-up-btn":
                dy = -step
            elif event.button.id == "preview-pan-down-btn":
                dy = step
            scroll.scroll_relative(x=dx, y=dy, animate=False)

    def on_resize(self, event) -> None:
        if self.image_path and self.image_path.exists():
            self.update_image(self.image_path)

    def update_image(self, path: Path | None):
        """Update the previewed image."""
        self.image_path = path
        content = self.query_one("#preview-content", Static)
        path_label = self.query_one("#preview-path", Static)
        path_input = self.query_one("#preview-path-input", Input)

        if path is None or not path.exists():
            content.update("No image loaded")
            path_label.update("Path: (none)")
            path_input.value = ""
            return
        path_label.update(f"Path: {path}")
        path_input.value = str(path)

        # Try to use climage for terminal preview
        try:
            import climage

            scroll = self.query_one("#preview-scroll", ScrollableContainer)
            base_width = max(20, scroll.size.width - 4)
            base_height = max(8, scroll.size.height - 2)
            scale = max(0.4, 1 + (self._zoom_step * 0.1))
            width = max(20, int(base_width * scale))
            height = max(8, int(base_height * scale))
            try:
                img_output = climage.convert(str(path), width=width, height=height, is_unicode=True)
            except TypeError:
                img_output = climage.convert(str(path), width=width, is_unicode=True)
            decoder = AnsiDecoder()
            chunks = list(decoder.decode(img_output))
            content.update(Text("\n").join(chunks) if chunks else "")
        except ImportError:
            content.update(f"Image: {path.name}\n(Install climage for preview)")
        except Exception as e:
            content.update(f"Error loading image: {e}")

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        app = getattr(self, "app", None)
        if app is not None:
            state = getattr(app, "state", None)
            if state is not None and hasattr(state, "workdir"):
                return Path(state.workdir) / path
        return Path.cwd() / path


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
        # Auto-scroll to latest entry
        self.scroll_end(animate=False, immediate=True)

    def clear(self):
        """Clear all log messages."""
        self.log_lines = []
        content = self.query_one("#log-content", Static)
        content.update("")
