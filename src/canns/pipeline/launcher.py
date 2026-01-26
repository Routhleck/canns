"""Launcher for selecting ASA or Gallery TUI."""

from __future__ import annotations

import os

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Button, Label


class ModePicker(App[str | None]):
    """Select which TUI to run."""

    TITLE = "CANNs TUI Launcher"

    CSS = """
    Screen {
        align: center middle;
    }

    #card {
        width: 68;
        height: 18;
        border: thick $accent;
        background: $surface;
        padding: 2;
    }

    #title {
        text-style: bold;
        margin-bottom: 1;
    }

    #subtitle {
        color: $text-muted;
        margin-bottom: 2;
    }

    Button {
        width: 100%;
        margin: 1 0;
    }
    """

    BINDINGS = [
        Binding("escape", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="card"):
            yield Label("Choose a workflow", id="title")
            yield Label("ASA analysis or Model Gallery", id="subtitle")
            with Vertical():
                yield Button("ASA (Attractor Structure Analyzer)", id="btn-asa", variant="primary")
                yield Button("Model Gallery", id="btn-gallery", variant="primary")
                yield Button("Quit", id="btn-quit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-asa":
            self.exit(result="asa")
        elif event.button.id == "btn-gallery":
            self.exit(result="gallery")
        elif event.button.id == "btn-quit":
            self.exit(result=None)


def main() -> None:
    """Entry point for the unified canns-tui launcher."""
    os.environ.setdefault("MPLBACKEND", "Agg")

    selection = ModePicker().run()
    if selection == "gallery":
        from .gallery import GalleryApp

        GalleryApp().run()
    elif selection == "asa":
        from .asa import ASAApp

        ASAApp().run()
