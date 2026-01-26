"""Main entry point for running the gallery TUI as a module."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from .app import GalleryApp

if __name__ == "__main__":
    app = GalleryApp()
    app.run()
