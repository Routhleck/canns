"""Main entry point for running ASA TUI as a module."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from .app import ASAApp

if __name__ == "__main__":
    app = ASAApp()
    app.run()
