"""Main entry point for running ASA TUI as a module."""

from .app import ASAApp

if __name__ == "__main__":
    app = ASAApp()
    app.run()
