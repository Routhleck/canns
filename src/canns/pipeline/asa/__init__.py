"""ASA TUI - Terminal User Interface for ASA Analysis.

This module provides a Textual-based TUI for running ASA (Attractor State Analysis)
with 7 analysis modules: TDA, CohoMap, PathCompare, CohoSpace, FR, FRM, and GridScore.
"""

from .app import ASAApp

__all__ = ["ASAApp", "main"]


def main():
    """Entry point for canns-tui command."""
    app = ASAApp()
    app.run()
