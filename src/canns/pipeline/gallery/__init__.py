"""Model gallery TUI."""

import os

__all__ = ["GalleryApp", "main"]


def main() -> None:
    """Entry point for the model gallery TUI."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    from .app import GalleryApp

    app = GalleryApp()
    app.run()


from .app import GalleryApp
