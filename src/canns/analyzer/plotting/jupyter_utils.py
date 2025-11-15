"""Utilities for Jupyter notebook integration with matplotlib animations."""

from __future__ import annotations


def is_jupyter_environment() -> bool:
    """
    Detect if code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in Jupyter/IPython notebook, False otherwise.
    """
    try:
        # Check if IPython is available and we're in a notebook
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False

        # Check if we're in a notebook environment (not just IPython terminal)
        # ZMQInteractiveShell is used by Jupyter notebooks
        # TerminalInteractiveShell is used by IPython terminal
        shell_class = ipython.__class__.__name__
        return shell_class == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False


def display_animation_in_jupyter(animation, format: str = "jshtml"):
    """
    Display a matplotlib animation in Jupyter notebook using HTML/JavaScript.

    Args:
        animation: matplotlib.animation.FuncAnimation object
        format: Display format - 'jshtml' (default) or 'html5' (video tag)

    Returns:
        IPython.display.HTML object if successful, None otherwise
    """
    try:
        from IPython.display import HTML

        if format == "html5":
            # Use HTML5 video tag (requires ffmpeg or similar)
            html_content = animation.to_html5_video()
        else:
            # Use JavaScript-based animation (default, no external dependencies)
            html_content = animation.to_jshtml()

        return HTML(html_content)
    except ImportError as e:
        print(f"Warning: Could not import IPython.display: {e}")
        return None
    except Exception as e:
        print(f"Warning: Could not render animation in Jupyter: {e}")
        return None
