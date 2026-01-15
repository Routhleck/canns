"""Utilities for Jupyter notebook integration with matplotlib animations."""

from __future__ import annotations


def is_jupyter_environment() -> bool:
    """Detect if code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.

    Examples:
        >>> from canns.analyzer.visualization.core.jupyter_utils import is_jupyter_environment
        >>> print(is_jupyter_environment() in {True, False})
        True
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


def display_animation_in_jupyter(animation, format: str = "html5"):
    """Display a matplotlib animation in a Jupyter notebook.

    Args:
        animation: ``matplotlib.animation.FuncAnimation`` instance.
        format: Display format - ``"html5"`` (default) or ``"jshtml"``.

    Returns:
        ``IPython.display.HTML`` object if successful, otherwise ``None``.

    Examples:
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from matplotlib.animation import FuncAnimation
        >>> from canns.analyzer.visualization.core.jupyter_utils import (
        ...     display_animation_in_jupyter,
        ...     is_jupyter_environment,
        ... )
        >>>
        >>> x = np.linspace(0, 2 * np.pi, 50)
        >>> fig, ax = plt.subplots()
        >>> (line,) = ax.plot([], [])
        >>>
        >>> def update(i):
        ...     line.set_data(x[: i + 1], np.sin(x[: i + 1]))
        ...     return (line,)
        >>>
        >>> anim = FuncAnimation(fig, update, frames=5, interval=50, blit=True)
        >>> if is_jupyter_environment():
        ...     _ = display_animation_in_jupyter(anim, format="jshtml")
        ... print(anim is not None)
        True
    """
    try:
        from IPython.display import HTML, display

        # Generate HTML content based on format
        use_autoplay = False

        if format == "html5":
            # Use HTML5 video tag (requires ffmpeg or similar)
            try:
                html_content = animation.to_html5_video()
            except Exception as e:
                # Fallback to jshtml if HTML5 video generation fails
                # (e.g., FFmpeg not installed)
                import warnings

                warnings.warn(
                    f"Failed to generate HTML5 video (FFmpeg may not be installed): {e}\n"
                    "Falling back to jshtml format. Install FFmpeg for better performance:\n"
                    "  conda install -c conda-forge ffmpeg  OR  brew install ffmpeg",
                    UserWarning,
                    stacklevel=2,
                )
                html_content = animation.to_jshtml()
                use_autoplay = True  # Add autoplay for jshtml fallback
        else:
            # Use JavaScript-based animation (no external dependencies)
            html_content = animation.to_jshtml()
            use_autoplay = True

        # Add autoplay functionality for jshtml
        if use_autoplay:
            autoplay_script = """
<script>
(function() {
    var attemptAutoplay = function(attempts) {
        if (attempts <= 0) {
            console.log('Autoplay: Max attempts reached');
            return;
        }

        try {
            // Find all animation containers
            var buttons = document.getElementsByClassName('anim-buttons');
            if (buttons.length === 0) {
                // Retry if animations not loaded yet
                setTimeout(function() { attemptAutoplay(attempts - 1); }, 200);
                return;
            }

            // Get the last animation (most recently added)
            var lastButtons = buttons[buttons.length - 1];
            var allButtons = lastButtons.getElementsByTagName('button');

            // matplotlib jshtml controls layout (9 buttons in total):
            // Button order: ... [3] = reverse play, ... [5] = forward play ...
            // We want index 5 for forward playback

            // Try to click the forward play button (index 5, the 6th button)
            if (allButtons.length > 5) {
                allButtons[5].click();
                console.log('Autoplay: Clicked forward play button (index 5)');
                return;
            }

            // Fallback: search for play button by content
            for (var i = 0; i < allButtons.length; i++) {
                var btn = allButtons[i];
                var btnText = btn.getAttribute('title') || btn.textContent || '';
                // Look for play (not reverse play)
                if (btnText === 'Play' || btnText.includes('▶') && !btnText.includes('◄')) {
                    btn.click();
                    console.log('Autoplay: Found and clicked play button at index ' + i);
                    return;
                }
            }

            console.log('Autoplay: Could not find play button');
        } catch(e) {
            console.log('Autoplay error:', e);
        }
    };

    // Start autoplay with retries
    setTimeout(function() { attemptAutoplay(5); }, 300);
})();
</script>
"""
            html_content = html_content + autoplay_script

        html_obj = HTML(html_content)
        display(html_obj)  # Actually display the animation in Jupyter
        return html_obj
    except ImportError as e:
        print(f"Warning: Could not import IPython.display: {e}")
        return None
    except Exception as e:
        print(f"Warning: Could not render animation in Jupyter: {e}")
        return None
