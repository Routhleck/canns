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
        from IPython.display import HTML, display

        if format == "html5":
            # Use HTML5 video tag (requires ffmpeg or similar)
            html_content = animation.to_html5_video()
        else:
            # Use JavaScript-based animation (default, no external dependencies)
            html_content = animation.to_jshtml()

            # Add autoplay functionality - automatically click the play button
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
