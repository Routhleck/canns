"""Version management for CANNs package."""

import re

try:
    # Try to get version from installed package
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("canns")
    except PackageNotFoundError:
        # Fallback for development installs
        __version__ = "0.5.1+dev"
except ImportError:
    # Fallback for Python < 3.8
    __version__ = "0.5.1+dev"


def parse_version_info(version_string):
    """
    Parse version string into tuple of integers.
    Handles various formats like:
    - "0.5.1"
    - "0.5.1+dev"
    - "0.5.1.dev6"
    - "0.5.1+dev6"
    """
    # Extract the base version (before any +, .dev, etc.)
    # First split on '+' to handle formats like "0.5.1+dev6"
    base_version = version_string.split("+")[0]

    # Then use regex to extract only the numeric parts
    # This handles formats like "0.5.1.dev6" -> "0.5.1"
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", base_version)
    if match:
        return tuple(int(x) for x in match.groups())
    else:
        # Fallback to manual parsing if regex fails
        parts = []
        for part in base_version.split("."):
            try:
                parts.append(int(part))
            except ValueError:
                # Stop at first non-numeric part
                break
        # Ensure we have at least 3 parts (major, minor, patch)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])


# Export the version info
version_info = parse_version_info(__version__)
