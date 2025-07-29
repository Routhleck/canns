"""Version management for CANNs package."""

try:
    # Try to get version from installed package
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("canns")
    except PackageNotFoundError:
        # Fallback for development installs
        __version__ = "0.5.1+dev"
except ImportError:
    # Fallback for Python < 3.8
    __version__ = "0.5.1+dev"

# Export the version
version_info = tuple(int(x) for x in __version__.split("+")[0].split("."))