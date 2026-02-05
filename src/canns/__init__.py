"""Top-level package for CANNs.

This module exposes the main namespaces so you can import them directly from
``canns`` (for example, ``canns.data`` or ``canns.trainer``). It also provides
simple version metadata.

Examples:
    >>> import canns
    >>> print(canns.__version__)
    >>> print(canns.version_info)
    >>> print(list(canns.data.DATASETS))
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Version information
try:
    from ._version import __version__ as _version
    from ._version import version_info as _version_info
except ImportError:
    # Fallback if _version.py is not available (e.g., during documentation build)
    _version = "unknown"
    _version_info = (0, 0, 0, "unknown")

__version__ = _version
"""Human-readable package version string.

This is usually derived from the installed package metadata. When that
information is unavailable, it falls back to ``"unknown"``.

Examples:
    >>> import canns
    >>> print(canns.__version__)
"""

version_info = _version_info
"""Version information as a tuple.

The tuple typically follows ``(major, minor, patch)``. A development or
documentation build may return a fallback value instead.

Examples:
    >>> import canns
    >>> print(canns.version_info)
"""

_LAZY_SUBMODULES = {
    "analyzer",
    "data",
    "models",
    "pipeline",
    "trainer",
    "utils",
}

if TYPE_CHECKING:  # pragma: no cover
    from . import analyzer, data, models, pipeline, trainer, utils


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "analyzer",
    "data",
    "models",
    "pipeline",
    "trainer",
    "utils",
    "__version__",
    "version_info",
]
