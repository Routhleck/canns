from . import analyzer as analyzer
from . import data as data
from . import models as models
from . import pipeline as pipeline
from . import trainer as trainer
from . import utils as utils

# Version information
try:
    from ._version import __version__, version_info
except ImportError:
    # Fallback if _version.py is not available (e.g., during documentation build)
    __version__ = "unknown"
    version_info = (0, 0, 0, "unknown")

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
