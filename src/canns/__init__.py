from . import analyzer as analyzer
from . import data as data
from . import models as models
from . import pipeline as pipeline
from . import trainer as trainer
from . import utils as utils

# Backward compatibility aliases for old import paths
_datasets = data  # For imports like: from canns import _datasets
misc = utils  # For imports like: from canns import misc

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
    # Backward compatibility
    "_datasets",
    "misc",
    # Version info
    "__version__",
    "version_info",
]
