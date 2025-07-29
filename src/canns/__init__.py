from . import analyzer as analyzer
from . import misc as misc
from . import models as models
from . import pipeline as pipeline
from . import trainer as trainer

# Version information
try:
    from ._version import __version__, version_info
except ImportError:
    # Fallback if _version.py is not available (e.g., during documentation build)
    __version__ = "unknown"
    version_info = (0, 0, 0, "unknown")

__all__ = [
    "analyzer",
    "misc", 
    "models",
    "pipeline",
    "trainer",
    "__version__",
    "version_info",
]
