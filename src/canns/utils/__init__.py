"""General utilities for CANNs.

This namespace provides small helpers that don't fit into a specific domain,
such as benchmarking utilities.

Examples:
    >>> from canns.utils import benchmark
    >>>
    >>> @benchmark(runs=3)
    ... def add():
    ...     return 1 + 1
    >>>
    >>> add()
"""

from .benchmark import benchmark
from .example_outputs import get_example_output_dir

__all__ = [
    "benchmark",
]
