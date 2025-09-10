"""Progress reporting system for CANNS training and prediction."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

__all__ = [
    "ProgressReporter",
    "SilentProgressReporter",
    "TqdmProgressReporter",
    "create_progress_reporter",
]


class ProgressReporter(ABC):
    """Abstract base class for progress reporting."""

    @abstractmethod
    def start_task(self, name: str, total: int | None = None, **kwargs) -> Any:
        """Start a new progress tracking task.

        Args:
            name: Name/description of the task
            total: Total number of steps (if known)
            **kwargs: Additional parameters for the reporter

        Returns:
            Task handle for updating progress
        """
        pass

    @abstractmethod
    def update(self, task_handle: Any, n: int = 1, **kwargs):
        """Update progress for a task.

        Args:
            task_handle: Handle returned by start_task
            n: Number of steps to advance
            **kwargs: Additional update parameters
        """
        pass

    @abstractmethod
    def finish_task(self, task_handle: Any):
        """Finish a progress tracking task.

        Args:
            task_handle: Handle returned by start_task
        """
        pass

    @contextmanager
    def progress(self, name: str, total: int | None = None, **kwargs):
        """Context manager for progress tracking.

        Usage:
            with reporter.progress("Training", total=100) as pbar:
                for i in range(100):
                    # do work
                    reporter.update(pbar)
        """
        task_handle = self.start_task(name, total, **kwargs)
        try:
            yield task_handle
        finally:
            self.finish_task(task_handle)


class SilentProgressReporter(ProgressReporter):
    """Progress reporter that does nothing (silent mode)."""

    def start_task(self, name: str, total: int | None = None, **kwargs) -> None:
        return None

    def update(self, task_handle: Any, n: int = 1, **kwargs):
        pass

    def finish_task(self, task_handle: Any):
        pass


class TqdmProgressReporter(ProgressReporter):
    """Progress reporter using tqdm library."""

    def __init__(self, nested: bool = True, **default_kwargs):
        """Initialize TqdmProgressReporter.

        Args:
            nested: Whether to support nested progress bars
            **default_kwargs: Default arguments for tqdm
        """
        if not TQDM_AVAILABLE:
            raise ImportError(
                "tqdm is required for TqdmProgressReporter. Install with: pip install tqdm"
            )

        self.nested = nested
        self.default_kwargs = default_kwargs
        self._active_bars = []

    def start_task(self, name: str, total: int | None = None, **kwargs) -> tqdm:
        """Start a new tqdm progress bar."""
        # Merge default kwargs with provided kwargs
        tqdm_kwargs = {**self.default_kwargs, **kwargs}

        # Set description and total
        tqdm_kwargs["desc"] = name
        if total is not None:
            tqdm_kwargs["total"] = total

        # Handle nested progress bars - improve display
        if self.nested and self._active_bars:
            tqdm_kwargs["leave"] = False
            # Add position to stack nested bars properly  
            tqdm_kwargs["position"] = len(self._active_bars)
            # Reduce refresh rate for nested bars to avoid conflicts
            if "miniters" not in tqdm_kwargs:
                tqdm_kwargs["miniters"] = 1

        pbar = tqdm(**tqdm_kwargs)
        self._active_bars.append(pbar)
        return pbar

    def update(self, task_handle: tqdm, n: int = 1, **kwargs):
        """Update tqdm progress bar."""
        if task_handle is not None:
            # Update postfix if provided
            if kwargs:
                task_handle.set_postfix(**kwargs)
            task_handle.update(n)

    def finish_task(self, task_handle: tqdm):
        """Close tqdm progress bar."""
        if task_handle is not None and task_handle in self._active_bars:
            task_handle.close()
            self._active_bars.remove(task_handle)


def create_progress_reporter(mode: str = "auto", nested: bool = True, **kwargs) -> ProgressReporter:
    """Factory function to create progress reporters.

    Args:
        mode: Progress reporting mode
            - "auto": Use tqdm if available, otherwise silent
            - "tqdm": Use tqdm (raises error if not available)
            - "silent": Silent mode (no progress reporting)
        nested: Whether to support nested progress bars
        **kwargs: Additional arguments for the reporter

    Returns:
        ProgressReporter instance
    """
    if mode == "silent":
        return SilentProgressReporter()
    elif mode == "tqdm":
        return TqdmProgressReporter(nested=nested, **kwargs)
    elif mode == "auto":
        if TQDM_AVAILABLE:
            return TqdmProgressReporter(nested=nested, **kwargs)
        else:
            return SilentProgressReporter()
    else:
        raise ValueError(f"Unknown progress mode: {mode}")
