"""Async worker infrastructure for ASA GUI.

This module provides QThread-based workers for running analysis
in the background without blocking the UI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal, Slot

if TYPE_CHECKING:
    from .state import WorkflowState

    from ..models.job import JobResult


class AnalysisWorker(QObject):
    """Background worker for analysis execution.

    Runs analysis in a separate thread and emits signals for
    progress updates, logging, and completion.
    """

    # Signals
    log = Signal(str)  # Log message
    progress = Signal(int)  # Progress percentage (0-100)
    finished = Signal(object)  # JobResult on success
    error = Signal(str)  # Error message on failure

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cancelled = False
        self._task: Callable[..., Any] | None = None
        self._args: tuple[Any, ...] = ()
        self._kwargs: dict[str, Any] = {}

    def setup(
        self,
        task: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Configure the task to run.

        Args:
            task: Callable to execute
            *args: Positional arguments for task
            **kwargs: Keyword arguments for task
        """
        self._task = task
        self._args = args
        self._kwargs = kwargs
        self._cancelled = False

    @Slot()
    def run(self) -> None:
        """Execute the configured task."""
        if self._task is None:
            self.error.emit("No task configured")
            return

        try:
            # Inject callbacks into kwargs
            self._kwargs["log_callback"] = self._emit_log
            self._kwargs["progress_callback"] = self._emit_progress
            self._kwargs["cancel_check"] = self._is_cancelled

            result = self._task(*self._args, **self._kwargs)

            if self._cancelled:
                self.error.emit("Cancelled by user")
            else:
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def request_cancel(self) -> None:
        """Request cancellation of running task."""
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    def _emit_log(self, msg: str) -> None:
        """Emit log signal (thread-safe)."""
        self.log.emit(msg)

    def _emit_progress(self, pct: int) -> None:
        """Emit progress signal (thread-safe)."""
        self.progress.emit(max(0, min(100, pct)))


class _UiDispatcher(QObject):
    """Dispatch worker signals onto the UI thread."""

    def __init__(
        self,
        manager: "WorkerManager",
        *,
        on_log: Callable[[str], None] | None,
        on_progress: Callable[[int], None] | None,
        on_finished: Callable[[Any], None] | None,
        on_error: Callable[[str], None] | None,
    ) -> None:
        super().__init__()
        self._manager = manager
        self._on_log = on_log
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._on_error = on_error

    @Slot(str)
    def handle_log(self, msg: str) -> None:
        if self._on_log:
            self._on_log(msg)

    @Slot(int)
    def handle_progress(self, pct: int) -> None:
        if self._on_progress:
            self._on_progress(pct)

    @Slot(object)
    def handle_finished(self, result: Any) -> None:
        try:
            if self._on_finished:
                self._on_finished(result)
        except Exception:
            import traceback

            traceback.print_exc()
        finally:
            self._manager._cleanup()

    @Slot(str)
    def handle_error(self, msg: str) -> None:
        try:
            if self._on_error:
                self._on_error(msg)
        except Exception:
            import traceback

            traceback.print_exc()
        finally:
            self._manager._cleanup()


class WorkerManager:
    """Manages worker thread lifecycle.

    Ensures only one worker runs at a time and handles
    proper cleanup on completion or cancellation.
    """

    def __init__(self) -> None:
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._dispatcher: _UiDispatcher | None = None
        self._on_cleanup: Callable[[], None] | None = None

    def is_running(self) -> bool:
        """Check if a worker is currently running."""
        return self._thread is not None and self._thread.isRunning()

    def start(
        self,
        task: Callable[..., Any],
        *args: Any,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[int], None] | None = None,
        on_finished: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_cleanup: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a task in a background thread.

        Args:
            task: Callable to execute
            *args: Positional arguments for task
            on_log: Callback for log messages
            on_progress: Callback for progress updates
            on_finished: Callback on successful completion
            on_error: Callback on error
            on_cleanup: Callback after thread cleanup
            **kwargs: Keyword arguments for task

        Raises:
            RuntimeError: If a task is already running
        """
        if self.is_running():
            raise RuntimeError("A task is already running")

        self._on_cleanup = on_cleanup

        # Create thread and worker
        self._thread = QThread()
        self._worker = AnalysisWorker()
        self._worker.setup(task, *args, **kwargs)
        self._worker.moveToThread(self._thread)

        # Dispatch signals onto UI thread
        self._dispatcher = _UiDispatcher(
            self,
            on_log=on_log,
            on_progress=on_progress,
            on_finished=on_finished,
            on_error=on_error,
        )
        app = QCoreApplication.instance()
        if app is not None:
            self._dispatcher.moveToThread(app.thread())

        self._worker.log.connect(self._dispatcher.handle_log)
        self._worker.progress.connect(self._dispatcher.handle_progress)
        self._worker.finished.connect(self._dispatcher.handle_finished)
        self._worker.error.connect(self._dispatcher.handle_error)

        # Start execution
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def request_cancel(self) -> None:
        """Request cancellation of running task."""
        if self._worker:
            self._worker.request_cancel()

    def wait(self, timeout_ms: int = 5000) -> bool:
        """Wait for worker to finish.

        Args:
            timeout_ms: Maximum time to wait in milliseconds

        Returns:
            True if worker finished, False if timeout
        """
        if self._thread:
            return self._thread.wait(timeout_ms)
        return True

    def _cleanup(self) -> None:
        """Clean up thread and worker after completion."""
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread.deleteLater()
            self._thread = None

        if self._worker:
            self._worker.deleteLater()
            self._worker = None
        if self._dispatcher:
            self._dispatcher.deleteLater()
            self._dispatcher = None

        if self._on_cleanup:
            self._on_cleanup()
            self._on_cleanup = None
