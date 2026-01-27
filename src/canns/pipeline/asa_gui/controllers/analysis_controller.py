"""Controller for analysis workflow."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject

from ..core import PipelineRunner, StateManager


class AnalysisController(QObject):
    def __init__(self, state_manager: StateManager, runner: PipelineRunner, parent=None) -> None:
        super().__init__(parent)
        self._state_manager = state_manager
        self._runner = runner

    def update_analysis(self, *, analysis_mode: str, analysis_params: dict) -> None:
        self._state_manager.batch_update(
            analysis_mode=analysis_mode,
            analysis_params=analysis_params,
        )

    def get_state(self):
        return self._state_manager.state

    def run_analysis(
        self,
        *,
        worker_manager,
        on_log: Callable[[str], None],
        on_progress: Callable[[int], None],
        on_finished: Callable[[object], None],
        on_error: Callable[[str], None],
        on_cleanup: Callable[[], None] | None = None,
    ) -> None:
        state = self._state_manager.state
        self._state_manager.update(is_running=True, current_stage="analysis")

        worker_manager.start(
            self._runner.run_analysis,
            state,
            on_log=on_log,
            on_progress=on_progress,
            on_finished=on_finished,
            on_error=on_error,
            on_cleanup=on_cleanup,
        )

    def finalize_analysis(self, artifacts: dict) -> None:
        self._state_manager.update(
            artifacts=artifacts,
            is_running=False,
            current_stage="",
            progress=0,
        )

    def mark_idle(self) -> None:
        self._state_manager.update(is_running=False, current_stage="", progress=0)
