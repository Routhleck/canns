"""Controller for preprocessing workflow."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject

from ..core import PipelineRunner, StateManager
from ..core.state import relative_path


class PreprocessController(QObject):
    def __init__(self, state_manager: StateManager, runner: PipelineRunner, parent=None) -> None:
        super().__init__(parent)
        self._state_manager = state_manager
        self._runner = runner

    def update_inputs(
        self,
        *,
        input_mode: str,
        preset: str,
        asa_file: str | None,
        neuron_file: str | None,
        traj_file: str | None,
        preprocess_method: str,
        preprocess_params: dict,
        preclass: str,
        preclass_params: dict,
    ) -> None:
        state = self._state_manager.state
        asa_path = self._to_path(asa_file)
        neuron_path = self._to_path(neuron_file)
        traj_path = self._to_path(traj_file)
        self._state_manager.batch_update(
            input_mode=input_mode,
            preset=preset,
            asa_file=relative_path(state, path=asa_path) if asa_path is not None else None,
            neuron_file=relative_path(state, path=neuron_path) if neuron_path is not None else None,
            traj_file=relative_path(state, path=traj_path) if traj_path is not None else None,
            preprocess_method=preprocess_method,
            preprocess_params=preprocess_params,
            preclass=preclass,
            preclass_params=preclass_params,
        )

    def run_preprocess(
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
        self._state_manager.update(is_running=True, current_stage="preprocess")

        worker_manager.start(
            self._runner.run_preprocessing,
            state,
            on_log=on_log,
            on_progress=on_progress,
            on_finished=on_finished,
            on_error=on_error,
            on_cleanup=on_cleanup,
        )

    def finalize_preprocess(self) -> None:
        self._state_manager.update(
            embed_data=self._runner.embed_data,
            aligned_pos=self._runner.aligned_pos,
            is_running=False,
            current_stage="",
            progress=0,
        )

    def mark_idle(self) -> None:
        self._state_manager.update(is_running=False, current_stage="", progress=0)

    @staticmethod
    def _to_path(path: str | None):
        if path is None:
            return None
        from pathlib import Path

        return Path(path)
