"""State management for ASA GUI.

This module provides centralized workflow state management with Qt signals
for reactive UI updates. All file paths are stored relative to the working
directory for portability.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class WorkflowState:
    """Centralized state for ASA analysis workflow.

    All file paths are relative to workdir for portability.
    """

    # Core paths
    workdir: Path = field(default_factory=lambda: Path(".").resolve())

    # Input configuration
    input_mode: str = "asa"  # "asa" | "neuron_traj" | "batch"
    preset: str = "grid"  # "grid" | "hd" | "none"

    # File paths (relative to workdir)
    asa_file: Path | None = None
    neuron_file: Path | None = None
    traj_file: Path | None = None

    # Preprocessing
    preprocess_method: str = "none"  # "none" | "embed_spike_trains"
    preprocess_params: dict[str, Any] = field(default_factory=dict)
    preclass: str = "none"  # "none" | "grid" | "hd"
    preclass_params: dict[str, Any] = field(default_factory=dict)

    # Preprocessed data (in-memory)
    embed_data: NDArray[np.floating] | None = None
    aligned_pos: dict[str, NDArray[np.floating]] | None = None

    # Analysis configuration
    analysis_mode: str = "tda"
    analysis_params: dict[str, Any] = field(default_factory=dict)

    # Results
    artifacts: dict[str, Path] = field(default_factory=dict)

    # Runtime state
    is_running: bool = False
    current_stage: str = ""
    progress: int = 0

    def copy(self) -> WorkflowState:
        """Create a shallow copy of the state (excluding large arrays)."""
        return WorkflowState(
            workdir=self.workdir,
            input_mode=self.input_mode,
            preset=self.preset,
            asa_file=self.asa_file,
            neuron_file=self.neuron_file,
            traj_file=self.traj_file,
            preprocess_method=self.preprocess_method,
            preprocess_params=deepcopy(self.preprocess_params),
            preclass=self.preclass,
            preclass_params=deepcopy(self.preclass_params),
            embed_data=None,  # Don't copy large arrays
            aligned_pos=None,
            analysis_mode=self.analysis_mode,
            analysis_params=deepcopy(self.analysis_params),
            artifacts=deepcopy(self.artifacts),
            is_running=self.is_running,
            current_stage=self.current_stage,
            progress=self.progress,
        )


class StateManager(QObject):
    """Reactive state manager with Qt signals.

    Emits signals when state changes to enable reactive UI updates.
    Supports undo/redo through state history.
    """

    # Signal emitted when any state field changes: (field_name, new_value)
    state_changed = Signal(str, object)

    # Signal emitted when state is fully replaced (e.g., undo/redo)
    state_replaced = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = WorkflowState()
        self._history: list[WorkflowState] = []
        self._history_index = -1
        self._max_history = 50

    @property
    def state(self) -> WorkflowState:
        """Get current workflow state."""
        return self._state

    def update(self, **kwargs: Any) -> None:
        """Update state fields and emit signals.

        Args:
            **kwargs: Field names and their new values
        """
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                old_value = getattr(self._state, key)
                if not self._is_equal(old_value, value):
                    setattr(self._state, key, value)
                    self.state_changed.emit(key, value)

    def batch_update(self, **kwargs: Any) -> None:
        """Update multiple fields without emitting individual signals.

        Emits state_replaced at the end.
        """
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        self.state_replaced.emit()

    def push_history(self) -> None:
        """Save current state for undo."""
        # Truncate forward history
        self._history = self._history[: self._history_index + 1]
        # Save state snapshot
        self._history.append(self._state.copy())
        # Limit history size
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        self._history_index = len(self._history) - 1

    def undo(self) -> bool:
        """Restore previous state.

        Returns:
            True if undo was successful
        """
        if self._history_index > 0:
            self._history_index -= 1
            self._restore(self._history[self._history_index])
            return True
        return False

    def redo(self) -> bool:
        """Restore next state.

        Returns:
            True if redo was successful
        """
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._restore(self._history[self._history_index])
            return True
        return False

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._history_index > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._history_index < len(self._history) - 1

    def _restore(self, snapshot: WorkflowState) -> None:
        """Restore state from snapshot."""
        # Preserve large arrays from current state
        embed_data = self._state.embed_data
        aligned_pos = self._state.aligned_pos

        self._state = snapshot.copy()
        self._state.embed_data = embed_data
        self._state.aligned_pos = aligned_pos

        self.state_replaced.emit()

    def reset(self) -> None:
        """Reset state to defaults."""
        self._state = WorkflowState()
        self._history.clear()
        self._history_index = -1
        self.state_replaced.emit()

    @staticmethod
    def _is_equal(a: Any, b: Any) -> bool:
        """Safe equality check that handles numpy arrays and containers."""
        if a is b:
            return True
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return np.array_equal(a, b)
            return False
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(StateManager._is_equal(a[k], b[k]) for k in a.keys())
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(StateManager._is_equal(x, y) for x, y in zip(a, b, strict=False))
        try:
            return a == b
        except Exception:
            return False


# --- Path utilities ---


def relative_path(state: WorkflowState, path: Path) -> Path:
    """Convert absolute path to workdir-relative path.

    Args:
        state: Current workflow state
        path: Absolute path to convert

    Returns:
        Path relative to workdir
    """
    try:
        return path.relative_to(state.workdir)
    except ValueError:
        # Path is not relative to workdir, return as-is
        return path


def resolve_path(state: WorkflowState, path: Path | None) -> Path | None:
    """Convert relative path to absolute path.

    Args:
        state: Current workflow state
        path: Relative path to convert

    Returns:
        Absolute path or None if path is None
    """
    if path is None:
        return None

    if path.is_absolute():
        return path

    return state.workdir / path


# --- Validation utilities ---


def validate_files(state: WorkflowState) -> tuple[bool, str]:
    """Check if required files exist and are valid.

    Args:
        state: Current workflow state

    Returns:
        Tuple of (is_valid, error_message)
    """
    if state.input_mode == "asa":
        if state.asa_file is None:
            return False, "ASA file not selected"

        asa_path = resolve_path(state, state.asa_file)
        if asa_path is None or not asa_path.exists():
            return False, f"ASA file not found: {asa_path}"

        # Validate .npz structure
        try:
            data = np.load(asa_path, allow_pickle=True)
            required_keys = ["spike", "t"]
            missing = [k for k in required_keys if k not in data.files]
            if missing:
                return False, f"ASA file missing required keys: {missing}"
        except Exception as e:
            return False, f"Failed to load ASA file: {e}"

    elif state.input_mode == "neuron_traj":
        if state.neuron_file is None:
            return False, "Neuron file not selected"
        if state.traj_file is None:
            return False, "Trajectory file not selected"

        neuron_path = resolve_path(state, state.neuron_file)
        traj_path = resolve_path(state, state.traj_file)

        if neuron_path is None or not neuron_path.exists():
            return False, f"Neuron file not found: {neuron_path}"
        if traj_path is None or not traj_path.exists():
            return False, f"Trajectory file not found: {traj_path}"

    return True, ""


def validate_preprocessing(state: WorkflowState) -> tuple[bool, str]:
    """Check if preprocessing is complete.

    Args:
        state: Current workflow state

    Returns:
        Tuple of (is_valid, error_message)
    """
    if state.preprocess_method == "none":
        # Need raw data loaded
        if state.embed_data is None:
            return False, "No data loaded"
    else:
        # Need preprocessed data
        if state.embed_data is None:
            return False, "Preprocessing not complete"

    return True, ""
