from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class Trainer(ABC):
    """Abstract base class for training utilities in CANNs."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        show_iteration_progress: bool = False,
        compiled_prediction: bool = True,
    ) -> None:
        self.model = model
        self.show_iteration_progress = show_iteration_progress
        self.compiled_prediction = compiled_prediction

    @abstractmethod
    def train(self, train_data: Iterable[Any]) -> None:
        """Train the associated model with the provided dataset."""

    @abstractmethod
    def predict(self, pattern: Any, *args: Any, **kwargs: Any) -> Any:
        """Predict an output for a single pattern."""

    def predict_batch(self, patterns: Iterable[Any], *args: Any, **kwargs: Any) -> list[Any]:
        """Predict outputs for multiple patterns using ``predict``."""
        return [self.predict(pattern, *args, **kwargs) for pattern in patterns]

    def configure_progress(
        self,
        *,
        show_iteration_progress: bool | None = None,
        compiled_prediction: bool | None = None,
    ) -> None:
        """Update progress-related flags for derived trainers."""
        if show_iteration_progress is not None:
            self.show_iteration_progress = show_iteration_progress
        if compiled_prediction is not None:
            self.compiled_prediction = compiled_prediction
