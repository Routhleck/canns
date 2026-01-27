"""Base classes for analysis modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PySide6.QtWidgets import QWidget


class AbstractAnalysisMode(ABC):
    name: str
    display_name: str

    @abstractmethod
    def create_params_widget(self) -> QWidget:
        """Create and return the parameter editor widget."""

    @abstractmethod
    def collect_params(self) -> dict[str, Any]:
        """Collect parameters from the widget into a dict."""

    def apply_preset(self, preset: str) -> None:
        """Apply preset hints (grid/hd) to parameters."""

    def apply_ranges(self, neuron_count: int | None, total_steps: int | None) -> None:
        """Apply neuron/time ranges based on loaded data."""
