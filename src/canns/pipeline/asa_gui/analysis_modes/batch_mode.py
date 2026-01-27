"""Batch analysis mode placeholder."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .base import AbstractAnalysisMode


class BatchMode(AbstractAnalysisMode):
    name = "batch"
    display_name = "Batch"

    def create_params_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Batch mode parameters will be added here."))
        return widget

    def collect_params(self) -> dict:
        return {}
