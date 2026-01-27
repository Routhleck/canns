"""Main window for ASA GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QMainWindow,
)

from .controllers import AnalysisController, PreprocessController
from .core import PipelineRunner, StateManager, WorkerManager
from .views.pages.analysis_page import AnalysisPage
from .views.pages.preprocess_page import PreprocessPage


class MainWindow(QMainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ASA GUI")
        self.resize(1200, 800)

        self._state_manager = StateManager()
        self._runner = PipelineRunner()
        self._workers = WorkerManager()

        self._preprocess_controller = PreprocessController(self._state_manager, self._runner)
        self._analysis_controller = AnalysisController(self._state_manager, self._runner)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        nav = QHBoxLayout()
        self.btn_preprocess = QPushButton("Preprocess")
        self.btn_analysis = QPushButton("Analysis")
        self.btn_preprocess.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        self.btn_analysis.clicked.connect(lambda: self._stack.setCurrentIndex(1))
        nav.addWidget(self.btn_preprocess)
        nav.addWidget(self.btn_analysis)
        nav.addStretch(1)
        layout.addLayout(nav)

        self._stack = QStackedWidget()
        self.preprocess_page = PreprocessPage(self._preprocess_controller, self._workers)
        self.analysis_page = AnalysisPage(self._analysis_controller, self._workers)

        self._stack.addWidget(self.preprocess_page)
        self._stack.addWidget(self.analysis_page)
        layout.addWidget(self._stack, 1)

        self.preprocess_page.preprocess_completed.connect(self._on_preprocess_completed)

        self.setCentralWidget(root)

    def _go_analysis(self) -> None:
        self._stack.setCurrentIndex(1)

    def _on_preprocess_completed(self) -> None:
        self._stack.setCurrentIndex(1)
        self.analysis_page.load_state(self._state_manager.state)


ASAGuiApp = MainWindow
