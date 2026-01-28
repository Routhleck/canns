"""Main window for ASA GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QMainWindow,
)

from .controllers import AnalysisController, PreprocessController
from .core import PipelineRunner, StateManager, WorkerManager
from .resources import load_theme_qss
from .views.pages.analysis_page import AnalysisPage
from .views.pages.preprocess_page import PreprocessPage


class MainWindow(QMainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ASA GUI")
        self.resize(1200, 800)

        self._settings = QSettings("canns", "asa_gui")

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
        self.btn_preprocess.setObjectName("navButton")
        self.btn_preprocess.setCheckable(True)
        self.btn_analysis = QPushButton("Analysis")
        self.btn_analysis.setObjectName("navButton")
        self.btn_analysis.setCheckable(True)
        self.btn_preprocess.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        self.btn_analysis.clicked.connect(lambda: self._stack.setCurrentIndex(1))
        nav.addWidget(self.btn_preprocess)
        nav.addWidget(self.btn_analysis)
        nav.addStretch(1)
        self.theme_switch = QComboBox()
        self.theme_switch.addItems(["Light", "Dark"])
        self.theme_switch.currentTextChanged.connect(self._apply_theme)
        nav.addWidget(self.theme_switch)
        layout.addLayout(nav)

        self._stack = QStackedWidget()
        self.preprocess_page = PreprocessPage(self._preprocess_controller, self._workers)
        self.analysis_page = AnalysisPage(self._analysis_controller, self._workers)

        self._stack.addWidget(self.preprocess_page)
        self._stack.addWidget(self.analysis_page)
        layout.addWidget(self._stack, 1)

        self.preprocess_page.preprocess_completed.connect(self._on_preprocess_completed)
        self._stack.currentChanged.connect(self._sync_nav)

        self.setCentralWidget(root)
        self._init_theme()
        self._init_icons(str(self.theme_switch.currentText()))
        self._sync_nav(self._stack.currentIndex())

    def _go_analysis(self) -> None:
        self._stack.setCurrentIndex(1)

    def _on_preprocess_completed(self) -> None:
        self._stack.setCurrentIndex(1)
        self.analysis_page.load_state(self._state_manager.state)

    def _sync_nav(self, index: int) -> None:
        self.btn_preprocess.setChecked(index == 0)
        self.btn_analysis.setChecked(index == 1)

    def _init_theme(self) -> None:
        theme = self._settings.value("theme", "Light")
        self.theme_switch.setCurrentText(str(theme))
        self._apply_theme(str(theme))

    def _apply_theme(self, theme: str) -> None:
        try:
            qss = load_theme_qss(theme)
            QApplication.instance().setStyleSheet(qss)
            self._settings.setValue("theme", theme)
            self._init_icons(theme)
        except Exception:
            pass

    def _init_icons(self, theme: str) -> None:
        try:
            import qtawesome as qta

            color = "#34d399" if str(theme).lower().startswith("dark") else "#10b981"
            self.btn_preprocess.setIcon(qta.icon("fa5s.sliders-h", color=color))
            self.btn_analysis.setIcon(qta.icon("fa5s.chart-area", color=color))
        except Exception:
            pass


ASAGuiApp = MainWindow
