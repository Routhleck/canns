"""Main window for ASA GUI."""

from __future__ import annotations

import sys

from PySide6.QtCore import QSettings, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .controllers import AnalysisController, PreprocessController
from .core import PipelineRunner, StateManager, WorkerManager
from .resources import load_theme_qss, resource_path
from .views.pages.analysis_page import AnalysisPage
from .views.pages.preprocess_page import PreprocessPage
from .views.widgets.popup_combo import PopupComboBox


class LogoLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:  # pragma: no cover - UI callback
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


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
        self.logo_label = LogoLabel()
        self.logo_label.setCursor(Qt.PointingHandCursor)
        logo_height = 32 if sys.platform == "darwin" else 40
        logo_pixmap = self._load_logo_pixmap(logo_height)
        if logo_pixmap is not None:
            self.logo_label.setPixmap(logo_pixmap)
            self.logo_label.setFixedSize(logo_pixmap.size())
            self.logo_label.setAlignment(Qt.AlignCenter)
            nav.addWidget(self.logo_label)
            self.logo_label.clicked.connect(self._open_homepage)
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
        self.theme_switch = PopupComboBox()
        self.theme_switch.addItem("Light", userData="Light")
        self.theme_switch.addItem("Dark", userData="Dark")
        self.theme_switch.currentIndexChanged.connect(self._apply_theme)
        self.theme_switch.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.theme_switch.setMinimumWidth(90)
        nav.addWidget(self.theme_switch)
        self.lang_switch = PopupComboBox()
        self.lang_switch.addItem("EN", userData="en")
        self.lang_switch.addItem("中文", userData="zh")
        self.lang_switch.currentIndexChanged.connect(self._apply_language)
        self.lang_switch.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.lang_switch.setMinimumWidth(90)
        nav.addWidget(self.lang_switch)
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
        self._init_language()
        self._init_icons(str(self.theme_switch.currentData() or "Light"))
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
        theme = str(self._settings.value("theme", "Light"))
        idx = 0 if theme.lower().startswith("light") else 1
        self.theme_switch.setCurrentIndex(idx)
        self._apply_theme()

    def _init_language(self) -> None:
        lang = str(self._settings.value("lang", "en")).lower()
        idx = 0 if lang.startswith("en") else 1
        self.lang_switch.setCurrentIndex(idx)
        self._apply_language()

    def _apply_theme(self) -> None:
        try:
            theme = str(self.theme_switch.currentData() or "Light")
            qss = load_theme_qss(theme)
            QApplication.instance().setStyleSheet(qss)
            self._settings.setValue("theme", theme)
            self._init_icons(theme)
        except Exception:
            pass

    def _apply_language(self) -> None:
        lang = self.lang_switch.currentData() or "en"
        self._settings.setValue("lang", str(lang))
        is_zh = str(lang).lower().startswith("zh")
        self.btn_preprocess.setText("预处理" if is_zh else "Preprocess")
        self.btn_analysis.setText("分析" if is_zh else "Analysis")
        self.preprocess_page.apply_language(str(lang))
        self.analysis_page.apply_language(str(lang))

    def _init_icons(self, theme: str) -> None:
        try:
            import qtawesome as qta

            color = "#34d399" if str(theme).lower().startswith("dark") else "#10b981"
            self.btn_preprocess.setIcon(qta.icon("fa5s.sliders-h", color=color))
            self.btn_analysis.setIcon(qta.icon("fa5s.chart-area", color=color))
        except Exception:
            pass

    def _load_logo_pixmap(self, height: int):
        logo_path = resource_path("logo.svg")
        if not logo_path.exists():
            from pathlib import Path

            logo_path = Path(__file__).resolve().parents[4] / "images" / "logo.svg"
            if not logo_path.exists():
                return None
        icon = QIcon(str(logo_path))
        if icon.isNull():
            return None
        size = max(64, int(height) * 4)
        pixmap = icon.pixmap(size, size)
        return pixmap.scaledToHeight(int(height), Qt.SmoothTransformation)

    def _open_homepage(self) -> None:
        QDesktopServices.openUrl(QUrl("https://github.com/Routhleck/canns"))


ASAGuiApp = MainWindow
