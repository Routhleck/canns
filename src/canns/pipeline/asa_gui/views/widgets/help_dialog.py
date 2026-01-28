"""Help dialog widget."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QTextBrowser


class HelpDialog(QDialog):
    def __init__(self, title: str, markdown: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(680, 520)

        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(markdown)
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def show_help_dialog(parent, title: str, markdown: str) -> None:
    dialog = HelpDialog(title=title, markdown=markdown, parent=parent)
    dialog.exec()
