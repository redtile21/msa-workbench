from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor, QFont

# --- Color Palette ---
COLOR_BACKGROUND = QColor("#FFFFFF")
COLOR_BACKGROUND_ALT = QColor("#F0F4F8")
COLOR_TEXT = QColor("#212529")
COLOR_TEXT_LIGHT = QColor("#6c757d")
COLOR_BORDER = QColor("#DEE2E6")

COLOR_PRIMARY = QColor("#007BFF")
COLOR_PRIMARY_LIGHT = QColor("#4DA3FF")
COLOR_PRIMARY_DARK = QColor("#0056B3")

COLOR_SUCCESS = QColor("#28A745")
COLOR_WARNING = QColor("#FFC107")
COLOR_ERROR = QColor("#DC3545")
COLOR_INFO = QColor("#17A2B8")


def apply_theme(app: QApplication):
    """Applies the custom theme to the QApplication instance."""
    
    # Set standard palette
    palette = QPalette()
    palette.setColor(QPalette.Window, COLOR_BACKGROUND)
    palette.setColor(QPalette.WindowText, COLOR_TEXT)
    palette.setColor(QPalette.Base, COLOR_BACKGROUND)
    palette.setColor(QPalette.AlternateBase, COLOR_BACKGROUND_ALT)
    palette.setColor(QPalette.ToolTipBase, COLOR_BACKGROUND)
    palette.setColor(QPalette.ToolTipText, COLOR_TEXT)
    palette.setColor(QPalette.Text, COLOR_TEXT)
    palette.setColor(QPalette.Button, COLOR_BACKGROUND_ALT)
    palette.setColor(QPalette.ButtonText, COLOR_TEXT)
    palette.setColor(QPalette.BrightText, COLOR_ERROR)
    palette.setColor(QPalette.Link, COLOR_PRIMARY)
    palette.setColor(QPalette.Highlight, COLOR_PRIMARY)
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    # Set font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Apply stylesheet
    stylesheet = f"""
        QMainWindow {{
            background-color: {COLOR_BACKGROUND.name()};
        }}
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {COLOR_BORDER.name()};
            border-radius: 4px;
            margin-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            left: 10px;
        }}
        QTabWidget::pane {{
            border-top: 1px solid {COLOR_BORDER.name()};
        }}
        QTabBar::tab {{
            padding: 8px 16px;
            font-weight: bold;
            color: {COLOR_TEXT_LIGHT.name()};
            background-color: transparent;
            border: none;
            border-bottom: 2px solid transparent;
        }}
        QTabBar::tab:selected {{
            color: {COLOR_PRIMARY.name()};
            border-bottom: 2px solid {COLOR_PRIMARY.name()};
        }}
        QTableView {{
            border: 1px solid {COLOR_BORDER.name()};
            gridline-color: {COLOR_BORDER.name()};
            alternate-background-color: {COLOR_BACKGROUND_ALT.name()};
        }}
        QHeaderView::section {{
            background-color: {COLOR_BACKGROUND_ALT.name()};
            padding: 4px;
            border: 1px solid {COLOR_BORDER.name()};
            font-weight: bold;
        }}
        QPushButton {{
            padding: 8px 16px;
            border: 1px solid {COLOR_BORDER.name()};
            border-radius: 4px;
            background-color: {COLOR_BACKGROUND_ALT.name()};
        }}
        QPushButton:hover {{
            background-color: #E2E6EA;
        }}
        QPushButton:pressed {{
            background-color: #D4DAE0;
        }}
        QPushButton[cssClass="primary"] {{
            background-color: {COLOR_PRIMARY.name()};
            color: white;
            font-weight: bold;
            border-color: {COLOR_PRIMARY.name()};
        }}
        QPushButton[cssClass="primary"]:hover {{
            background-color: {COLOR_PRIMARY_LIGHT.name()};
        }}
        QPushButton[cssClass="primary"]:pressed {{
            background-color: {COLOR_PRIMARY_DARK.name()};
        }}
        QLineEdit, QTextEdit, QSpinBox, QComboBox {{
            padding: 5px;
            border: 1px solid {COLOR_BORDER.name()};
            border-radius: 4px;
        }}
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
            border: 1px solid {COLOR_PRIMARY.name()};
        }}
    """
    app.setStyleSheet(stylesheet)
