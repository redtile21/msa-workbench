from qt_material import apply_stylesheet
from PySide6.QtGui import QColor

# Define colors for StatusBadge to ensure it works with qdarktheme
COLOR_SUCCESS = QColor("#28a745")  # Bootstrap green
COLOR_WARNING = QColor("#ffc107")  # Bootstrap yellow
COLOR_ERROR = QColor("#dc3545")    # Bootstrap red
COLOR_INFO = QColor("#17a2b8")     # Bootstrap cyan
COLOR_TEXT_LIGHT = QColor("#6c757d") # Bootstrap gray for default/light text

def apply_theme(app):
    """
    Applies a theme to the application.
    """
    apply_stylesheet(app, theme='dark_blue.xml')