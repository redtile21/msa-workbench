from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from ..theme import COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR, COLOR_INFO, COLOR_TEXT_LIGHT

class StatusBadge(QLabel):
    """A label widget to display a status with a colored background."""
    
    STATUS_STYLES = {
        "good": f"background-color: {COLOR_SUCCESS.name()}; color: white; padding: 2px 6px; border-radius: 4px;",
        "average": f"background-color: {COLOR_WARNING.name()}; color: #212529; padding: 2px 6px; border-radius: 4px;",
        "poor": f"background-color: {COLOR_ERROR.name()}; color: white; padding: 2px 6px; border-radius: 4px;",
        "info": f"background-color: {COLOR_INFO.name()}; color: white; padding: 2px 6px; border-radius: 4px;",
        "default": f"background-color: {COLOR_TEXT_LIGHT.name()}; color: white; padding: 2px 6px; border-radius: 4px;",
    }

    def __init__(self, text="", status="default", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.set_status(status)

    def set_status(self, status: str):
        """Sets the status and updates the badge's style."""
        style = self.STATUS_STYLES.get(status.lower(), self.STATUS_STYLES["default"])
        self.setStyleSheet(style)

    def set_text_and_status(self, text: str, status: str):
        """Sets both the text and the status of the badge."""
        self.setText(text)
        self.set_status(status)
