from PySide6.QtWidgets import QStyledItemDelegate
from PySide6.QtCore import Qt

class IndentationDelegate(QStyledItemDelegate):
    """
    A delegate to draw text with an indent based on a simple rule.
    Indents text if it contains a ':' character.
    """
    def paint(self, painter, option, index):
        text = index.model().data(index, Qt.DisplayRole)
        
        if ":" in text:
            # Simple rule: indent if it's an interaction term
            option.rect.adjust(20, 0, 0, 0)
        
        super().paint(painter, option, index)
