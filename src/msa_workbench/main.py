import sys
from PySide6.QtWidgets import QApplication
from msa_workbench.ui.main_window import MainWindow
from msa_workbench.ui.theme import apply_theme

def main():
    print("Starting QApplication...")
    app = QApplication(sys.argv)
    # apply_theme(app)
    print("Creating MainWindow...")
    window = MainWindow()
    print("Showing MainWindow...")
    window.show()
    print("Starting event loop...")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
