import sys
import os
from PySide6.QtWidgets import QApplication
from msa_workbench.ui.main_window import MainWindow
from msa_workbench.ui.theme import apply_theme

def main():
    print("DEBUG: msa_workbench.main.py main() function executed.")
    print("Starting QApplication...")
    app = QApplication(sys.argv)
    apply_theme(app)
    print("Creating MainWindow...")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    window = MainWindow(project_root=project_root)
    print("Showing MainWindow...")
    window.show()
    print("Starting event loop...")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
