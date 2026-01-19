from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QTextEdit,
    QSpinBox,
    QRadioButton,
    QCheckBox,
    QPushButton,
    QTableView,
    QLabel,
    QMessageBox,
    QGroupBox,
    QSplitter,
    QFileDialog
)
from PySide6.QtCore import Qt
import pandas as pd

from msa_workbench.ui.dataframe_model import DataFrameModel
from msa_workbench.builder.run_table import (
    validate_factors,
    build_run_table,
    sort_run_table_left_to_right,
    randomize_run_table,
    export_run_table_csv,
)

class BuilderPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.df_generated: Optional[pd.DataFrame] = None

        # Main layout
        main_splitter = QSplitter(Qt.Horizontal)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(main_splitter)

        # --- Left Panel (Controls) ---
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        main_splitter.addWidget(controls_panel)

        # --- Right Panel (Preview) ---
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        main_splitter.addWidget(preview_panel)
        
        main_splitter.setSizes([400, 600])

        # Controls
        self._create_factor_widgets(controls_layout)
        self._create_options_widgets(controls_layout)
        
        # Preview
        self._create_preview_widgets(preview_layout)
        
        controls_layout.addStretch()

    def _create_factor_widgets(self, layout):
        group = QGroupBox("Factors (1-4)")
        form_layout = QFormLayout(group)
        
        self.factor_widgets = []
        for i in range(4):
            name_edit = QLineEdit()
            levels_edit = QTextEdit()
            levels_edit.setPlaceholderText("One level per line")
            levels_edit.setFixedHeight(80)
            
            form_layout.addRow(f"Factor {i+1} Name:", name_edit)
            form_layout.addRow(f"Factor {i+1} Levels:", levels_edit)
            self.factor_widgets.append({"name_widget": name_edit, "levels_widget": levels_edit})
            
        layout.addWidget(group)

    def _create_options_widgets(self, layout):
        # Replicates
        replicates_group = QGroupBox("Replicates")
        replicates_layout = QFormLayout(replicates_group)
        self.replicates_spinbox = QSpinBox()
        self.replicates_spinbox.setMinimum(1)
        self.replicates_spinbox.setValue(3)
        replicates_layout.addRow("Replicates per combination:", self.replicates_spinbox)
        layout.addWidget(replicates_group)

        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)
        self.sort_left_to_right_radio = QRadioButton("Left-to-right (by factor)")
        self.sort_randomized_radio = QRadioButton("Randomized")
        self.sort_left_to_right_radio.setChecked(True)
        
        self.seed_checkbox = QCheckBox("Use seed")
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setMaximum(999999)
        self.seed_spinbox.setEnabled(False)
        self.seed_checkbox.toggled.connect(self.seed_spinbox.setEnabled)
        
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(self.seed_checkbox)
        seed_layout.addWidget(self.seed_spinbox)

        output_layout.addWidget(self.sort_left_to_right_radio)
        output_layout.addWidget(self.sort_randomized_radio)
        output_layout.addLayout(seed_layout)
        layout.addWidget(output_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate / Update Table")
        self.generate_button.clicked.connect(self.generate_table)
        self.export_button = QPushButton("Export CSV...")
        self.export_button.clicked.connect(self.export_csv)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.export_button)
        layout.addLayout(button_layout)

    def _create_preview_widgets(self, layout):
        self.preview_label = QLabel("Generated Table Preview")
        self.info_label = QLabel("Combinations: 0, Total Rows: 0")
        
        self.table_view = QTableView()
        self.df_model = DataFrameModel()
        self.table_view.setModel(self.df_model)
        
        layout.addWidget(self.preview_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.table_view)

    def generate_table(self):
        try:
            factors = []
            for fw in self.factor_widgets:
                name = fw["name_widget"].text().strip()
                levels = [line.strip() for line in fw["levels_widget"].toPlainText().strip().split('\n') if line.strip()]
                if name and levels:
                    factors.append({"name": name, "levels": levels})
            
            validated_factors = validate_factors(factors)
            factor_names = [f["name"] for f in validated_factors]
            replicates = self.replicates_spinbox.value()
            
            df = build_run_table(validated_factors, replicates)
            
            if self.sort_randomized_radio.isChecked():
                seed = self.seed_spinbox.value() if self.seed_checkbox.isChecked() else None
                df = randomize_run_table(df, seed)
            else:
                df = sort_run_table_left_to_right(df, factor_names)

            self.df_generated = df
            self.df_model.setDataFrame(df)
            
            # Update info
            num_combinations = len(df) / replicates
            self.info_label.setText(f"Combinations: {int(num_combinations)}, Total Rows: {len(df)}")
            self.export_button.setEnabled(True)

        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            self.export_button.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")
            self.export_button.setEnabled(False)

    def export_csv(self):
        if self.df_generated is None:
            QMessageBox.warning(self, "Warning", "No table has been generated yet.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "msa_run_table.csv", "CSV Files (*.csv)")
        if path:
            try:
                export_run_table_csv(self.df_generated, path)
                QMessageBox.information(self, "Success", f"Table exported to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")