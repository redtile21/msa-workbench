import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QTableView,
    QLabel,
    QFormLayout,
    QComboBox,
    QListWidget,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QMessageBox,
    QTabWidget,
    QSplitter,
    QTextEdit,
    QHeaderView
)
from PySide6.QtCore import Qt

from msa_workbench.ui.dataframe_model import DataFrameModel
from msa_workbench.engine.msa_engine import MSAConfig, run_crossed_msa, MSAResult
from msa_workbench.ui.widgets.mpl_canvas import MplCanvas
from msa_workbench.ui.widgets.status_badge import StatusBadge
from msa_workbench.ui.widgets.indentation_delegate import IndentationDelegate
from msa_workbench.reporting.pdf_report import save_pdf_report
from msa_workbench.plotting import get_variability_chart, get_stddev_chart
from msa_workbench.reporting.analysis_notes import get_variation_impact_analysis
from msa_workbench.ui.pages.builder_page import BuilderPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MSA Workbench")
        self.setGeometry(100, 100, 1400, 800)

        # State
        self.df = None
        self.result: MSAResult | None = None

        # Main layout
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)

        # --- Create Pages/Tabs ---
        self.builder_page = BuilderPage()
        self.analysis_page = QWidget()
        self.results_page = QWidget()

        self.main_tabs.addTab(self.builder_page, "MSA Builder")
        self.main_tabs.addTab(self.analysis_page, "MSA Analysis")
        self.main_tabs.addTab(self.results_page, "Analysis Results")

        self._setup_analysis_page()
        self._setup_results_page()

    def _setup_analysis_page(self):
        # Main layout for the analysis tab
        page_layout = QHBoxLayout(self.analysis_page)
        splitter = QSplitter(Qt.Horizontal)
        page_layout.addWidget(splitter)

        # --- Left panel for controls ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # --- Right panel for data preview ---
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        splitter.addWidget(controls_widget)
        splitter.addWidget(preview_widget)
        splitter.setSizes([450, 550])

        # --- Data Input Group ---
        data_group = QGroupBox("Data Input")
        data_layout = QVBoxLayout(data_group)
        self.load_button = QPushButton("Load CSV...")
        self.load_button.clicked.connect(self.load_csv)
        self.df_info_label = QLabel("Load a CSV file to begin.")
        data_layout.addWidget(self.load_button)
        data_layout.addWidget(self.df_info_label)
        controls_layout.addWidget(data_group)

        # --- Model Configuration Group ---
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout(model_group)
        self.response_combo = QComboBox()
        self.response_combo.setToolTip("Select the measurement column.")
        self.response_suggestion_label = QLabel("Suggested")
        self.response_suggestion_label.setStyleSheet("color: #6c757d; font-style: italic;")
        model_layout.addRow("Response:", self.response_combo)
        model_layout.addRow("", self.response_suggestion_label)

        self.factors_button = QPushButton("Select Factors...")
        self.factors_button.setToolTip("Choose 2-4 categorical columns for the analysis.")
        self.factors_button.clicked.connect(self.select_factors)
        self.factors_label = QLabel("None selected")
        self.factors_suggestion_label = QLabel("Suggested")
        self.factors_suggestion_label.setStyleSheet("color: #6c757d; font-style: italic;")
        model_layout.addRow("Factors:", self.factors_button)
        model_layout.addRow("", self.factors_label)
        model_layout.addRow("", self.factors_suggestion_label)
        self.selected_factors = []

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Crossed", "Main Effects"])
        self.model_type_combo.setToolTip("Crossed model includes interactions (2-3 factors).\nMain Effects model is for 4 factors or to exclude interactions.")
        model_layout.addRow("Model Type:", self.model_type_combo)

        self.part_combo = QComboBox()
        self.part_combo.setToolTip("Select the column representing the part/sample.")
        self.part_suggestion_label = QLabel("Suggested")
        self.part_suggestion_label.setStyleSheet("color: #6c757d; font-style: italic;")
        model_layout.addRow("Part:", self.part_combo)
        model_layout.addRow("", self.part_suggestion_label)

        self.operator_combo = QComboBox()
        self.operator_combo.setToolTip("Select the column representing the operator/appraiser.")
        self.operator_suggestion_label = QLabel("Suggested")
        self.operator_suggestion_label.setStyleSheet("color: #6c757d; font-style: italic;")
        model_layout.addRow("Operator:", self.operator_combo)
        model_layout.addRow("", self.operator_suggestion_label)
        controls_layout.addWidget(model_group)

        # --- Specification Limits Group ---
        spec_group = QGroupBox("Specification Limits (Optional)")
        spec_layout = QFormLayout(spec_group)
        self.lsl_input = QLineEdit()
        self.lsl_input.setToolTip("Lower Specification Limit.\nUsed with USL to calculate %Tolerance.")
        spec_layout.addRow("LSL:", self.lsl_input)
        
        self.usl_input = QLineEdit()
        self.usl_input.setToolTip("Upper Specification Limit.\nUsed with LSL to calculate %Tolerance.")
        spec_layout.addRow("USL:", self.usl_input)

        self.tolerance_input = QLineEdit()
        self.tolerance_input.setToolTip("Manual Tolerance value.\nOverrides USL-LSL if provided.")
        spec_layout.addRow("Tolerance:", self.tolerance_input)
        controls_layout.addWidget(spec_group)

        # --- Actions Group ---
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setProperty("cssClass", "primary")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setEnabled(False)
        actions_layout.addWidget(self.run_button)
        controls_layout.addWidget(actions_group)
        
        controls_layout.addStretch()

        # --- Preview Table ---
        preview_layout.addWidget(QLabel("Data Preview"))
        self.table_view = QTableView()
        self.df_model = DataFrameModel()
        self.table_view.setModel(self.df_model)
        preview_layout.addWidget(self.table_view)

    def _setup_results_page(self):
        layout = QVBoxLayout(self.results_page)
        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)

        summary_tab = QWidget()
        warnings_tab = QWidget()
        var_comp_tab = QWidget()
        anova_tab = QWidget()
        charts_tab = QWidget()
        
        self.results_tabs.addTab(summary_tab, "Summary")
        self.results_tabs.addTab(warnings_tab, "Warnings")
        self.results_tabs.addTab(var_comp_tab, "Variance Components")
        self.results_tabs.addTab(anova_tab, "ANOVA")
        self.results_tabs.addTab(charts_tab, "Charts")

        # Summary Tab
        summary_layout = QFormLayout(summary_tab)
        self.grr_sv_label = QLabel("N/A")
        self.grr_tol_label = QLabel("N/A")
        self.ndc_label = QLabel("N/A")
        self.interpretation_badge = StatusBadge("N/A")
        summary_layout.addRow("GRR % Study Var:", self.grr_sv_label)
        summary_layout.addRow("GRR % Tolerance:", self.grr_tol_label)
        summary_layout.addRow("NDC:", self.ndc_label)
        summary_layout.addRow("Interpretation:", self.interpretation_badge)

        # Warnings Tab
        warnings_layout = QVBoxLayout(warnings_tab)
        self.warnings_text = QTextEdit()
        self.warnings_text.setReadOnly(True)
        warnings_layout.addWidget(self.warnings_text)

        # Var Comp Tab
        var_comp_layout = QVBoxLayout(var_comp_tab)
        var_comp_title = QLabel("Variance Components (Study Variation Breakdown)")
        var_comp_title.setStyleSheet("font-weight: bold;")
        var_comp_layout.addWidget(var_comp_title)
        self.var_comp_table = QTableView()
        self.var_comp_model = DataFrameModel()
        self.var_comp_table.setModel(self.var_comp_model)
        self.var_comp_table.setItemDelegateForColumn(0, IndentationDelegate(self))
        var_comp_layout.addWidget(self.var_comp_table)
        self.impact_analysis_text = QTextEdit()
        self.impact_analysis_text.setReadOnly(True)
        var_comp_layout.addWidget(self.impact_analysis_text)

        # ANOVA Tab
        anova_layout = QVBoxLayout(anova_tab)
        anova_title = QLabel("ANOVA Table (Model Fit)")
        anova_title.setStyleSheet("font-weight: bold;")
        anova_layout.addWidget(anova_title)
        self.anova_table = QTableView()
        self.anova_model = DataFrameModel()
        self.anova_table.setModel(self.anova_model)
        self.anova_table.setItemDelegateForColumn(0, IndentationDelegate(self))
        anova_layout.addWidget(self.anova_table)

        # Charts Tab
        charts_layout = QVBoxLayout(charts_tab)
        self.variability_chart_canvas = MplCanvas(self)
        self.stddev_chart_canvas = MplCanvas(self)
        charts_layout.addWidget(self.variability_chart_canvas)
        charts_layout.addWidget(self.stddev_chart_canvas)
        
        # Export button
        self.export_button = QPushButton("Export PDF...")
        self.export_button.clicked.connect(self.export_pdf)
        self.export_button.setEnabled(False)
        layout.addWidget(self.export_button)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.df = pd.read_csv(path)
                self.df_model.setDataFrame(self.df)
                self.df_info_label.setText(f"Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
                self._update_config_options()
                self.run_button.setEnabled(True)
                self._auto_populate_fields()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

    def _update_config_options(self):
        if self.df is None:
            return
        
        cols = self.df.columns.tolist()
        self.response_combo.clear()
        self.response_combo.addItems(cols)

    def _auto_populate_fields(self):
        if self.df is None:
            return

        cols = self.df.columns.tolist()
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Hide all suggestion labels initially
        for label in [self.response_suggestion_label, self.factors_suggestion_label, self.part_suggestion_label, self.operator_suggestion_label]:
            label.setVisible(False)

        # 1. Auto-select Response
        response_candidates = [c for c in ["measurement", "result", "value", "response", "y"] if c.lower() in [col.lower() for col in cols]]
        if response_candidates:
            self.response_combo.setCurrentText(response_candidates[0])
            self.response_suggestion_label.setVisible(True)
        elif numeric_cols:
            self.response_combo.setCurrentText(numeric_cols[-1])
            self.response_suggestion_label.setVisible(True)
        
        # 2. Suggest Factors
        excluded_patterns = ["date", "time", "timestamp", "run", "order", "id"]
        suggested_factors = [c for c in categorical_cols if not any(pat in c.lower() for pat in excluded_patterns)]
        if suggested_factors:
            self.selected_factors = suggested_factors[:4] # Limit to 4
            self.factors_label.setText(", ".join(self.selected_factors))
            self.factors_suggestion_label.setVisible(True)
            self.part_combo.clear()
            self.part_combo.addItems(self.selected_factors)
            self.operator_combo.clear()
            self.operator_combo.addItems(self.selected_factors)

        # 3. Auto-select Part
        part_candidates = [c for c in self.selected_factors if any(p in c.lower() for p in ["part", "sample", "material", "specimen", "item", "unit", "lot"])]
        if part_candidates:
            self.part_combo.setCurrentText(part_candidates[0])
            self.part_suggestion_label.setVisible(True)

        # 4. Auto-select Operator
        operator_candidates = [c for c in self.selected_factors if any(o in c.lower() for o in ["operator", "user", "tech", "technician", "appraiser", "analyst"])]
        if operator_candidates:
            self.operator_combo.setCurrentText(operator_candidates[0])
            self.operator_suggestion_label.setVisible(True)
        
        # Update model type based on factor count
        if len(self.selected_factors) >= 4:
            self.model_type_combo.setCurrentText("Main Effects")
            self.model_type_combo.setEnabled(False)
        else:
            self.model_type_combo.setEnabled(True)

    def select_factors(self):
        # Hide suggestion labels when user manually changes selection
        self.factors_suggestion_label.setVisible(False)
        self.part_suggestion_label.setVisible(False)
        self.operator_suggestion_label.setVisible(False)

        if self.df is None:
            QMessageBox.warning(self, "Warning", "Load data first.")
            return

        cols = [c for c in self.df.columns if c != self.response_combo.currentText()]
        dialog = QListWidgetDialog(cols, self.selected_factors, self)
        if dialog.exec():
            self.selected_factors = dialog.selected_items()
            self.factors_label.setText(", ".join(self.selected_factors))
            self.part_combo.clear()
            self.part_combo.addItems(self.selected_factors)
            self.operator_combo.clear()
            self.operator_combo.addItems(self.selected_factors)

            if len(self.selected_factors) >= 4:
                self.model_type_combo.setCurrentText("Main Effects")
                self.model_type_combo.setEnabled(False)
            else:
                self.model_type_combo.setEnabled(True)

    def run_analysis(self):
        if self.df is None:
            return

        try:
            lsl = float(self.lsl_input.text()) if self.lsl_input.text() else None
            usl = float(self.usl_input.text()) if self.usl_input.text() else None
            tolerance = float(self.tolerance_input.text()) if self.tolerance_input.text() else None

            config = MSAConfig(
                response_col=self.response_combo.currentText(),
                factor_cols=self.selected_factors,
                part_col=self.part_combo.currentText(),
                operator_col=self.operator_combo.currentText(),
                lsl=lsl,
                usl=usl,
                tolerance=tolerance,
                model_type=self.model_type_combo.currentText().lower()
            )
            self.result = run_crossed_msa(self.df.copy(), config)
            self._update_results_ui()
            self.export_button.setEnabled(True)
            self.main_tabs.setCurrentWidget(self.results_page)
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to run MSA: {e}")

    def _update_results_ui(self):
        if self.result is None:
            return

        def _format_sig(val):
            if val is None or not isinstance(val, (int, float)):
                return "N/A"
            return "{:.4g}".format(val)

        # Summary
        summary = self.result.grr_summary
        interpretation = summary.interpretation.lower()
        status = "default"
        if "excellent" in interpretation or "good" in interpretation:
            status = "good"
        elif "acceptable" in interpretation or "marginal" in interpretation:
            status = "average"
        elif "poor" in interpretation or "unacceptable" in interpretation:
            status = "poor"

        self.grr_sv_label.setText(f"{_format_sig(summary.total_gage_rr_pct_study_var)}%")
        self.grr_tol_label.setText(f"{_format_sig(summary.total_gage_rr_pct_tolerance)}%" if summary.total_gage_rr_pct_tolerance is not None else "N/A")
        self.ndc_label.setText(_format_sig(summary.ndc))
        self.interpretation_badge.set_text_and_status(summary.interpretation, status)

        # Warnings
        self.warnings_text.setText("\n".join(self.result.warnings) or "No warnings.")

        # Var Comp
        var_comp_df = pd.DataFrame(self.result.var_components)
        for col in var_comp_df.columns:
            if pd.api.types.is_numeric_dtype(var_comp_df[col]):
                var_comp_df[col] = var_comp_df[col].apply(_format_sig)
        self.var_comp_model.setDataFrame(var_comp_df)
        self.var_comp_table.resizeColumnsToContents()
        
        impacts = get_variation_impact_analysis(self.result)
        impact_html = ""
        for type_, msg in impacts:
            color = "green" if type_ == "success" else ("red" if type_ == "error" else "black")
            impact_html += f'<p style="color:{color};">{msg}</p>'
        self.impact_analysis_text.setHtml(impact_html)

        # ANOVA
        anova_df = pd.DataFrame(self.result.anova_table)
        for col in anova_df.columns:
            if pd.api.types.is_numeric_dtype(anova_df[col]):
                anova_df[col] = anova_df[col].apply(_format_sig)
        self.anova_model.setDataFrame(anova_df)
        self.anova_table.resizeColumnsToContents()

        # Charts
        self.variability_chart_canvas.figure.clear()
        ax_var = self.variability_chart_canvas.figure.add_subplot(111)
        get_variability_chart(self.result, ax_var)
        self.variability_chart_canvas.draw()

        self.stddev_chart_canvas.figure.clear()
        ax_std = self.stddev_chart_canvas.figure.add_subplot(111)
        get_stddev_chart(self.result, ax_std)
        self.stddev_chart_canvas.draw()


    def export_pdf(self):
        if self.result is None:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "msa_report.pdf", "PDF Files (*.pdf)")
        if path:
            try:
                save_pdf_report(self.result, path)
                QMessageBox.information(self, "Success", f"Report saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save PDF: {e}")


class QListWidgetDialog(QDialog):
    def __init__(self, items, selected_items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Factors")
        
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.list_widget.addItems(items)
        for item in selected_items:
            find_items = self.list_widget.findItems(item, Qt.MatchExactly)
            if find_items:
                find_items[0].setSelected(True)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def selected_items(self):
        return [item.text() for item in self.list_widget.selectedItems()]
