import pandas as pd
import numpy as np
import os
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
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
    QHeaderView,
    QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QPixmap

from msa_workbench.ui.dataframe_model import DataFrameModel
from msa_workbench.engine.msa_engine import MSAConfig, run_msa, MSAResult
from msa_workbench.ui.widgets.status_badge import StatusBadge
from msa_workbench.ui.widgets.indentation_delegate import IndentationDelegate
from msa_workbench.reporting.pdf_report import save_pdf_report
from msa_workbench.reporting.pptx_report import save_pptx_report
from msa_workbench.plotting import get_variability_chart, get_stddev_chart
from msa_workbench.reporting.analysis_notes import get_variation_impact_analysis
from msa_workbench.ui.pages.builder_page import BuilderPage
from msa_workbench.ui.pages.charts_page import ChartsPage


class MainWindow(QMainWindow):
    def __init__(self, project_root: str):
        super().__init__()
        self.setWindowTitle("MSA Workbench")
        self.setGeometry(100, 100, 1400, 800)
        self.project_root = project_root

        # State
        self.df = None
        self.result: MSAResult | None = None

        # Timer for debounced chart rendering
        self._charts_rerender_timer = QTimer(self)
        self._charts_rerender_timer.setSingleShot(True)
        self._charts_rerender_timer.timeout.connect(self._render_charts)
        self._last_chart_render_state = None

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
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        # --- Right panel for data preview ---
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setSpacing(10)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        splitter.addWidget(controls_widget)
        splitter.addWidget(preview_widget)
        splitter.setSizes([450, 550])

        # --- Data Input Group ---
        data_group = QGroupBox("Data Input")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(10)
        self.load_button = QPushButton("Load CSV...")
        self.load_button.clicked.connect(self.load_csv)
        self.df_info_label = QLabel("Load a CSV file to begin.")
        data_layout.addWidget(self.load_button)
        data_layout.addWidget(self.df_info_label)
        controls_layout.addWidget(data_group)

        # --- Model Configuration Group ---
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout(model_group)
        model_layout.setVerticalSpacing(15)
        model_layout.setHorizontalSpacing(20)
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
        spec_layout.setVerticalSpacing(15)
        spec_layout.setHorizontalSpacing(20)
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
        actions_layout.setSpacing(10)
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

        self.results_tabs.currentChanged.connect(self._on_results_tab_changed)

        # Summary Tab
        summary_layout_main = QVBoxLayout(summary_tab)
        summary_scroll_area = QScrollArea()
        summary_scroll_area.setWidgetResizable(True)
        summary_layout_main.addWidget(summary_scroll_area)

        summary_content_widget = QWidget()
        summary_layout = QVBoxLayout(summary_content_widget)
        summary_layout.setSpacing(20) # Add some space between sections
        summary_scroll_area.setWidget(summary_content_widget)

        # Table A - Gauge R&R Summary
        grr_summary_group = QGroupBox("Gauge R&R Summary")
        grr_summary_layout = QVBoxLayout(grr_summary_group)
        self.summary_table_a = QTableView()
        self.summary_model_a = DataFrameModel()
        self.summary_table_a.setModel(self.summary_model_a)
        grr_summary_layout.addWidget(self.summary_table_a)
        summary_layout.addWidget(grr_summary_group)
        self.summary_table_a.setToolTip("Summary statistics for Gauge R&R analysis.")


        # Table B - Measurement Source Breakdown
        msb_group = QGroupBox("Measurement Source Breakdown")
        msb_layout = QVBoxLayout(msb_group)
        self.summary_table_b = QTableView()
        self.summary_model_b = DataFrameModel()
        self.summary_table_b.setModel(self.summary_model_b)
        msb_layout.addWidget(self.summary_table_b)
        summary_layout.addWidget(msb_group)
        self.summary_table_b.setToolTip("Breakdown of variation by measurement source.")
        
        # Interpretation
        interpretation_group = QGroupBox("Interpretation")
        interpretation_layout = QFormLayout(interpretation_group)
        interpretation_layout.setVerticalSpacing(15)
        interpretation_layout.setHorizontalSpacing(20)
        self.interpretation_badge = StatusBadge("N/A")
        interpretation_layout.addRow("Overall Assessment:", self.interpretation_badge)
        summary_layout.addWidget(interpretation_group)

        # Warnings Tab
        warnings_layout = QVBoxLayout(warnings_tab)
        warnings_layout.setSpacing(10)
        self.warnings_text = QTextEdit()
        self.warnings_text.setReadOnly(True)
        warnings_layout.addWidget(self.warnings_text)

        # Var Comp Tab
        var_comp_layout = QVBoxLayout(var_comp_tab)
        var_comp_layout.setSpacing(10)
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
        anova_layout.setSpacing(10)
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
        self.charts_scroll_area = QScrollArea()
        self.charts_scroll_area.setWidgetResizable(True)
        charts_layout.addWidget(self.charts_scroll_area)

        charts_content = QWidget()
        charts_content_layout = QVBoxLayout(charts_content)
        charts_content_layout.setSpacing(10)
        charts_content_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.charts_scroll_area.setWidget(charts_content)
        self.charts_scroll_area.viewport().installEventFilter(self)
        self.variability_chart_label = QLabel()
        self.variability_chart_label.setAlignment(Qt.AlignCenter)
        self.stddev_chart_label = QLabel()
        self.stddev_chart_label.setAlignment(Qt.AlignCenter)
        charts_content_layout.addWidget(self.variability_chart_label)
        charts_content_layout.addWidget(self.stddev_chart_label)
                # Export buttons
        self.export_button = QPushButton("Export PDF...")
        self.export_button.clicked.connect(self.export_pdf)
        self.export_button.setEnabled(False)
        
        self.export_pptx_button = QPushButton("Export PPTX...")
        self.export_pptx_button.clicked.connect(self.export_pptx)
        self.export_pptx_button.setEnabled(False)

        export_layout = QHBoxLayout()
        export_layout.addStretch()
        export_layout.addWidget(self.export_button)
        export_layout.addWidget(self.export_pptx_button)
        
        layout.addLayout(export_layout)

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
            
            # Notify user and switch tabs before running analysis
            msg_box = QMessageBox(QMessageBox.Information, "Analysis Running", "The MSA analysis is now running...")
            msg_box.setStandardButtons(QMessageBox.NoButton)
            msg_box.setModal(False)
            msg_box.show()
            
            self.main_tabs.setCurrentWidget(self.results_page)
            QApplication.processEvents() # Allow UI to update

            self.result = run_msa(self.df.copy(), config)
            self._update_results_ui()
            self.export_button.setEnabled(True)
            self.export_pptx_button.setEnabled(True)
            
            msg_box.close() # Close the notification

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to run MSA: {e}")
            if 'msg_box' in locals():
                msg_box.close()

    def _format_sig(self, val):
        if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
            return "N/A"
        return "{:.4g}".format(val)

    def _format_pct(self, val):
        if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
            return "N/A"
        return f"{val:.2f}%"

    def _get_summary_tables_data(self):
        if self.result is None:
            return pd.DataFrame(), pd.DataFrame()

        summary = self.result.grr_summary
        config = self.result.config
        
        # Helper for calculating square root, handling None
        def safe_sqrt(x):
            return np.sqrt(x) if x is not None and x >= 0 else np.nan

        # Extract variance components and std_devs directly from result.var_components
        # This ensures we use the same values as the MSA engine output
        def get_vc_by_source(source_str: str, default_val: float = np.nan):
            vc = next((vc for vc in self.result.var_components if vc.source == source_str), None)
            return (vc.var_comp, vc.std_dev, vc.variability, vc.pct_tolerance) if vc else (default_val, default_val, default_val, default_val)

        v_repeat_raw, sd_repeat, _, pct_tol_repeat = get_vc_by_source("Repeatability")
        v_gage_raw, sd_gage, _, pct_tol_gage = get_vc_by_source("Gage R&R")
        v_part_raw, sd_part, _, pct_tol_part = get_vc_by_source(f"Part-to-Part ({self.result.config.part_col})")
        v_total_raw, sd_total, _, pct_tol_total = get_vc_by_source("Total Variation")

        # Use the extracted standard deviations
        rr_sd = sd_gage
        tv_sd = sd_total

        # Reproducibility terms for table B
        reproducibility_terms_for_table_b = []
        for vc in self.result.var_components:
            if vc.source.startswith('Reproducibility:'): # Corrected to match "Reproducibility: <factor>"
                reproducibility_terms_for_table_b.append((vc.source, vc.std_dev, vc.pct_tolerance))

        # Tolerance
        lsl = config.lsl
        usl = config.usl
        tolerance = config.tolerance
        if tolerance is None:
            tolerance = usl - lsl if usl is not None and lsl is not None else np.nan
        
        tolerance_str = self._format_sig(tolerance) if not np.isnan(tolerance) else "N/A"
        lsl_str = self._format_sig(lsl) if lsl is not None else "N/A"
        usl_str = self._format_sig(usl) if usl is not None else "N/A"

        # Table A - Gauge R&R Summary
        table_a_data = []

        # % Gauge R&R
        pct_gauge_rr = (rr_sd / tv_sd) * 100 if tv_sd > 0 else np.nan
        table_a_data.append(["% Gauge R&R", self._format_pct(pct_gauge_rr), "% Gauge R&R = 100 * (RR / TV)"])

        # Precision to Part Variation
        prec_to_part_var = rr_sd / sd_part if sd_part > 0 else np.nan
        table_a_data.append(["Precision to Part Variation", self._format_sig(prec_to_part_var), "Precision to Part Variation = RR / PV"])

        # Number of Distinct Categories
        ndc = np.floor(np.sqrt(2) * (sd_part / rr_sd)) if rr_sd > 0 else np.nan
        table_a_data.append(["Number of Distinct Categories", self._format_sig(ndc), "NDC = floor(sqrt(2) * (PV / RR))"])
        
        table_a_data.append(["Lower Tolerance (LT)", lsl_str, "Lower Specification Limit (LSL)"])
        table_a_data.append(["Upper Tolerance (UT)", usl_str, "Upper Specification Limit (USL)"])
        table_a_data.append(["Tolerance", tolerance_str, "Tolerance = UT - LT"])

        # Precision/Tolerance Ratio
        prec_tol_ratio = rr_sd / tolerance if tolerance > 0 else np.nan
        table_a_data.append(["Precision/Tolerance Ratio", self._format_sig(prec_tol_ratio), "Precision/Tolerance Ratio = RR / (UT - LT)"])

        df_table_a = pd.DataFrame(table_a_data, columns=["Statistic", "Value", "Formula (optional tooltip)"])
        print("\n--- df_table_a (Gauge R&R Summary) ---")
        print(df_table_a)

        # Table B - Measurement Source Breakdown
        table_b_data = []

        # Repeatability (EV)
        ev_6sd = 6 * sd_repeat
        table_b_data.append([
            "Repeatability (EV)",
            self._format_sig(ev_6sd),
            self._format_pct(pct_tol_repeat),
            "V(Repeatability)"
        ])

        # Reproducibility terms
        for source, sd_repro, repro_pct_tolerance in reproducibility_terms_for_table_b:
            repro_6sd = 6 * sd_repro
            table_b_data.append([
                source,
                self._format_sig(repro_6sd),
                self._format_pct(repro_pct_tolerance),
                f"V({source.replace('Reproducibility: ', '')})"
            ])

        # Gauge R&R (RR)
        rr_6sd = 6 * rr_sd
        table_b_data.append([
            "Gauge R&R (RR)",
            self._format_sig(rr_6sd),
            self._format_pct(pct_tol_gage),
            "V(Gage R&R)"
        ])

        # Part Variation (PV)
        pv_6sd = 6 * sd_part
        table_b_data.append([
            "Part Variation (PV)",
            self._format_sig(pv_6sd),
            self._format_pct(pct_tol_part),
            f"V(Part-to-Part ({self.result.config.part_col}))"
        ])

        # Total Variation (TV)
        tv_6sd = 6 * tv_sd
        table_b_data.append([
            "Total Variation (TV)",
            self._format_sig(tv_6sd),
            self._format_pct(pct_tol_total),
            "V(Total Variation)"
        ])

        df_table_b = pd.DataFrame(table_b_data, columns=["Measurement Source", "Variation (6 * StdDev)", "% of Tolerance", "which is 6*sqrt of"])
        print("\n--- df_table_b (Measurement Source Breakdown) ---")
        print(df_table_b)

        return df_table_a, df_table_b

    def _update_results_ui(self):
        if self.result is None:
            print("DEBUG: self.result is None in _update_results_ui. Tables will not be updated.")
            return

        print("DEBUG: self.result is NOT None. Attempting to update summary tables.")

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
        self.interpretation_badge.set_text_and_status(summary.interpretation, status)

        # Populate new summary tables
        df_table_a, df_table_b = self._get_summary_tables_data()
        self.summary_table_a.horizontalHeader().setWordWrap(True)
        self.summary_model_a.setDataFrame(df_table_a)
        self.summary_table_a.resizeColumnsToContents()
        self.summary_table_a.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.summary_table_a.horizontalHeader().setStretchLastSection(True)

        self.summary_table_b.horizontalHeader().setWordWrap(True)
        self.summary_model_b.setDataFrame(df_table_b)
        self.summary_table_b.resizeColumnsToContents()
        self.summary_table_b.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.summary_table_b.horizontalHeader().setStretchLastSection(True)

        # Warnings
        self.warnings_text.setText("\n".join(self.result.warnings) or "No warnings.")

        # Var Comp
        var_comp_df = pd.DataFrame(self.result.var_components)
        var_comp_df.rename(columns={
            "source": "Source",
            "var_comp": "Variance Comp.",
            "std_dev": "Std. Dev.",
            "variability": "6 * Std. Dev.",
            "pct_contribution": "% Contribution",
            "pct_study_var": "% Study Var",
            "pct_tolerance": "% Tolerance",
        }, inplace=True)
        self.var_comp_table.horizontalHeader().setWordWrap(True)
        self.var_comp_model.setDataFrame(var_comp_df)
        self.var_comp_table.resizeColumnsToContents()
        self.var_comp_table.horizontalHeader().setWordWrap(True) # Added this line
        self.var_comp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.var_comp_table.horizontalHeader().setStretchLastSection(True)
        
        impacts = get_variation_impact_analysis(self.result)
        impact_html = ""
        for type_, msg in impacts:
            color = "green" if type_ == "success" else ("red" if type_ == "error" else "white")
            impact_html += f'<p style="color:{color};">{msg}</p>'
        self.impact_analysis_text.setHtml(impact_html)

        # ANOVA
        anova_df = pd.DataFrame(self.result.anova_table)
        anova_df.rename(columns={
            "term": "Source",
            "df": "DF",
            "ss": "Sum of Sq.",
            "ms": "Mean Sq.",
            "f": "F-Value",
            "p": "P-Value",
            "mean": "Mean",
            "std_dev": "Std. Dev.",
        }, inplace=True)
        for col in anova_df.columns:
            if pd.api.types.is_numeric_dtype(anova_df[col]):
                anova_df[col] = anova_df[col].apply(self._format_sig)
        self.anova_model.setDataFrame(anova_df)
        self.anova_table.resizeColumnsToContents()
        self.anova_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.anova_table.horizontalHeader().setStretchLastSection(True)

                # Charts
        self._schedule_chart_render()


    def eventFilter(self, obj, event):
        # Debounced re-render on viewport resize. This preserves scroll wheel behavior because
        # we still render to PNG/QPixmap rather than embedding a Matplotlib canvas widget.
        if hasattr(self, "charts_scroll_area") and obj is self.charts_scroll_area.viewport():
            if event.type() == QEvent.Resize:
                self._schedule_chart_render(delay_ms=120)
        return super().eventFilter(obj, event)

    def _on_results_tab_changed(self, idx: int):
        # Only render when the Charts tab is active (helps avoid rendering when viewport is 0px wide).
        try:
            if self.results_tabs.tabText(idx) == "Charts":
                self._schedule_chart_render(delay_ms=0)
        except Exception:
            pass

    def _schedule_chart_render(self, delay_ms: int = 0):
        if self.result is None:
            return
        if not hasattr(self, "_charts_rerender_timer") or self._charts_rerender_timer is None:
            return
        self._charts_rerender_timer.start(max(0, int(delay_ms)))

    def _get_chart_group_count(self) -> int:
        # Match the x-axis grouping logic used by the plotting functions:
        # operator (+ optional extra factor like instrument) + part
        try:
            if self.result is None or self.result.chart_data is None or self.result.chart_data.variability is None:
                return 10
            cfg = self.result.config
            dfv = self.result.chart_data.variability

            part_col = cfg.part_col
            op_col = cfg.operator_col

            other_factors = [f for f in cfg.factor_cols if f not in (part_col, op_col) and f in dfv.columns]
            inst_col = other_factors[0] if other_factors else None

            cols = [op_col]
            if inst_col:
                cols.append(inst_col)
            cols.append(part_col)

            cols = [c for c in cols if c in dfv.columns]
            if not cols:
                return 10

            return int(dfv[cols].drop_duplicates().shape[0])
        except Exception:
            return 10

    @staticmethod
    def _figure_to_pixmap(fig, dpi: int = 100):
        import io
        from PySide6.QtGui import QPixmap
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        buf.close()
        return pixmap

    def _render_charts(self):
        if self.result is None:
            return
        if not hasattr(self, "charts_scroll_area"):
            return

        viewport_w = self.charts_scroll_area.viewport().width()
        if viewport_w <= 50:
            # If not visible yet, don't render at a bogus size.
            return

        n_groups = self._get_chart_group_count()

        # Sizing rules:
        # - React to viewport width
        # - Ensure enough width per x-group
        # - Always wider than tall (no fixed ratio required)
        dpi = 100
        px_per_group = 60
        min_width_px = 600
        max_width_px = 3200
        side_padding_px = 60

        available_w = max(300, viewport_w - side_padding_px)
        width_px = max(min_width_px, available_w, n_groups * px_per_group)
        width_px = min(width_px, max_width_px)

        # Height: enforce "wider than tall" by targeting ~0.45*width and clamping
        min_height_px = 360
        max_height_px = 900
        height_px = int(width_px * 0.45)
        height_px = max(min_height_px, min(height_px, max_height_px))
        height_px = min(height_px, width_px - 1)

        state = (width_px, height_px, n_groups, id(self.result))
        if state == getattr(self, "_last_chart_render_state", None):
            return
        self._last_chart_render_state = state

        import matplotlib.pyplot as plt

        # Variability chart
        fig_var = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        ax_var = fig_var.add_subplot(111)
        get_variability_chart(self.result, ax_var)
        fig_var.tight_layout(pad=2.0)
        pixmap_var = self._figure_to_pixmap(fig_var, dpi=dpi)
        self.variability_chart_label.setPixmap(pixmap_var)
        self.variability_chart_label.setFixedSize(pixmap_var.size())
        plt.close(fig_var)

        # Stddev chart
        fig_std = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        ax_std = fig_std.add_subplot(111)
        get_stddev_chart(self.result, ax_std)
        fig_std.tight_layout(pad=2.0)
        pixmap_std = self._figure_to_pixmap(fig_std, dpi=dpi)
        self.stddev_chart_label.setPixmap(pixmap_std)
        self.stddev_chart_label.setFixedSize(pixmap_std.size())
        plt.close(fig_std)


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

    def export_pptx(self):
        if self.result is None:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save PowerPoint", "msa_report.pptx", "PowerPoint Files (*.pptx)")
        if path:
            try:
                template_path = os.path.join(self.project_root, "template_slide.pptx")
                save_pptx_report(self.result, template_path, path)
                QMessageBox.information(self, "Success", f"Report saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save PPTX: {e}")


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
