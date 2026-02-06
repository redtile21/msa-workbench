import pandas as pd
import numpy as np
from unittest.mock import Mock
from msa_workbench.ui.main_window import MainWindow
from msa_workbench.engine.msa_engine import MSAConfig, MSAResult

def test_get_summary_tables_data_basic():
    # Mock a simple MSAResult
    mock_config = MSAConfig(
        response_col="measurement",
        factor_cols=["part", "operator"],
        part_col="part",
        operator_col="operator",
        lsl=90.0,
        usl=110.0,
        tolerance=None # Will be calculated as 20.0
    )

    mock_result = MSAResult(
        config=mock_config,
        grr_summary=Mock(
            interpretation="Acceptable" # Just a placeholder
        ),
        var_components=[
            {"source": "Repeatability", "var_comp": 0.25},  # V_within = 0.25
            {"source": "Reproducibility - Operator", "var_comp": 0.50}, # V_operator = 0.50
            {"source": "Part-to-Part", "var_comp": 4.0},     # V_sample = 4.0
            {"source": "Total Variation", "var_comp": 4.75}
        ],
        anova_table=[], # Not needed for this test
        warnings=[],    # Not needed for this test
        chart_data=None, # Not needed for this test
        diagnostics={} # Added missing argument
    )

    # Instantiate MainWindow (only to access the method, UI won't be rendered)
    window = MainWindow(project_root=".")
    window.result = mock_result

    df_table_a, df_table_b = window._get_summary_tables_data()

    # --- Assertions for Table A (Gauge R&R Summary) ---
    assert isinstance(df_table_a, pd.DataFrame)
    assert len(df_table_a) == 7 # % Gauge R&R, Precision to Part, NDC, LT, UT, Tolerance, Precision/Tolerance Ratio
    assert list(df_table_a.columns) == ["Statistic", "Value", "Formula (optional tooltip)"]

    # Expected values for Table A calculations
    # V_within = 0.25 => SD_within = 0.5
    # V_operator = 0.50 => SD_operator = sqrt(0.5) ~ 0.707
    # V_sample = 4.0 => SD_part = 2.0
    # Tolerance = 110.0 - 90.0 = 20.0

    # RR_sd = sqrt(V_within + V_operator) = sqrt(0.25 + 0.50) = sqrt(0.75) ~ 0.866
    # TV_sd = sqrt(V_within + V_operator + V_sample) = sqrt(0.25 + 0.50 + 4.0) = sqrt(4.75) ~ 2.179
    # PV_sd = SD_part = 2.0

    # % Gauge R&R = (RR_sd / TV_sd) * 100 = (0.866 / 2.179) * 100 ~ 39.74%
    # Precision to Part Variation = RR_sd / PV_sd = 0.866 / 2.0 = 0.433
    # NDC = floor(sqrt(2) * (PV_sd / RR_sd)) = floor(1.414 * (2.0 / 0.866)) = floor(1.414 * 2.309) = floor(3.265) = 3
    # Precision/Tolerance Ratio = RR_sd / Tolerance = 0.866 / 20.0 = 0.0433

    # Check specific values for Table A
    # Using '4g' format for numeric values, '2f' for percentage
    assert df_table_a.loc[0, "Value"] == "39.74%"
    assert df_table_a.loc[1, "Value"] == "0.433"
    assert df_table_a.loc[2, "Value"] == "3"
    assert df_table_a.loc[3, "Value"] == "90" # LSL
    assert df_table_a.loc[4, "Value"] == "110" # USL
    assert df_table_a.loc[5, "Value"] == "20" # Tolerance
    assert df_table_a.loc[6, "Value"] == "0.0433"

    # --- Assertions for Table B (Measurement Source Breakdown) ---
    assert isinstance(df_table_b, pd.DataFrame)
    assert len(df_table_b) == 5 # Repeatability, Repro-Op, GRR, Part, Total
    assert list(df_table_b.columns) == ["Measurement Source", "Variation (6 * StdDev)", "% of Tolerance", "which is 6*sqrt of"]

    # Expected values for Table B calculations
    # Tolerance = 20.0

    # Repeatability (EV)
    # 6 * SD_within = 6 * 0.5 = 3.0
    # % of Tolerance = (3.0 / 20.0) * 100 = 15.00%

    # Reproducibility - Operator
    # 6 * SD_operator = 6 * 0.707 = 4.242
    # % of Tolerance = (4.242 / 20.0) * 100 = 21.21%

    # Gauge R&R (RR)
    # 6 * RR_sd = 6 * 0.866 = 5.196
    # % of Tolerance = (5.196 / 20.0) * 100 = 25.98%

    # Part Variation (PV)
    # 6 * SD_part = 6 * 2.0 = 12.0
    # % of Tolerance = (12.0 / 20.0) * 100 = 60.00%

    # Total Variation (TV)
    # 6 * TV_sd = 6 * 2.179 = 13.074
    # % of Tolerance = (13.074 / 20.0) * 100 = 65.37%

    # Check specific values for Table B
    assert df_table_b.loc[0, "Measurement Source"] == "Repeatability (EV)"
    assert df_table_b.loc[0, "Variation (6 * StdDev)"] == "3"
    assert df_table_b.loc[0, "% of Tolerance"] == "15.00%"
    assert df_table_b.loc[0, "which is 6*sqrt of"] == "V(Within)"

    assert df_table_b.loc[1, "Measurement Source"] == "Reproducibility - Operator"
    assert df_table_b.loc[1, "Variation (6 * StdDev)"] == "4.243" # Adjusted for 4g format
    assert df_table_b.loc[1, "% of Tolerance"] == "21.21%"
    assert df_table_b.loc[1, "which is 6*sqrt of"] == "V(Operator)"

    assert df_table_b.loc[2, "Measurement Source"] == "Gauge R&R (RR)"
    assert df_table_b.loc[2, "Variation (6 * StdDev)"] == "5.196"
    assert df_table_b.loc[2, "% of Tolerance"] == "25.98%"
    assert df_table_b.loc[2, "which is 6*sqrt of"] == "V(Within) + V(Operator)"

    assert df_table_b.loc[3, "Measurement Source"] == "Part Variation (PV)"
    assert df_table_b.loc[3, "Variation (6 * StdDev)"] == "12"
    assert df_table_b.loc[3, "% of Tolerance"] == "60.00%"
    assert df_table_b.loc[3, "which is 6*sqrt of"] == "V(Sample)"

    assert df_table_b.loc[4, "Measurement Source"] == "Total Variation (TV)"
    assert df_table_b.loc[4, "Variation (6 * StdDev)"] == "13.07" # Adjusted for 4g format
    assert df_table_b.loc[4, "% of Tolerance"] == "65.37%"
    assert df_table_b.loc[4, "which is 6*sqrt of"] == "V(Within) + V(Operator) + V(Sample)"
