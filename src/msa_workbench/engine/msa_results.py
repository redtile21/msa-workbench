# msa_results.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .msa_utils import design_diagnostics, shapiro_safe


def interpret_grr(val: float) -> str:
    if val < 10:
        return "Excellent"
    if val < 30:
        return "Acceptable"
    return "Poor"


def build_chart_data(df: pd.DataFrame, response_col: str, factor_cols: List[str], part_col: str, operator_col: Optional[str]):
    # Variability view is raw data (subset)
    cols = list(set(list(factor_cols) + [response_col]))
    variability = df[cols].copy()

    # Std dev per Part x Operator cell (for classical MSA visuals)
    if operator_col is not None and part_col is not None and part_col in df.columns and operator_col in df.columns:
        grouped = df.groupby([part_col, operator_col], observed=True)[response_col]
        stddev = grouped.agg(
            cell_mean="mean",
            cell_std="std",
            n="count",
        ).reset_index()
    else:
        stddev = pd.DataFrame()

    return variability, stddev


def build_result_object(
    df: pd.DataFrame,
    config,
    vc_map: Dict[str, float],
    anova_rows: List[Any],
    resid: pd.Series,
    warnings: List[str],
    diag: Optional[Dict[str, Any]],
    VarianceComponentRow,
    GRRSummary,
    ChartData,
    MSAResult,
):
    part = config.part_col
    op = config.operator_col

    sig2_repeat = float(vc_map.get("Repeatability", 0.0))
    sig2_part = float(vc_map.get(part, 0.0))

    if abs(sig2_part) < 1e-12 and config.model_type == "main effects":
        warnings.append(
            "WARNING: Part-to-Part variation is zero. This may indicate that interaction effects "
            "(not included in a main effects model) are significant and are inflating the residual error, "
            "masking the true part variation."
        )

    if config.model_type == "crossed":
        sig2_repro = float(sum(v for k, v in vc_map.items() if k != part and k != "Repeatability"))
    else:
        repro_factors = [f for f in config.factor_cols if f != part]
        sig2_repro = float(sum(vc_map.get(f, 0.0) for f in repro_factors))

    sig2_gage = sig2_repeat + sig2_repro
    sig2_total = sig2_gage + sig2_part

    # Defensive: ensure non-negative before sqrt
    def sqrt_nn(x: float) -> float:
        return float(np.sqrt(max(0.0, x)))

    sigma_repeat = sqrt_nn(sig2_repeat)
    sigma_gage = sqrt_nn(sig2_gage)
    sigma_part = sqrt_nn(sig2_part)
    sigma_total = sqrt_nn(sig2_total)

    sigmas = {k: sqrt_nn(float(v)) for k, v in vc_map.items()}

    def get_pct_contrib(var_i: float) -> float:
        return (var_i / sig2_total * 100.0) if sig2_total > 0 else 0.0

    def get_pct_sv(sigma_i: float) -> float:
        return (sigma_i / sigma_total * 100.0) if sigma_total > 0 else 0.0

    def get_pct_tol(sigma_i: float) -> Optional[float]:
        tol = config.tolerance_value
        if tol is None or tol <= 0:
            return None
        return (6.0 * sigma_i) / float(tol) * 100.0

    vc_rows: List[Any] = []

    vc_rows.append(
        VarianceComponentRow(
            "Repeatability",
            sig2_repeat,
            sigmas.get("Repeatability", 0.0),
            6.0 * sigmas.get("Repeatability", 0.0),
            get_pct_contrib(sig2_repeat),
            get_pct_sv(sigmas.get("Repeatability", 0.0)),
            get_pct_tol(sigmas.get("Repeatability", 0.0)),
        )
    )

    if config.model_type == "crossed":
        repro_keys = [k for k in vc_map.keys() if k != part and k != "Repeatability"]
    else:
        repro_keys = [f for f in config.factor_cols if f != part]

    for k in repro_keys:
        vc_rows.append(
            VarianceComponentRow(
                f"Reproducibility: {k}",
                float(vc_map.get(k, 0.0)),
                sigmas.get(k, 0.0),
                6.0 * sigmas.get(k, 0.0),
                get_pct_contrib(float(vc_map.get(k, 0.0))),
                get_pct_sv(sigmas.get(k, 0.0)),
                get_pct_tol(sigmas.get(k, 0.0)),
            )
        )

    vc_rows.append(
        VarianceComponentRow(
            "Gage R&R",
            sig2_gage,
            sigma_gage,
            6.0 * sigma_gage,
            get_pct_contrib(sig2_gage),
            get_pct_sv(sigma_gage),
            get_pct_tol(sigma_gage),
        )
    )

    vc_rows.append(
        VarianceComponentRow(
            f"Part-to-Part ({part})",
            sig2_part,
            sigma_part,
            6.0 * sigma_part,
            get_pct_contrib(sig2_part),
            get_pct_sv(sigma_part),
            get_pct_tol(sigma_part),
        )
    )

    vc_rows.append(
        VarianceComponentRow(
            "Total Variation",
            sig2_total,
            sigma_total,
            6.0 * sigma_total,
            100.0,
            100.0,
            get_pct_tol(sigma_total),
        )
    )

    ndc = 1.41 * (sigma_part / sigma_gage) if sigma_gage > 0 else 0.0
    grr_pct_sv = get_pct_sv(sigma_gage)

    summary = GRRSummary(
        total_gage_rr_pct_study_var=float(grr_pct_sv),
        total_gage_rr_pct_tolerance=get_pct_tol(sigma_gage),
        ndc=float(ndc),
        interpretation=interpret_grr(float(grr_pct_sv)),
    )

    variability, stddev = build_chart_data(df, config.response_col, config.factor_cols, config.part_col, config.operator_col)
    chart_data = ChartData(variability, stddev)

    final_diag: Dict[str, Any] = dict(diag or {})
    final_diag["design"] = design_diagnostics(df, config.factor_cols)
    final_diag.setdefault("residual_normality_pvalue", shapiro_safe(resid))

    return MSAResult(config, anova_rows, vc_rows, summary, chart_data, final_diag, warnings)
