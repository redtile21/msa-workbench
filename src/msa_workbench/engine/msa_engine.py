# msa_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
from itertools import combinations, permutations


# =========================
# Dataclasses / result types
# =========================

@dataclass
class MSAConfig:
    response_col: str
    factor_cols: List[str]
    part_col: str
    operator_col: Optional[str] = None
    lsl: Optional[float] = None
    usl: Optional[float] = None
    tolerance: Optional[float] = None
    model_type: str = "crossed"

    def __post_init__(self):
        n_factors = len(self.factor_cols)
        if n_factors < 2 or n_factors > 4:
            raise ValueError("Implementation supports between 2 and 4 factors.")

        if self.part_col not in self.factor_cols:
            raise ValueError("part_col must be one of factor_cols.")

        if self.operator_col is None:
            others = [f for f in self.factor_cols if f != self.part_col]
            self.operator_col = others[0] if others else None
        
        if self.model_type == "crossed" and n_factors > 3:
            raise ValueError("Crossed models are only supported for 2 or 3 factors.")

    @property
    def tolerance_value(self) -> Optional[float]:
        if self.tolerance is not None:
            return self.tolerance
        if self.lsl is not None and self.usl is not None:
            return float(self.usl - self.lsl)
        return None


@dataclass
class ANOVATableRow:
    term: str
    df: float
    ss: float
    ms: float
    f: Optional[float]
    p: Optional[float]


@dataclass
class VarianceComponentRow:
    source: str
    var_comp: float
    std_dev: float
    variability: float
    pct_contribution: float
    pct_study_var: float
    pct_tolerance: Optional[float]


@dataclass
class GRRSummary:
    total_gage_rr_pct_study_var: float
    total_gage_rr_pct_tolerance: Optional[float]
    ndc: Optional[float]
    interpretation: str


@dataclass
class ChartData:
    variability: pd.DataFrame
    stddev: pd.DataFrame


@dataclass
class MSAResult:
    config: MSAConfig
    anova_table: List[ANOVATableRow]
    var_components: List[VarianceComponentRow]
    grr_summary: GRRSummary
    chart_data: ChartData
    diagnostics: Dict[str, Any]
    warnings: List[str]


# =========================
# Public API
# =========================

def run_crossed_msa(df: pd.DataFrame, config: MSAConfig) -> MSAResult:
    """
    Dispatch to the appropriate implementation based on number of factors and model type.
    """
    n_factors = len(config.factor_cols)

    if config.model_type == 'main effects':
        return _run_main_effects_msa(df, config)
    
    if n_factors == 2:
        return _run_crossed_msa_2factor(df, config)
    elif n_factors == 3:
        return _run_crossed_msa_3factor(df, config)
    else: # 4 factors
        return _run_main_effects_msa(df, config)


# =========================
# 2-FACTOR IMPLEMENTATION
# =========================

def _run_crossed_msa_2factor(df: pd.DataFrame, config: MSAConfig) -> MSAResult:
    warnings: List[str] = []
    _validate_dataframe(df, config, warnings)

    part = config.part_col
    op = config.operator_col
    y = config.response_col

    # 1. Fit OLS for Initial ANOVA
    formula = f'Q("{y}") ~ C(Q("{part}")) + C(Q("{op}")) + C(Q("{part}")):C(Q("{op}"))'
    model = smf.ols(formula, data=df).fit()
    anova = anova_lm(model, typ=2)

    # 2. Extract Terms
    term_part = _find_term(anova, part)
    term_op = _find_term(anova, op)
    term_int = _find_term(anova, [part, op])
    term_res = "Residual"

    # 3. Extract MS and DF
    ms_part, df_part = _get_ms_df(anova, term_part)
    ms_op, df_op = _get_ms_df(anova, term_op)
    ms_int, df_int = _get_ms_df(anova, term_int)
    ms_res, df_res = _get_ms_df(anova, term_res)

    # 4. Correct F-Tests (Random Model)
    #    Part F = MS_Part / MS_Int
    #    Op F   = MS_Op   / MS_Int
    _update_anova_f_test(anova, term_part, ms_part, ms_int, df_part, df_int)
    _update_anova_f_test(anova, term_op, ms_op, ms_int, df_op, df_int)
    #    Int F  = MS_Int  / MS_Res (Already correct in OLS, but we refresh it)
    _update_anova_f_test(anova, term_int, ms_int, ms_res, df_int, df_res)

    # 5. Clean Output
    anova_clean = _clean_anova_index(anova)
    anova_rows = _build_anova_rows(anova_clean)

    # --- Variance Components ---
    n_parts = df[part].nunique()
    n_ops = df[op].nunique()
    reps_per_cell = df.groupby([part, op])[y].count().mean()
    n_reps = int(round(reps_per_cell))

    if abs(reps_per_cell - n_reps) > 0.05:
        warnings.append(f"Unbalanced design ({reps_per_cell:.2f} reps/cell). Results may be approximate.")

    sig2_repeat = max(ms_res, 0.0)
    sig2_part_op = max((ms_int - ms_res) / n_reps, 0.0)
    sig2_op = max((ms_op - ms_int) / (n_parts * n_reps), 0.0)
    sig2_part = max((ms_part - ms_int) / (n_ops * n_reps), 0.0)

    vc_map = {
        "Repeatability": sig2_repeat,
        f"{op}": sig2_op,
        f"{part}:{op}": sig2_part_op,
        f"{part}": sig2_part
    }

    return _build_result_object(df, config, vc_map, anova_rows, model.resid, warnings)


# =========================
# 3-FACTOR IMPLEMENTATION
# =========================

def _run_crossed_msa_3factor(df: pd.DataFrame, config: MSAConfig) -> MSAResult:
    warnings: List[str] = []
    _validate_dataframe(df, config, warnings)

    fac_a = config.part_col
    fac_b = config.operator_col
    all_factors = config.factor_cols
    others = [f for f in all_factors if f != fac_a and f != fac_b]
    fac_c = others[0]
    y = config.response_col

    # 1. Fit OLS for Initial ANOVA
    formula = (f'Q("{y}") ~ C(Q("{fac_a}")) + C(Q("{fac_b}")) + C(Q("{fac_c}")) + '
               f'C(Q("{fac_a}")):C(Q("{fac_b}")) + C(Q("{fac_a}")):C(Q("{fac_c}")) + C(Q("{fac_b}")):C(Q("{fac_c}")) + '
               f'C(Q("{fac_a}")):C(Q("{fac_b}")):C(Q("{fac_c}"))')

    model = smf.ols(formula, data=df).fit()
    anova = anova_lm(model, typ=2)

    # 2. Extract Terms (for consistent lookup)
    t_A = _find_term(anova, fac_a)
    t_B = _find_term(anova, fac_b)
    t_C = _find_term(anova, fac_c)
    t_AB = _find_term(anova, [fac_a, fac_b])
    t_AC = _find_term(anova, [fac_a, fac_c])
    t_BC = _find_term(anova, [fac_b, fac_c])
    t_ABC = _find_term(anova, [fac_a, fac_b, fac_c])
    t_Res = "Residual"

    # 3. Extract Mean Squares and DFs
    ms_A, df_A = _get_ms_df(anova, t_A)
    ms_B, df_B = _get_ms_df(anova, t_B)
    ms_C, df_C = _get_ms_df(anova, t_C)

    ms_AB, df_AB = _get_ms_df(anova, t_AB)
    ms_AC, df_AC = _get_ms_df(anova, t_AC)
    ms_BC, df_BC = _get_ms_df(anova, t_BC)

    ms_ABC, df_ABC = _get_ms_df(anova, t_ABC)
    ms_Res, df_Res = _get_ms_df(anova, t_Res)

    # 4. Correct F-Tests (3-Factor Random Model)
    #    Level 3: 3-Way Interaction (ABC) vs Residual
    _update_anova_f_test(anova, t_ABC, ms_ABC, ms_Res, df_ABC, df_Res)

    #    Level 2: 2-Way Interactions (AB, AC, BC) vs 3-Way Interaction (ABC)
    _update_anova_f_test(anova, t_AB, ms_AB, ms_ABC, df_AB, df_ABC)
    _update_anova_f_test(anova, t_AC, ms_AC, ms_ABC, df_AC, df_ABC)
    _update_anova_f_test(anova, t_BC, ms_BC, ms_ABC, df_BC, df_ABC)

    #    Level 1: Main Effects vs Synthesized Denominator
    #    Denominator = MS_AB + MS_AC - MS_ABC (Approximation for A)
    #    Note: Depending on the specific pair (e.g., A is usually tested against AB+AC-ABC)

    #    Synthesized Denominators:
    #    Denom_A = MS_AB + MS_AC - MS_ABC
    ms_denom_A = ms_AB + ms_AC - ms_ABC
    df_denom_A = _satterthwaite_df(ms_AB, df_AB, ms_AC, df_AC, ms_ABC, df_ABC)
    _update_anova_f_test(anova, t_A, ms_A, ms_denom_A, df_A, df_denom_A)

    #    Denom_B = MS_AB + MS_BC - MS_ABC
    ms_denom_B = ms_AB + ms_BC - ms_ABC
    df_denom_B = _satterthwaite_df(ms_AB, df_AB, ms_BC, df_BC, ms_ABC, df_ABC)
    _update_anova_f_test(anova, t_B, ms_B, ms_denom_B, df_B, df_denom_B)

    #    Denom_C = MS_AC + MS_BC - MS_ABC
    ms_denom_C = ms_AC + ms_BC - ms_ABC
    df_denom_C = _satterthwaite_df(ms_AC, df_AC, ms_BC, df_BC, ms_ABC, df_ABC)
    _update_anova_f_test(anova, t_C, ms_C, ms_denom_C, df_C, df_denom_C)

    # 5. Clean Output
    anova_clean = _clean_anova_index(anova)
    anova_rows = _build_anova_rows(anova_clean)

    # --- Variance Components ---
    n_a = df[fac_a].nunique()
    n_b = df[fac_b].nunique()
    n_c = df[fac_c].nunique()
    reps_per_cell = df.groupby([fac_a, fac_b, fac_c])[y].count().mean()
    n_r = int(round(reps_per_cell))

    if abs(reps_per_cell - n_r) > 0.05:
        warnings.append(f"Unbalanced design ({reps_per_cell:.2f} reps/cell). ANOVA estimates may be biased.")

    # EMS Solvers
    var_e = max(0.0, ms_Res)
    var_abc = max(0.0, (ms_ABC - ms_Res) / n_r)

    var_ab = max(0.0, (ms_AB - ms_ABC) / (n_r * n_c))
    var_ac = max(0.0, (ms_AC - ms_ABC) / (n_r * n_b))
    var_bc = max(0.0, (ms_BC - ms_ABC) / (n_r * n_a))

    var_a = max(0.0, (ms_A - ms_AB - ms_AC + ms_ABC) / (n_r * n_b * n_c))
    var_b = max(0.0, (ms_B - ms_AB - ms_BC + ms_ABC) / (n_r * n_a * n_c))
    var_c = max(0.0, (ms_C - ms_AC - ms_BC + ms_ABC) / (n_r * n_a * n_b))

    vc_map = {
        "Repeatability": var_e,
        f"{fac_a}": var_a,
        f"{fac_b}": var_b,
        f"{fac_c}": var_c,
        f"{fac_a}:{fac_b}": var_ab,
        f"{fac_a}:{fac_c}": var_ac,
        f"{fac_b}:{fac_c}": var_bc,
        f"{fac_a}:{fac_b}:{fac_c}": var_abc
    }

    return _build_result_object(df, config, vc_map, anova_rows, model.resid, warnings)


def _run_main_effects_msa(df: pd.DataFrame, config: MSAConfig) -> MSAResult:
    warnings: List[str] = ["Running Main Effects Model."]
    _validate_dataframe(df, config, warnings)

    y = config.response_col
    factors = config.factor_cols
    
    formula = f'Q("{y}") ~ {" + ".join([f"C(Q(\'{f}\'))" for f in factors])}'
    
    model = smf.ols(formula, data=df).fit()
    anova = anova_lm(model, typ=2)

    # Correct F-Tests
    ms_res, df_res = _get_ms_df(anova, "Residual")
    for factor in factors:
        term = _find_term(anova, factor)
        ms_fac, df_fac = _get_ms_df(anova, term)
        _update_anova_f_test(anova, term, ms_fac, ms_res, df_fac, df_res)

    anova_clean = _clean_anova_index(anova)
    anova_rows = _build_anova_rows(anova_clean)

    # Variance Components
    vc_map = {"Repeatability": max(ms_res, 0.0)}
    n_total = len(df)
    
    # Check for unbalance
    reps_per_cell = df.groupby(factors)[y].count()
    if reps_per_cell.nunique() > 1:
        warnings.append(f"Unbalanced design ({reps_per_cell.mean():.2f} reps/cell). Results may be approximate.")

    for factor in factors:
        term = _find_term(anova, factor)
        ms_fac, df_fac = _get_ms_df(anova, term)
        
        n_prime = n_total / df[factor].nunique()

        sig2_fac = max((ms_fac - ms_res) / n_prime, 0.0)
        vc_map[factor] = sig2_fac

    return _build_result_object(df, config, vc_map, anova_rows, model.resid, warnings)


def _run_crossed_msa_mixed(df: pd.DataFrame, config: MSAConfig) -> MSAResult:
    warnings: List[str] = ["Using Iterative Mixed Model (4+ factors). Convergence may vary."]
    _validate_dataframe(df, config, warnings)
    raise NotImplementedError("4+ factor support is experimental. Please stick to 2 or 3 factors.")

# =========================
# HELPER FUNCTIONS
# =========================

def _clean_anova_index(anova: pd.DataFrame) -> pd.DataFrame:
    """Removes the C(Q("...")) wrapper from ANOVA index names for display."""
    clean_df = anova.copy()
    new_index = []

    for idx in clean_df.index:
        name = str(idx)
        # Regex to strip C(Q("Name")) -> Name
        clean_name = re.sub(r'C\(Q\("([^"]+)"\)\)', r'\1', name)
        new_index.append(clean_name)

    clean_df.index = new_index
    return clean_df


def _find_term(anova: pd.DataFrame, cols) -> str:
    """Finds the exact index string in the ANOVA table for a set of columns."""
    if isinstance(cols, str): cols = [cols]

    c_terms = [f'C(Q("{c}"))' for c in cols]

    possible_names = []
    for p in permutations(c_terms):
        possible_names.append(":".join(p))

    for name in possible_names:
        if name in anova.index:
            return name

    return ""  # Should not happen


def _get_ms_df(anova, term):
    """Safely extract Mean Square and DF for a term."""
    if term not in anova.index:
        return 0.0, 1.0
    row = anova.loc[term]

    # Calculate MS if missing (Statsmodels usually provides it)
    if "mean_sq" in row:
        ms = float(row["mean_sq"])
    else:
        ss = float(row["sum_sq"] if "sum_sq" in row else row["ss"])
        df_val = float(row["df"])
        ms = ss / df_val if df_val > 0 else 0.0

    df_val = float(row["df"])
    return ms, df_val


def _satterthwaite_df(ms1, df1, ms2, df2, ms3, df3):
    """
    Calculates approx DF for linear combination L = MS1 + MS2 - MS3.
    Formula: DF = (MS1 + MS2 - MS3)^2 / [ (MS1^2/df1) + (MS2^2/df2) + (MS3^2/df3) ]
    """
    numerator = (ms1 + ms2 - ms3) ** 2

    denom = 0.0
    if df1 > 0: denom += (ms1 ** 2) / df1
    if df2 > 0: denom += (ms2 ** 2) / df2
    if df3 > 0: denom += (ms3 ** 2) / df3

    if denom == 0: return 1.0
    return numerator / denom


def _update_anova_f_test(anova, term, ms_num, ms_denom, df_num, df_denom):
    """Calculates F-ratio and p-value and updates the ANOVA table."""
    if term not in anova.index:
        return

    # If denominator is <= 0 (can happen with negative MS estimates), F is undefined/infinite
    if ms_denom <= 0:
        anova.loc[term, "F"] = np.nan
        anova.loc[term, "PR(>F)"] = np.nan
        return

    f_value = ms_num / ms_denom
    p_value = stats.f.sf(f_value, df_num, df_denom)

    anova.loc[term, "F"] = f_value
    anova.loc[term, "PR(>F)"] = p_value


def _build_result_object(df, config, vc_map, anova_rows, resid, warnings):
    part = config.part_col
    op = config.operator_col

    sig2_repeat = vc_map.get("Repeatability", 0.0)
    sig2_part = vc_map.get(part, 0.0)

    if abs(sig2_part) < 1e-9 and config.model_type == 'main effects':
        warnings.append("WARNING: Part-to-Part variation is zero. This may indicate that interaction effects (not included in a main effects model) are significant and are inflating the residual error, masking the true part variation.")
    
    if config.model_type == 'crossed':
        sig2_repro = sum(v for k, v in vc_map.items() if k != part and k != "Repeatability")
    else: # main effects
        repro_factors = [f for f in config.factor_cols if f != part]
        sig2_repro = sum(vc_map.get(f, 0.0) for f in repro_factors)

    sig2_gage = sig2_repeat + sig2_repro
    sig2_total = sig2_gage + sig2_part

    sigmas = {k: np.sqrt(v) for k, v in vc_map.items()}
    sigma_gage = np.sqrt(sig2_gage)
    sigma_part = np.sqrt(sig2_part)
    sigma_total = np.sqrt(sig2_total)

    def get_pct_contrib(var_i):
        return (var_i / sig2_total * 100.0) if sig2_total > 0 else 0.0

    def get_pct_sv(sigma_i):
        return (sigma_i / sigma_total * 100.0) if sigma_total > 0 else 0.0

    def get_pct_tol(sigma_i):
        tol = config.tolerance_value
        if tol is None or tol <= 0: return None
        return (6.0 * sigma_i) / tol * 100.0

    vc_rows = []

    vc_rows.append(VarianceComponentRow(
        "Repeatability", sig2_repeat, sigmas["Repeatability"], 6.0 * sigmas["Repeatability"],
        get_pct_contrib(sig2_repeat), get_pct_sv(sigmas["Repeatability"]), get_pct_tol(sigmas["Repeatability"])
    ))
    
    if config.model_type == 'crossed':
        repro_keys = [k for k in vc_map.keys() if k != part and k != "Repeatability"]
    else: # main effects
        repro_keys = [f for f in config.factor_cols if f != part]

    for k in repro_keys:
        vc_rows.append(VarianceComponentRow(
            f"Reproducibility ({k})", vc_map[k], sigmas[k], 6.0 * sigmas[k],
            get_pct_contrib(vc_map[k]), get_pct_sv(sigmas[k]), get_pct_tol(sigmas[k])
        ))

    vc_rows.append(VarianceComponentRow(
        "Gage R&R", sig2_gage, sigma_gage, 6.0 * sigma_gage,
        get_pct_contrib(sig2_gage), get_pct_sv(sigma_gage), get_pct_tol(sigma_gage)
    ))

    vc_rows.append(VarianceComponentRow(
        f"Part-to-Part ({part})", sig2_part, sigma_part, 6.0 * sigma_part,
        get_pct_contrib(sig2_part), get_pct_sv(sigma_part), get_pct_tol(sigma_part)
    ))

    vc_rows.append(VarianceComponentRow(
        "Total Variation", sig2_total, sigma_total, 6.0 * sigma_total,
        100.0, 100.0, get_pct_tol(sigma_total)
    ))

    ndc = 1.41 * (sigma_part / sigma_gage) if sigma_gage > 0 else 0.0
    grr_pct_sv = get_pct_sv(sigma_gage)

    summary = GRRSummary(
        total_gage_rr_pct_study_var=float(grr_pct_sv),
        total_gage_rr_pct_tolerance=get_pct_tol(sigma_gage),
        ndc=float(ndc),
        interpretation=_interpret_grr(grr_pct_sv)
    )

    chart_data = _build_chart_data(df, config)

    n_ops = df[op].nunique() if op and op in df.columns else np.nan

    diag = {
        "n_parts": df[part].nunique(),
        "n_operators": n_ops,
        "n_reps_mean": df.groupby(config.factor_cols)[config.response_col].count().mean(),
        "residual_normality_pvalue": _shapiro_safe(resid, warnings)
    }

    return MSAResult(config, anova_rows, vc_rows, summary, chart_data, diag, warnings)


def _get_ms_fuzzy(anova, cols) -> float:
    if isinstance(cols, str): cols = [cols]
    return _get_ms(anova, _find_term(anova, cols))


def _get_ms(anova: pd.DataFrame, term: str) -> float:
    try:
        row = anova.loc[term]
        ss = float(row["sum_sq"] if "sum_sq" in anova.columns else row["ss"])
        df_val = float(row["df"])
        if "mean_sq" in anova.columns:
            return float(row["mean_sq"])
        return ss / df_val if df_val > 0 else 0.0
    except KeyError:
        return 0.0


def _validate_dataframe(df, config, warnings):
    if config.response_col not in df.columns:
        raise KeyError(f"Response column '{config.response_col}' missing from dataframe.")
    for col in config.factor_cols:
        if col not in df.columns:
            raise KeyError(f"Factor column '{col}' missing from dataframe.")

    df[config.response_col] = pd.to_numeric(df[config.response_col], errors='coerce')
    df.dropna(subset=[config.response_col], inplace=True)

    for col in config.factor_cols:
        df[col] = df[col].astype(str).astype("category")


def _build_anova_rows(anova):
    rows = []
    for term, r in anova.iterrows():
        df_val = r["df"]
        ss = r.get("sum_sq", r.get("ss"))
        ms = r.get("mean_sq", ss / df_val)
        f = r.get("F", None)
        p = r.get("PR(>F)", None)
        rows.append(ANOVATableRow(term, df_val, ss, ms, f, p))
    return rows


def _build_chart_data(df, config):
    y = config.response_col
    cols = list(config.factor_cols) + [y]
    cols = list(set(cols))
    variability = df[cols].copy()

    part = config.part_col
    op = config.operator_col
    grouped = df.groupby([part, op])[y]
    stddev = grouped.agg(
        cell_mean="mean",
        cell_std="std",
        n="count",
    ).reset_index()

    return ChartData(variability, stddev)


def _shapiro_safe(resid, warnings):
    try:
        from scipy.stats import shapiro
        if len(resid) > 5000: resid = np.random.choice(resid, 5000)
        return float(shapiro(resid)[1])
    except:
        return None


def _interpret_grr(val):
    if val < 10: return "Excellent"
    if val < 30: return "Acceptable"
    return "Poor"
