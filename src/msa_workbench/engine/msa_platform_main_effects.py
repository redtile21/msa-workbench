# msa_platform_main_effects.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from .msa_utils import (
    validate_dataframe,
    design_diagnostics,
    is_balanced_and_complete,
    clean_anova_index,
    find_term,
    get_ms_df,
    update_anova_f_test,
    build_anova_rows,
)
from .msa_bayes import gibbs_random_intercepts_main_effects
from .msa_results import build_result_object


def _ems_main_effects_vc(df: pd.DataFrame, response_col: str, factor_cols: List[str]) -> Tuple[Dict[str, float], List[Any], pd.Series, Dict[str, Any]]:
    """
    Balanced/complete ANOVA + EMS for additive random effects (main effects only).
    Returns (vc_map, anova_df_rows, resid, diag).
    """
    y = response_col
    factors = factor_cols

    formula = f'Q("{y}") ~ ' + " + ".join([f'C(Q("{f}"))' for f in factors])
    model = smf.ols(formula, data=df).fit()
    anova = anova_lm(model, typ=2)

    # residual
    ms_res, df_res = get_ms_df(anova, "Residual")
    # replicate count per full cell
    reps = df.groupby(factors, observed=True).size()
    r = float(reps.mean()) if len(reps) else 1.0
    r = max(1.0, r)

    # level counts
    n_levels = {f: int(df[f].nunique()) for f in factors}
    prod_all = 1
    for f in factors:
        prod_all *= max(1, n_levels[f])

    vc_map: Dict[str, float] = {"Repeatability": float(max(ms_res, 0.0))}

    # For each factor i: EMS coefficient for sigma_i^2 is r * product(levels of all other factors)
    for f in factors:
        term = find_term(anova, f)
        ms_f, df_f = get_ms_df(anova, term)
        # F-test vs residual (reference)
        update_anova_f_test(anova, term, ms_f, ms_res, df_f, df_res)

        denom = r
        for g in factors:
            if g != f:
                denom *= max(1, n_levels[g])
        vc_map[f] = float((ms_f - ms_res) / denom) if denom > 0 else 0.0

    anova_rows = build_anova_rows(clean_anova_index(anova), None)  # placeholder; caller will rebuild w/ dataclass
    # We'll return the anova DataFrame itself via diag to rebuild rows properly in caller
    diag = {
        "ems": {
            "r": float(r),
            "n_levels": n_levels,
            "ms_residual": float(ms_res),
        },
        "anova_raw": anova,
    }
    return vc_map, [], model.resid, diag


def run_main_effects(df: pd.DataFrame, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult):
    warnings: List[str] = []
    df2 = validate_dataframe(df, config.response_col, config.factor_cols)

    y = config.response_col
    factors = config.factor_cols

    diag: Dict[str, Any] = {"platform": "main_effects"}
    design_diag = design_diagnostics(df2, factors)
    diag["design"] = design_diag

    # Choose method: EMS (balanced & complete), else Bayesian directly.
    use_ems = is_balanced_and_complete(design_diag, std_tol=1e-8, missing_tol_pct=1e-6)

    vc_map: Dict[str, float] = {}
    anova_rows: List[Any] = []
    resid = pd.Series(dtype=float)

    if use_ems:
        warnings.append("Balanced design detected. Using ANOVA/EMS for initial variance components.")
        # OLS ANOVA
        formula = f'Q("{y}") ~ ' + " + ".join([f'C(Q("{f}"))' for f in factors])
        model = smf.ols(formula, data=df2).fit()
        anova = anova_lm(model, typ=2)

        ms_res, df_res = get_ms_df(anova, "Residual")
        reps = df2.groupby(factors, observed=True).size()
        r = float(reps.mean()) if len(reps) else 1.0
        r = max(1.0, r)

        n_levels = {f: int(df2[f].nunique()) for f in factors}

        vc_map = {"Repeatability": float(max(ms_res, 0.0))}
        raw_vc = {}
        for f in factors:
            term = find_term(anova, f)
            ms_f, df_f = get_ms_df(anova, term)
            update_anova_f_test(anova, term, ms_f, ms_res, df_f, df_res)
            denom = r
            for g in factors:
                if g != f:
                    denom *= max(1, n_levels[g])
            est = float((ms_f - ms_res) / denom) if denom > 0 else 0.0
            raw_vc[f] = est
            vc_map[f] = est

        # detect negative components
        negatives = {k: v for k, v in raw_vc.items() if v < 0}
        if negatives:
            warnings.append(
                "Switching to Bayesian estimates because of negative EMS variance component(s): "
                + ", ".join([f"{k}={v:.6g}" for k, v in negatives.items()])
            )
            # Bayesian VC (posterior means)
            post_means, gibbs_diag = gibbs_random_intercepts_main_effects(
                df2, y, factors,
                seed=0, draws=3000, burn=1000,
            )
            vc_map = {"Repeatability": float(post_means["Residual"])}
            for f in factors:
                vc_map[f] = float(post_means[f])

            diag["bayes"] = gibbs_diag.__dict__
            diag["method"] = "bayesian"
        else:
            # truncate any small negative due to numeric noise
            for f in factors:
                vc_map[f] = float(max(0.0, vc_map[f]))
            diag["method"] = "ems"

        anova_rows = build_anova_rows(clean_anova_index(anova), ANOVATableRow)
        resid = model.resid
        diag["ems"] = {"r": float(r), "n_levels": n_levels, "ms_residual": float(ms_res)}
    else:
        warnings.append("Unbalanced or incomplete design detected. Using Bayesian variance components (robust).")
        post_means, gibbs_diag = gibbs_random_intercepts_main_effects(
            df2, y, factors,
            seed=0, draws=3000, burn=1000,
        )
        vc_map = {"Repeatability": float(post_means["Residual"])}
        for f in factors:
            vc_map[f] = float(post_means[f])

        diag["bayes"] = gibbs_diag.__dict__
        diag["method"] = "bayesian"

        # reference-only ANOVA (fixed effects)
        try:
            formula = f'Q("{y}") ~ ' + " + ".join([f'C(Q("{f}"))' for f in factors])
            model = smf.ols(formula, data=df2).fit()
            anova = anova_lm(model, typ=2)
            anova_rows = build_anova_rows(clean_anova_index(anova), ANOVATableRow)
            resid = model.resid
            warnings.append("ANOVA table is for reference only and not used for VC estimation.")
        except Exception as e:
            warnings.append(f"Could not generate reference ANOVA table: {e}")
            anova_rows = []
            resid = pd.Series(dtype=float)

    return build_result_object(
        df2, config, vc_map, anova_rows, resid, warnings, diag,
        VarianceComponentRow, GRRSummary, ChartData, MSAResult
    )
