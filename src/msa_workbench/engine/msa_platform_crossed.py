from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from .msa_bayes import gibbs_random_intercepts_main_effects
from .msa_results import build_result_object
from .msa_utils import (
    build_anova_rows,
    canonical_vc_term,
    clean_anova_index,
    design_diagnostics,
    find_term,
    get_ms_df,
    is_balanced_and_complete,
    update_anova_f_test,
    validate_dataframe,
)


def _one_way_random_effects(
    df: pd.DataFrame,
    y: str,
    factor: str,
    *,
    min_rep: int = 2,
) -> Tuple[Dict[str, float], List[Any], pd.Series, List[str], Dict[str, Any]]:
    """One-way random effects (factor + repeatability) via ANOVA/EMS.

    Handles edge cases:
      - If average replication < min_rep, factor VC is not identifiable; returns factor VC=0 and
        repeatability as sample variance around the grand mean.
    """
    warnings: List[str] = []
    diag: Dict[str, Any] = {"platform": "crossed_1way"}

    reps = df.groupby([factor], observed=True).size()
    r_mean = float(reps.mean()) if len(reps) else 1.0
    r = int(round(r_mean)) if r_mean > 0 else 1
    diag["replicates_per_level_mean"] = float(r_mean)
    diag["replicates_per_level_rounded"] = int(r)

    if r < min_rep:
        # No replication => cannot separate factor from error. Treat all variability as repeatability.
        warnings.append(
            f"No/insufficient replication for one-way random effects on '{factor}' (r≈{r_mean:.3g}). "
            "Reporting factor variance as 0 and using overall variance as repeatability."
        )
        vc_map = {
            "Repeatability": float(np.var(df[y].astype(float), ddof=1)) if len(df) > 1 else 0.0,
            str(factor): 0.0,
        }
        return vc_map, [], df[y] - df[y].mean(), warnings, diag

    formula = f'Q("{y}") ~ C(Q("{factor}"))'
    model = smf.ols(formula, data=df).fit()
    anova = anova_lm(model, typ=2)
    term = find_term(anova, factor)
    ms_f, df_f = get_ms_df(anova, term)
    ms_e, df_e = get_ms_df(anova, "Residual")

    update_anova_f_test(anova, term, ms_f, ms_e, df_f, df_e)
    sig2_e = float(ms_e)
    sig2_f = float((ms_f - ms_e) / float(r))

    vc_map = {
        "Repeatability": float(max(0.0, sig2_e)),
        str(factor): float(max(0.0, sig2_f)),
    }
    # Caller currently uses a reference ANOVA (or none) for these degenerate edge cases.
    return vc_map, [], model.resid, warnings, diag


def run_crossed_2factor(
    df: pd.DataFrame,
    config,
    ANOVATableRow,
    VarianceComponentRow,
    GRRSummary,
    ChartData,
    MSAResult,
):
    """2-factor crossed (Operator × Part) with replication.

    Implements:
      - Balance detection
      - EMS for balanced designs
      - EMS negative variance trigger -> REML MixedLM
      - REML convergence/boundary -> Bayesian Gibbs fallback
      - Edge-case handling: single operator / single part / no replication
    """
    warnings: List[str] = []
    df2 = validate_dataframe(df, config.response_col, config.factor_cols)

    part = config.part_col
    op = config.operator_col
    y = config.response_col

    diag: Dict[str, Any] = {"platform": "crossed_2factor"}
    design_diag = design_diagnostics(df2, config.factor_cols)
    diag["design"] = design_diag

    n_parts = int(design_diag.get("level_counts", {}).get(part, 1) or 1)
    n_ops = int(design_diag.get("level_counts", {}).get(op, 1) or 1)
    r_mean = float(design_diag.get("replicate_dist", {}).get("mean", 1) or 1)
    r = int(round(r_mean)) if r_mean > 0 else 1
    diag["replicates_per_cell_mean"] = float(r_mean)
    diag["replicates_per_cell_rounded"] = int(r)

    # -------------------------
    # Degenerate cases
    # -------------------------
    if n_parts < 2 and n_ops < 2:
        warnings.append(
            "Only a single Part and a single Operator are present. This reduces to a pure repeatability study."
        )
        diag["method"] = "repeatability_only"
        sig2_e = float(np.var(df2[y].astype(float), ddof=1)) if len(df2) > 1 else 0.0
        vc_map = {"Repeatability": max(0.0, sig2_e), part: 0.0, op: 0.0}
        return build_result_object(
            df2,
            config,
            vc_map,
            [],
            pd.Series(df2[y] - df2[y].mean(), index=df2.index, dtype=float),
            warnings,
            diag,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
        )

    if n_parts < 2:
        warnings.append(
            "Only a single Part is present. Reproducibility (Operator) can be estimated, but Part-to-Part is 0."
        )
        vc_map, anova_rows, resid, warn1, diag1 = _one_way_random_effects(df2, y, op)
        warnings.extend(warn1)
        diag.update(diag1)
        diag["method"] = "one_way_ems"
        vc_map.setdefault(part, 0.0)
        return build_result_object(
            df2,
            config,
            vc_map,
            [],
            resid,
            warnings,
            diag,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
        )

    if n_ops < 2:
        warnings.append(
            "Only a single Operator is present. This reduces to a one-way random effects study on Parts."
        )
        vc_map, anova_rows, resid, warn1, diag1 = _one_way_random_effects(df2, y, part)
        warnings.extend(warn1)
        diag.update(diag1)
        diag["method"] = "one_way_ems"
        vc_map.setdefault(op, 0.0)
        return build_result_object(
            df2,
            config,
            vc_map,
            [],
            resid,
            warnings,
            diag,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
        )

    # -------------------------
    # Balance/router
    # -------------------------
    # Balanced & complete => classical ANOVA/EMS; else MixedLM
    if not is_balanced_and_complete(design_diag, std_tol=1e-8, missing_tol_pct=1e-6):
        warnings.append(
            "Unbalanced or incomplete design detected. Using Mixed-Effects (REML → Bayes) estimation."
        )
        return run_crossed_mixed(
            df2,
            config,
            ANOVATableRow,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
            warnings=warnings,
        )

    # No replication: cannot separate interaction from repeatability.
    if r < 2:
        warnings.append(
            "Balanced design with ~1 observation per Part×Operator cell. "
            "Interaction and repeatability are confounded; estimating using the additive (no-interaction) model."
        )
        diag["method"] = "ems_no_replication"

        # Additive model: y ~ Part + Operator. Residual captures Part×Operator (and any error).
        formula = f'Q("{y}") ~ C(Q("{part}")) + C(Q("{op}"))'
        model = smf.ols(formula, data=df2).fit()
        anova = anova_lm(model, typ=2)

        term_part = find_term(anova, part)
        term_op = find_term(anova, op)
        ms_part, df_part = get_ms_df(anova, term_part)
        ms_op, df_op = get_ms_df(anova, term_op)
        ms_res, df_res = get_ms_df(anova, "Residual")

        update_anova_f_test(anova, term_part, ms_part, ms_res, df_part, df_res)
        update_anova_f_test(anova, term_op, ms_op, ms_res, df_op, df_res)

        anova_rows = build_anova_rows(
            clean_anova_index(anova), ANOVATableRow, df2, y, config.factor_cols
        )

        sig2_repeat = float(max(0.0, ms_res))
        sig2_part = float(max(0.0, (ms_part - ms_res) / float(n_ops)))
        sig2_op = float(max(0.0, (ms_op - ms_res) / float(n_parts)))
        int_key = canonical_vc_term(f"{part}:{op}", factor_order=config.factor_cols)
        vc_map = {
            "Repeatability": sig2_repeat,
            str(op): sig2_op,
            int_key: 0.0,
            str(part): sig2_part,
        }
        return build_result_object(
            df2,
            config,
            vc_map,
            anova_rows,
            model.resid,
            warnings,
            diag,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
        )

    # -------------------------
    # EMS (with interaction)
    # -------------------------
    warnings.append("Balanced design detected. Using ANOVA/EMS method.")
    diag["method"] = "ems"

    formula = f'Q("{y}") ~ C(Q("{part}")) + C(Q("{op}")) + C(Q("{part}")):C(Q("{op}"))'
    model = smf.ols(formula, data=df2).fit()
    anova = anova_lm(model, typ=2)

    term_part = find_term(anova, part)
    term_op = find_term(anova, op)
    term_int = find_term(anova, [part, op])
    term_res = "Residual"

    ms_part, df_part = get_ms_df(anova, term_part)
    ms_op, df_op = get_ms_df(anova, term_op)
    ms_int, df_int = get_ms_df(anova, term_int)
    ms_res, df_res = get_ms_df(anova, term_res)

    update_anova_f_test(anova, term_part, ms_part, ms_int, df_part, df_int)
    update_anova_f_test(anova, term_op, ms_op, ms_int, df_op, df_int)
    update_anova_f_test(anova, term_int, ms_int, ms_res, df_int, df_res)

    anova_rows = build_anova_rows(
        clean_anova_index(anova), ANOVATableRow, df2, y, config.factor_cols
    )

    # Raw EMS estimates (do not truncate until after negative-check)
    sig2_repeat_raw = float(ms_res)
    sig2_int_raw = float((ms_int - ms_res) / float(r))
    sig2_op_raw = float((ms_op - ms_int) / (float(n_parts) * float(r)))
    sig2_part_raw = float((ms_part - ms_int) / (float(n_ops) * float(r)))

    raw = {
        "Repeatability": sig2_repeat_raw,
        str(op): sig2_op_raw,
        str(part): sig2_part_raw,
        "__int__": sig2_int_raw,
    }
    neg = {k: v for k, v in raw.items() if v < -1e-12}

    if neg:
        warnings.append(
            "Negative EMS variance component(s) detected; switching to Mixed-Effects (REML → Bayes): "
            + ", ".join([f"{k}={v:.6g}" for k, v in neg.items()])
        )
        diag["ems_raw_vc"] = {k: float(v) for k, v in neg.items()}
        return run_crossed_mixed(
            df2,
            config,
            ANOVATableRow,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
            warnings=warnings,
        )

    int_key = canonical_vc_term(f"{part}:{op}", factor_order=config.factor_cols)
    vc_map = {
        "Repeatability": float(max(0.0, sig2_repeat_raw)),
        str(op): float(max(0.0, sig2_op_raw)),
        int_key: float(max(0.0, sig2_int_raw)),
        str(part): float(max(0.0, sig2_part_raw)),
    }

    return build_result_object(
        df2,
        config,
        vc_map,
        anova_rows,
        model.resid,
        warnings,
        diag,
        VarianceComponentRow,
        GRRSummary,
        ChartData,
        MSAResult,
    )


def run_crossed_3factor(
    df: pd.DataFrame,
    config,
    ANOVATableRow,
    VarianceComponentRow,
    GRRSummary,
    ChartData,
    MSAResult,
):
    """3-factor crossed (Part × Operator × Instrument-like third factor).

    Balanced full-factorial with replication uses EMS on the full factorial ANOVA.
    Otherwise routes to MixedLM (REML → Bayes).

    Notes:
      - If any factor has <2 levels or replication <2, we skip EMS and go to MixedLM.
      - Negative EMS VCs trigger MixedLM.
    """
    warnings: List[str] = []
    df2 = validate_dataframe(df, config.response_col, config.factor_cols)

    fac_a = config.part_col
    fac_b = config.operator_col
    all_factors = list(config.factor_cols)
    others = [f for f in all_factors if f != fac_a and f != fac_b]
    if not others:
        raise ValueError("3-factor crossed model requires a 3rd factor besides part and operator.")
    fac_c = others[0]
    y = config.response_col

    diag: Dict[str, Any] = {"platform": "crossed_3factor"}
    design_diag = design_diagnostics(df2, config.factor_cols)
    diag["design"] = design_diag

    n_a = int(design_diag.get("level_counts", {}).get(fac_a, 1) or 1)
    n_b = int(design_diag.get("level_counts", {}).get(fac_b, 1) or 1)
    n_c = int(design_diag.get("level_counts", {}).get(fac_c, 1) or 1)
    r_mean = float(design_diag.get("replicate_dist", {}).get("mean", 1) or 1)
    r = int(round(r_mean)) if r_mean > 0 else 1
    diag["replicates_per_cell_mean"] = float(r_mean)
    diag["replicates_per_cell_rounded"] = int(r)

    # If degenerate or no replication, route to MixedLM
    if min(n_a, n_b, n_c) < 2 or r < 2:
        if min(n_a, n_b, n_c) < 2:
            warnings.append(
                "One or more factors have a single observed level. Skipping EMS and using Mixed-Effects (REML → Bayes)."
            )
        else:
            warnings.append(
                "Single-replicate cells detected (r≈1). EMS is not available; using Mixed-Effects (REML → Bayes)."
            )
        return run_crossed_mixed(
            df2,
            config,
            ANOVATableRow,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
            warnings=warnings,
        )

    if not is_balanced_and_complete(design_diag, std_tol=1e-8, missing_tol_pct=1e-6):
        warnings.append(
            "Unbalanced or incomplete design detected. Using Mixed-Effects (REML → Bayes) estimation."
        )
        return run_crossed_mixed(
            df2,
            config,
            ANOVATableRow,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
            warnings=warnings,
        )

    warnings.append("Balanced design detected. Using ANOVA/EMS method.")
    diag["method"] = "ems"

    formula = (
        f'Q("{y}") ~ C(Q("{fac_a}")) + C(Q("{fac_b}")) + C(Q("{fac_c}")) + '
        f'C(Q("{fac_a}")):C(Q("{fac_b}")) + C(Q("{fac_a}")):C(Q("{fac_c}")) + C(Q("{fac_b}")):C(Q("{fac_c}")) + '
        f'C(Q("{fac_a}")):C(Q("{fac_b}")):C(Q("{fac_c}"))'
    )
    model = smf.ols(formula, data=df2).fit()
    anova = anova_lm(model, typ=2)

    t_A = find_term(anova, fac_a)
    t_B = find_term(anova, fac_b)
    t_C = find_term(anova, fac_c)
    t_AB = find_term(anova, [fac_a, fac_b])
    t_AC = find_term(anova, [fac_a, fac_c])
    t_BC = find_term(anova, [fac_b, fac_c])
    t_ABC = find_term(anova, [fac_a, fac_b, fac_c])
    t_Res = "Residual"

    ms_A, df_A = get_ms_df(anova, t_A)
    ms_B, df_B = get_ms_df(anova, t_B)
    ms_C, df_C = get_ms_df(anova, t_C)
    ms_AB, df_AB = get_ms_df(anova, t_AB)
    ms_AC, df_AC = get_ms_df(anova, t_AC)
    ms_BC, df_BC = get_ms_df(anova, t_BC)
    ms_ABC, df_ABC = get_ms_df(anova, t_ABC)
    ms_Res, df_Res = get_ms_df(anova, t_Res)

    # Error terms: use higher-order interaction as denominator
    update_anova_f_test(anova, t_ABC, ms_ABC, ms_Res, df_ABC, df_Res)
    update_anova_f_test(anova, t_AB, ms_AB, ms_ABC, df_AB, df_ABC)
    update_anova_f_test(anova, t_AC, ms_AC, ms_ABC, df_AC, df_ABC)
    update_anova_f_test(anova, t_BC, ms_BC, ms_ABC, df_BC, df_ABC)
    update_anova_f_test(anova, t_A, ms_A, ms_AB, df_A, df_AB)
    update_anova_f_test(anova, t_B, ms_B, ms_AB, df_B, df_AB)
    update_anova_f_test(anova, t_C, ms_C, ms_AC, df_C, df_AC)

    anova_rows = build_anova_rows(
        clean_anova_index(anova), ANOVATableRow, df2, y, config.factor_cols
    )

    # EMS for balanced 3-factor crossed
    #
    # Reference (one common form):
    #   MS_Res = σ²_e
    #   MS_ABC = σ²_e + r σ²_ABC
    #   MS_AB  = σ²_e + r σ²_ABC + r c σ²_AB
    #   MS_AC  = σ²_e + r σ²_ABC + r b σ²_AC
    #   MS_BC  = σ²_e + r σ²_ABC + r a σ²_BC
    #   MS_A   = σ²_e + r σ²_ABC + r c σ²_AB + r b c σ²_A
    #   MS_B   = σ²_e + r σ²_ABC + r c σ²_AB + r a c σ²_B
    #   MS_C   = σ²_e + r σ²_ABC + r b σ²_AC + r a b σ²_C
    #
    # We solve using the standard EMS differences.
    var_e = float(ms_Res)
    var_abc = float((ms_ABC - ms_Res) / float(r))
    var_ab = float((ms_AB - ms_ABC) / (float(r) * float(n_c)))
    var_ac = float((ms_AC - ms_ABC) / (float(r) * float(n_b)))
    var_bc = float((ms_BC - ms_ABC) / (float(r) * float(n_a)))
    var_a = float((ms_A - ms_AB) / (float(r) * float(n_b) * float(n_c)))
    var_b = float((ms_B - ms_AB) / (float(r) * float(n_a) * float(n_c)))
    var_c = float((ms_C - ms_AC) / (float(r) * float(n_a) * float(n_b)))

    raw_vc = {
        "Repeatability": var_e,
        fac_a: var_a,
        fac_b: var_b,
        fac_c: var_c,
        f"{fac_a}:{fac_b}": var_ab,
        f"{fac_a}:{fac_c}": var_ac,
        f"{fac_b}:{fac_c}": var_bc,
        f"{fac_a}:{fac_b}:{fac_c}": var_abc,
    }
    neg = {k: v for k, v in raw_vc.items() if v < -1e-12}
    if neg:
        warnings.append(
            "Negative EMS variance component(s) detected; switching to Mixed-Effects (REML → Bayes): "
            + ", ".join([f"{k}={v:.6g}" for k, v in neg.items()])
        )
        diag["ems_raw_vc"] = {k: float(v) for k, v in neg.items()}
        return run_crossed_mixed(
            df2,
            config,
            ANOVATableRow,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
            warnings=warnings,
        )

    vc_map = {
        "Repeatability": float(max(0.0, var_e)),
        str(fac_a): float(max(0.0, var_a)),
        str(fac_b): float(max(0.0, var_b)),
        str(fac_c): float(max(0.0, var_c)),
        canonical_vc_term(f"{fac_a}:{fac_b}", factor_order=config.factor_cols): float(max(0.0, var_ab)),
        canonical_vc_term(f"{fac_a}:{fac_c}", factor_order=config.factor_cols): float(max(0.0, var_ac)),
        canonical_vc_term(f"{fac_b}:{fac_c}", factor_order=config.factor_cols): float(max(0.0, var_bc)),
        canonical_vc_term(f"{fac_a}:{fac_b}:{fac_c}", factor_order=config.factor_cols): float(max(0.0, var_abc)),
    }

    return build_result_object(
        df2,
        config,
        vc_map,
        anova_rows,
        model.resid,
        warnings,
        diag,
        VarianceComponentRow,
        GRRSummary,
        ChartData,
        MSAResult,
    )


def run_crossed_mixed(
    df: pd.DataFrame,
    config,
    ANOVATableRow,
    VarianceComponentRow,
    GRRSummary,
    ChartData,
    MSAResult,
    warnings: Optional[List[str]] = None,
):
    """Mixed model variance components for crossed designs.

    Primary use:
      - Unbalanced/incomplete designs
      - EMS negative variance triggers
      - No-replication/degenerate cases where EMS is not applicable

    Estimation cascade:
      1) MixedLM (REML)
      2) Bayesian Gibbs fallback (weakly informative prior) if MixedLM fails or is unstable/boundary
    """
    warnings = list(warnings or [])
    diag: Dict[str, Any] = {"platform": "crossed_mixed"}

    df2 = df
    y = config.response_col

    all_factors = list(config.factor_cols)
    # Drop constant factors to avoid singular random-effects design matrices
    active_factors: List[str] = []
    dropped: List[str] = []
    for f in all_factors:
        try:
            if int(df2[f].nunique()) < 2:
                dropped.append(f)
            else:
                active_factors.append(f)
        except Exception:
            active_factors.append(f)

    diag["active_factors"] = list(active_factors)
    diag["dropped_factors"] = list(dropped)
    if dropped:
        warnings.append(
            "One or more factors have a single observed level and were omitted from MixedLM estimation: "
            + ", ".join([str(d) for d in dropped])
            + ". Their variance components are reported as 0."
        )

    warnings.append("Using Mixed-Effects Model (REML) for crossed design variance components.")

    factors = list(active_factors)

    # Variance components specification
    vc_formula: Dict[str, str] = {}
    for factor in factors:
        vc_formula[str(factor)] = f"0 + C(Q('{factor}'))"

    # Always include 2-way interactions
    for fac1, fac2 in combinations(factors, 2):
        vc_formula[f"{fac1}:{fac2}"] = f"0 + C(Q('{fac1}')):C(Q('{fac2}'))"

    # Include 3-way interaction for exactly 3 active factors when feasible
    include_3way = False
    if len(factors) == 3:
        try:
            n_cells_3 = int(df2.groupby(factors, observed=True).ngroups)
        except Exception:
            n_cells_3 = 0
        max_3way_levels = int(getattr(config, "max_3way_levels", 2500) or 2500)
        if 0 < n_cells_3 <= max_3way_levels:
            a, b, c = factors
            vc_formula[f"{a}:{b}:{c}"] = f"0 + C(Q('{a}')):C(Q('{b}')):C(Q('{c}'))"
            include_3way = True
        else:
            warnings.append(
                "3-way interaction term was omitted in MixedLM due to high cardinality; "
                "any 3-way interaction is absorbed into repeatability/residual."
            )
        diag["3way_levels"] = int(n_cells_3)
    diag["include_3way"] = bool(include_3way)

    # Fit MixedLM
    df_fit = df2.assign(dummy_group=1)
    model = smf.mixedlm(f'Q("{y}") ~ 1', df_fit, vc_formula=vc_formula, groups="dummy_group")

    fit_res = None
    fit_errors: List[str] = []

    tries = [
        ("lbfgs", dict(maxiter=2000, disp=False)),
        ("powell", dict(maxiter=4000, disp=False)),
        ("cg", dict(maxiter=4000, disp=False)),
        ("nm", dict(maxiter=6000, disp=False)),
    ]

    for method, kw in tries:
        try:
            import warnings as py_warnings
            from statsmodels.tools.sm_exceptions import ConvergenceWarning

            with py_warnings.catch_warnings(record=True) as wlist:
                py_warnings.simplefilter("always")
                res = model.fit(reml=True, method=method, **kw)

            fit_res = res
            diag["optimizer"] = method
            diag["converged"] = bool(getattr(res, "converged", True))
            diag["llf"] = float(getattr(res, "llf", np.nan))
            diag["fit_warnings"] = [
                {"category": getattr(w.category, "__name__", str(w.category)), "message": str(w.message)}
                for w in wlist
            ]

            unstable = False
            if not diag.get("converged", True):
                unstable = True
            for w in wlist:
                msg = str(w.message).lower()
                if issubclass(w.category, ConvergenceWarning):
                    unstable = True
                if "not positive definite" in msg and "hessian" in msg:
                    unstable = True
                if "boundary" in msg:
                    unstable = True
            diag["unstable"] = bool(unstable)
            if diag["unstable"]:
                warnings.append(
                    "MixedLM fit emitted convergence/boundary warnings; variance components may be unreliable."
                )
            break
        except Exception as e:
            fit_errors.append(f"{method}: {e}")

    # ------------------------------------------------------------
    # If MixedLM fails, go directly to Bayes
    # ------------------------------------------------------------
    if fit_res is None:
        warnings.append("MixedLM REML fit failed with all attempted optimizers; switching to Bayesian fallback.")
        for msg in fit_errors[:4]:
            warnings.append(f"  - {msg}")
        diag["method"] = "bayesian"
        vc_map = _bayes_fallback_crossed(df2, y, factors, all_factors, dropped, config, warnings, diag)
        anova_rows, resid = _reference_anova_crossed(df2, y, factors, ANOVATableRow, warnings)
        return build_result_object(
            df2,
            config,
            vc_map,
            anova_rows,
            resid,
            warnings,
            diag,
            VarianceComponentRow,
            GRRSummary,
            ChartData,
            MSAResult,
        )

    # ------------------------------------------------------------
    # Variance components from MixedLM
    # ------------------------------------------------------------
    vc_map: Dict[str, float] = {"Repeatability": float(max(getattr(fit_res, "scale", 0.0), 0.0))}

    vcomp_values = np.asarray(getattr(fit_res, "vcomp", []), dtype=float)
    names = (
        list(getattr(getattr(fit_res, "model", None), "exog_vc", object()).names)
        if hasattr(getattr(fit_res, "model", None), "exog_vc")
        else []
    )

    diag["vcomp_names"] = [str(n) for n in names]
    diag["vcomp_values_raw"] = [float(v) for v in vcomp_values.tolist()] if vcomp_values.size else []

    mapped = False
    if names and len(names) == int(vcomp_values.size):
        mapped = True
        name_map: Dict[str, str] = {}
        for name, val in zip(names, vcomp_values):
            raw_name = str(name)
            canon_name = canonical_vc_term(raw_name, factor_order=factors)
            name_map[raw_name] = canon_name
            vc_map[canon_name] = float(max(val, 0.0))
            if val < 0:
                warnings.append(f"Variance component for '{canon_name}' was negative and truncated to 0.")
        diag["vcomp_name_map"] = name_map

    if not mapped:
        warnings.append(
            "WARNING: Could not reliably map MixedLM variance components by name; falling back to vc_formula key order. "
            "This may lead to swapped components if statsmodels internal ordering differs."
        )
        keys = [canonical_vc_term(k, factor_order=factors) for k in list(vc_formula.keys())]
        for i, key in enumerate(keys):
            if i >= vcomp_values.size:
                break
            vc_map[key] = float(max(vcomp_values[i], 0.0))
            if vcomp_values[i] < 0:
                warnings.append(f"Variance component for '{key}' was negative and truncated to 0.")
        diag["vcomp_mapping_fallback"] = True

    # Ensure dropped/constant factors are present (0 variance)
    for f in dropped:
        vc_map.setdefault(str(f), 0.0)

    # Boundary / instability trigger -> Bayes fallback
    raw_vals = np.asarray(getattr(fit_res, "vcomp", []), dtype=float)
    if bool(diag.get("unstable")) or (raw_vals.size and np.any(raw_vals < -1e-12)):
        warnings.append("MixedLM solution appears on/near a boundary; switching to Bayesian variance components.")
        diag["method"] = "bayesian"
        vc_map = _bayes_fallback_crossed(df2, y, factors, all_factors, dropped, config, warnings, diag)
    else:
        diag["method"] = "mixedlm_reml"

    warnings.append("ANOVA table is for reference only and not used for VC estimation.")
    anova_rows, resid = _reference_anova_crossed(df2, y, factors, ANOVATableRow, warnings)

    return build_result_object(
        df2,
        config,
        vc_map,
        anova_rows,
        resid,
        warnings,
        diag,
        VarianceComponentRow,
        GRRSummary,
        ChartData,
        MSAResult,
    )


def _reference_anova_crossed(
    df: pd.DataFrame,
    y: str,
    factors: List[str],
    ANOVATableRow,
    warnings: List[str],
) -> Tuple[List[Any], pd.Series]:
    """Fixed-effects reference ANOVA table for crossed designs.

    Uses Type III for unbalanced designs when possible; falls back to Type II.
    """
    anova_rows: List[Any] = []
    resid = pd.Series(dtype=float)
    try:
        interaction_terms = [f"C(Q('{f1}')):C(Q('{f2}'))" for f1, f2 in combinations(factors, 2)]
        main_terms = [f"C(Q('{f}'))" for f in factors]
        ols_formula = f'Q("{y}") ~ ' + " + ".join(main_terms)
        if interaction_terms:
            ols_formula += " + " + " + ".join(interaction_terms)
        ols_model = smf.ols(ols_formula, data=df).fit()

        # Prefer Type III for unbalanced / incomplete designs
        try:
            anova = anova_lm(ols_model, typ=3)
        except Exception:
            anova = anova_lm(ols_model, typ=2)
        anova_rows = build_anova_rows(
            clean_anova_index(anova), ANOVATableRow, df, y, factors
        )
        resid = ols_model.resid
    except Exception as e:
        warnings.append(f"Could not generate reference ANOVA table: {e}")
    return anova_rows, resid


def _bayes_fallback_crossed(
    df: pd.DataFrame,
    y: str,
    active_factors: List[str],
    all_factors: List[str],
    dropped: List[str],
    config,
    warnings: List[str],
    diag: Dict[str, Any],
) -> Dict[str, float]:
    """Bayesian Gibbs fallback for crossed designs.

    Uses an additive random-intercepts model over:
      - main effects (active_factors)
      - 2-way interactions between active_factors
      - 3-way interaction when len(active_factors)==3

    This is used only as a *fallback* when REML fails or is on/near the boundary.
    """
    max_levels_total = int(getattr(config, "max_bayes_levels", 2500) or 2500)
    df_b = df.copy()

    gibbs_cols: List[str] = []
    col_to_key: Dict[str, str] = {}

    for f in active_factors:
        gibbs_cols.append(f)
        col_to_key[f] = str(f)

    # Interaction columns
    for f1, f2 in combinations(active_factors, 2):
        key = canonical_vc_term(f"{f1}:{f2}", factor_order=all_factors)
        col = f"__int__{key}"
        df_b[col] = (df_b[f1].astype(str) + ":" + df_b[f2].astype(str)).astype("category")
        gibbs_cols.append(col)
        col_to_key[col] = key

    if len(active_factors) == 3:
        a, b, c = active_factors
        key3 = canonical_vc_term(f"{a}:{b}:{c}", factor_order=all_factors)
        col3 = f"__int__{key3}"
        df_b[col3] = (df_b[a].astype(str) + ":" + df_b[b].astype(str) + ":" + df_b[c].astype(str)).astype(
            "category"
        )
        gibbs_cols.append(col3)
        col_to_key[col3] = key3

    total_levels = 0
    for c in gibbs_cols:
        try:
            total_levels += int(df_b[c].nunique())
        except Exception:
            pass

    if total_levels > max_levels_total:
        warnings.append(
            "Bayesian fallback skipped due to high total random-effect cardinality ("
            + str(total_levels)
            + "). Returning conservative repeatability-only estimate instead."
        )
        diag["bayes"] = {"skipped": True, "total_levels": int(total_levels), "max_levels_total": int(max_levels_total)}
        vc_map: Dict[str, float] = {
            "Repeatability": float(np.var(df_b[y].astype(float), ddof=1)) if len(df_b) > 1 else 0.0
        }
        for f in all_factors:
            vc_map.setdefault(str(f), 0.0)
        return vc_map

    warnings.append("Using Bayesian Gibbs sampler for variance components (fallback).")
    post, gibbs_diag = gibbs_random_intercepts_main_effects(
        df_b,
        y,
        gibbs_cols,
        seed=int(getattr(config, "random_seed", 0) or 0),
        draws=3000,
        burn=1000,
        a0=2.1,
        summary_stat="mean",
    )

    vc_map: Dict[str, float] = {"Repeatability": float(post.get("Residual", 0.0))}
    for c in gibbs_cols:
        key = col_to_key.get(c, c)
        vc_map[str(key)] = float(post.get(c, 0.0))

    for f in dropped:
        vc_map.setdefault(str(f), 0.0)

    diag["bayes"] = gibbs_diag.__dict__
    return vc_map
