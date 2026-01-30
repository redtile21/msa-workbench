from __future__ import annotations

from typing import Dict, Any, List, Optional
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from .msa_utils import (
    validate_dataframe,
    design_diagnostics,
    is_balanced_and_complete,
    canonical_vc_term,
    clean_anova_index,
    find_term,
    get_ms_df,
    satterthwaite_df,
    update_anova_f_test,
    build_anova_rows,
)
from .msa_results import build_result_object


def run_crossed_2factor(df: pd.DataFrame, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult):
    warnings: List[str] = []
    df2 = validate_dataframe(df, config.response_col, config.factor_cols)

    part = config.part_col
    op = config.operator_col
    y = config.response_col

    diag: Dict[str, Any] = {"platform": "crossed_2factor"}
    design_diag = design_diagnostics(df2, config.factor_cols)
    diag["design"] = design_diag

    # Balanced & complete => classical ANOVA/EMS; else MixedLM
    if not is_balanced_and_complete(design_diag, std_tol=1e-8, missing_tol_pct=1e-6):
        warnings.append("Unbalanced or incomplete design detected. Falling back to Mixed Model estimation.")
        return run_crossed_mixed(df2, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult, warnings=warnings)

    warnings.append("Balanced design detected. Using ANOVA/EMS method.")
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

    anova_rows = build_anova_rows(clean_anova_index(anova), ANOVATableRow)

    n_parts = design_diag.get("level_counts", {}).get(part, 1) or 1
    n_ops = design_diag.get("level_counts", {}).get(op, 1) or 1
    n_reps = int(round(design_diag.get("replicate_dist", {}).get("mean", 1) or 1))

    sig2_repeat = max(ms_res, 0.0)
    sig2_part_op = max((ms_int - ms_res) / n_reps, 0.0) if n_reps > 0 else 0.0
    sig2_op = max((ms_op - ms_int) / (n_parts * n_reps), 0.0) if n_parts > 0 and n_reps > 0 else 0.0
    sig2_part = max((ms_part - ms_int) / (n_ops * n_reps), 0.0) if n_ops > 0 and n_reps > 0 else 0.0

    int_key = canonical_vc_term(f"{part}:{op}", factor_order=config.factor_cols)
    vc_map = {
        "Repeatability": sig2_repeat,
        f"{op}": sig2_op,
        int_key: sig2_part_op,
        f"{part}": sig2_part,
    }

    return build_result_object(
        df2, config, vc_map, anova_rows, model.resid, warnings, diag,
        VarianceComponentRow, GRRSummary, ChartData, MSAResult
    )


def run_crossed_3factor(df: pd.DataFrame, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult):
    warnings: List[str] = []
    df2 = validate_dataframe(df, config.response_col, config.factor_cols)

    fac_a = config.part_col
    fac_b = config.operator_col
    all_factors = config.factor_cols
    others = [f for f in all_factors if f != fac_a and f != fac_b]
    if not others:
        raise ValueError("3-factor crossed model requires a 3rd factor besides part and operator.")
    fac_c = others[0]
    y = config.response_col

    diag: Dict[str, Any] = {"platform": "crossed_3factor"}
    design_diag = design_diagnostics(df2, config.factor_cols)
    diag["design"] = design_diag

    if not is_balanced_and_complete(design_diag, std_tol=1e-8, missing_tol_pct=1e-6):
        warnings.append("Unbalanced or incomplete design detected. Falling back to Mixed Model estimation.")
        return run_crossed_mixed(df2, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult, warnings=warnings)

    warnings.append("Balanced design detected. Using ANOVA/EMS method.")

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

    update_anova_f_test(anova, t_ABC, ms_ABC, ms_Res, df_ABC, df_Res)
    update_anova_f_test(anova, t_AB, ms_AB, ms_ABC, df_AB, df_ABC)
    update_anova_f_test(anova, t_AC, ms_AC, ms_ABC, df_AC, df_ABC)
    update_anova_f_test(anova, t_BC, ms_BC, ms_ABC, df_BC, df_ABC)

    ms_denom_A = ms_AB + ms_AC - ms_ABC
    df_denom_A = satterthwaite_df(ms_AB, df_AB, ms_AC, df_AC, ms_ABC, df_ABC)
    update_anova_f_test(anova, t_A, ms_A, ms_denom_A, df_A, df_denom_A)

    ms_denom_B = ms_AB + ms_BC - ms_ABC
    df_denom_B = satterthwaite_df(ms_AB, df_AB, ms_BC, df_BC, ms_ABC, df_ABC)
    update_anova_f_test(anova, t_B, ms_B, ms_denom_B, df_B, df_denom_B)

    ms_denom_C = ms_AC + ms_BC - ms_ABC
    df_denom_C = satterthwaite_df(ms_AC, df_AC, ms_BC, df_BC, ms_ABC, df_ABC)
    update_anova_f_test(anova, t_C, ms_C, ms_denom_C, df_C, df_denom_C)

    anova_rows = build_anova_rows(clean_anova_index(anova), ANOVATableRow)

    n_a = design_diag.get("level_counts", {}).get(fac_a, 1) or 1
    n_b = design_diag.get("level_counts", {}).get(fac_b, 1) or 1
    n_c = design_diag.get("level_counts", {}).get(fac_c, 1) or 1
    n_r = int(round(design_diag.get("replicate_dist", {}).get("mean", 1) or 1))

    var_e = max(0.0, ms_Res)
    var_abc = max(0.0, (ms_ABC - ms_Res) / n_r) if n_r > 0 else 0.0
    var_ab = max(0.0, (ms_AB - ms_ABC) / (n_r * n_c)) if n_r > 0 and n_c > 0 else 0.0
    var_ac = max(0.0, (ms_AC - ms_ABC) / (n_r * n_b)) if n_r > 0 and n_b > 0 else 0.0
    var_bc = max(0.0, (ms_BC - ms_ABC) / (n_r * n_a)) if n_r > 0 and n_a > 0 else 0.0
    var_a = max(0.0, (ms_A - ms_AB - ms_AC + ms_ABC) / (n_r * n_b * n_c)) if n_r > 0 and n_b > 0 and n_c > 0 else 0.0
    var_b = max(0.0, (ms_B - ms_AB - ms_BC + ms_ABC) / (n_r * n_a * n_c)) if n_r > 0 and n_a > 0 and n_c > 0 else 0.0
    var_c = max(0.0, (ms_C - ms_AC - ms_BC + ms_ABC) / (n_r * n_a * n_b)) if n_r > 0 and n_a > 0 and n_b > 0 else 0.0

    vc_map = {
        "Repeatability": var_e,
        f"{fac_a}": var_a,
        f"{fac_b}": var_b,
        f"{fac_c}": var_c,
        canonical_vc_term(f"{fac_a}:{fac_b}", factor_order=config.factor_cols): var_ab,
        canonical_vc_term(f"{fac_a}:{fac_c}", factor_order=config.factor_cols): var_ac,
        canonical_vc_term(f"{fac_b}:{fac_c}", factor_order=config.factor_cols): var_bc,
        canonical_vc_term(f"{fac_a}:{fac_b}:{fac_c}", factor_order=config.factor_cols): var_abc,
    }

    return build_result_object(
        df2, config, vc_map, anova_rows, model.resid, warnings, diag,
        VarianceComponentRow, GRRSummary, ChartData, MSAResult
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
    """Mixed model variance components for unbalanced/incomplete crossed designs.

    Keeps interaction terms limited to 2-way (same behavior as prior engine).
    """
    warnings = list(warnings or [])
    warnings.extend(
        [
            "Using Mixed-Effects Model (REML) due to unbalanced or incomplete design.",
            "Interaction terms beyond 2-way are not included.",
        ]
    )
    diag: Dict[str, Any] = {"platform": "crossed_mixed"}

    df2 = df
    y = config.response_col
    factors = config.factor_cols

    # Variance components specification
    vc_formula: Dict[str, str] = {}
    for factor in factors:
        vc_formula[factor] = f"0 + C(Q('{factor}'))"
    for fac1, fac2 in combinations(factors, 2):
        vc_formula[f"{fac1}:{fac2}"] = f"0 + C(Q('{fac1}')):C(Q('{fac2}'))"

    df_fit = df2.assign(dummy_group=1)

    # MixedLM can be finicky; keep a small, robust set of optimizers with caps.
    model = smf.mixedlm(f'Q("{y}") ~ 1', df_fit, vc_formula=vc_formula, groups="dummy_group")
    fit_res = None
    fit_errors = []

    tries = [
        ("lbfgs", dict(maxiter=2000, disp=False)),
        ("powell", dict(maxiter=4000, disp=False)),
        ("cg", dict(maxiter=4000, disp=False)),
        ("nm", dict(maxiter=6000, disp=False)),
    ]

    for method, kw in tries:
        try:
            res = model.fit(reml=True, method=method, **kw)
            fit_res = res
            diag["optimizer"] = method
            diag["converged"] = bool(getattr(res, "converged", True))
            diag["llf"] = float(getattr(res, "llf", np.nan))
            break
        except Exception as e:
            fit_errors.append(f"{method}: {e}")

    if fit_res is None:
        warnings.append("MixedLM REML fit failed with all attempted optimizers.")
        for msg in fit_errors[:4]:
            warnings.append(f"  - {msg}")
        # Return empty VCs but still provide charts/diagnostics
        return build_result_object(
            df2, config, {}, [], pd.Series(dtype=float), warnings, diag,
            VarianceComponentRow, GRRSummary, ChartData, MSAResult
        )

    # ------------------------------------------------------------
    # Variance components
    # ------------------------------------------------------------
    # statsmodels returns variance components in the order of `fit_res.model.exog_vc.names`.
    # This order is *not guaranteed* to match the insertion order of `vc_formula`.
    vc_map: Dict[str, float] = {"Repeatability": float(max(getattr(fit_res, "scale", 0.0), 0.0))}

    vcomp_values = np.asarray(getattr(fit_res, "vcomp", []), dtype=float)
    names = list(getattr(getattr(fit_res, "model", None), "exog_vc", object()).names) if hasattr(getattr(fit_res, "model", None), "exog_vc") else []

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
        # Fallback: map by vc_formula keys, but still canonicalize terms.
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

    # Reference ANOVA table (fixed effects); do not use for estimation
    warnings.append("ANOVA table is for reference only and not used for VC estimation.")
    anova_rows: List[Any] = []
    resid = pd.Series(dtype=float)

    try:
        interaction_terms = [f"C(Q('{f1}')):C(Q('{f2}'))" for f1, f2 in combinations(factors, 2)]
        main_terms = [f"C(Q('{f}'))" for f in factors]
        ols_formula = f'Q("{y}") ~ {" + ".join(main_terms)}'
        if interaction_terms:
            ols_formula += " + " + " + ".join(interaction_terms)
        ols_model = smf.ols(ols_formula, data=df2).fit()
        anova = anova_lm(ols_model, typ=2)
        anova_rows = build_anova_rows(clean_anova_index(anova), ANOVATableRow)
        resid = ols_model.resid
    except Exception as e:
        warnings.append(f"Could not generate reference ANOVA table: {e}")

    return build_result_object(
        df2, config, vc_map, anova_rows, resid, warnings, diag,
        VarianceComponentRow, GRRSummary, ChartData, MSAResult
    )
