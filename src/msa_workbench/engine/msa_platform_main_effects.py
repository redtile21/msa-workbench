from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
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


def _fit_mixedlm_main_effects(
    df: pd.DataFrame,
    response_col: str,
    factor_cols: List[str],
) -> Tuple[Optional[Dict[str, float]], Dict[str, Any], List[str]]:
    """Fit a random-intercepts MixedLM (REML) for a main-effects (additive) random model.

    Returns (vc_map | None, diag, warnings).
    """
    warnings: List[str] = []
    diag: Dict[str, Any] = {}

    y = response_col
    factors = list(factor_cols)

    df_fit = df.assign(dummy_group=1)

    # Variance components specification (no interactions for a main-effects model)
    vc_formula: Dict[str, str] = {f: f"0 + C(Q('{f}'))" for f in factors}

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
        return None, diag, warnings

    # -------------------------
    # Variance component mapping
    # -------------------------
    vc_map: Dict[str, float] = {"Repeatability": float(max(getattr(fit_res, "scale", 0.0), 0.0))}

    vcomp_values = np.asarray(getattr(fit_res, "vcomp", []), dtype=float)
    names = list(getattr(getattr(fit_res, "model", None), "exog_vc", object()).names) if hasattr(getattr(fit_res, "model", None), "exog_vc") else []

    diag["vcomp_names"] = [str(n) for n in names]
    diag["vcomp_values_raw"] = [float(v) for v in vcomp_values.tolist()] if vcomp_values.size else []

    mapped = False
    if names and len(names) == int(vcomp_values.size):
        mapped = True
        for name, val in zip(names, vcomp_values):
            key = str(name)
            vc_map[key] = float(max(val, 0.0))
            if val < 0:
                warnings.append(f"Variance component for '{key}' was negative and truncated to 0.")

    if not mapped:
        # Fallback: map by vc_formula keys in insertion order (should be stable for most statsmodels versions)
        warnings.append(
            "WARNING: Could not reliably map MixedLM variance components by name; falling back to vc_formula key order."
        )
        keys = list(vc_formula.keys())
        for i, key in enumerate(keys):
            if i >= vcomp_values.size:
                break
            vc_map[key] = float(max(vcomp_values[i], 0.0))
            if vcomp_values[i] < 0:
                warnings.append(f"Variance component for '{key}' was negative and truncated to 0.")
        diag["vcomp_mapping_fallback"] = True

    return vc_map, diag, warnings


def run_main_effects(df: pd.DataFrame, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult):
    warnings: List[str] = []
    df2 = validate_dataframe(df, config.response_col, config.factor_cols)

    y = config.response_col
    factors = config.factor_cols

    diag: Dict[str, Any] = {"platform": "main_effects"}
    design_diag = design_diagnostics(df2, factors)
    diag["design"] = design_diag

    # Always compute the reference (fixed-effects) ANOVA table for reporting.
    anova_rows: List[Any] = []
    resid = pd.Series(dtype=float)
    try:
        formula = f'Q("{y}") ~ ' + " + ".join([f'C(Q("{f}"))' for f in factors])
        ols_model = smf.ols(formula, data=df2).fit()
        anova = anova_lm(ols_model, typ=2)

        # Update F-tests vs residual (reference)
        ms_res, df_res = get_ms_df(anova, "Residual")
        for f in factors:
            term = find_term(anova, f)
            ms_f, df_f = get_ms_df(anova, term)
            update_anova_f_test(anova, term, ms_f, ms_res, df_f, df_res)

        anova_rows = build_anova_rows(clean_anova_index(anova), ANOVATableRow)
        resid = ols_model.resid
    except Exception as e:
        warnings.append(f"Could not generate reference ANOVA table: {e}")
        anova_rows = []
        resid = pd.Series(dtype=float)

    # -------------------------
    # Variance component method
    # -------------------------
    use_ems = is_balanced_and_complete(design_diag, std_tol=1e-8, missing_tol_pct=1e-6)

    vc_map: Dict[str, float] = {}

    if use_ems:
        warnings.append("Balanced design detected. Using ANOVA/EMS for initial variance components.")
        # EMS via OLS ANOVA mean squares
        formula = f'Q("{y}") ~ ' + " + ".join([f'C(Q("{f}"))' for f in factors])
        model = smf.ols(formula, data=df2).fit()
        anova = anova_lm(model, typ=2)

        ms_res, df_res = get_ms_df(anova, "Residual")
        reps = df2.groupby(factors, observed=True).size()
        r = float(reps.mean()) if len(reps) else 1.0
        r = max(1.0, r)

        n_levels = {f: int(df2[f].nunique()) for f in factors}

        vc_map = {"Repeatability": float(max(ms_res, 0.0))}
        raw_vc: Dict[str, float] = {}

        for f in factors:
            term = find_term(anova, f)
            ms_f, df_f = get_ms_df(anova, term)

            denom = r
            for g in factors:
                if g != f:
                    denom *= max(1, n_levels[g])
            est = float((ms_f - ms_res) / denom) if denom > 0 else 0.0
            raw_vc[f] = est
            vc_map[f] = est

        negatives = {k: v for k, v in raw_vc.items() if v < 0}
        if negatives:
            warnings.append(
                "Negative EMS variance component(s) detected; switching to Mixed-Effects (REML) estimation: "
                + ", ".join([f"{k}={v:.6g}" for k, v in negatives.items()])
            )
            mixed_vc, mixed_diag, mixed_warn = _fit_mixedlm_main_effects(df2, y, factors)
            warnings.extend(mixed_warn)
            if mixed_vc is not None:
                vc_map = mixed_vc
                diag["mixedlm"] = mixed_diag
                diag["method"] = "mixedlm_reml"
            else:
                warnings.append("MixedLM failed; falling back to Bayesian variance components (median).")
                post, gibbs_diag = gibbs_random_intercepts_main_effects(
                    df2, y, factors,
                    seed=0, draws=3000, burn=1000,
                    summary_stat="median",
                )
                vc_map = {"Repeatability": float(post["Residual"])}
                for f in factors:
                    vc_map[f] = float(post[f])
                diag["bayes"] = gibbs_diag.__dict__
                diag["method"] = "bayesian"
        else:
            # truncate any small negatives due to numeric noise
            for f in factors:
                vc_map[f] = float(max(0.0, vc_map[f]))
            diag["method"] = "ems"
            diag["ems"] = {"r": float(r), "n_levels": n_levels, "ms_residual": float(ms_res)}
    else:
        warnings.append("Unbalanced or incomplete design detected. Using Mixed-Effects Model (REML).")
        mixed_vc, mixed_diag, mixed_warn = _fit_mixedlm_main_effects(df2, y, factors)
        warnings.extend(mixed_warn)

        if mixed_vc is not None:
            vc_map = mixed_vc
            diag["mixedlm"] = mixed_diag
            diag["method"] = "mixedlm_reml"
        else:
            warnings.append("MixedLM failed; falling back to Bayesian variance components (median).")
            post, gibbs_diag = gibbs_random_intercepts_main_effects(
                df2, y, factors,
                seed=0, draws=3000, burn=1000,
                summary_stat="median",
            )
            vc_map = {"Repeatability": float(post["Residual"])}
            for f in factors:
                vc_map[f] = float(post[f])

            diag["bayes"] = gibbs_diag.__dict__
            diag["method"] = "bayesian"
            warnings.append("ANOVA table is for reference only and not used for VC estimation.")

    return build_result_object(
        df2, config, vc_map, anova_rows, resid, warnings, diag,
        VarianceComponentRow, GRRSummary, ChartData, MSAResult
    )
