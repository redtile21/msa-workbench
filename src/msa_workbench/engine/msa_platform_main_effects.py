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
            # Capture statsmodels convergence/boundary warnings; these are a strong signal that
            # the REML solution is on/near the boundary and variance components can be inflated.
            import warnings as py_warnings
            from statsmodels.tools.sm_exceptions import ConvergenceWarning

            with py_warnings.catch_warnings(record=True) as wlist:
                py_warnings.simplefilter("always")
                res = model.fit(reml=True, method=method, **kw)

            fit_res = res
            diag["optimizer"] = method
            diag["converged"] = bool(getattr(res, "converged", True))
            diag["llf"] = float(getattr(res, "llf", np.nan))

            # Persist the warning messages for diagnostics
            diag["fit_warnings"] = [
                {"category": getattr(w.category, "__name__", str(w.category)), "message": str(w.message)}
                for w in wlist
            ]

            # Heuristic stability flag: any ConvergenceWarning / boundary / non-PD Hessian
            unstable = False
            if not diag.get("converged", True):
                unstable = True
            for w in wlist:
                msg = str(w.message)
                if issubclass(w.category, ConvergenceWarning):
                    unstable = True
                if "not positive definite" in msg.lower() and "hessian" in msg.lower():
                    unstable = True
                if "boundary" in msg.lower():
                    unstable = True
            diag["unstable"] = bool(unstable)

            if diag["unstable"]:
                warnings.append(
                    "MixedLM fit emitted convergence/boundary warnings (e.g., non-positive definite Hessian). "
                    "Variance components may be unreliable."
                )
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
                "Negative EMS variance component(s) detected; attempting Mixed-Effects (REML) estimation: "
                + ", ".join([f"{k}={v:.6g}" for k, v in negatives.items()])
            )

            mixed_vc, mixed_diag, mixed_warn = _fit_mixedlm_main_effects(df2, y, factors)
            warnings.extend(mixed_warn)

            # If the REML fit is unstable / on the boundary (common when the true VC is ~0),
            # prefer Bayesian shrinkage estimates. This matches JMP's behavior when it reports:
            # "Switching to Bayesian estimates because of negative REML variance component(s)."
            use_bayes = False

            if mixed_vc is not None:
                # If EMS says a component is negative, it's usually near zero. If MixedLM returns
                # a comparatively large VC for such a factor, treat it as suspicious.
                rep = float(mixed_vc.get("Repeatability", 0.0))
                suspicious: List[str] = []
                for k in negatives.keys():
                    est = float(mixed_vc.get(k, 0.0))
                    if rep > 0 and est > 0.5 * rep:
                        suspicious.append(f"{k}={est:.6g}")

                if bool(mixed_diag.get("unstable")):
                    use_bayes = True
                    warnings.append(
                        "MixedLM fit appears unstable (convergence/boundary warnings). "
                        "Switching to Bayesian variance component estimates (JMP-like fallback)."
                    )
                elif suspicious:
                    use_bayes = True
                    warnings.append(
                        "MixedLM produced large variance component(s) for factor(s) with negative EMS estimates: "
                        + ", ".join(suspicious)
                        + ". Switching to Bayesian variance component estimates (JMP-like fallback)."
                    )
                else:
                    vc_map = mixed_vc
                    diag["mixedlm"] = mixed_diag
                    diag["method"] = "mixedlm_reml"
            else:
                use_bayes = True
                warnings.append("MixedLM failed; switching to Bayesian variance components (JMP-like fallback).")

            if use_bayes:
                post, gibbs_diag = gibbs_random_intercepts_main_effects(
                    df2, y, factors,
                    seed=0, draws=3000, burn=1000,
                    a0=2.1,
                    summary_stat="mean",
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

        if mixed_vc is not None and not bool(mixed_diag.get("unstable")):
            vc_map = mixed_vc
            diag["mixedlm"] = mixed_diag
            diag["method"] = "mixedlm_reml"
        else:
            if mixed_vc is None:
                warnings.append("MixedLM failed; falling back to Bayesian variance components (JMP-like fallback).")
            else:
                warnings.append(
                    "MixedLM fit appears unstable (convergence/boundary warnings). "
                    "Falling back to Bayesian variance components (JMP-like fallback)."
                )

            post, gibbs_diag = gibbs_random_intercepts_main_effects(
                df2, y, factors,
                seed=0, draws=3000, burn=1000,
                a0=2.1,
                summary_stat="mean",
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
