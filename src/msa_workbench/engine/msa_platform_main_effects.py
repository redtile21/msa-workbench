from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from .msa_bayes import gibbs_random_intercepts_main_effects
from .msa_results import build_result_object
from .msa_utils import (
    branching_diagnostics,
    build_anova_rows,
    clean_anova_index,
    design_diagnostics,
    find_term,
    get_ms_df,
    infer_nesting,
    infer_nesting_chain,
    is_balanced_and_complete,
    is_balanced_main_effects,
    update_anova_f_test,
    validate_dataframe,
)


def _prepare_vc_columns(
    df: pd.DataFrame,
    factors: List[str],
    nesting_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    """Create/return (df_with_ids, vc_columns, nesting_info).

    vc_columns maps output factor name -> column used in variance-component formulas.

    If nesting is detected (e.g., Part nested in Operator, and Operator labels repeat across a higher-level
    factor), we build recursive IDs so that nested groups remain unique.
    """
    nesting_info = nesting_info or infer_nesting(df, factors)
    parent_map: Dict[str, str] = dict(nesting_info.get("parent_map", {}))

    df_out = df.copy()
    vc_columns: Dict[str, str] = {}
    visiting: set[str] = set()

    def _id_col(f: str) -> str:
        if f in vc_columns:
            return vc_columns[f]
        if f in visiting:
            # cycle; fall back to raw column
            vc_columns[f] = f
            return f
        visiting.add(f)

        parent = parent_map.get(f)
        if not parent:
            vc_columns[f] = f
            visiting.discard(f)
            return f

        parent_id = _id_col(parent)
        new_col = f"__{f}__in__{parent}"

        base = df_out[parent_id].astype(str) if parent_id in df_out.columns else df_out[parent].astype(str)
        df_out[new_col] = (base + "|" + df_out[f].astype(str)).astype("category")

        vc_columns[f] = new_col
        visiting.discard(f)
        return new_col

    for f in factors:
        _id_col(str(f))

    return df_out, vc_columns, nesting_info


def _fit_mixedlm_main_effects(
    df: pd.DataFrame,
    response_col: str,
    factor_cols: List[str],
    *,
    vc_columns: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, float]], Dict[str, Any], List[str]]:
    """Fit a random-intercepts MixedLM (REML) for an additive random-effects model.

    Returns (vc_map | None, diag, warnings).
    """
    warnings: List[str] = []
    diag: Dict[str, Any] = {}

    y = response_col
    factors = list(factor_cols)
    vc_columns = dict(vc_columns or {f: f for f in factors})

    # Drop constant factors from MixedLM for identifiability (caller will report them as 0)
    active_factors: List[str] = []
    dropped: List[str] = []
    for f in factors:
        col = vc_columns.get(f, f)
        try:
            if int(df[col].nunique()) < 2:
                dropped.append(f)
            else:
                active_factors.append(f)
        except Exception:
            active_factors.append(f)

    diag["dropped_factors"] = list(dropped)
    diag["active_factors"] = list(active_factors)

    df_fit = df.assign(dummy_group=1)

    # Variance components specification (main effects only)
    vc_formula: Dict[str, str] = {
        f: f"0 + C(Q('{vc_columns.get(f, f)}'))" for f in active_factors
    }

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

    if fit_res is None:
        warnings.append("MixedLM REML fit failed with all attempted optimizers.")
        for msg in fit_errors[:4]:
            warnings.append(f"  - {msg}")
        return None, diag, warnings

    # Map variance components
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
        for name, val in zip(names, vcomp_values):
            key = str(name)
            vc_map[key] = float(max(val, 0.0))
            if val < 0:
                warnings.append(f"Variance component for '{key}' was negative and truncated to 0.")

    if not mapped:
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

    # Ensure dropped factors are present as 0
    for f in dropped:
        vc_map.setdefault(str(f), 0.0)

    return vc_map, diag, warnings


def run_main_effects(
    df: pd.DataFrame,
    config,
    ANOVATableRow,
    VarianceComponentRow,
    GRRSummary,
    ChartData,
    MSAResult,
):
    """Main-effects MSA platform.

    Supports:
      - Fully crossed additive random effects designs (balanced -> EMS)
      - Hierarchical/nested main-effects designs (balanced hierarchical -> nested EMS)
      - Unbalanced/sparse designs -> MixedLM (REML) -> Bayesian Gibbs fallback
    """
    warnings: List[str] = []
    df2 = validate_dataframe(df, config.response_col, config.factor_cols)

    y = config.response_col
    factors = list(config.factor_cols)

    diag: Dict[str, Any] = {"platform": "main_effects"}
    design_diag = design_diagnostics(df2, factors)
    diag["design"] = design_diag

    # Nesting inference (used for routing and for MixedLM/Bayes encoding)
    nesting_info = infer_nesting(df2, factors)
    diag["nesting"] = nesting_info

    # Prepare nested id columns for mixed/bayes if needed
    df_vc, vc_columns, _ = _prepare_vc_columns(df2, factors, nesting_info=nesting_info)
    diag["vc_columns"] = dict(vc_columns)

    # Replication and balance indicators
    balanced_cells = is_balanced_main_effects(design_diag, std_tol=1e-8)
    full_factorial = is_balanced_and_complete(design_diag, std_tol=1e-8, missing_tol_pct=1e-6)
    r_mean = float(design_diag.get("replicate_dist", {}).get("mean", 1.0) or 1.0)
    r = int(round(r_mean)) if r_mean > 0 else 1
    diag["replicates_per_cell_mean"] = float(r_mean)
    diag["replicates_per_cell_rounded"] = int(r)

    # Attempt a strict nesting chain inference (for nested EMS)
    chain = infer_nesting_chain(df2, factors)
    diag["nesting_chain"] = chain

    branch_stats: Dict[str, Any] = {}
    nested_balanced = False
    if chain:
        for parent, child in zip(chain[:-1], chain[1:]):
            st = branching_diagnostics(df2, parent, child)
            branch_stats[f"{child} per {parent}"] = st
        nested_balanced = all(float(st.get("std", 0.0) or 0.0) <= 1e-8 for st in branch_stats.values())
    diag["branching"] = branch_stats
    diag["nested_balanced"] = bool(nested_balanced)

    # ---------------------------------
    # Reference ANOVA (for reporting)
    # ---------------------------------
    anova_rows: List[Any] = []
    resid = pd.Series(dtype=float)
    try:
        formula_ref = f'Q("{y}") ~ ' + " + ".join([f'C(Q("{f}"))' for f in factors])
        ols_ref = smf.ols(formula_ref, data=df2).fit()
        try:
            anova_ref = anova_lm(ols_ref, typ=(2 if full_factorial else 3))
        except Exception:
            anova_ref = anova_lm(ols_ref, typ=2)

        ms_res, df_res = get_ms_df(anova_ref, "Residual")
        for f in factors:
            term = find_term(anova_ref, f)
            ms_f, df_f = get_ms_df(anova_ref, term)
            update_anova_f_test(anova_ref, term, ms_f, ms_res, df_f, df_res)

        anova_rows = build_anova_rows(
            clean_anova_index(anova_ref), ANOVATableRow, df2, y, factors
        )
        resid = ols_ref.resid
    except Exception as e:
        warnings.append(f"Could not generate reference ANOVA table: {e}")

    # ---------------------------------
    # Variance component estimation
    # ---------------------------------
    vc_map: Dict[str, float] = {}

    # EMS requires replication for an error term
    ems_possible = balanced_cells and (r >= 2)

    # 1) Balanced, full factorial: standard EMS (additive)
    if ems_possible and full_factorial:
        warnings.append("Balanced full-factorial design detected. Using ANOVA/EMS method.")
        diag["method"] = "ems"

        formula = f'Q("{y}") ~ ' + " + ".join([f'C(Q("{f}"))' for f in factors])
        model = smf.ols(formula, data=df2).fit()
        anova = anova_lm(model, typ=2)

        ms_res, _ = get_ms_df(anova, "Residual")
        n_levels = {f: int(df2[f].nunique()) for f in factors}

        vc_map = {"Repeatability": float(ms_res)}
        raw_vc: Dict[str, float] = {}
        for f in factors:
            term = find_term(anova, f)
            ms_f, _ = get_ms_df(anova, term)
            denom = float(r)
            for g in factors:
                if g != f:
                    denom *= float(max(1, n_levels[g]))
            est = float((ms_f - ms_res) / denom) if denom > 0 else 0.0
            raw_vc[f] = est
            vc_map[f] = est

        neg = {k: v for k, v in raw_vc.items() if v < -1e-12}
        if neg:
            warnings.append(
                "Negative EMS variance component(s) detected; switching to Mixed-Effects (REML): "
                + ", ".join([f"{k}={v:.6g}" for k, v in neg.items()])
            )
            vc_map = {}

    # 2) Balanced hierarchical (strict chain): nested EMS
    if not vc_map and ems_possible and chain and nested_balanced:
        warnings.append("Balanced hierarchical design detected. Using nested ANOVA/EMS method.")
        diag["method"] = "ems_nested"

        # Build nested ANOVA terms using recursive IDs (avoids brittle interaction coding).
        # Example (Part nested in Operator): y ~ C(Operator) + C(Part_in_Operator)
        terms: List[str] = []
        for f in chain:
            col = vc_columns.get(f, f)
            terms.append(f'C(Q("{col}"))')
        formula_nested = f'Q("{y}") ~ ' + " + ".join(terms)

        nested_model = smf.ols(formula_nested, data=df_vc).fit()
        anova_nested = anova_lm(nested_model, typ=1)

        ms_terms: List[float] = [get_ms_df(anova_nested, t)[0] for t in terms]
        ms_res, _ = get_ms_df(anova_nested, "Residual")

        # Overwrite reference ANOVA with nested ANOVA for this EMS path
        anova_rows = build_anova_rows(
            clean_anova_index(anova_nested), ANOVATableRow, df2, y, factors
        )
        resid = nested_model.resid

        # Branching means along the chain edges
        branch_means: List[int] = []
        for parent, child in zip(chain[:-1], chain[1:]):
            mean_ = float(branch_stats.get(f"{child} per {parent}", {}).get("mean", 0.0) or 0.0)
            branch_means.append(max(1, int(round(mean_))))

        # Precompute products of deeper branching counts
        cumprod = [1] * len(chain)
        running = 1
        for i in range(len(chain) - 2, -1, -1):
            running *= branch_means[i]
            cumprod[i] = running

        vc_map = {"Repeatability": float(ms_res)}

        # Deepest nested factor
        vc_map[chain[-1]] = float((ms_terms[-1] - ms_res) / float(r))

        # Work upward: Var(F_i) = (MS_i - MS_{i+1}) / (r * prod(deeper branch))
        for i in range(len(chain) - 2, -1, -1):
            denom = float(r) * float(cumprod[i])
            vc_map[chain[i]] = float((ms_terms[i] - ms_terms[i + 1]) / denom) if denom > 0 else 0.0

        neg = {k: v for k, v in vc_map.items() if k != "Repeatability" and v < -1e-12}
        if vc_map.get("Repeatability", 0.0) < -1e-12:
            neg["Repeatability"] = float(vc_map["Repeatability"])
        if neg:
            warnings.append(
                "Negative nested EMS variance component(s) detected; switching to Mixed-Effects (REML): "
                + ", ".join([f"{k}={v:.6g}" for k, v in neg.items()])
            )
            vc_map = {}

    # 3) MixedLM REML (unbalanced, incomplete, no replication, or EMS negative)
    if not vc_map:
        if not balanced_cells:
            warnings.append("Unbalanced replication detected. Using Mixed-Effects (REML) estimation.")
        elif r < 2:
            warnings.append("No replication detected (r≈1). Using Mixed-Effects (REML) estimation.")
        else:
            warnings.append("Using Mixed-Effects (REML) estimation.")

        mixed_vc, mixed_diag, mixed_warn = _fit_mixedlm_main_effects(
            df_vc, y, factors, vc_columns=vc_columns
        )
        warnings.extend(mixed_warn)
        diag["mixedlm"] = mixed_diag

        if mixed_vc is not None and not bool(mixed_diag.get("unstable", False)):
            vc_map = mixed_vc
            diag["method"] = "mixedlm_reml"
        else:
            warnings.append(
                "MixedLM REML appears unstable or failed; switching to Bayesian Gibbs sampler for variance components."
            )
            diag["method"] = "bayesian"

            active_cols = [vc_columns[f] for f in factors if f not in set(mixed_diag.get("dropped_factors", []))]
            post, gibbs_diag = gibbs_random_intercepts_main_effects(
                df_vc,
                y,
                active_cols,
                seed=int(getattr(config, "random_seed", 0) or 0),
                draws=3000,
                burn=1000,
                a0=2.1,
                summary_stat="mean",
            )

            vc_map = {"Repeatability": float(post.get("Residual", 0.0))}
            for f in factors:
                col = vc_columns[f]
                vc_map[f] = float(post.get(col, 0.0))

            for f in mixed_diag.get("dropped_factors", []):
                vc_map[str(f)] = 0.0

            diag["bayes"] = gibbs_diag.__dict__

    # Final sanitation: truncate small negatives (should not persist after routing)
    vc_map["Repeatability"] = float(max(0.0, vc_map.get("Repeatability", 0.0)))
    for f in factors:
        vc_map[f] = float(max(0.0, vc_map.get(f, 0.0)))

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
