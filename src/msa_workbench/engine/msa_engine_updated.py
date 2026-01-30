import os, textwrap, pathlib, zipfile, pandas as pd, numpy as np, math, json, re
base = pathlib.Path("/mnt/data/msa_refactor")
base.mkdir(exist_ok=True)

# -------------------------
# msa_utils.py
# -------------------------
msa_utils = r'''
# msa_utils.py
from __future__ import annotations

from typing import List, Optional, Dict, Any, Iterable, Tuple, Union
from itertools import permutations
import re

import numpy as np
import pandas as pd
from scipy import stats


def validate_dataframe(df: pd.DataFrame, response_col: str, factor_cols: List[str]) -> pd.DataFrame:
    """
    Returns a sanitized copy of df:
      - coerces response to numeric
      - drops NA response rows
      - casts factor cols to category (string-categories)
    """
    if response_col not in df.columns:
        raise KeyError(f"Response column '{response_col}' missing from dataframe.")
    for col in factor_cols:
        if col not in df.columns:
            raise KeyError(f"Factor column '{col}' missing from dataframe.")

    df2 = df.copy()
    df2[response_col] = pd.to_numeric(df2[response_col], errors="coerce")
    df2 = df2.dropna(subset=[response_col])

    for col in factor_cols:
        df2[col] = df2[col].astype(str).astype("category")

    return df2


def design_diagnostics(df: pd.DataFrame, factor_cols: List[str]) -> Dict[str, Any]:
    level_counts = {f: int(df[f].nunique()) for f in factor_cols}
    expected_cells = int(np.prod(list(level_counts.values()))) if level_counts else 0
    reps_per_cell = df.groupby(factor_cols, observed=True).size()
    actual_cells = int(len(reps_per_cell))
    rep_min = float(reps_per_cell.min()) if actual_cells > 0 else 0.0
    rep_mean = float(reps_per_cell.mean()) if actual_cells > 0 else 0.0
    rep_max = float(reps_per_cell.max()) if actual_cells > 0 else 0.0
    rep_std = float(reps_per_cell.std()) if actual_cells > 1 else 0.0
    missing_cells_pct = float((1 - actual_cells / expected_cells) * 100) if expected_cells > 0 else 0.0

    return {
        "level_counts": level_counts,
        "replicate_dist": {"min": rep_min, "mean": rep_mean, "max": rep_max, "std": rep_std},
        "expected_cells": expected_cells,
        "actual_cells": actual_cells,
        "missing_cells_pct": missing_cells_pct,
    }


def is_balanced_and_complete(design_diag: Dict[str, Any], std_tol: float = 1e-8, missing_tol_pct: float = 1e-6) -> bool:
    rep_std = float(design_diag.get("replicate_dist", {}).get("std", 0.0) or 0.0)
    missing_pct = float(design_diag.get("missing_cells_pct", 0.0) or 0.0)
    return (rep_std <= std_tol) and (missing_pct <= missing_tol_pct)


def clean_anova_index(anova: pd.DataFrame) -> pd.DataFrame:
    """Removes the C(Q("...")) wrapper from ANOVA index names for display."""
    clean_df = anova.copy()
    new_index = []
    for idx in clean_df.index:
        name = str(idx)
        clean_name = re.sub(r'C\(Q\("([^"]+)"\)\)', r"\1", name)
        clean_name = re.sub(r"C\(Q\('([^']+)'\)\)", r"\1", clean_name)
        new_index.append(clean_name)
    clean_df.index = new_index
    return clean_df


def find_term(anova: pd.DataFrame, cols: Union[str, List[str]]) -> str:
    """Find the exact index string in the ANOVA table for a set of columns."""
    if isinstance(cols, str):
        cols = [cols]
    c_terms = [f'C(Q("{c}"))' for c in cols]
    possible = [":".join(p) for p in permutations(c_terms)]
    for name in possible:
        if name in anova.index:
            return name
    # fallback: sometimes formula uses single quotes
    c_terms2 = [f"C(Q('{c}'))" for c in cols]
    possible2 = [":".join(p) for p in permutations(c_terms2)]
    for name in possible2:
        if name in anova.index:
            return name
    return ""


def get_ms_df(anova: pd.DataFrame, term: str) -> Tuple[float, float]:
    """Safely extract Mean Square and DF for a term."""
    if term not in anova.index:
        return 0.0, 1.0
    row = anova.loc[term]
    df_val = float(row["df"])
    if "mean_sq" in row:
        ms = float(row["mean_sq"])
    else:
        ss = float(row.get("sum_sq", row.get("ss", 0.0)))
        ms = ss / df_val if df_val > 0 else 0.0
    return float(ms), float(df_val)


def satterthwaite_df(ms1: float, df1: float, ms2: float, df2: float, ms3: float, df3: float) -> float:
    """
    Approx DF for linear combination L = MS1 + MS2 - MS3.
    DF = (MS1 + MS2 - MS3)^2 / [ (MS1^2/df1) + (MS2^2/df2) + (MS3^2/df3) ]
    """
    numerator = (ms1 + ms2 - ms3) ** 2
    denom = 0.0
    if df1 > 0:
        denom += (ms1 ** 2) / df1
    if df2 > 0:
        denom += (ms2 ** 2) / df2
    if df3 > 0:
        denom += (ms3 ** 2) / df3
    if denom <= 0:
        return 1.0
    return float(numerator / denom)


def update_anova_f_test(anova: pd.DataFrame, term: str, ms_num: float, ms_denom: float, df_num: float, df_denom: float) -> None:
    """Calculates F-ratio and p-value and updates the ANOVA table in-place."""
    if term not in anova.index:
        return
    if ms_denom <= 0:
        anova.loc[term, "F"] = np.nan
        anova.loc[term, "PR(>F)"] = np.nan
        return
    f_value = ms_num / ms_denom
    p_value = stats.f.sf(f_value, df_num, df_denom)
    anova.loc[term, "F"] = float(f_value)
    anova.loc[term, "PR(>F)"] = float(p_value)


def build_anova_rows(anova: pd.DataFrame, ANOVATableRow) -> List[Any]:
    rows = []
    for term, r in anova.iterrows():
        df_val = float(r["df"])
        ss = float(r.get("sum_sq", r.get("ss", 0.0)))
        ms = float(r.get("mean_sq", ss / df_val if df_val > 0 else 0.0))
        f = r.get("F", None)
        p = r.get("PR(>F)", None)
        rows.append(ANOVATableRow(str(term), df_val, ss, ms, None if pd.isna(f) else float(f), None if pd.isna(p) else float(p)))
    return rows


def shapiro_safe(resid: Union[pd.Series, np.ndarray]) -> Optional[float]:
    try:
        from scipy.stats import shapiro
        arr = np.asarray(resid)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None
        if arr.size > 5000:
            rng = np.random.default_rng(0)
            arr = rng.choice(arr, 5000, replace=False)
        return float(shapiro(arr)[1])
    except Exception:
        return None
'''
(base / "msa_utils.py").write_text(textwrap.dedent(msa_utils))

# -------------------------
# msa_bayes.py
# -------------------------
msa_bayes = r'''
# msa_bayes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class GibbsDiagnostics:
    draws: int
    burn: int
    seed: int
    a0: float
    b0: float
    post_means: Dict[str, float]
    post_sds: Dict[str, float]


def gibbs_random_intercepts_main_effects(
    df: pd.DataFrame,
    response_col: str,
    factor_cols: List[str],
    seed: int = 0,
    draws: int = 3000,
    burn: int = 1000,
    a0: float = 1e-3,
    b0: Optional[float] = None,
) -> Tuple[Dict[str, float], GibbsDiagnostics]:
    """
    Gibbs sampler for:
      y = mu + sum_j u_j[level_j] + e
      u_j ~ N(0, sigma_j^2 I)
      e ~ N(0, sigma_e^2 I)

    Priors (conjugate):
      sigma^2 ~ InvGamma(a0, b0)

    Returns posterior means of variance components (sigma^2), including:
      - "Residual" (sigma_e^2)
      - each factor name (sigma_j^2)
    """
    rng = np.random.default_rng(seed)

    y = np.asarray(df[response_col], dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("No rows to analyze after cleaning.")

    # Encode each factor as integer indices 0..(L-1)
    level_index = {}
    n_levels = {}
    for f in factor_cols:
        cats = df[f].cat.categories if hasattr(df[f], "cat") else pd.Categorical(df[f]).categories
        cat_to_i = {c: i for i, c in enumerate(list(cats))}
        idx = np.asarray([cat_to_i[v] for v in df[f].astype(str).values], dtype=int)
        level_index[f] = idx
        n_levels[f] = int(len(cats))

    # Hyperparameter scaling
    s2 = float(np.var(y, ddof=1)) if n > 1 else float(np.var(y))
    if b0 is None:
        b0 = max(1e-12, 1e-3 * s2)

    # State
    mu = float(np.mean(y))
    u = {f: np.zeros(n_levels[f], dtype=float) for f in factor_cols}
    sig2 = {f: max(1e-12, 0.1 * s2) for f in factor_cols}
    sig2_e = max(1e-12, 0.1 * s2)

    # Precompute observation indices per level for each factor
    obs_by_level = {}
    for f in factor_cols:
        L = n_levels[f]
        groups = [[] for _ in range(L)]
        idx = level_index[f]
        for i in range(n):
            groups[idx[i]].append(i)
        obs_by_level[f] = [np.asarray(g, dtype=int) for g in groups]

    # Storage
    store = {f: [] for f in ["Residual"] + factor_cols}

    # Helper to compute fitted and residual
    def compute_residual(mu_val: float) -> np.ndarray:
        fitted = np.full(n, mu_val, dtype=float)
        for f in factor_cols:
            fitted += u[f][level_index[f]]
        return y - fitted

    resid = compute_residual(mu)

    # Diffuse prior on mu
    mu_var0 = 1e12

    for it in range(draws + burn):
        # ----- mu | rest -----
        # y - sum u
        y_minus_u = y.copy()
        for f in factor_cols:
            y_minus_u -= u[f][level_index[f]]
        prec = (n / sig2_e) + (1.0 / mu_var0)
        var_mu = 1.0 / prec
        mean_mu = var_mu * ((np.sum(y_minus_u) / sig2_e) + (0.0 / mu_var0))
        mu = rng.normal(mean_mu, np.sqrt(var_mu))

        # ----- u_j | rest -----
        # Update sequentially; keep residual in sync incrementally.
        # resid = y - mu - sum u
        resid = y - mu
        for f in factor_cols:
            resid -= u[f][level_index[f]]

        for f in factor_cols:
            # For updating u_f, define target = y - mu - sum_{k!=f} u_k
            # We can compute target via: target = resid + u_f[level]
            idx_f = level_index[f]
            u_old = u[f].copy()
            sigma2_f = sig2[f]

            for lev, obs_idx in enumerate(obs_by_level[f]):
                if obs_idx.size == 0:
                    u[f][lev] = 0.0
                    continue
                # target for these obs: resid + u_old[lev]
                t = resid[obs_idx] + u_old[lev]
                n_l = obs_idx.size
                prec_l = (n_l / sig2_e) + (1.0 / sigma2_f)
                var_l = 1.0 / prec_l
                mean_l = var_l * (np.sum(t) / sig2_e)
                u[f][lev] = rng.normal(mean_l, np.sqrt(var_l))

            # Update resid to reflect new u[f]
            resid += u_old[idx_f] - u[f][idx_f]

        # ----- sigma^2_j | u -----
        for f in factor_cols:
            a_post = a0 + n_levels[f] / 2.0
            b_post = b0 + 0.5 * float(np.sum(u[f] ** 2))
            # sample inv-gamma by sampling gamma for precision
            prec_sample = rng.gamma(shape=a_post, scale=1.0 / b_post)
            sig2[f] = 1.0 / prec_sample

        # ----- sigma^2_e | resid -----
        a_post = a0 + n / 2.0
        b_post = b0 + 0.5 * float(np.sum(resid ** 2))
        prec_sample = rng.gamma(shape=a_post, scale=1.0 / b_post)
        sig2_e = 1.0 / prec_sample

        if it >= burn:
            store["Residual"].append(sig2_e)
            for f in factor_cols:
                store[f].append(sig2[f])

    post_means = {k: float(np.mean(v)) for k, v in store.items()}
    post_sds = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in store.items()}

    diag = GibbsDiagnostics(
        draws=draws,
        burn=burn,
        seed=seed,
        a0=float(a0),
        b0=float(b0),
        post_means=post_means,
        post_sds=post_sds,
    )
    return post_means, diag
'''
(base / "msa_bayes.py").write_text(textwrap.dedent(msa_bayes))

# -------------------------
# msa_results.py
# -------------------------
msa_results = r'''
# msa_results.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from msa_utils import design_diagnostics, shapiro_safe


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
'''
(base / "msa_results.py").write_text(textwrap.dedent(msa_results))

# -------------------------
# msa_platform_crossed.py
# -------------------------
msa_platform_crossed = r'''
# msa_platform_crossed.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from msa_utils import (
    validate_dataframe,
    design_diagnostics,
    is_balanced_and_complete,
    clean_anova_index,
    find_term,
    get_ms_df,
    satterthwaite_df,
    update_anova_f_test,
    build_anova_rows,
)
from msa_results import build_result_object


def run_crossed_2factor(df: pd.DataFrame, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult):
    warnings: List[str] = []
    df2 = validate_dataframe(df, config.response_col, config.factor_cols)

    part = config.part_col
    op = config.operator_col
    y = config.response_col

    diag = {"platform": "crossed_2factor"}
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

    vc_map = {"Repeatability": sig2_repeat, f"{op}": sig2_op, f"{part}:{op}": sig2_part_op, f"{part}": sig2_part}

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
        f"{fac_a}:{fac_b}": var_ab,
        f"{fac_a}:{fac_c}": var_ac,
        f"{fac_b}:{fac_c}": var_bc,
        f"{fac_a}:{fac_b}:{fac_c}": var_abc,
    }

    return build_result_object(
        df2, config, vc_map, anova_rows, model.resid, warnings, diag,
        VarianceComponentRow, GRRSummary, ChartData, MSAResult
    )


def run_crossed_mixed(df: pd.DataFrame, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult, warnings: Optional[List[str]] = None):
    """
    Mixed model variance components for unbalanced/incomplete crossed designs.
    Keeps interaction terms limited to 2-way (same behavior as prior engine).
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

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

    vc_map: Dict[str, float] = {"Repeatability": float(max(fit_res.scale, 0.0))}
    vcomp_values = np.asarray(getattr(fit_res, "vcomp", []), dtype=float)
    keys = list(vc_formula.keys())
    for i, key in enumerate(keys):
        if i >= vcomp_values.size:
            break
        vc_map[key] = float(max(vcomp_values[i], 0.0))
        if vcomp_values[i] < 0:
            warnings.append(f"Variance component for '{key}' was negative and truncated to 0.")

    # Reference ANOVA table (fixed effects); do not use for estimation
    warnings.append("ANOVA table is for reference only and not used for VC estimation.")
    anova_rows = []
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
'''
(base / "msa_platform_crossed.py").write_text(textwrap.dedent(msa_platform_crossed))

# -------------------------
# msa_platform_main_effects.py
# -------------------------
msa_platform_main_effects = r'''
# msa_platform_main_effects.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from msa_utils import (
    validate_dataframe,
    design_diagnostics,
    is_balanced_and_complete,
    clean_anova_index,
    find_term,
    get_ms_df,
    update_anova_f_test,
    build_anova_rows,
)
from msa_bayes import gibbs_random_intercepts_main_effects
from msa_results import build_result_object


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
'''
(base / "msa_platform_main_effects.py").write_text(textwrap.dedent(msa_platform_main_effects))

# -------------------------
# msa_engine.py (dispatcher + public types)
# -------------------------
msa_engine = r'''
# msa_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import warnings

import pandas as pd


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
    model_type: str = "crossed"  # "crossed" (2-3 factors) or "main effects" (2-4 factors)

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
    Public entry point (kept stable):
      - config.model_type == "crossed": supports 2- or 3-factor crossed studies.
      - config.model_type == "main effects": supports 2- to 4-factor additive random effects.
    """
    n_factors = len(config.factor_cols)

    # Lazy imports to avoid circular imports and keep msa_engine lightweight.
    from msa_platform_crossed import run_crossed_2factor, run_crossed_3factor
    from msa_platform_main_effects import run_main_effects

    if config.model_type.lower() in {"main effects", "main_effects", "maineffects"}:
        return run_main_effects(df, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)

    # Default: crossed
    if n_factors == 2:
        return run_crossed_2factor(df, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)
    elif n_factors == 3:
        return run_crossed_3factor(df, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)

    # Should be unreachable due to __post_init__
    raise ValueError("Crossed models are only supported for 2 or 3 factors. Use model_type='main effects' for 4 factors.")
'''
(base / "msa_engine.py").write_text(textwrap.dedent(msa_engine))