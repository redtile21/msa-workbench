from __future__ import annotations

from typing import List, Optional, Dict, Any, Iterable, Tuple, Union, Sequence
from itertools import permutations
import re

import numpy as np
import pandas as pd
from scipy import stats


def canonical_vc_term(term: str, factor_order: Optional[Sequence[str]] = None, sep: str = ":") -> str:
    """Canonicalize a variance-component term name.

    This is used to ensure interaction terms are consistently labeled regardless of:
      - statsmodels internal ordering (e.g., returning "Instrument:Sample" vs "Sample:Instrument")
      - user factor ordering

    Rules:
      - Non-interaction terms are returned unchanged.
      - Interaction terms (containing `sep`) are split and re-joined in a deterministic order.
      - If `factor_order` is provided, the interaction components are ordered by their index in `factor_order`
        (unknown terms fall back to alphabetical).
      - If `factor_order` is not provided, components are sorted alphabetically.
    """
    if term is None:
        return ""
    s = str(term).strip()
    if sep not in s:
        return s

    parts = [p.strip() for p in s.split(sep) if str(p).strip()]
    if len(parts) <= 1:
        return s

    if factor_order:
        order = {str(f): i for i, f in enumerate(list(factor_order))}

        def _key(p: str):
            return (order.get(p, 10**9), p)

        parts_sorted = sorted(parts, key=_key)
    else:
        parts_sorted = sorted(parts)

    return sep.join(parts_sorted)


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
        rows.append(
            ANOVATableRow(
                str(term),
                df_val,
                ss,
                ms,
                None if pd.isna(f) else float(f),
                None if pd.isna(p) else float(p),
            )
        )
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
