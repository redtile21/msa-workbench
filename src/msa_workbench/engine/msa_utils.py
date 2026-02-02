from __future__ import annotations

from itertools import permutations
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def canonical_vc_term(term: str, factor_order: Optional[Sequence[str]] = None, sep: str = ":") -> str:
    """Canonicalize a variance-component term name.

    Ensures interaction terms are consistently labeled regardless of internal ordering
    (e.g., statsmodels returning "B:A" vs "A:B").
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


def _n_levels_including_unused(df: pd.DataFrame, col: str) -> int:
    """Return the number of levels for a factor column.

    If the column is categorical, include unused categories. This enables diagnostics
    for missing levels/cells when the input data uses predefined categories.
    """
    if col not in df.columns:
        return 0
    s = df[col]
    try:
        if pd.api.types.is_categorical_dtype(s):
            return int(len(s.cat.categories))
    except Exception:
        pass
    return int(s.nunique())


def validate_dataframe(df: pd.DataFrame, response_col: str, factor_cols: List[str]) -> pd.DataFrame:
    """Return a sanitized copy of df.

    - coerces response to numeric
    - drops NA response rows
    - casts factor cols to categorical strings
    - preserves pre-specified categories when the input column is already categorical
      (including unused categories)
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
        s = df2[col]
        # If already categorical, preserve its category set.
        if pd.api.types.is_categorical_dtype(s):
            cats = s.cat.categories.astype(str)
            df2[col] = pd.Categorical(s.astype(str), categories=cats)
        else:
            df2[col] = df2[col].astype(str).astype("category")

    return df2


def design_diagnostics(df: pd.DataFrame, factor_cols: List[str]) -> Dict[str, Any]:
    level_counts = {f: _n_levels_including_unused(df, f) for f in factor_cols}
    expected_cells = int(np.prod(list(level_counts.values()))) if level_counts else 0
    reps_per_cell = df.groupby(factor_cols, observed=True).size()
    actual_cells = int(len(reps_per_cell))
    rep_min = float(reps_per_cell.min()) if actual_cells > 0 else 0.0
    rep_mean = float(reps_per_cell.mean()) if actual_cells > 0 else 0.0
    rep_max = float(reps_per_cell.max()) if actual_cells > 0 else 0.0
    rep_std = float(reps_per_cell.std()) if actual_cells > 1 else 0.0
    missing_cells_pct = float((1 - actual_cells / expected_cells) * 100) if expected_cells > 0 else 0.0

    return {
        "level_counts": {k: int(v) for k, v in level_counts.items()},
        "replicate_dist": {"min": rep_min, "mean": rep_mean, "max": rep_max, "std": rep_std},
        "expected_cells": expected_cells,
        "actual_cells": actual_cells,
        "missing_cells_pct": missing_cells_pct,
    }


def is_balanced_and_complete(
    design_diag: Dict[str, Any],
    std_tol: float = 1e-8,
    missing_tol_pct: float = 1e-6,
) -> bool:
    rep_std = float(design_diag.get("replicate_dist", {}).get("std", 0.0) or 0.0)
    missing_pct = float(design_diag.get("missing_cells_pct", 0.0) or 0.0)
    return (rep_std <= std_tol) and (missing_pct <= missing_tol_pct)


def is_balanced_main_effects(design_diag: Dict[str, Any], std_tol: float = 1e-8) -> bool:
    """Balance predicate for main-effects/nested designs.

    For main-effects designs, "completeness" of a full factorial is not expected.
    We therefore only require equal replication per *observed* cell.
    """
    rep_std = float(design_diag.get("replicate_dist", {}).get("std", 0.0) or 0.0)
    return rep_std <= std_tol


def infer_nesting(df: pd.DataFrame, factor_cols: List[str]) -> Dict[str, Any]:
    """Infer simple nesting relationships among factor columns.

    Returns:
      {
        "parent_map": {child: parent, ...}  # chosen immediate parent
        "candidates": {child: [parents...]}
        "stats": {(child,parent): {"max_parents_per_child": ..., "levels_child": ..., "levels_parent": ...}}
      }

    A column 'child' is considered nested in 'parent' if each child level appears under
    <= 1 parent level (i.e., max nunique(parent) within child is 1).
    If multiple parents satisfy, we choose the "closest" parent as the one with the
    largest number of levels (typically the immediate parent in a hierarchy).
    """
    stats_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    candidates: Dict[str, List[str]] = {c: [] for c in factor_cols}

    levels = {c: _n_levels_including_unused(df, c) for c in factor_cols}

    for child in factor_cols:
        for parent in factor_cols:
            if parent == child:
                continue
            try:
                parent_counts = df.groupby(child, observed=True)[parent].nunique()
                max_parents = int(parent_counts.max()) if len(parent_counts) else 0
            except Exception:
                max_parents = 999999

            stats_map[(child, parent)] = {
                "max_parents_per_child": max_parents,
                "levels_child": int(levels.get(child, 0)),
                "levels_parent": int(levels.get(parent, 0)),
            }

            if max_parents <= 1:
                # plausible nesting; keep as candidate
                candidates[child].append(parent)

    parent_map: Dict[str, str] = {}
    for child, parents in candidates.items():
        if not parents:
            continue
        # choose the most specific parent (largest number of levels)
        best = max(parents, key=lambda p: (levels.get(p, 0), str(p)))
        # avoid self/degenerate
        if best != child:
            parent_map[child] = best

    # Clean simple 1-1 cycles (rare but possible)
    for child, parent in list(parent_map.items()):
        if parent_map.get(parent) == child:
            # break the cycle by dropping the weaker (smaller-level) edge
            if levels.get(child, 0) >= levels.get(parent, 0):
                parent_map.pop(parent, None)
            else:
                parent_map.pop(child, None)

    return {"parent_map": parent_map, "candidates": candidates, "stats": stats_map}


def infer_nesting_chain(df: pd.DataFrame, factor_cols: List[str]) -> Optional[List[str]]:
    """Infer a strict nesting *chain* root -> ... -> leaf, if one exists.

    Returns ordered list of factors if:
      - exactly one root exists
      - each node has at most one child
      - the chain covers all factors
    Otherwise returns None.
    """
    info = infer_nesting(df, factor_cols)
    parent_map: Dict[str, str] = dict(info.get("parent_map", {}))

    roots = [f for f in factor_cols if f not in parent_map]
    if len(roots) != 1:
        return None
    root = roots[0]

    # build parent -> children map
    children_for: Dict[str, List[str]] = {f: [] for f in factor_cols}
    for child, parent in parent_map.items():
        children_for.setdefault(parent, []).append(child)

    chain: List[str] = [root]
    cur = root
    visited = set(chain)
    while True:
        kids = children_for.get(cur, [])
        if len(kids) == 0:
            break
        if len(kids) > 1:
            return None
        nxt = kids[0]
        if nxt in visited:
            return None
        chain.append(nxt)
        visited.add(nxt)
        cur = nxt

    if len(chain) != len(factor_cols):
        return None
    return chain


def branching_diagnostics(df: pd.DataFrame, parent: str, child: str) -> Dict[str, Any]:
    """Diagnostics for a nesting edge parent -> child.

    Computes the number of child levels per parent level.
    In a balanced hierarchical design, this should be constant (std ~ 0).
    """
    try:
        counts = df.groupby(parent, observed=True)[child].nunique()
        if len(counts) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(counts.mean()),
            "std": float(counts.std()) if len(counts) > 1 else 0.0,
            "min": float(counts.min()),
            "max": float(counts.max()),
        }
    except Exception:
        return {"mean": 0.0, "std": float("nan"), "min": 0.0, "max": 0.0}


def clean_anova_index(anova: pd.DataFrame) -> pd.DataFrame:
    """Remove the C(Q("...")) wrapper from ANOVA index names for display."""
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
    """Approx DF for linear combination L = MS1 + MS2 - MS3."""
    numerator = (ms1 + ms2 - ms3) ** 2
    denom = 0.0
    if df1 > 0:
        denom += (ms1**2) / df1
    if df2 > 0:
        denom += (ms2**2) / df2
    if df3 > 0:
        denom += (ms3**2) / df3
    if denom <= 0:
        return 1.0
    return float(numerator / denom)


def update_anova_f_test(
    anova: pd.DataFrame,
    term: str,
    ms_num: float,
    ms_denom: float,
    df_num: float,
    df_denom: float,
) -> None:
    """Calculate F-ratio and p-value and update the ANOVA table in-place."""
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
