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

    # Optional bootstrap CIs (percentile). Results are stored under result.diagnostics["bootstrap_ci"].
    bootstrap_iters: int = 0
    ci_level: float = 0.95
    # Shared RNG seed for simulation/bootstrapping/Bayesian fallbacks.
    random_seed: int = 0

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

        if self.bootstrap_iters is None:
            self.bootstrap_iters = 0
        if int(self.bootstrap_iters) < 0:
            raise ValueError("bootstrap_iters must be >= 0")
        if self.ci_level is None:
            self.ci_level = 0.95
        if not (0.0 < float(self.ci_level) < 1.0):
            raise ValueError("ci_level must be between 0 and 1")

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
    ndc: int
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

def run_msa(df: pd.DataFrame, config: MSAConfig) -> MSAResult:
    """Run a Measurement Systems Analysis (MSA) study.

    This dispatches to the appropriate platform (crossed or main effects) and optionally
    computes bootstrap confidence intervals (percentile) if config.bootstrap_iters > 0.
    """

    def _run_msa_core(df_in: pd.DataFrame, cfg: MSAConfig) -> MSAResult:
        n_factors = len(cfg.factor_cols)

        # Lazy imports to avoid circular imports and keep msa_engine lightweight.
        from .msa_platform_crossed import run_crossed_2factor, run_crossed_3factor
        from .msa_platform_main_effects import run_main_effects

        if cfg.model_type.lower() in {"main effects", "main_effects", "maineffects"}:
            return run_main_effects(df_in, cfg, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)

        # Default: crossed
        if n_factors == 2:
            return run_crossed_2factor(df_in, cfg, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)
        if n_factors == 3:
            return run_crossed_3factor(df_in, cfg, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)
        if n_factors >= 4:
            warnings.warn(
                "Crossed models with >3 factors are not supported. Falling back to Main Effects model.",
                UserWarning,
            )
            return run_main_effects(df_in, cfg, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)

        raise ValueError("Unsupported number of factors for MSA analysis.")

    # Core run
    res = _run_msa_core(df, config)

    # Optional bootstrap CIs (percentile). Stored in diagnostics to avoid changing public result schema.
    n_boot = int(getattr(config, "bootstrap_iters", 0) or 0)
    if n_boot <= 0:
        return res

    import numpy as np
    from dataclasses import replace as _dc_replace

    alpha = (1.0 - float(getattr(config, "ci_level", 0.95) or 0.95)) / 2.0
    low_q = 100.0 * alpha
    high_q = 100.0 * (1.0 - alpha)

    rng = np.random.default_rng(int(getattr(config, "random_seed", 0) or 0))
    boot_cfg = _dc_replace(config, bootstrap_iters=0)

    # Collect bootstrap distributions for var components (by row.source) and key summary metrics.
    dist: Dict[str, List[float]] = {}
    grr_dist: Dict[str, List[float]] = {
        "total_gage_rr_pct_study_var": [],
        "total_gage_rr_pct_tolerance": [],
        "ndc": [],
    }

    for b in range(n_boot):
        df_b = df.sample(n=len(df), replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
        try:
            r_b = _run_msa_core(df_b, boot_cfg)
        except Exception:
            continue

        for row in r_b.var_components:
            dist.setdefault(str(row.source), []).append(float(row.var_comp))

        try:
            grr_dist["total_gage_rr_pct_study_var"].append(float(r_b.grr_summary.total_gage_rr_pct_study_var))
        except Exception:
            pass
        try:
            grr_dist["total_gage_rr_pct_tolerance"].append(
                float(r_b.grr_summary.total_gage_rr_pct_tolerance) if r_b.grr_summary.total_gage_rr_pct_tolerance is not None else float("nan")
            )
        except Exception:
            pass
        try:
            grr_dist["ndc"].append(float(r_b.grr_summary.ndc))
        except Exception:
            pass

    def _pct_ci(vals: List[float]) -> Optional[Dict[str, float]]:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < max(10, n_boot // 4):
            return None
        lo = float(np.percentile(arr, low_q))
        hi = float(np.percentile(arr, high_q))
        return {"low": lo, "high": hi, "n": int(arr.size)}

    vc_ci: Dict[str, Any] = {k: _pct_ci(v) for k, v in dist.items()}
    grr_ci: Dict[str, Any] = {k: _pct_ci(v) for k, v in grr_dist.items()}

    res.diagnostics = dict(res.diagnostics)
    res.diagnostics["bootstrap_iters"] = n_boot
    res.diagnostics["ci_level"] = float(getattr(config, "ci_level", 0.95) or 0.95)
    res.diagnostics["bootstrap_ci"] = {"variance_components": vc_ci, "grr_summary": grr_ci}

    return res