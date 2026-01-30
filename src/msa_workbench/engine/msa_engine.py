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
    """
    Dispatch to the appropriate implementation based on number of factors and model type.
    """
    n_factors = len(config.factor_cols)

    # Lazy imports to avoid circular imports and keep msa_engine lightweight.
    from .msa_platform_crossed import run_crossed_2factor, run_crossed_3factor
    from .msa_platform_main_effects import run_main_effects

    if config.model_type.lower() in {"main effects", "main_effects", "maineffects"}:
        return run_main_effects(df, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)

    # Default: crossed
    if n_factors == 2:
        return run_crossed_2factor(df, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)
    elif n_factors == 3:
        return run_crossed_3factor(df, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)
    elif n_factors >= 4:
        # Fallback to main effects model for 4+ factors, as crossed models are not supported beyond 3.
        # This implicitly handles the warning from the previous version of msa_engine.py
        warnings.warn("Crossed models with >3 factors are not supported. "
                      "Falling back to Main Effects model.", UserWarning)
        return run_main_effects(df, config, ANOVATableRow, VarianceComponentRow, GRRSummary, ChartData, MSAResult)

    # Should be unreachable due to __post_init__ or previous checks
    raise ValueError("Unsupported number of factors for MSA analysis.")