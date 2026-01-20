# tests/test_msa_engine.py

import pytest
import pandas as pd
import numpy as np
from itertools import product

from msa_workbench.engine.msa_engine import (
    MSAConfig,
    run_crossed_msa,
)

# =========================
# Synthetic Data Generator
# =========================

def generate_synthetic_msa_data(
    factors: dict,
    sigmas: dict,
    n_reps: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a synthetic MSA dataset from known variance components.
    This generator creates data based on a MAIN EFFECTS ONLY model.
    """
    rng = np.random.default_rng(seed)

    factor_names = list(factors.keys())
    factor_levels = list(factors.values())

    cells = list(product(*factor_levels))
    records = []
    
    cell_effects = {name: rng.normal(0, sigmas[name], size=len(levels)) 
                    for name, levels in factors.items()}

    for cell in cells:
        cell_sum_of_effects = 0
        for i, factor_name in enumerate(factor_names):
            level = cell[i]
            level_idx = factors[factor_name].index(level)
            cell_sum_of_effects += cell_effects[factor_name][level_idx]

        for _ in range(n_reps):
            measurement = (
                10.0
                + cell_sum_of_effects
                + rng.normal(0, sigmas["Repeatability"])
            )
            record = dict(zip(factor_names, cell))
            record["Measurement"] = measurement
            records.append(record)

    return pd.DataFrame(records)

# =========================
# Tests
# =========================

def test_main_effects_request_falls_back_to_2_factor_crossed():
    """
    Tests that a 'main effects' model request correctly falls back to a 
    2-factor crossed model using Part and Operator, and gives a warning.
    """
    factors = {
        "Part": [f"P{i}" for i in range(10)],
        "Operator": [f"Op{i}" for i in range(3)],
        "Gage": [f"G{i}" for i in range(2)], # This factor should be ignored
    }
    
    sigmas = {
        "Part": 2.0,
        "Operator": 0.5,
        "Gage": 10.0,  # Make Gage variance huge to see if it affects results
        "Repeatability": 0.2
    }
    df = generate_synthetic_msa_data(factors, sigmas, n_reps=3)

    config = MSAConfig(
        response_col="Measurement",
        factor_cols=["Part", "Operator", "Gage"],
        part_col="Part",
        operator_col="Operator",
        model_type="main effects" # Request the model that now falls back
    )

    result = run_crossed_msa(df, config)

    # 1. Check for the warning
    assert any("Running a 2-factor crossed model" in w.message for w in result.warnings), \
        "Expected a warning about falling back to a 2-factor model."

    # 2. Check that the results are reasonable for a 2-factor model
    # The Gage variance is ignored, so the total variance will be lower
    # than the true total variance of the generated data. We expect the
    # Part and Operator variances to be estimated reasonably well by the
    # ANOVA method since the design is balanced.
    part_vc_row = next((r for r in result.var_components if "Part-to-Part" in r.source), None)
    op_vc_row = next((r for r in result.var_components if "Reproducibility: Operator" in r.source), None)

    assert part_vc_row is not None
    assert op_vc_row is not None
    
    # ANOVA estimates on this balanced data should be close to the true inputs
    # for the factors included in the model.
    true_part_var = sigmas["Part"]**2
    true_op_var = sigmas["Operator"]**2
    
    assert true_part_var * 0.5 <= part_vc_row.var_comp <= true_part_var * 1.5
    assert true_op_var * 0.5 <= op_vc_row.var_comp <= true_op_var * 1.5


def test_unbalanced_design_triggers_mixed_model_fallback():
    """
    Tests that an unbalanced design correctly triggers the MixedLM fallback
    and produces a result.
    """
    factors = {
        "Part": [f"P{i}" for i in range(5)],
        "Operator": [f"Op{i}" for i in range(2)],
    }
    sigmas = {"Part": 2.0, "Operator": 0.5, "Repeatability": 0.3}
    df = generate_synthetic_msa_data(factors, sigmas, n_reps=3)

    # Make the data unbalanced by dropping a few rows
    df_unbalanced = df.iloc[:-2]

    config = MSAConfig(
        response_col="Measurement",
        factor_cols=["Part", "Operator"],
        part_col="Part",
        operator_col="Operator",
        model_type="crossed"
    )

    result = run_crossed_msa(df_unbalanced, config)

    # 1. Check for the fallback warning
    assert any("Falling back to Mixed Model estimation" in w for w in result.warnings), \
        "Expected a warning about falling back to MixedLM."
        
    # 2. Check that we got a result
    assert len(result.var_components) > 0
    part_vc_row = next((r for r in result.var_components if "Part-to-Part" in r.source), None)
    assert part_vc_row is not None
    assert part_vc_row.var_comp > 0 # Just check that it's not zero