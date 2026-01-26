# tests/test_msa_modeling_fixes.py

import pandas as pd
import numpy as np
import pytest
from itertools import product

from msa_workbench.engine.msa_engine import run_msa, MSAConfig

def generate_synthetic_msa_data(
    factors: dict,
    replicates: int,
    component_sigmas: dict,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic data for an MSA study from known variance components.
    """
    rng = np.random.default_rng(seed)

    factor_names = list(factors.keys())
    levels = [factors[name] for name in factor_names]
    
    design = list(product(*levels))
    df = pd.DataFrame(design, columns=factor_names)

    df = pd.concat([df] * replicates, ignore_index=True)
    df.sort_values(by=factor_names, inplace=True, ignore_index=True)

    y = np.zeros(len(df))

    for factor in factor_names:
        if factor in component_sigmas and component_sigmas[factor] > 0:
            sigma = component_sigmas[factor]
            factor_levels = df[factor].unique()
            effects = pd.Series(rng.normal(0, sigma, len(factor_levels)), index=factor_levels)
            y += df[factor].map(effects).values

    interaction_keys = [k for k in component_sigmas if ":" in k]
    for key in interaction_keys:
        parts = key.split(":")
        if len(parts) == 2 and parts[0] in factor_names and parts[1] in factor_names:
            sigma = component_sigmas[key]
            if sigma > 0:
                interaction_levels = df.groupby(parts).size().reset_index()[parts]
                effects = pd.Series(rng.normal(0, sigma, len(interaction_levels)),
                                    index=pd.MultiIndex.from_frame(interaction_levels))
                
                multi_index = pd.MultiIndex.from_frame(df[parts])
                y += multi_index.map(effects).values
    
    sigma_repeatability = component_sigmas.get("Repeatability", 1.0)
    y += rng.normal(0, sigma_repeatability, len(df))

    df["Measurement"] = y
    return df


# Placeholder for future tests
def test_crossed_model_robust_on_unbalanced_data():
    pass

def test_crossed_model_robust_with_missing_cells():
    pass

def test_model_handles_degenerate_cases():
    pass