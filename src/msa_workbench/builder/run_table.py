from typing import List, Dict, Optional
import pandas as pd
from itertools import product
import numpy as np

def validate_factors(factors: List[Dict]) -> List[Dict]:
    """
    Validates a list of factor dictionaries.

    Args:
        factors: A list of dictionaries, where each dictionary represents a factor.
                 Each dictionary should have "name" (str) and "levels" (List[str]).

    Returns:
        The validated list of factors.

    Raises:
        ValueError: If validation fails.
    """
    if not 1 <= len(factors) <= 4:
        raise ValueError("Please provide between 1 and 4 factors.")

    names = set()
    for factor in factors:
        name = factor.get("name")
        if not name or not name.strip():
            raise ValueError("Factor names cannot be empty.")
        
        name_lower = name.lower()
        if name_lower in names:
            raise ValueError(f"Factor name '{name}' is not unique (case-insensitive).")
        names.add(name_lower)

        levels = factor.get("levels")
        if not levels:
            raise ValueError(f"Factor '{name}' must have at least one level.")
        if any(not level or not level.strip() for level in levels):
            raise ValueError(f"Levels for factor '{name}' cannot be empty.")
            
    return factors

def build_run_table(factors: List[Dict], replicates: int) -> pd.DataFrame:
    """
    Builds a run table DataFrame from factors and replicates.

    Args:
        factors: A list of validated factor dictionaries.
        replicates: The number of replicates per combination (must be >= 1).

    Returns:
        A pandas DataFrame representing the run table.
    """
    if replicates < 1:
        raise ValueError("Number of replicates must be at least 1.")

    validated_factors = validate_factors(factors)
    
    factor_names = [f["name"] for f in validated_factors]
    level_lists = [f["levels"] for f in validated_factors]

    cartesian_product = list(product(*level_lists))
    
    rows = []
    for combo in cartesian_product:
        for i in range(1, replicates + 1):
            row = dict(zip(factor_names, combo))
            row["Replicate"] = i
            rows.append(row)
            
    df = pd.DataFrame(rows)
    df["RunID"] = range(1, len(df) + 1)
    
    # Reorder columns
    cols = ["RunID"] + factor_names + ["Replicate"]
    df = df[cols]
    
    return df

def sort_run_table_left_to_right(df: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
    """
    Sorts the run table by factor columns, then Replicate.

    Args:
        df: The run table DataFrame.
        factor_names: The list of factor names in the desired sort order.

    Returns:
        The sorted DataFrame with a recomputed RunID.
    """
    sorted_df = df.sort_values(by=factor_names + ["Replicate"]).reset_index(drop=True)
    sorted_df["RunID"] = range(1, len(sorted_df) + 1)
    return sorted_df

def randomize_run_table(df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Randomizes the order of the run table.

    Args:
        df: The run table DataFrame.
        seed: An optional seed for reproducible randomization.

    Returns:
        The randomized DataFrame with a new "RunOrder" column.
    """
    randomized_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    randomized_df["RunOrder"] = range(1, len(randomized_df) + 1)
    return randomized_df

def export_run_table_csv(df: pd.DataFrame, path: str):
    """
    Exports the DataFrame to a CSV file.

    Args:
        df: The DataFrame to export.
        path: The file path to save to.
    """
    df.to_csv(path, index=False, encoding='utf-8')

if __name__ == '__main__':
    # Self-check block
    print("Running self-checks for run_table.py...")

    # Test Case 1: Basic 2-factor, 2-level, 3-replicate build
    factors_1 = [
        {"name": "Operator", "levels": ["Op A", "Op B"]},
        {"name": "Part", "levels": ["Part 1", "Part 2"]},
    ]
    replicates_1 = 3
    df1 = build_run_table(factors_1, replicates_1)
    assert len(df1) == 2 * 2 * 3  # 12 rows
    assert list(df1.columns) == ["RunID", "Operator", "Part", "Replicate"]
    print("Test 1 Passed: Basic build.")

    # Test Case 2: Left-to-right sort
    factor_names_2 = ["Part", "Operator"]
    df2 = sort_run_table_left_to_right(df1, factor_names_2)
    assert df2.iloc[0]["Part"] == "Part 1"
    assert df2.iloc[0]["Operator"] == "Op A"
    print("Test 2 Passed: Left-to-right sort.")

    # Test Case 3: Randomized with seed
    df3a = randomize_run_table(df1, seed=42)
    df3b = randomize_run_table(df1, seed=42)
    pd.testing.assert_frame_equal(df3a, df3b)
    assert "RunOrder" in df3a.columns
    print("Test 3 Passed: Seeded randomization is reproducible.")
    
    # Test Case 4: Validation
    try:
        validate_factors([{"name": "F1", "levels": []}])
    except ValueError as e:
        assert "must have at least one level" in str(e)
        print("Test 4.1 Passed: Catches empty levels.")

    try:
        validate_factors([{"name": "F1", "levels": ["L1"]}, {"name": "f1", "levels": ["L2"]}])
    except ValueError as e:
        assert "not unique" in str(e)
        print("Test 4.2 Passed: Catches duplicate names.")
        
    try:
        build_run_table(factors_1, 0)
    except ValueError as e:
        assert "at least 1" in str(e)
        print("Test 4.3 Passed: Catches replicates < 1.")

    print("\nAll self-checks passed!")
    print("\nSample table (randomized):")
    print(df3a)