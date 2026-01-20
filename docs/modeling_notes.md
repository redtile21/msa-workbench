# MSA Engine: Modeling Notes (Initial Investigation)

This document summarizes the initial state of the MSA modeling code in `msa_engine.py`.

## 1. Model Dispatching

The primary entry point is `run_crossed_msa(df, config)`. It dispatches as follows:

- **`config.model_type == 'main effects'`**: Always uses `_run_main_effects_msa`.
- **`config.model_type == 'crossed'`**:
    - 2 factors: `_run_crossed_msa_2factor`
    - 3 factors: `_run_crossed_msa_3factor`
    - 4 factors: **Falls back to `_run_main_effects_msa`**. This is a critical detail, as a 4-factor "crossed" analysis is not actually performed.

## 2. Crossed Models (2 and 3 Factors)

### Method
- **Analysis of Variance (ANOVA)** using Ordinary Least Squares (OLS) from `statsmodels.formula.api.ols`.
- `typ=2` sums of squares are used.
- **Variance Components (VCs)** are calculated via the **Method of Moments** (solving Expected Mean Squares equations).

### Model Formulas
- **2-Factor:** `y ~ C(Part) + C(Operator) + C(Part):C(Operator)`
- **3-Factor:** `y ~ C(A) + C(B) + C(C) + C(A):C(B) + C(A):C(C) + C(B):C(C) + C(A):C(B):C(C)`

### Assumptions & Known Issues
- **Balanced Design:** The VC formulas are only exact for fully balanced designs (equal number of replicates in every single cell). The code attempts a check but proceeds even if unbalanced, leading to approximate/biased results.
- **No Missing Cells:** The OLS model with full interactions assumes all combinations of factors exist. Missing cells will cause issues with `typ=2` ANOVA and the interpretation of results.
- **Negative Variances:** The code uses `max(0.0, ...)` to truncate negative variance estimates, which is a sign that the Method of Moments is failing, often due to unbalance or model misspecification.
- **F-Tests:** Denominators for F-tests are manually adjusted to be correct for a random-effects model, which is good practice. The 3-factor model correctly uses the Satterthwaite approximation for main effect tests.

## 3. Main Effects Model (`_run_main_effects_msa`)

This implementation appears to be the primary source of the "not working" bug.

### Method
- Also uses `statsmodels.formula.api.ols` with `typ=2` ANOVA.

### Model Formula
- `y ~ C(Factor1) + C(Factor2) + ...` (no interactions)

### Failure Analysis
The method used to calculate variance components is fundamentally flawed for this purpose.

1.  **Pooled Error Term**: In an OLS model with only main effects, the `MS_Residual` (error term) incorrectly pools together true random error (Repeatability) **and all real interaction effects**.
2.  **Incorrect VC Formula**: The code calculates variance for a factor as `sig2_fac = (ms_fac - ms_res) / n_prime`.
    - Because `ms_res` is inflated with interaction effects, `ms_fac - ms_res` is often negative, leading to VC estimates of 0 after truncation.
    - This is especially problematic for the Part-to-Part variance, as the Part*Operator interaction (a key component of Gage R&R) gets absorbed into the error term, artificially deflating the part variance.
3.  **Incorrect `n_prime`**: The term `n_prime` is calculated as `n_total / df[factor].nunique()`. This is a naive formula that does not correctly account for the number of levels of the other factors in the experiment, making the variance calculation incorrect even if there were no interactions.

**Conclusion:** The main-effects model is not a valid method for estimating random effects variance components as currently implemented. It should be replaced entirely with a true variance component estimation method like `statsmodels.MixedLM`.
