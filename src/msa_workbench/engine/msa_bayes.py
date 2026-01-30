from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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
    post_medians: Dict[str, float]
    post_sds: Dict[str, float]
    summary_stat: str


def gibbs_random_intercepts_main_effects(
    df: pd.DataFrame,
    response_col: str,
    factor_cols: List[str],
    seed: int = 0,
    draws: int = 3000,
    burn: int = 1000,
    a0: float = 2.0,
    b0: Optional[float] = None,
    summary_stat: str = "mean",
) -> Tuple[Dict[str, float], GibbsDiagnostics]:
    """Gibbs sampler for an additive random-intercepts model.

    Model:
      y = mu + sum_j u_j[level_j] + e
      u_j ~ N(0, sigma_j^2 I)
      e   ~ N(0, sigma_e^2 I)

    Priors (conjugate):
      sigma^2 ~ InvGamma(a0, b0)

    Returns a point estimate for variance components (sigma^2), including:
      - "Residual" (sigma_e^2)
      - each factor name (sigma_j^2)

    Note:
      The default prior (a0=2.0) yields a finite posterior mean and tends to better match JMP's
      'Bayesian estimates' fallback when REML produces negative/unstable variance components.
      If you want a more diffuse prior, you can lower `a0`, but for small numbers of levels
      (e.g., a 2-level factor), the posterior mean of sigma^2 can be unstable.
      When matching JMP/REML results, `summary_stat="median"` is often a better choice.
    """
    summary_stat = str(summary_stat).strip().lower()
    if summary_stat not in {"mean", "median"}:
        raise ValueError("summary_stat must be 'mean' or 'median'.")

    rng = np.random.default_rng(seed)

    y = np.asarray(df[response_col], dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("No rows to analyze after cleaning.")

    # Encode each factor as integer indices 0..(L-1)
    level_index: Dict[str, np.ndarray] = {}
    n_levels: Dict[str, int] = {}
    for f in factor_cols:
        cats = df[f].cat.categories if hasattr(df[f], "cat") else pd.Categorical(df[f]).categories
        cat_to_i = {str(c): i for i, c in enumerate(list(cats))}
        idx = np.asarray([cat_to_i[str(v)] for v in df[f].astype(str).values], dtype=int)
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
    obs_by_level: Dict[str, List[np.ndarray]] = {}
    for f in factor_cols:
        L = n_levels[f]
        groups: List[List[int]] = [[] for _ in range(L)]
        idx = level_index[f]
        for i in range(n):
            groups[int(idx[i])].append(i)
        obs_by_level[f] = [np.asarray(g, dtype=int) for g in groups]

    # Storage
    store: Dict[str, List[float]] = {k: [] for k in ["Residual"] + factor_cols}

    # Diffuse prior on mu
    mu_var0 = 1e12

    for it in range(draws + burn):
        # ----- mu | rest -----
        y_minus_u = y.copy()
        for f in factor_cols:
            y_minus_u -= u[f][level_index[f]]
        prec = (n / sig2_e) + (1.0 / mu_var0)
        var_mu = 1.0 / prec
        mean_mu = var_mu * ((np.sum(y_minus_u) / sig2_e) + (0.0 / mu_var0))
        mu = rng.normal(mean_mu, np.sqrt(var_mu))

        # ----- u_j | rest -----
        resid = y - mu
        for f in factor_cols:
            resid -= u[f][level_index[f]]

        for f in factor_cols:
            idx_f = level_index[f]
            u_old = u[f].copy()
            sigma2_f = sig2[f]

            for lev, obs_idx in enumerate(obs_by_level[f]):
                if obs_idx.size == 0:
                    u[f][lev] = 0.0
                    continue
                # target = y - mu - sum_{k!=f} u_k
                t = resid[obs_idx] + u_old[lev]
                n_l = obs_idx.size
                prec_l = (n_l / sig2_e) + (1.0 / sigma2_f)
                var_l = 1.0 / prec_l
                mean_l = var_l * (np.sum(t) / sig2_e)
                u[f][lev] = rng.normal(mean_l, np.sqrt(var_l))

            # keep resid consistent incrementally
            resid += u_old[idx_f] - u[f][idx_f]

        # ----- sigma^2_j | u -----
        for f in factor_cols:
            a_post = a0 + n_levels[f] / 2.0
            b_post = b0 + 0.5 * float(np.sum(u[f] ** 2))
            prec_sample = rng.gamma(shape=a_post, scale=1.0 / b_post)
            sig2[f] = 1.0 / prec_sample

        # ----- sigma^2_e | resid -----
        a_post = a0 + n / 2.0
        b_post = b0 + 0.5 * float(np.sum(resid ** 2))
        prec_sample = rng.gamma(shape=a_post, scale=1.0 / b_post)
        sig2_e = 1.0 / prec_sample

        if it >= burn:
            store["Residual"].append(float(sig2_e))
            for f in factor_cols:
                store[f].append(float(sig2[f]))

    post_means = {k: float(np.mean(v)) for k, v in store.items()}
    post_medians = {k: float(np.median(v)) for k, v in store.items()}
    post_sds = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in store.items()}

    diag = GibbsDiagnostics(
        draws=draws,
        burn=burn,
        seed=seed,
        a0=float(a0),
        b0=float(b0),
        post_means=post_means,
        post_medians=post_medians,
        post_sds=post_sds,
        summary_stat=summary_stat,
    )

    point = post_means if summary_stat == "mean" else post_medians
    return point, diag
