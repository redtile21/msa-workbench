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
