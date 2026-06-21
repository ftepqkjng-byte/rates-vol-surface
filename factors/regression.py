"""Regression and pattern-projection helpers.

* ``regress`` — generic OLS of any target panel on any factor panel
  (per-cell time-series regression). The workhorse for "I have a factor
  design X, how well does it explain Y?".
* ``project_onto_patterns`` — cross-sectional dual: decompose each day's
  surface move into exposures on K hand-drawn spatial patterns.
  Dispatches over OLS / ridge / NNLS / lasso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regress(
    targets: pd.DataFrame,
    factors: pd.DataFrame,
    add_intercept: bool = True,
) -> dict[str, pd.DataFrame | pd.Series]:
    """OLS-regress every column of ``targets`` on the same factor panel.

    ``targets``  : (date, target_name) — e.g. the diff'd cube cells.
    ``factors``  : (date, factor_name) — e.g. ``stack_block_scores(...)``,
                   bucket means, vanilla PCs, hand-crafted spreads —
                   anything that's a date-indexed numeric panel.

    Returns a dict with four DataFrames/Series, all aligned on the
    intersection of ``targets.index`` and ``factors.index``:

    * ``betas``     — (n_factors[+1] × n_targets) — coefficients per
                       target; first row is ``intercept`` if requested.
    * ``fitted``    — (date × target).
    * ``residuals`` — (date × target).
    * ``r2``        — (target,) Series of in-sample R².

    Designed to be the workhorse for 'I have a factor design X, how
    well does it explain Y?' — block PCs vs cube cells is the first
    use case, but the same call handles any other factor pattern.
    """
    common = targets.index.intersection(factors.index)
    Y = targets.loc[common].dropna(axis=1)
    F_raw = factors.loc[common]
    F = F_raw.values
    if add_intercept:
        F = np.column_stack([np.ones(len(F)), F])
        beta_index = ["intercept", *F_raw.columns]
    else:
        beta_index = list(F_raw.columns)

    beta, *_ = np.linalg.lstsq(F, Y.values, rcond=None)
    fitted_arr = F @ beta
    fitted = pd.DataFrame(fitted_arr, index=common, columns=Y.columns)
    residuals = Y - fitted

    ss_res = (residuals.values ** 2).sum(axis=0)
    ss_tot = ((Y.values - Y.values.mean(axis=0)) ** 2).sum(axis=0)
    safe_tot = np.where(ss_tot == 0, 1.0, ss_tot)
    r2 = 1.0 - ss_res / safe_tot

    return {
        "betas":     pd.DataFrame(beta, index=beta_index, columns=Y.columns),
        "fitted":    fitted,
        "residuals": residuals,
        "r2":        pd.Series(r2, index=Y.columns, name="r2"),
    }


def project_onto_patterns(
    target: pd.DataFrame,
    patterns: pd.DataFrame,
    method: str = "ols",
    ridge_alpha: float = 1.0,
    lasso_alpha: float = 0.01,
    standardize_patterns: bool = True,
) -> dict:
    """Cross-sectional decomposition of each row of ``target`` onto a set
    of fixed ``patterns``::

        x_t  ≈  patterns^T @ exposures_t  +  residual_t

    For every date, solve a linear system across cells to recover how
    much each pattern explains that day's move. The dual of ``regress``:
    ``regress`` finds per-cell betas on factor *time series*; this finds
    per-date exposures on *spatial* patterns.

    Parameters
    ----------
    target               : (T, N) DataFrame — typically a diff'd panel
                           (rate diff, vol diff, …). MultiIndex columns.
    patterns             : (K, N) DataFrame — one row per pattern, same
                           column convention as ``sparse_pca_warm``'s
                           prior and ``pattern_creator.py``'s saved pkl.
    method               : ``'ols'``, ``'ridge'``, ``'nnls'``, or
                           ``'lasso'``. See module docstring for guidance.
    ridge_alpha          : L2 penalty for ``method='ridge'``.
    lasso_alpha          : L1 penalty for ``method='lasso'``.
    standardize_patterns : unit-L2-normalise each pattern row before
                           fitting. Makes exposures comparable across
                           patterns of different support sizes; on by
                           default since hand-drawn 0/±1 patterns vary
                           wildly in raw norm.

    Returns
    -------
    dict with keys
        ``exposures``        : (T, K) DataFrame.
        ``fitted``           : (T, N) DataFrame, ``exposures @ patterns_used``.
        ``residuals``        : (T, N) DataFrame.
        ``r2_total``         : float — variance share explained overall.
        ``r2_per_cell``      : (N,) Series, indexed by ``(expiry, tenor)``.
        ``r2_per_date``      : (T,) Series.
        ``exposure_corr``    : (K, K) DataFrame — corr of exposures over time.
        ``pattern_corr``     : (K, K) DataFrame — corr of (standardised)
                               patterns as vectors in cell-space.
        ``condition_number`` : float — ``cond(P P^T)``; > ~1e6 ⇒ patterns
                               are effectively collinear, prefer ridge/lasso.
        ``patterns_used``    : (K, N) DataFrame — patterns after standardisation.
        ``method``           : str.
    """
    # Preserve MultiIndex on cell columns so per-cell outputs can still
    # unstack by (expiry, tenor).
    cols = patterns.columns.intersection(target.columns)
    if len(cols) == 0:
        raise ValueError("patterns and target share no columns")
    X_df = target[cols].dropna(axis=0)
    X = X_df.values
    P = patterns[cols].values.astype(float).copy()
    pattern_names = list(patterns.index)
    K = P.shape[0]
    T = X.shape[0]

    if standardize_patterns:
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        P = P / norms

    if method == "ols":
        # Solve P^T @ A^T = X^T in LS sense via lstsq (numerically stable).
        A_T, *_ = np.linalg.lstsq(P.T, X.T, rcond=None)
        A = A_T.T
    elif method == "ridge":
        Gram = P @ P.T
        A = X @ P.T @ np.linalg.solve(Gram + ridge_alpha * np.eye(K),
                                      np.eye(K))
    elif method == "nnls":
        from scipy.optimize import nnls
        A = np.zeros((T, K))
        Pt = P.T
        for t in range(T):
            A[t], _ = nnls(Pt, X[t])
    elif method == "lasso":
        from sklearn.linear_model import Lasso
        A = np.zeros((T, K))
        Pt = P.T
        las = Lasso(alpha=lasso_alpha, fit_intercept=False, max_iter=5000)
        for t in range(T):
            las.fit(Pt, X[t])
            A[t] = las.coef_
    else:
        raise ValueError(
            f"method must be one of 'ols' | 'ridge' | 'nnls' | 'lasso', "
            f"got {method!r}"
        )

    fitted = A @ P                                     # (T, N)
    resid = X - fitted

    total_var = float((X ** 2).sum())
    resid_var = float((resid ** 2).sum())
    r2_total = 1.0 - resid_var / total_var if total_var > 0 else 0.0

    per_cell_total = (X ** 2).sum(axis=0)
    per_cell_resid = (resid ** 2).sum(axis=0)
    r2_cell = 1.0 - per_cell_resid / np.where(
        per_cell_total == 0, 1.0, per_cell_total
    )

    per_date_total = (X ** 2).sum(axis=1)
    per_date_resid = (resid ** 2).sum(axis=1)
    r2_date = 1.0 - per_date_resid / np.where(
        per_date_total == 0, 1.0, per_date_total
    )

    cond_num = float(np.linalg.cond(P @ P.T))

    # Intrinsic pattern correlation (constant across the choice of method).
    P_centered = P - P.mean(axis=1, keepdims=True)
    P_norms = np.linalg.norm(P_centered, axis=1, keepdims=True)
    P_norms[P_norms == 0] = 1.0
    Pn = P_centered / P_norms
    pattern_corr = pd.DataFrame(
        Pn @ Pn.T, index=pattern_names, columns=pattern_names
    )

    A_df = pd.DataFrame(A, index=X_df.index, columns=pattern_names)
    return {
        "method":           method,
        "exposures":        A_df,
        "fitted":           pd.DataFrame(fitted, index=X_df.index, columns=cols),
        "residuals":        pd.DataFrame(resid, index=X_df.index, columns=cols),
        "r2_total":         r2_total,
        "r2_per_cell":      pd.Series(r2_cell, index=cols, name="r2_per_cell"),
        "r2_per_date":      pd.Series(r2_date, index=X_df.index, name="r2_per_date"),
        "exposure_corr":    A_df.corr(),
        "pattern_corr":     pattern_corr,
        "condition_number": cond_num,
        "patterns_used":    pd.DataFrame(P, index=pattern_names, columns=cols),
    }
