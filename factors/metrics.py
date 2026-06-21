"""Comparison metrics across factor-construction methods.

* ``variance_retained``     — cumulative explained-variance share at each K.
* ``loading_sparsity``      — Gini coefficient of |loading| per PC.
* ``rolling_stability``     — |cosine similarity| of rolling-window loadings
                              vs full-sample loadings.
* ``replication_residual``  — residual-variance fraction after K-factor
                              reconstruction (hedge-replication proxy).
* ``metrics_table``         — one-row-per-method summary covering the four.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pca import run_pca


def variance_retained(explained: pd.Series) -> pd.Series:
    """Cumulative explained-variance share at each K. Direct passthrough
    for the cumulative scree — included so every metric has a function."""
    return explained.cumsum()


def loading_sparsity(loadings: pd.DataFrame) -> pd.Series:
    """Gini coefficient over |loading| per PC. 0 = uniformly spread, 1 =
    all mass on one cell. Higher means the factor lives on fewer cube cells.
    """
    out = {}
    for pc in loadings.index:
        v = np.sort(np.abs(loadings.loc[pc].values))
        n = len(v)
        s = v.sum()
        if s == 0 or n == 0:
            out[pc] = 0.0
            continue
        cum = np.cumsum(v)
        out[pc] = (n + 1 - 2 * cum.sum() / s) / n
    return pd.Series(out, name="gini")


def rolling_stability(
    wide: pd.DataFrame,
    n_components: int = 3,
    window: int = 250,
    step: int = 10,
    standardize: bool = True,
) -> pd.DataFrame:
    """Refit PCA on each rolling window, compare loadings to the
    full-sample fit by absolute cosine similarity (sign is arbitrary in
    PCA). Returns a (window_end_date × PC) DataFrame in [0, 1]; values
    near 1 mean the factor is stable.
    """
    _, full_loadings, _ = run_pca(wide, n_components=n_components, standardize=standardize)
    X = wide.dropna(axis=1)
    cols = full_loadings.columns
    rows = []
    dates = []
    for end in range(window, len(X) + 1, step):
        sub = X.iloc[end - window:end]
        try:
            _, L, _ = run_pca(sub, n_components=n_components, standardize=standardize)
        except ValueError:
            continue
        shared = cols.intersection(L.columns)
        a = full_loadings[shared].values
        b = L[shared].values
        sims = np.abs(
            (a * b).sum(axis=1)
            / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        )
        rows.append(sims)
        dates.append(X.index[end - 1])
    return pd.DataFrame(rows, index=pd.Index(dates, name="window_end"),
                        columns=full_loadings.index)


def replication_residual(
    wide_original: pd.DataFrame,
    recon: pd.DataFrame,
) -> float:
    """Residual variance fraction of ``wide_original - recon``, summed
    across cube cells and normalised by total surface variance. Method-
    agnostic: works for any reconstruction expressed in the same units as
    ``wide_original`` (vanilla / varimax / block).
    """
    cols = recon.columns
    X = wide_original.loc[recon.index, cols]
    total = X.var().sum()
    if total == 0:
        return 0.0
    return float((X - recon).var().sum() / total)


def metrics_table(
    name: str,
    wide: pd.DataFrame,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    explained: pd.Series,
    replication: dict[int, float] | None = None,
    k_grid: tuple[int, ...] = (1, 3, 5),
    stability_window: int = 250,
    stability_step: int = 25,
) -> pd.DataFrame:
    """One-row-per-method summary covering the four comparison metrics.

    ``replication`` is an optional ``{k: residual_fraction}`` map already
    computed by the caller (since the reconstruction recipe differs by
    method). When absent, the ``resid_at_K`` columns are simply omitted.
    Designed for vanilla / varimax / single-panel PCA; block PCA has its
    own ``block_summary`` because per-block scores aren't directly
    comparable to a single panel-wide PCA.
    """
    cum = variance_retained(explained)
    spars = loading_sparsity(loadings)
    stab = rolling_stability(
        wide, n_components=min(3, scores.shape[1]),
        window=stability_window, step=stability_step,
    )
    row = {
        "method": name,
        "loading_gini_mean": float(spars.mean()),
        "stability_mean": float(stab.mean().mean()) if len(stab) else float("nan"),
    }
    for k in k_grid:
        if k <= len(cum):
            row[f"var_at_{k}"] = float(cum.iloc[k - 1])
        if replication is not None and k in replication:
            row[f"resid_at_{k}"] = float(replication[k])
    return pd.DataFrame([row])
