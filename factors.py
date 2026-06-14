"""Factor-construction extensions to vanilla ``pca.py``.

Four research tracks from the README, plus the four comparison metrics
shared across all of them. Every function takes wide-format DataFrames
(``date`` index, ``(expiry, tenor)`` MultiIndex columns) and is pure —
no global state, no fitted classes.

Tracks
------
* ``varimax`` + ``rotate_scores``      — orthogonal rotation of PCA loadings
                                          for per-factor sparsity.
* ``hierarchical_pca`` +
  ``reconstruct_hierarchical``         — strip per-date cube mean, PCA the
                                          residual, and reassemble.
* ``bucket_factors`` + ``regress_out`` +
  ``bucket_residual_pca`` +
  ``reconstruct_bucket_residual``      — hand-defined economic buckets,
                                          OLS-out, PCA the residual,
                                          reassemble.
* ``cross_surface_cca`` + ``lagged_corr`` — CCA between two surfaces' PC
                                            scores (or any pair of score
                                            panels).

Metrics
-------
* ``variance_retained``        — cumulative explained-variance share.
* ``loading_sparsity``         — Gini coefficient of |loading| per PC.
* ``rolling_stability``        — |cosine similarity| of rolling-window
                                 loadings vs full-sample loadings.
* ``replication_residual``     — residual-variance fraction after K-PC
                                 reconstruction (hedge-replication proxy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

from pca import EXPIRY_LABELS, TENOR_LABELS, reconstruct, run_pca


# ---- Track 1: Varimax rotation ---------------------------------------------
def varimax(
    loadings: pd.DataFrame,
    gamma: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> pd.DataFrame:
    """Kaiser varimax rotation. Input ``loadings`` is (k, p) — rows are PCs,
    columns the cube cells. Returns rotated loadings with the same shape /
    index / columns. Rotation is orthogonal so total variance is preserved.
    """
    L = loadings.values.T          # (p, k)
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        Lambda = L @ R
        B = Lambda ** 3 - (gamma / p) * Lambda @ np.diag(
            (Lambda ** 2).sum(axis=0)
        )
        U, S, Vt = np.linalg.svd(L.T @ B, full_matrices=False)
        R = U @ Vt
        d = S.sum()
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            break
    rotated = (L @ R).T            # (k, p)
    return pd.DataFrame(rotated, index=loadings.index, columns=loadings.columns)


def rotate_scores(scores: pd.DataFrame, loadings: pd.DataFrame,
                  rotated_loadings: pd.DataFrame) -> pd.DataFrame:
    """Apply the same rotation to the score series so the score basis
    matches the rotated loadings. Solves ``rotated = R.T @ loadings`` for
    ``R`` and right-multiplies the scores by ``R``.
    """
    R = np.linalg.lstsq(loadings.values.T, rotated_loadings.values.T, rcond=None)[0]
    rotated = scores.values @ R
    return pd.DataFrame(rotated, index=scores.index, columns=rotated_loadings.index)


# ---- Track 2: Hierarchical PCA ---------------------------------------------
def hierarchical_pca(
    wide: pd.DataFrame,
    n_components: int = 5,
    standardize: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Subtract per-date cube mean (the 'level' factor), then PCA the
    residual. Returns
    ``(level_series, residual_panel, scores, loadings, explained_variance_ratio)``.

    ``level_series`` is the cross-sectional mean across all
    ``(expiry, tenor)`` cells on each date — the L0 factor in the
    hierarchy. Subsequent PCs describe slope / curvature / region modes
    without competing with the level absorber. The residual panel is
    returned so ``reconstruct(...)`` can be called on it.
    """
    X = wide.dropna(axis=1)
    level = X.mean(axis=1)
    residual = X.sub(level, axis=0)
    scores, loadings, explained = run_pca(
        residual, n_components=n_components, standardize=standardize
    )
    return level, residual, scores, loadings, explained


def reconstruct_hierarchical(
    level: pd.Series,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    residual_panel: pd.DataFrame,
    n_pcs: int,
) -> pd.DataFrame:
    """Level + first-``n_pcs`` reconstruction of the residual, on the
    original surface's scale. ``n_pcs = 0`` returns just the level
    broadcast across cube cells.
    """
    cols = loadings.columns
    base = pd.DataFrame(
        np.broadcast_to(level.values[:, None], (len(level), len(cols))).copy(),
        index=level.index, columns=cols,
    )
    if n_pcs == 0:
        return base
    pc_recon = reconstruct(scores, loadings, residual_panel, n_components=n_pcs)
    return base + pc_recon


# ---- Track 3: Bucket factors + residual PCA --------------------------------
def default_buckets() -> dict[str, list[tuple[str, str]]]:
    """A small starter set of economically meaningful regions over the
    canonical ``(expiry, tenor)`` cube. Buckets are illustrative; the
    real desk version should be edited per the trader's hedge axes.

    * ``short_level``    — short-expiry × short-tenor cells.
    * ``long_level``     — long-expiry  × long-tenor cells.
    * ``slope_2s10s``    — 2Y vs 10Y tenor wing average (uses tag, not
                           literal subtraction — pair with another bucket
                           in a regression for the spread).
    * ``belly``          — mid-expiry × mid-tenor square.
    """
    short_exp = ["1M", "2M", "3M", "6M"]
    long_exp  = ["5Y", "7Y", "10Y", "15Y", "20Y", "25Y", "30Y"]
    short_t   = ["1Y", "2Y", "3Y"]
    long_t    = ["10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]
    mid_exp   = ["1Y", "2Y", "3Y"]
    mid_t     = ["3Y", "4Y", "5Y", "7Y"]
    return {
        "short_level": [(e, t) for e in short_exp for t in short_t],
        "long_level":  [(e, t) for e in long_exp  for t in long_t],
        "slope_2s10s": [(e, t) for e in EXPIRY_LABELS for t in ("2Y", "10Y")],
        "belly":       [(e, t) for e in mid_exp for t in mid_t],
    }


def bucket_factors(
    wide: pd.DataFrame,
    buckets: dict[str, list[tuple[str, str]]],
) -> pd.DataFrame:
    """Build a (date × bucket) factor panel by averaging the cells in
    each bucket. Cells missing from ``wide`` are silently skipped.
    """
    out: dict[str, pd.Series] = {}
    for name, pairs in buckets.items():
        cols = [p for p in pairs if p in wide.columns]
        if cols:
            out[name] = wide[cols].mean(axis=1)
    return pd.DataFrame(out)


def regress_out(
    wide: pd.DataFrame,
    factors: pd.DataFrame,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """OLS each ``wide`` column on ``factors`` jointly; return the residual
    panel (same shape as ``wide.dropna(axis=1)``).
    """
    X = wide.dropna(axis=1)
    F = factors.loc[X.index].values
    if add_intercept:
        F = np.column_stack([np.ones(len(F)), F])
    beta, *_ = np.linalg.lstsq(F, X.values, rcond=None)
    resid = X.values - F @ beta
    return pd.DataFrame(resid, index=X.index, columns=X.columns)


def bucket_residual_pca(
    wide: pd.DataFrame,
    buckets: dict[str, list[tuple[str, str]]],
    n_components: int = 5,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Convenience: build bucket factors, regress them out of ``wide``,
    PCA the residual. Returns
    ``(bucket_panel, residual_panel, scores, loadings, explained_variance_ratio)``.

    The residual panel is returned so callers can use
    ``reconstruct_bucket_residual`` (or call ``reconstruct(...)`` directly
    on the residual panel) to evaluate the additive model.
    """
    F = bucket_factors(wide, buckets)
    residual = regress_out(wide, F)
    scores, loadings, explained = run_pca(
        residual, n_components=n_components, standardize=standardize
    )
    return F, residual, scores, loadings, explained


def reconstruct_bucket_residual(
    wide: pd.DataFrame,
    residual_panel: pd.DataFrame,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    n_pcs: int,
) -> pd.DataFrame:
    """Bucket-OLS fit + first-``n_pcs`` reconstruction of the residual.
    The bucket fit is recovered as ``wide_cleaned - residual_panel``, so no
    betas need to be threaded through. ``n_pcs = 0`` returns just the
    bucket projection.
    """
    cols = residual_panel.columns
    fitted = wide[cols].loc[residual_panel.index] - residual_panel
    if n_pcs == 0:
        return fitted
    pc_recon = reconstruct(scores, loadings, residual_panel, n_components=n_pcs)
    return fitted + pc_recon


# ---- Track 4: Joint rate-vol structure -------------------------------------
def cross_surface_cca(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    n_components: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Canonical correlation between two score panels (any pair —
    typically rate PCs vs vol PCs). Returns (canonical_a, canonical_b,
    canonical_correlations).

    Use this to ask 'is there a linear combination of vol PCs that the
    rate PCs predict?' — answers the README's 'shared latent drivers'
    question more cleanly than naked PC-vs-PC correlations.
    """
    common = scores_a.index.intersection(scores_b.index)
    A = scores_a.loc[common].values
    B = scores_b.loc[common].values
    k = min(n_components, A.shape[1], B.shape[1])
    cca = CCA(n_components=k, max_iter=1000)
    Ac, Bc = cca.fit_transform(A, B)
    names = [f"CC{i + 1}" for i in range(k)]
    canonical_a = pd.DataFrame(Ac, index=common, columns=names)
    canonical_b = pd.DataFrame(Bc, index=common, columns=names)
    corrs = pd.Series(
        [np.corrcoef(Ac[:, i], Bc[:, i])[0, 1] for i in range(k)],
        index=names, name="canonical_corr",
    )
    return canonical_a, canonical_b, corrs


def lagged_corr(
    a: pd.Series,
    b: pd.Series,
    lags: range = range(-5, 6),
) -> pd.Series:
    """Cross-correlation between ``a`` and ``b`` at integer lags. Positive
    lag = ``a`` leads ``b``. Useful as a quick check before full CCA.
    """
    common = a.index.intersection(b.index)
    a = a.loc[common]; b = b.loc[common]
    return pd.Series(
        {k: a.corr(b.shift(-k)) for k in lags}, name="lagged_corr"
    )


# ---- Comparison metrics ----------------------------------------------------
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
    agnostic: works for vanilla / varimax / hierarchical / bucket-residual
    as long as the caller passes a reconstruction expressed in the same
    units as ``wide_original``.
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
    method — vanilla/varimax just call ``reconstruct``, hierarchical /
    bucket call their dedicated reconstructors). When absent, the
    ``resid_at_K`` columns are simply omitted.
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
