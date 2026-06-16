"""Factor-construction extensions to vanilla ``pca.py``.

All work below operates on **daily diffs** of the wide panels — we
care about cube moves, not levels. The notebook builds the diff panels
once at setup; every helper here is unit-agnostic and simply takes a
wide DataFrame.

Tracks
------
* ``varimax`` + ``rotate_scores``  — orthogonal rotation of PCA loadings
                                     for per-factor sparsity.
* ``block_pca`` + ``reconstruct_block`` + ``stack_block_scores``
                                   — partition the cube into a grid of
                                     blocks, PCA each block independently,
                                     reassemble. Localises factors to
                                     hedge-mappable regions.
* ``sparse_pca_warm``              — warm-started sparse PCA. Start from
                                     an artificial pattern, iterate so
                                     loadings fit the market better while
                                     staying close to the prior. Sequential
                                     power iteration with deflation;
                                     unit-norm loadings.
* ``soft_constrained_pca`` +
  ``lambda_search``                — joint-ALS matrix factorisation with a
                                     Tikhonov penalty on ``V - V0``. Fits
                                     all k factors simultaneously; loadings
                                     are not unit-norm. ``lambda_search``
                                     sweeps λ on a train/val time split.
* ``regress``                      — generic OLS of any target panel on
                                     any factor panel. Use block PCs (or
                                     any other factor design) to explain
                                     surface cells one-shot.
* ``cross_surface_cca`` + ``lagged_corr``
                                   — CCA between two surfaces' PC scores
                                     (or any pair of score panels).

Metrics
-------
* ``variance_retained``     — cumulative explained-variance share.
* ``loading_sparsity``      — Gini coefficient of |loading| per PC.
* ``rolling_stability``     — |cosine similarity| of rolling-window
                              loadings vs full-sample loadings.
* ``replication_residual``  — residual-variance fraction after K-factor
                              reconstruction (hedge-replication proxy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

from pca import reconstruct, run_pca


# ---- Varimax rotation ------------------------------------------------------
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


# ---- Block PCA -------------------------------------------------------------
# Default 3x3 partition. Edit (or pass your own) for finer / coarser grids.
DEFAULT_EXPIRY_BLOCKS: list[tuple[str, list[str]]] = [
    ("short_exp", ["1M", "2M", "3M", "6M", "9M"]),
    ("mid_exp",   ["1Y", "2Y", "3Y", "4Y", "5Y"]),
    ("long_exp",  ["7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]),
]
DEFAULT_TENOR_BLOCKS: list[tuple[str, list[str]]] = [
    ("short_ten", ["1Y", "2Y", "3Y"]),
    ("mid_ten",   ["4Y", "5Y", "7Y", "10Y"]),
    ("long_ten",  ["12Y", "15Y", "20Y", "25Y", "30Y"]),
]


def make_blocks(
    expiry_groups: list[tuple[str, list[str]]] = DEFAULT_EXPIRY_BLOCKS,
    tenor_groups:  list[tuple[str, list[str]]] = DEFAULT_TENOR_BLOCKS,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Cartesian product of expiry × tenor groups. Returns a dict mapping
    ``(expiry_block_name, tenor_block_name)`` to the list of
    ``(expiry, tenor)`` cells in that block. Default 3×3 grid gives 9
    blocks covering the canonical cube.
    """
    return {
        (e_name, t_name): [(e, t) for e in e_list for t in t_list]
        for e_name, e_list in expiry_groups
        for t_name, t_list in tenor_groups
    }


def block_pca(
    wide: pd.DataFrame,
    blocks: dict[tuple[str, str], list[tuple[str, str]]] | None = None,
    n_components: int = 2,
    standardize: bool = True,
) -> dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]]:
    """Run PCA independently inside each block of the partition.

    Returns ``{block_key: {"scores": s, "loadings": L, "explained": e}}``.
    Each block's PCA is fit only on the cells in that block, so factors
    are local to that region of the cube. Per-block component count is
    capped at the block's column count.
    """
    if blocks is None:
        blocks = make_blocks()
    out: dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]] = {}
    for key, cells in blocks.items():
        cols = [c for c in cells if c in wide.columns]
        if not cols:
            continue
        sub = wide[cols].dropna(axis=1)
        if sub.shape[1] == 0:
            continue
        k = min(n_components, sub.shape[1])
        s, L, e = run_pca(sub, n_components=k, standardize=standardize)
        out[key] = {"scores": s, "loadings": L, "explained": e}
    return out


def reconstruct_block(
    wide: pd.DataFrame,
    block_results: dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]],
    n_components: int | None = None,
) -> pd.DataFrame:
    """Per-block reconstruction stitched back into the full grid.

    ``n_components`` truncates each block to its first ``k`` factors
    (same ``k`` across blocks). ``None`` uses every fitted PC. The result
    has the same row index as the per-block scores and only the columns
    that were actually fit (cells outside the partition are dropped).
    """
    parts = []
    for res in block_results.values():
        s, L = res["scores"], res["loadings"]
        k = s.shape[1] if n_components is None else min(n_components, s.shape[1])
        cols = L.columns
        scaled = s.iloc[:, :k].values @ L.iloc[:k].values
        means = wide[cols].mean().values
        stds = wide[cols].std().values
        parts.append(pd.DataFrame(scaled * stds + means,
                                  index=s.index, columns=cols))
    return pd.concat(parts, axis=1)


def stack_block_scores(
    block_results: dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]],
    top_k: int | None = None,
) -> pd.DataFrame:
    """Flatten per-block scores into one (date × factor) DataFrame.
    Column names are ``{expiry_block}|{tenor_block}|PC{i}``. Convenient
    for feeding the block factors into a downstream model (e.g. CCA
    against vol-surface factors).
    """
    parts = []
    for (e_name, t_name), res in block_results.items():
        s = res["scores"]
        k = s.shape[1] if top_k is None else min(top_k, s.shape[1])
        sub = s.iloc[:, :k].copy()
        sub.columns = [f"{e_name}|{t_name}|{c}" for c in sub.columns]
        parts.append(sub)
    return pd.concat(parts, axis=1)


def block_summary(
    block_results: dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]],
) -> pd.DataFrame:
    """One row per block: shape and cumulative variance at each retained
    PC. Use to decide where the partition is too coarse (PC1 dominant) or
    too fine (variance spread thinly across local PCs)."""
    rows = []
    for (e_name, t_name), res in block_results.items():
        e = res["explained"]
        row = {
            "expiry_block": e_name,
            "tenor_block":  t_name,
            "n_cells":      res["loadings"].shape[1],
            "n_pcs":        len(e),
        }
        for i in range(len(e)):
            row[f"cum_var_at_{i + 1}"] = float(e.iloc[: i + 1].sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---- Anchored sparse PCA (warm-started from an artificial pattern) ---------
def sparse_pca_warm(
    wide: pd.DataFrame,
    prior: pd.Series | pd.DataFrame,
    anchor: float = 1.0,
    l1: float = 0.0,
    max_iter: int = 200,
    tol: float = 1e-7,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Sparse PCA started from an artificial loading pattern.

    Initialises loadings at ``prior`` (the desk's hand-drawn factor —
    e.g. a 2s10s steepener mask, a short-end shock, …) and iterates an
    alternating score/loading update. The loading update penalises both
    deviation from ``prior`` (Tikhonov anchor) and absolute size (L1),
    so the result is a factor that explains more market variance than
    the raw pattern but stays visually close to it.

    Parameters
    ----------
    wide       : (date × (expiry, tenor)) panel — typically already diffed.
    prior      : pd.Series indexed by ``wide.columns`` (one factor) **or**
                 pd.DataFrame whose columns match ``wide.columns`` and
                 whose rows are individual factor priors (one row → one
                 factor). The prior is normalised to unit L2 norm.
    anchor     : weight on ``||w - w_prior||²``, scaled internally by N
                 (sample size) so it's comparable to the data-fit term
                 regardless of panel length. ``0`` ≈ unconstrained PCA
                 with a warm start; ``~0.1`` lightly steers toward the
                 prior; ``~1`` is balanced; ``~10`` ≈ loadings frozen.
    l1         : optional L1 soft-threshold for extra sparsity, scaled
                 internally by N. ``0`` disables; values in ``[0, ~1]``
                 zero out cells whose data-projection magnitude is below
                 the threshold.
    max_iter   : per-factor iteration cap.
    tol        : L2 tolerance on the loading delta for early stopping.
    standardize: standardise ``wide`` cell-wise before fitting (matches
                 ``run_pca``).

    Returns
    -------
    (scores, loadings, explained) — same shapes as ``run_pca``. Loadings
    are unit-norm. ``explained`` is ``||f_i||² / ||X||_F²`` against the
    standardised panel; for multiple anchored factors the entries can
    overlap (no enforced orthogonality) so the sum may exceed each
    individual share but isn't a clean cumulative.
    """
    prior_df = prior.to_frame().T if isinstance(prior, pd.Series) else prior.copy()

    X_full = wide.dropna(axis=1)
    cols = [c for c in prior_df.columns if c in X_full.columns]
    if not cols:
        raise ValueError("prior columns do not overlap wide.columns")
    X = X_full[cols]
    prior_df = prior_df[cols]

    Xs = (StandardScaler().fit_transform(X.values) if standardize
          else X.values - X.values.mean(axis=0))

    W_prior = prior_df.values.astype(float).copy()
    norms = np.linalg.norm(W_prior, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W_prior = W_prior / norms

    k = W_prior.shape[0]
    W = np.zeros_like(W_prior)
    F = np.zeros((Xs.shape[0], k))

    N = Xs.shape[0]
    eff_anchor = anchor * N
    eff_l1 = l1 * N
    X_residual = Xs.copy()
    for i in range(k):
        w = W_prior[i].copy()
        w_prior = W_prior[i]
        for _ in range(max_iter):
            f = X_residual @ w
            denom = float(f @ f) + eff_anchor
            num = X_residual.T @ f + eff_anchor * w_prior
            if eff_l1 > 0:
                num = np.sign(num) * np.maximum(np.abs(num) - eff_l1, 0.0)
            w_new = num / denom if denom > 0 else w
            n = np.linalg.norm(w_new)
            if n > 0:
                w_new = w_new / n
            if np.linalg.norm(w_new - w) < tol:
                w = w_new
                break
            w = w_new
        f = X_residual @ w
        W[i] = w
        F[:, i] = f
        X_residual = X_residual - np.outer(f, w)

    total_var = float((Xs ** 2).sum())
    evr = (F ** 2).sum(axis=0) / total_var if total_var > 0 else np.zeros(k)

    factor_names = list(prior_df.index)
    return (
        pd.DataFrame(F, index=X.index, columns=factor_names),
        pd.DataFrame(W, index=factor_names, columns=cols),
        pd.Series(evr, index=factor_names, name="explained_variance_ratio"),
    )


# ---- Joint-ALS soft-constrained PCA ----------------------------------------
def soft_constrained_pca(
    wide: pd.DataFrame,
    V0: pd.Series | pd.DataFrame,
    lam: float = 1.0,
    lambda_per_factor: np.ndarray | pd.Series | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    init: str = "prior",
    standardize: bool = True,
) -> dict:
    """Joint ALS for the soft-constrained matrix-factorisation objective::

        min_{V, F}  ||X - F V^T||_F^2  +  sum_j λ_j ||v_j - v_j^(0)||^2

    Different from ``sparse_pca_warm`` (sequential power iteration with
    deflation, unit-norm loadings) — this fits all k factors *jointly* as
    a regularised matrix factorisation. Loadings are not constrained to
    unit norm; scores carry the scale.

    Parameters
    ----------
    wide              : (T, N) DataFrame — date-indexed, MultiIndex columns.
    V0                : prior loadings. (k, N) DataFrame, one row per
                        factor, **or** (N,) Series for k=1. Codebase
                        convention — same shape as ``run_pca`` loadings.
    lam               : scalar regulariser.
    lambda_per_factor : optional length-k array; overrides ``lam`` so
                        each factor can be anchored at its own strength.
    max_iter, tol     : ALS controls — stop on relative ``||ΔV||_F / ||V||_F``.
    init              : ``'prior'`` (V = V0) or ``'pca'`` (V = top-k right
                        singular vectors of the standardised panel).
    standardize       : standardise X cell-wise before fitting.

    Returns
    -------
    dict with keys
        ``V``                    : (k, N) DataFrame, fitted loadings.
        ``F``                    : (T, k) DataFrame, fitted scores.
        ``reconstruction_error`` : ``||X - F V^T||_F^2``.
        ``loading_deviation``    : unweighted ``||V - V0||_F^2``.
        ``objective_history``    : objective per iteration (recon +
                                   λ-weighted deviation).
    """
    V0_df = V0.to_frame().T if isinstance(V0, pd.Series) else V0.copy()
    cols = [c for c in V0_df.columns if c in wide.columns]
    if not cols:
        raise ValueError("V0 columns do not overlap wide.columns")
    X_full = wide[cols].dropna(axis=0)
    V0_df = V0_df[cols]
    factor_names = list(V0_df.index)
    k = len(factor_names)

    Xs = (StandardScaler().fit_transform(X_full.values) if standardize
          else X_full.values - X_full.values.mean(axis=0))

    if lambda_per_factor is None:
        lam_vec = np.full(k, float(lam))
    else:
        lam_vec = np.asarray(lambda_per_factor, dtype=float).reshape(-1)
        if len(lam_vec) != k:
            raise ValueError(
                f"lambda_per_factor length {len(lam_vec)} != k={k}"
            )
    Lam = np.diag(lam_vec)

    V0_arr = V0_df.values.T.astype(float).copy()       # (N, k)

    if init == "prior":
        V = V0_arr.copy()
    elif init == "pca":
        _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
        V = Vt[:k].T.copy()
    else:
        raise ValueError(f"init must be 'prior' or 'pca', got {init!r}")

    def _objective(V_: np.ndarray, F_: np.ndarray) -> float:
        recon = float(np.sum((Xs - F_ @ V_.T) ** 2))
        weighted_dev = float(
            np.sum(lam_vec * np.sum((V_ - V0_arr) ** 2, axis=0))
        )
        return recon + weighted_dev

    jitter = 1e-12 * np.eye(k)
    obj_hist: list[float] = []
    V_prev = V.copy()
    F = np.zeros((Xs.shape[0], k))

    for _ in range(max_iter):
        # Step 1: solve (V^T V) F^T = V^T X^T ⇒ F = (V^T V)^{-1} V^T X^T, transposed
        VtV = V.T @ V
        F = np.linalg.solve(VtV + jitter, V.T @ Xs.T).T

        # Step 2: solve (F^T F + Lam) V^T = F^T X + Lam V0^T
        FtF = F.T @ F
        rhs = F.T @ Xs + lam_vec[:, None] * V0_arr.T   # (k, N)
        V = np.linalg.solve(FtF + Lam + jitter, rhs).T  # (N, k)

        obj_hist.append(_objective(V, F))
        if np.linalg.norm(V - V_prev) / (np.linalg.norm(V_prev) + 1e-12) < tol:
            break
        V_prev = V.copy()

    # Final F update so scores are consistent with the returned V.
    VtV = V.T @ V
    F = np.linalg.solve(VtV + jitter, V.T @ Xs.T).T

    recon_err = float(np.sum((Xs - F @ V.T) ** 2))
    load_dev = float(np.sum((V - V0_arr) ** 2))

    return {
        "V": pd.DataFrame(V.T, index=factor_names, columns=X_full.columns),
        "F": pd.DataFrame(F, index=X_full.index, columns=factor_names),
        "reconstruction_error": recon_err,
        "loading_deviation": load_dev,
        "objective_history": obj_hist,
    }


def lambda_search(
    wide: pd.DataFrame,
    V0: pd.Series | pd.DataFrame,
    lambda_grid: list[float] | np.ndarray,
    val_frac: float = 0.2,
    init: str = "prior",
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Chronological train/val split, fit ``soft_constrained_pca`` for
    each λ in ``lambda_grid`` on the train fold, evaluate out-of-sample
    reconstruction on the val fold. Standardisation is fit on train only.

    Returns a DataFrame with columns ``lambda``,
    ``val_reconstruction_error``, ``loading_deviation`` — the Pareto
    data. **Does not auto-select λ**; pick from the curve by eye.
    """
    V0_df = V0.to_frame().T if isinstance(V0, pd.Series) else V0
    cols = [c for c in V0_df.columns if c in wide.columns]
    X_full = wide[cols].dropna(axis=0)
    V0_arr = V0_df[cols].values.T.astype(float)        # (N, k)

    n_val = max(1, int(len(X_full) * val_frac))
    X_train = X_full.iloc[:-n_val]
    X_val = X_full.iloc[-n_val:]

    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values)
    X_val_s = scaler.transform(X_val.values)

    train_df = pd.DataFrame(X_train_s, index=X_train.index,
                            columns=X_train.columns)

    rows = []
    for lam in lambda_grid:
        fit = soft_constrained_pca(
            train_df, V0_df, lam=float(lam), init=init,
            max_iter=max_iter, tol=tol, standardize=False,
        )
        V = fit["V"].values.T                          # (N, k)
        k = V.shape[1]
        VtV = V.T @ V
        F_val = np.linalg.solve(
            VtV + 1e-12 * np.eye(k), V.T @ X_val_s.T
        ).T
        val_recon = float(np.sum((X_val_s - F_val @ V.T) ** 2))
        load_dev = float(np.sum((V - V0_arr) ** 2))
        rows.append({
            "lambda": float(lam),
            "val_reconstruction_error": val_recon,
            "loading_deviation": load_dev,
        })
    return pd.DataFrame(rows)


# ---- Generic regression interface ------------------------------------------
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


# ---- Joint rate-vol structure ----------------------------------------------
def cross_surface_cca(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    n_components: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Canonical correlation between two score panels (any pair —
    typically rate PCs vs vol PCs, or rate block scores vs vol block
    scores). Returns (canonical_a, canonical_b, canonical_correlations).
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
