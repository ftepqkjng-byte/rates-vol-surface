"""Anchor-point regression factor model.

Factors are **observable cube moves** at a small set of fixed
``(expiry, tenor)`` grid points (the "anchors"). Every other cell's
daily diff is regressed onto the anchor moves, so the beta panel is a
direct hedge ratio: a move at anchor ``a`` of size 1 implies a move of
``beta_{cell, a}`` at every other cell, by construction.

Unlike PCA / sparse PCA / block PCA, the factors here are not latent
components extracted from the data — they are concrete tradeable
instruments picked up front (or chosen greedily). This makes the
resulting hedge directly executable and the beta panel immediately
interpretable to a trader.

Helpers in this module
----------------------
* ``get_anchor_slice``    — pull the anchor cells' time series out of a
                            wide diff panel.
* ``residual_panel``      — ``diff − anchor_moves @ betas.T``.
* ``fit_ridge``           — per-cell ridge regression onto the anchors
                            (``lam=0`` ⇒ plain OLS).
* ``fit_nnls``            — per-cell non-negative LSQ (``β ≥ 0``); no
                            ridge term needed — non-negativity alone is
                            usually enough to keep the fit well-posed.
* ``fit_simplex``         — per-cell simplex-constrained LSQ
                            (``β ≥ 0`` and ``Σβ = 1``); each cell's move
                            is modelled as a convex combination of
                            anchor moves.
* ``fit_ridge_rolling`` /
  ``fit_nnls_rolling`` /
  ``fit_simplex_rolling`` — strictly-past rolling-window versions.
* ``anchor_diagnostics``  — correlation matrix, Gram condition number,
                            VIFs — flags multicollinear anchor sets.
* ``greedy_anchor_select``— forward greedy search over the cube for the
                            ``k`` anchors that maximise explained
                            variance on the non-anchor cells.
* ``r2_heatmap``          — per-cell in-sample R², reshaped to
                            ``(expiry × tenor)`` for plotting.
* ``constraint_cost_heatmap`` — per-cell R² under all three modes side
                            by side; the cost of each constraint layer.
* ``convex_hull_flags``   — boolean grid: cells where the simplex
                            constraint costs >5 R² points vs NNLS;
                            these cells lie outside the convex hull of
                            the anchor moves.
* ``beta_allocation_table``— canonical-order presentation of a beta
                            panel as a hedge-allocation table.
* ``beta_stability``      — std of each ``(cell, anchor)`` rolling beta
                            over time.
* ``anchor_metrics_row``  — single-row summary slottable into
                            ``factors.metrics_table``; reports ridge /
                            NNLS / simplex R² and the hull-breach %.

Every helper operates on the existing wide diff panels with
``MultiIndex(expiry, tenor)`` columns produced by ``data/pipeline.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize

from config import EXPIRY_RANK, TENOR_RANK
from .metrics import loading_sparsity


def _anchor_label(expiry: str, tenor: str) -> str:
    """Canonical ``{expiry}|{tenor}`` flat label for a cube cell."""
    return f"{expiry}|{tenor}"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_anchor_slice(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
) -> pd.DataFrame:
    """Extract the move time series at the specified anchor points.

    Parameters
    ----------
    diff
        Wide daily-diff panel (``date × MultiIndex(expiry, tenor)``).
    anchors
        List of ``(expiry_label, tenor_label)`` tuples. Every tuple must
        appear in ``diff.columns``.

    Returns
    -------
    DataFrame of shape ``(n_dates, n_anchors)`` whose columns are the
    flat labels ``'{expiry}|{tenor}'``, preserving anchor order.
    """
    missing = [a for a in anchors if a not in diff.columns]
    if missing:
        raise KeyError(f"anchor cells absent from diff.columns: {missing}")
    out = diff[anchors].copy()
    out.columns = [_anchor_label(e, t) for e, t in anchors]
    return out


def residual_panel(
    diff: pd.DataFrame,
    anchor_moves: pd.DataFrame,
    betas: pd.DataFrame,
) -> pd.DataFrame:
    """Residual cube ``diff − anchor_moves @ betas.T``.

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    anchor_moves
        Output of :func:`get_anchor_slice` — columns are the flat anchor
        labels in the order they were fit.
    betas
        ``(n_cells × n_anchors)`` panel from :func:`fit_ridge` —
        ``betas.index`` matches ``diff.columns`` and ``betas.columns``
        matches ``anchor_moves.columns``.

    Returns
    -------
    DataFrame the same shape as ``diff`` (restricted to dates in
    ``anchor_moves`` and cells in ``betas.index``). Anchor cells have
    residual exactly zero because their beta row is one-hot.
    """
    common = diff.index.intersection(anchor_moves.index)
    am = anchor_moves.loc[common, betas.columns].values
    b = betas[betas.columns].values  # (N, K)
    fitted = am @ b.T  # (T, N)
    fitted_df = pd.DataFrame(fitted, index=common, columns=betas.index)
    return diff.loc[common, betas.index] - fitted_df


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

def fit_ridge(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    lam: float = 0.0,
) -> pd.DataFrame:
    """Per-cell ridge regression of each cell's diff onto the anchor diffs.

    Closed-form solve ``β = (AᵀA + lam·I)⁻¹ Aᵀy`` for every cell, with
    anchor cells overridden to a one-hot row so the beta panel covers
    the entire cube uniformly (``β = e_i`` for the ``i``-th anchor).

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    anchors
        ``(expiry, tenor)`` cells to use as observable factors.
    lam
        L2 ridge penalty. ``0`` is plain OLS. Raise ``ValueError`` if
        the anchor Gram matrix is rank-deficient *and* ``lam == 0`` —
        bumping ``lam`` slightly (e.g. ``1e-4``) or dropping a
        collinear anchor (see :func:`anchor_diagnostics`) fixes it.

    Returns
    -------
    DataFrame of shape ``(n_cells, n_anchors)``. Index is the same
    ``MultiIndex(expiry, tenor)`` as ``diff.columns``; columns are the
    flat ``'{expiry}|{tenor}'`` anchor labels.

    Notes
    -----
    NaN handling is per-cell: rows where any anchor is NaN are dropped
    once up-front; cells with sparse NaN values are re-solved on their
    own observed rows. For typical post-burn-in diff panels (no NaN)
    the solve is a single vectorised lstsq across all cells.
    """
    anchor_labels = [_anchor_label(e, t) for e, t in anchors]
    K = len(anchors)

    A_df = diff[anchors].dropna()
    if len(A_df) < K:
        raise ValueError(
            f"need at least K={K} clean anchor rows, got {len(A_df)}"
        )
    A = A_df.values  # (T, K)
    Y_df = diff.loc[A_df.index]
    Y = Y_df.values  # (T, N)

    AtA = A.T @ A
    if lam <= 0.0 and np.linalg.matrix_rank(AtA) < K:
        raise ValueError(
            f"anchor Gram matrix is singular at lam=0 (rank-deficient, "
            f"K={K}). Use a positive lam or drop a collinear anchor — "
            f"see anchor_diagnostics()."
        )
    M = AtA + lam * np.eye(K)

    cell_has_nan = np.isnan(Y).any(axis=0)
    betas = np.full((Y.shape[1], K), np.nan)

    # Vectorised matrix solve for fully-observed cells.
    if (~cell_has_nan).any():
        Yc = Y[:, ~cell_has_nan]
        betas[~cell_has_nan] = np.linalg.solve(M, A.T @ Yc).T

    # Sparse-NaN cells: per-cell solve on their observed rows.
    for j in np.where(cell_has_nan)[0]:
        m = ~np.isnan(Y[:, j])
        if m.sum() < K:
            continue
        A_j, y_j = A[m], Y[m, j]
        AtA_j = A_j.T @ A_j + lam * np.eye(K)
        try:
            betas[j] = np.linalg.solve(AtA_j, A_j.T @ y_j)
        except np.linalg.LinAlgError:
            continue

    out = pd.DataFrame(betas, index=Y_df.columns, columns=anchor_labels)

    # Anchor cells are explained by themselves, exactly.
    for i, anchor in enumerate(anchors):
        if anchor in out.index:
            out.loc[anchor] = 0.0
            out.loc[anchor, anchor_labels[i]] = 1.0

    return out


def fit_ridge_rolling(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    lam: float = 0.0,
    window: int = 126,
) -> pd.DataFrame:
    """Rolling-window version of :func:`fit_ridge` (strict past, no leakage).

    For each date ``t`` with index position ``i ≥ window``, fit on the
    window ``diff.iloc[i-window:i]`` (i.e. the last ``window`` business
    days *before* ``t``). This gives a panel of betas that can be
    inspected for stability — large per-(cell, anchor) std implies the
    hedge ratio drifts over time.

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    anchors
        Same as :func:`fit_ridge`.
    lam
        Ridge penalty. Rank-deficient windows are skipped silently
        when ``lam == 0`` (preferred over raising mid-loop).
    window
        Rolling lookback. ``126`` ≈ 6 months of business days.

    Returns
    -------
    DataFrame of shape ``(n_fit_dates, n_cells * n_anchors)`` with a
    three-level ``MultiIndex(expiry, tenor, anchor)`` on the columns —
    the first two levels mirror ``diff.columns`` so per-cell results
    unstack straight onto the cube grid. Dates with fewer than
    ``window`` past observations are dropped, hence the first ``window``
    rows of ``diff`` produce no output.

    Notes
    -----
    Leakage-free: beta at date ``t`` is built from
    ``diff.iloc[i-window:i]`` which does **not** include ``t`` itself.
    """
    anchor_labels = [_anchor_label(e, t) for e, t in anchors]
    K = len(anchors)
    N = diff.shape[1]

    A_all = diff[anchors].values
    Y_all = diff.values
    dates = diff.index
    cell_pos = {c: i for i, c in enumerate(diff.columns)}

    rows: dict = {}
    for t_idx in range(window, len(dates)):
        win_A = A_all[t_idx - window:t_idx]
        win_Y = Y_all[t_idx - window:t_idx]

        mask = (~np.isnan(win_A).any(axis=1)) & (~np.isnan(win_Y).any(axis=1))
        if mask.sum() < K:
            continue
        A = win_A[mask]
        Y = win_Y[mask]

        AtA = A.T @ A
        if lam <= 0.0 and np.linalg.matrix_rank(AtA) < K:
            continue  # rank-deficient window, skip silently
        M = AtA + lam * np.eye(K)
        try:
            B = np.linalg.solve(M, A.T @ Y).T  # (N, K)
        except np.linalg.LinAlgError:
            continue

        for i, anchor in enumerate(anchors):
            if anchor in cell_pos:
                ci = cell_pos[anchor]
                B[ci] = 0.0
                B[ci, i] = 1.0

        rows[dates[t_idx]] = B.flatten()

    col_index = pd.MultiIndex.from_tuples(
        [(*cell, a) for cell in diff.columns for a in anchor_labels],
        names=["expiry", "tenor", "anchor"],
    )
    if not rows:
        return pd.DataFrame(columns=col_index)
    return pd.DataFrame.from_dict(rows, orient="index", columns=col_index)


# ---------------------------------------------------------------------------
# Constrained solvers — NNLS and simplex
# ---------------------------------------------------------------------------

def _prepare_anchor_panels(
    diff: pd.DataFrame, anchors: list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Common pre-processing for the constrained fitters: drop rows where
    any anchor is NaN, return ``(A, Y, Y_df, anchor_labels)``."""
    anchor_labels = [_anchor_label(e, t) for e, t in anchors]
    K = len(anchors)
    A_df = diff[anchors].dropna()
    if len(A_df) < K:
        raise ValueError(
            f"need at least K={K} clean anchor rows, got {len(A_df)}"
        )
    Y_df = diff.loc[A_df.index]
    return A_df.values, Y_df.values, Y_df, anchor_labels


def fit_nnls(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
) -> pd.DataFrame:
    """Per-cell non-negative least squares onto the anchor moves.

    For every cube cell ``(i, j)`` solve

    .. math::
        \\min_{\\beta \\ge 0}\\; \\| y_{i,j} - A\\beta \\|^2

    using :func:`scipy.optimize.nnls`. Anchor cells get a one-hot beta
    row (``β = e_i`` for the ``i``-th anchor), matching the
    :func:`fit_ridge` convention so beta panels from all fit modes are
    directly comparable.

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    anchors
        ``(expiry, tenor)`` cells used as observable factors.

    Returns
    -------
    DataFrame of shape ``(n_cells, n_anchors)``. Same index/column
    convention as :func:`fit_ridge`.

    Notes
    -----
    No regularisation parameter is needed — non-negativity alone
    keeps the problem well-posed. Unlike OLS, NNLS does **not** raise
    on collinear anchors: it gracefully zeroes the dominated anchor
    (active-set termination). The cost of that robustness is that
    NNLS has no closed-form solution — every cell is a separate small
    iterative solve.
    """
    A, Y, Y_df, anchor_labels = _prepare_anchor_panels(diff, anchors)
    K = len(anchors)
    cell_has_nan = np.isnan(Y).any(axis=0)
    betas = np.full((Y.shape[1], K), np.nan)

    for j in range(Y.shape[1]):
        if cell_has_nan[j]:
            m = ~np.isnan(Y[:, j])
            if m.sum() < K:
                continue
            A_j, y_j = A[m], Y[m, j]
        else:
            A_j, y_j = A, Y[:, j]
        try:
            sol, _ = nnls(A_j, y_j)
            betas[j] = sol
        except RuntimeError:
            continue

    out = pd.DataFrame(betas, index=Y_df.columns, columns=anchor_labels)
    for i, anchor in enumerate(anchors):
        if anchor in out.index:
            out.loc[anchor] = 0.0
            out.loc[anchor, anchor_labels[i]] = 1.0
    return out


def fit_simplex(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
) -> pd.DataFrame:
    """Per-cell simplex-constrained least squares onto the anchor moves.

    For every cube cell ``(i, j)`` solve the convex QP

    .. math::
        \\min_{\\beta}\\; \\| y_{i,j} - A\\beta \\|^2
        \\quad \\text{s.t.}\\quad \\beta \\ge 0,\\ \\mathbf{1}^T\\beta = 1

    using :func:`scipy.optimize.minimize` with ``method='SLSQP'``. Each
    cell's daily move is modelled as a **convex combination** of the
    anchor moves — a strong interpretability assumption that holds
    naturally for cube cells inside the convex hull of the anchors but
    is restrictive for cells whose moves have larger amplitude than
    any anchor (use :func:`convex_hull_flags` to spot those).

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    anchors
        ``(expiry, tenor)`` cells used as observable factors.

    Returns
    -------
    DataFrame of shape ``(n_cells, n_anchors)``. Every non-anchor row
    is non-negative and sums to exactly 1; anchor rows are one-hot
    (which also satisfies both constraints).

    Notes
    -----
    Each SLSQP solve is warm-started from the NNLS solution
    (normalised to sum to 1, or uniform ``1/K`` if NNLS returned all
    zeros), which roughly halves iteration counts vs a cold start.
    Small negative entries from numerical SLSQP slack are clipped to
    zero and the row is re-normalised so ``Σβ = 1`` holds exactly in
    the returned DataFrame.
    """
    A, Y, Y_df, anchor_labels = _prepare_anchor_panels(diff, anchors)
    K = len(anchors)
    cell_has_nan = np.isnan(Y).any(axis=0)
    betas = np.full((Y.shape[1], K), np.nan)

    # NNLS warm-start — reuses fit_nnls so anchor cells already one-hot.
    nnls_betas = fit_nnls(diff, anchors).values

    bounds = [(0.0, None)] * K
    constraints = {"type": "eq", "fun": lambda x: x.sum() - 1.0,
                   "jac": lambda x: np.ones(K)}

    for j in range(Y.shape[1]):
        if cell_has_nan[j]:
            m = ~np.isnan(Y[:, j])
            if m.sum() < K:
                continue
            A_j, y_j = A[m], Y[m, j]
        else:
            A_j, y_j = A, Y[:, j]

        x0 = nnls_betas[j]
        s = float(np.nansum(x0))
        if not np.isfinite(s) or s <= 1e-12:
            x0 = np.ones(K) / K
        else:
            x0 = np.clip(x0, 0.0, None) / s

        def loss(x, A_j=A_j, y_j=y_j):
            r = A_j @ x - y_j
            return 0.5 * float(r @ r)

        def grad(x, A_j=A_j, y_j=y_j):
            return A_j.T @ (A_j @ x - y_j)

        try:
            res = minimize(
                loss, x0, jac=grad, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 100},
            )
            x = np.clip(res.x, 0.0, None)
            s_x = x.sum()
            if s_x > 1e-12:
                betas[j] = x / s_x
            else:
                betas[j] = np.ones(K) / K
        except Exception:
            continue

    out = pd.DataFrame(betas, index=Y_df.columns, columns=anchor_labels)
    for i, anchor in enumerate(anchors):
        if anchor in out.index:
            out.loc[anchor] = 0.0
            out.loc[anchor, anchor_labels[i]] = 1.0
    return out


def _fit_rolling_constrained(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    window: int,
    fit_fn,
) -> pd.DataFrame:
    """Shared rolling-window driver for ``fit_nnls_rolling`` and
    ``fit_simplex_rolling``. Strictly past — at date ``t`` we fit on
    ``diff.iloc[i-window:i]`` (no inclusion of ``t``)."""
    anchor_labels = [_anchor_label(e, t) for e, t in anchors]
    K = len(anchors)
    dates = diff.index
    rows: dict = {}
    for t_idx in range(window, len(dates)):
        win_df = diff.iloc[t_idx - window:t_idx]
        try:
            betas = fit_fn(win_df, anchors)
        except (ValueError, np.linalg.LinAlgError):
            continue
        rows[dates[t_idx]] = betas.values.flatten()
    col_index = pd.MultiIndex.from_tuples(
        [(*cell, a) for cell in diff.columns for a in anchor_labels],
        names=["expiry", "tenor", "anchor"],
    )
    if not rows:
        return pd.DataFrame(columns=col_index)
    return pd.DataFrame.from_dict(rows, orient="index", columns=col_index)


def fit_nnls_rolling(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    window: int = 126,
) -> pd.DataFrame:
    """Rolling-window version of :func:`fit_nnls`, strict past only.

    Same return shape as :func:`fit_ridge_rolling`. Dates with fewer
    than ``window`` past observations are dropped. Use the output with
    :func:`beta_stability` to spot cells whose non-negative hedge
    weights drift through time.
    """
    return _fit_rolling_constrained(diff, anchors, window, fit_nnls)


def fit_simplex_rolling(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    window: int = 126,
) -> pd.DataFrame:
    """Rolling-window version of :func:`fit_simplex`, strict past only.

    Same return shape as :func:`fit_ridge_rolling`. Slower than the
    NNLS / ridge variants because each window runs an SLSQP solve per
    cube cell; budget ~1–2 min for a year of business days on the
    full 200-cell cube.
    """
    return _fit_rolling_constrained(diff, anchors, window, fit_simplex)


# ---------------------------------------------------------------------------
# Multicollinearity diagnostics
# ---------------------------------------------------------------------------

def anchor_diagnostics(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
) -> dict:
    """Multicollinearity health-check on a candidate anchor set.

    Returns
    -------
    dict with three entries:

    * ``'corr'``             : ``(K × K)`` Pearson correlation matrix.
    * ``'condition_number'`` : ``cond(AᵀA)`` — > ~30 starts to bite,
                               > ~1e6 means OLS is effectively singular.
    * ``'vif'``              : Series of variance-inflation factors per
                               anchor. ``VIF_i = 1 / (1 − R²_i)`` where
                               ``R²_i`` is from regressing anchor ``i``
                               on the others. > ~10 is the standard
                               flag.

    High condition number or high VIF means the anchor set is
    redundant — either bump ``lam`` on :func:`fit_ridge` or reselect.
    """
    anchor_labels = [_anchor_label(e, t) for e, t in anchors]
    A_df = diff[anchors].dropna()
    A_df.columns = anchor_labels
    A = A_df.values
    K = len(anchors)

    corr = A_df.corr()
    cond = float(np.linalg.cond(A.T @ A))

    vif = {}
    for i, name in enumerate(anchor_labels):
        if K == 1:
            vif[name] = 1.0
            continue
        y = A[:, i]
        X = np.delete(A, i, axis=1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif[name] = float("inf") if (1.0 - r2) < 1e-12 else 1.0 / (1.0 - r2)

    return {
        "corr": corr,
        "condition_number": cond,
        "vif": pd.Series(vif, name="vif"),
    }


# ---------------------------------------------------------------------------
# Greedy forward selection
# ---------------------------------------------------------------------------

def greedy_anchor_select(
    diff: pd.DataFrame,
    k: int,
    lam: float = 0.0,
    exclude: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Forward greedy search for the ``k`` best anchor cells.

    At each step, pick the cube cell whose addition to the current
    anchor set maximises the *reduction* in residual variance across
    all cells. Equivalently: pick the cell that maximises cumulative
    R² (with anchor cells contributing zero residual, since their
    beta row is one-hot).

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    k
        Number of anchors to select.
    lam
        Ridge penalty passed to every intermediate :func:`fit_ridge`
        call. Use a small positive value (e.g. ``1e-4``) if you expect
        candidate cells to be highly correlated.
    exclude
        Optional list of ``(expiry, tenor)`` cells forbidden as anchors
        (e.g. illiquid cells that can't be traded as a hedge).

    Returns
    -------
    List of ``k`` dicts in selection order, each with keys
    ``'anchor'``, ``'marginal_r2'``, ``'cumulative_r2'``. Prints a
    progress line per step since the search runs ``k · n_cells`` ridge
    fits in total.
    """
    forbidden = set(tuple(c) for c in (exclude or []))
    cells = list(diff.columns)

    # Fixed denominator across steps — total centered variance of every cell.
    centered = diff - diff.mean()
    total_var = float((centered ** 2).sum().sum())
    if total_var <= 0.0:
        raise ValueError("diff has zero total variance; nothing to explain")

    selected: list[tuple[str, str]] = []
    history: list[dict] = []
    prev_r2 = 0.0

    for step in range(k):
        candidates = [c for c in cells if c not in selected and c not in forbidden]
        if not candidates:
            break
        best_cand = None
        best_r2 = -np.inf
        for cand in candidates:
            trial = selected + [cand]
            try:
                betas = fit_ridge(diff, trial, lam=lam)
            except ValueError:
                continue  # singular gram with this candidate, skip
            am = get_anchor_slice(diff, trial)
            resid = residual_panel(diff, am, betas)
            resid_ss = float((resid.values ** 2).sum())
            r2 = 1.0 - resid_ss / total_var
            if r2 > best_r2:
                best_r2 = r2
                best_cand = cand

        if best_cand is None:
            break

        marginal = best_r2 - prev_r2
        selected.append(best_cand)
        history.append({
            "anchor":        best_cand,
            "marginal_r2":   marginal,
            "cumulative_r2": best_r2,
        })
        print(
            f"greedy step {step + 1}/{k}: picked {best_cand}, "
            f"cumulative R² = {best_r2:.4f}, marginal R² = {marginal:.4f}"
        )
        prev_r2 = best_r2

    return history


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def r2_heatmap(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    lam: float = 0.0,
) -> pd.DataFrame:
    """Per-cell in-sample R², reshaped to ``(expiry × tenor)``.

    Centered R² (``1 − SS_resid / SS_tot`` with ``SS_tot`` measured
    around the column mean) for every cell, then pivoted into a grid
    with rows ordered by ``EXPIRY_RANK`` and columns by ``TENOR_RANK``
    so heatmap axes follow the canonical cube ordering.

    Anchor cells score exactly ``1.0`` by construction (one-hot beta).
    """
    betas = fit_ridge(diff, anchors, lam=lam)
    am = get_anchor_slice(diff, anchors)
    resid = residual_panel(diff, am, betas)

    aligned = diff.loc[resid.index, resid.columns]
    ss_resid = (resid ** 2).sum(axis=0)
    ss_total = ((aligned - aligned.mean()) ** 2).sum(axis=0)
    safe_total = ss_total.where(ss_total > 0, 1.0)
    r2 = 1.0 - ss_resid / safe_total

    heatmap = r2.unstack(level="tenor")
    row_order = sorted(heatmap.index, key=lambda e: EXPIRY_RANK.get(e, 1e9))
    col_order = sorted(heatmap.columns, key=lambda t: TENOR_RANK.get(t, 1e9))
    return heatmap.reindex(index=row_order, columns=col_order)


def _r2_per_cell(
    diff: pd.DataFrame, anchors: list[tuple[str, str]], betas: pd.DataFrame,
) -> pd.Series:
    """Centered per-cell R² of a given beta panel (shared helper for the
    constraint-comparison diagnostics)."""
    am = get_anchor_slice(diff, anchors)
    resid = residual_panel(diff, am, betas)
    aligned = diff.loc[resid.index, resid.columns]
    ss_resid = (resid ** 2).sum(axis=0)
    ss_total = ((aligned - aligned.mean()) ** 2).sum(axis=0)
    safe_total = ss_total.where(ss_total > 0, 1.0)
    return 1.0 - ss_resid / safe_total


def constraint_cost_heatmap(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    lam: float = 0.0,
) -> pd.DataFrame:
    """Per-cell R² under ridge / NNLS / simplex, side by side.

    Returns
    -------
    DataFrame indexed by ``MultiIndex(expiry, tenor)`` (i.e. one row
    per cube cell, ``n_expiry · n_tenor`` rows in total) with columns
    ``['r2_ridge', 'r2_nnls', 'r2_simplex']``. Differences between
    columns quantify the explained-variance cost of each constraint
    layer:

    * ``r2_ridge − r2_nnls``    : cost of forbidding negative hedge ratios.
    * ``r2_nnls  − r2_simplex`` : cost of forcing the hedge ratios to
                                  sum to one (convex-combination
                                  assumption).
    """
    ridge_betas   = fit_ridge(diff, anchors, lam=lam)
    nnls_betas    = fit_nnls(diff, anchors)
    simplex_betas = fit_simplex(diff, anchors)
    return pd.DataFrame({
        "r2_ridge":   _r2_per_cell(diff, anchors, ridge_betas),
        "r2_nnls":    _r2_per_cell(diff, anchors, nnls_betas),
        "r2_simplex": _r2_per_cell(diff, anchors, simplex_betas),
    })


def convex_hull_flags(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
) -> pd.DataFrame:
    """Flag cells outside the convex hull of the anchor moves.

    A cell is flagged ``True`` when its simplex-constrained R² falls
    more than 5 percentage points below its NNLS R² — i.e. the
    ``Σβ = 1`` constraint costs real fit, which can only happen when
    the cell's move cannot be written as a convex combination of the
    anchor moves (its amplitude or direction lies outside the
    anchors' convex hull).

    Returns
    -------
    Boolean DataFrame of shape ``(n_expiry, n_tenor)`` with rows
    ordered by ``EXPIRY_RANK`` and columns by ``TENOR_RANK``. Anchor
    cells are always ``False`` (their one-hot beta satisfies both
    constraints trivially, so the R² gap is zero).
    """
    nnls_betas    = fit_nnls(diff, anchors)
    simplex_betas = fit_simplex(diff, anchors)
    r2_nnls    = _r2_per_cell(diff, anchors, nnls_betas)
    r2_simplex = _r2_per_cell(diff, anchors, simplex_betas)
    breach = (r2_simplex < r2_nnls - 0.05)

    grid = breach.unstack(level="tenor")
    row_order = sorted(grid.index, key=lambda e: EXPIRY_RANK.get(e, 1e9))
    col_order = sorted(grid.columns, key=lambda t: TENOR_RANK.get(t, 1e9))
    return grid.reindex(index=row_order, columns=col_order).astype(bool)


def beta_allocation_table(
    betas: pd.DataFrame,
    anchors: list[tuple[str, str]],
) -> pd.DataFrame:
    """Canonical-order presentation of a beta panel as an allocation table.

    Re-orders the rows of ``betas`` by ``(EXPIRY_RANK, TENOR_RANK)``
    and the columns by the order in ``anchors``. For
    :func:`fit_simplex` output, each row sums to 1 and represents how
    a unit of notional in cell ``(expiry, tenor)`` is allocated across
    the anchor instruments — read row-wise as a hedge recipe.

    The function does not transform values; it's purely a re-indexing
    helper so any fit panel (ridge / NNLS / simplex) displays in
    grid order with the anchor columns in the user's chosen order.
    """
    anchor_labels = [_anchor_label(e, t) for e, t in anchors]
    sorted_idx = sorted(
        betas.index,
        key=lambda c: (EXPIRY_RANK.get(c[0], 1e9), TENOR_RANK.get(c[1], 1e9)),
    )
    return betas.reindex(index=sorted_idx, columns=anchor_labels)


def beta_stability(
    rolling_betas: pd.DataFrame,
    anchors: list[tuple[str, str]],
) -> pd.DataFrame:
    """Std of each ``(cell, anchor)`` rolling beta over time.

    Takes the output of :func:`fit_ridge_rolling` (MultiIndex columns
    ``[cell, anchor]``) and collapses across dates into a single value
    per cell × anchor.

    Returns
    -------
    DataFrame of shape ``(n_cells, n_anchors)``. High values flag cells
    whose hedge ratio onto a given anchor drifts over time.
    """
    if rolling_betas.empty:
        anchor_labels = [_anchor_label(e, t) for e, t in anchors]
        return pd.DataFrame(columns=anchor_labels)
    stds = rolling_betas.std(axis=0)  # (n_cells * n_anchors,)
    return stds.unstack(level="anchor")


def anchor_metrics_row(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    lam: float = 0.0,
    window: int = 126,
    mode: str = "ridge",
) -> dict:
    """Single-row summary slottable into the ``factors.metrics_table`` sheet.

    Always fits all three regression modes (ridge / NNLS / simplex)
    on the supplied anchor set so the row reports the full
    R²-cost-of-constraint picture in one call. ``mode`` only selects
    which mode's R² takes the ``'variance_retained'`` headline slot
    and which rolling fit feeds ``'rolling_stability'``.

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    anchors
        ``(expiry, tenor)`` cells used as observable factors.
    lam
        Ridge penalty (used by the ridge fit and the ridge rolling fit).
    window
        Rolling lookback for the stability metric.
    mode
        One of ``'ridge'`` / ``'nnls'`` / ``'simplex'``. Selects the
        ``method`` label, the ``variance_retained`` headline, and the
        rolling fitter used for ``rolling_stability``.

    Returns
    -------
    dict with keys

    * ``'method'``                : ``'anchor_<mode>_k{K}[_lam{lam}]'``.
    * ``'variance_retained'``     : headline mean per-cell R² (depends
                                    on ``mode``).
    * ``'loading_sparsity_gini'`` : mean Gini of ``|beta|`` per anchor
                                    column on the **selected mode**'s
                                    betas.
    * ``'rolling_stability'``     : ``1 − mean(beta_std) / mean(|beta|)``,
                                    clipped to ``[0, 1]``, on the
                                    selected mode's rolling betas.
    * ``'hedge_residual_r2'``     : ``1 − variance_retained``
                                    (sanity check).
    * ``'r2_ridge_mean'`` /
      ``'r2_nnls_mean'`` /
      ``'r2_simplex_mean'``       : per-mode mean per-cell R² — always
                                    populated so the row supports
                                    cross-mode comparison regardless
                                    of which mode was selected.
    * ``'hull_breach_pct'``       : fraction of non-anchor cells flagged
                                    by :func:`convex_hull_flags`. ``0``
                                    means every non-anchor cell stays
                                    inside the anchors' convex hull.
    """
    ridge_betas   = fit_ridge(diff, anchors, lam=lam)
    nnls_betas    = fit_nnls(diff, anchors)
    simplex_betas = fit_simplex(diff, anchors)

    r2_ridge_pc   = _r2_per_cell(diff, anchors, ridge_betas)
    r2_nnls_pc    = _r2_per_cell(diff, anchors, nnls_betas)
    r2_simplex_pc = _r2_per_cell(diff, anchors, simplex_betas)

    r2_ridge_mean   = float(r2_ridge_pc.mean())
    r2_nnls_mean    = float(r2_nnls_pc.mean())
    r2_simplex_mean = float(r2_simplex_pc.mean())

    # Hull breach % on non-anchor cells (anchor cells are trivially OK).
    anchor_set = set(anchors)
    non_anchor = [c for c in r2_nnls_pc.index if c not in anchor_set]
    if non_anchor:
        breach = (r2_simplex_pc.loc[non_anchor]
                  < r2_nnls_pc.loc[non_anchor] - 0.05)
        hull_breach_pct = float(breach.mean())
    else:
        hull_breach_pct = 0.0

    # Mode-dependent dispatch for the headline metrics.
    K = len(anchors)
    if mode == "ridge":
        method   = f"anchor_ridge_k{K}_lam{lam}"
        headline = r2_ridge_mean
        betas    = ridge_betas
        rolling_fn = lambda: fit_ridge_rolling(diff, anchors, lam=lam, window=window)
    elif mode == "nnls":
        method   = f"anchor_nnls_k{K}"
        headline = r2_nnls_mean
        betas    = nnls_betas
        rolling_fn = lambda: fit_nnls_rolling(diff, anchors, window=window)
    elif mode == "simplex":
        method   = f"anchor_simplex_k{K}"
        headline = r2_simplex_mean
        betas    = simplex_betas
        rolling_fn = lambda: fit_simplex_rolling(diff, anchors, window=window)
    else:
        raise ValueError(
            f"mode must be 'ridge' / 'nnls' / 'simplex', got {mode!r}"
        )

    gini_per_anchor = loading_sparsity(betas.T)
    mean_gini = float(gini_per_anchor.mean())

    try:
        rolling = rolling_fn()
        if len(rolling) > 1:
            std_panel = beta_stability(rolling, anchors)
            mean_abs_beta = float(betas.abs().mean().mean())
            scale = max(mean_abs_beta, 1e-12)
            rel_instability = float(std_panel.mean().mean()) / scale
            stability = float(max(0.0, 1.0 - min(1.0, rel_instability)))
        else:
            stability = float("nan")
    except (ValueError, np.linalg.LinAlgError):
        stability = float("nan")

    return {
        "method":                method,
        "variance_retained":     headline,
        "loading_sparsity_gini": mean_gini,
        "rolling_stability":     stability,
        "hedge_residual_r2":     1.0 - headline,
        "r2_ridge_mean":         r2_ridge_mean,
        "r2_nnls_mean":          r2_nnls_mean,
        "r2_simplex_mean":       r2_simplex_mean,
        "hull_breach_pct":       hull_breach_pct,
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def anchor_zscore(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    window: int = 63,
) -> pd.DataFrame:
    """Rolling strict-past z-score of each anchor's daily move.

    For date ``t`` and anchor ``a``::

        z_{a,t} = (move_{a,t} − mean_past) / std_past

    where ``mean_past`` and ``std_past`` are taken over the prior
    ``window`` business days (``diff.iloc[t-window:t]``); the
    ``.shift(1)`` on the rolling stats keeps date ``t`` itself out.

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    anchors
        ``(expiry, tenor)`` cells whose moves we are z-scoring.
    window
        Lookback for the rolling mean and std. ``63`` ≈ 3 months.

    Returns
    -------
    DataFrame of shape ``(n_dates, n_anchors)`` with the flat
    ``'{expiry}|{tenor}'`` anchor labels as columns. Large ``|z|``
    (typically ``> 3``) marks dates where an anchor moved by an
    unusual amount versus its recent history.
    """
    am = get_anchor_slice(diff, anchors)
    mean = am.rolling(window=window).mean().shift(1)
    std  = am.rolling(window=window).std().shift(1)
    safe_std = std.where(std > 0)
    return (am - mean) / safe_std


def residual_zscore(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    lam: float = 0.0,
    window: int = 63,
) -> pd.Series:
    """Rolling strict-past z-score of the per-date residual energy.

    For date ``t``::

        energy_t = RMS_cells( residual_{cell,t} )
        z_t      = (energy_t − mean_past) / std_past

    The residual panel is built from full-sample :func:`fit_ridge`
    betas (in-sample analysis): a spike at date ``t`` means *the
    surface at ``t`` does not match the regression model the rest of
    the sample agrees on*.

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    anchors
        ``(expiry, tenor)`` cells used as observable factors.
    lam
        Ridge penalty for the underlying :func:`fit_ridge` call.
    window
        Rolling lookback for the residual-energy mean / std.

    Returns
    -------
    Series indexed by date. Combine with :func:`anchor_zscore` via
    :func:`anomaly_report` to spot dates where an anchor moved
    abnormally *and* the regression model failed there (the
    "decoupling" case).
    """
    betas  = fit_ridge(diff, anchors, lam=lam)
    am     = get_anchor_slice(diff, anchors)
    resid  = residual_panel(diff, am, betas)
    energy = np.sqrt((resid ** 2).mean(axis=1))
    mean   = energy.rolling(window=window).mean().shift(1)
    std    = energy.rolling(window=window).std().shift(1)
    safe_std = std.where(std > 0)
    return ((energy - mean) / safe_std).rename("residual_z")


def anomaly_report(
    diff: pd.DataFrame,
    anchors: list[tuple[str, str]],
    z_threshold: float = 3.0,
    lam: float = 0.0,
    window: int = 63,
) -> pd.DataFrame:
    """Per-date anomaly flags from anchor-move and residual z-scores.

    Combines :func:`anchor_zscore` (per-anchor z) and
    :func:`residual_zscore` (per-date residual-energy z) into one
    table with three boolean flags. The most actionable is
    ``decoupling``: dates where an anchor moved abnormally **and**
    the rest of the cube did not follow — a signal that either the
    anchor decoupled from the surface (regime shift, idiosyncratic
    event at that cell) or a factor the model can't capture moved.

    Parameters
    ----------
    diff
        Wide daily-diff panel.
    anchors
        ``(expiry, tenor)`` cells used as observable factors.
    z_threshold
        Flag dates with ``|z| > z_threshold``. ``3.0`` ≈ 0.3%
        false-positive rate under iid Gaussian moves.
    lam
        Ridge penalty for the underlying fit.
    window
        Rolling lookback for both z-score components.

    Returns
    -------
    DataFrame indexed by date with columns

    * ``max_anchor_z``     : ``max_a |z_{a,t}|`` across anchors.
    * ``worst_anchor``     : the anchor label achieving the max.
    * ``residual_z``       : signed residual-energy z (large positive
                             = surprisingly large residual).
    * ``anchor_anomaly``   : ``max_anchor_z > z_threshold``.
    * ``residual_anomaly`` : ``residual_z   > z_threshold``.
    * ``decoupling``       : both flags fire on the same date.

    Burn-in rows (dates where the rolling stats aren't defined yet)
    are dropped before returning.
    """
    az = anchor_zscore(diff, anchors, window=window)
    rz = residual_zscore(diff, anchors, lam=lam, window=window)
    abs_z = az.abs().dropna(how="all")
    out = pd.DataFrame({
        "max_anchor_z": abs_z.max(axis=1),
        "worst_anchor": abs_z.idxmax(axis=1),
        "residual_z":   rz,
    })
    out = out.dropna(subset=["max_anchor_z", "residual_z"])
    out["anchor_anomaly"]   = out["max_anchor_z"] > z_threshold
    out["residual_anomaly"] = out["residual_z"]   > z_threshold
    out["decoupling"]       = out["anchor_anomaly"] & out["residual_anomaly"]
    return out
