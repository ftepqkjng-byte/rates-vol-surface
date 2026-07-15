"""Book-vega hedging against a fixed set of trained patterns.

Distinct from ``factors.anchor``: there the factors are *concrete
tradeable cube cells* chosen to replicate the whole surface. Here the
factors are a small number of already-trained latent patterns (e.g.
from ``sparse_pca_warm`` / ``decorr_constrained_pca``), and the
question is the other direction — given a book's vega and each cell's
regression beta on the pattern scores, how much book exposure does the
book have to each pattern, and what minimal-notional hedge at a set of
liquid grid points brings every pattern exposure back within
tolerance.

Helpers in this module
-----------------------
* ``book_pattern_exposure`` — aggregate a vega panel into book exposure
  per pattern, via ``vega @ betas``.
* ``pattern_epsilon``       — per-pattern risk tolerance from the
  diagonal of a pattern-score covariance matrix.
* ``liquid_hedge_candidates`` — filter a cube index down to the liquid
  hedge universe (``config.LIQUID_EXPIRY_LABELS`` ×
  ``config.LIQUID_TENOR_LABELS``).
* ``sparse_hedge``          — the hedge LP, two dual formulations
  selected via ``method``:

  * ``"min_notional"`` (default) — minimise total weighted notional
    subject to each pattern's residual exposure staying within its
    epsilon band.
  * ``"min_residual"`` — minimise total residual exposure (summed
    across patterns) subject to every position staying under a fixed
    per-instrument notional cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from config import LIQUID_EXPIRY_LABELS, LIQUID_TENOR_LABELS


def book_pattern_exposure(vega: pd.Series, betas: pd.DataFrame) -> pd.Series:
    """Aggregate book vega into exposure per pattern.

    ``book_F_k = sum_i vega_i * beta_{i,k}`` — the book's dollar-vega
    sensitivity to a unit move in pattern ``k``'s score. Uses the full
    panels (not restricted to any liquid subset): the book can carry
    vega anywhere on the grid.

    Parameters
    ----------
    vega  : Series indexed by ``MultiIndex(expiry, tenor)``.
    betas : ``(n_cells, n_patterns)`` DataFrame, same index convention,
            columns are pattern names.

    Returns
    -------
    Series of length ``n_patterns``, indexed by ``betas.columns``.
    """
    common = vega.index.intersection(betas.index)
    if len(common) == 0:
        raise ValueError("vega and betas share no (expiry, tenor) cells")
    return betas.loc[common].mul(vega.loc[common], axis=0).sum(axis=0)


def pattern_epsilon(cov: pd.DataFrame, z: float = 1.0) -> pd.Series:
    """Per-pattern risk tolerance from the diagonal of a covariance matrix.

    ``eps_k = z * sqrt(cov.loc[k, k])``. Kept separate from
    ``sparse_hedge`` so a caller can supply a hand-picked epsilon
    without needing a covariance matrix at all.

    Parameters
    ----------
    cov : ``(n_patterns, n_patterns)`` covariance of the pattern
          scores (e.g. ``F.cov()``).
    z   : risk multiplier — ``1.0`` ≈ one standard deviation of daily
          pattern-score move.

    Returns
    -------
    Series indexed by ``cov.index``.
    """
    diag = pd.Series(np.diag(cov.values), index=cov.index)
    return z * np.sqrt(diag)


def liquid_hedge_candidates(index: pd.MultiIndex) -> pd.MultiIndex:
    """Filter a ``(expiry, tenor)`` index down to the liquid hedge universe.

    Keeps only cells whose expiry is in ``config.LIQUID_EXPIRY_LABELS``
    and whose tenor is in ``config.LIQUID_TENOR_LABELS``.
    """
    expiries = index.get_level_values("expiry")
    tenors = index.get_level_values("tenor")
    mask = expiries.isin(LIQUID_EXPIRY_LABELS) & tenors.isin(LIQUID_TENOR_LABELS)
    return index[mask]


def sparse_hedge(
    book_exposure: pd.Series,
    betas: pd.DataFrame,
    epsilon: pd.Series | None = None,
    position_cap: float | pd.Series | None = None,
    cost: pd.Series | None = None,
    candidates: pd.MultiIndex | None = None,
    tol: float = 1e-6,
    method: str = "min_notional",
) -> dict:
    """Hedge LP — two dual formulations of the same fit-vs-cost trade-off.

    ``method="min_notional"`` (default) — smallest weighted notional
    that brings every pattern's residual exposure within ``epsilon``::

        min_alpha   sum_i cost_i * |alpha_i|
        s.t.        |book_exposure_k - sum_i alpha_i * beta_{i,k}| <= epsilon_k
                    for every pattern k

    ``method="min_residual"`` — smallest total residual exposure
    (summed across patterns) given a fixed notional cap per
    instrument::

        min_alpha   sum_k |book_exposure_k - sum_i alpha_i * beta_{i,k}|
        s.t.        |alpha_i| * cost_i <= position_cap_i   for every candidate i

    Both are linearised into a standard-form LP by splitting
    ``alpha_i = alpha_plus_i - alpha_minus_i`` (both ``>= 0``), so
    ``|alpha_i| = alpha_plus_i + alpha_minus_i``; ``min_residual``
    additionally uses one epigraph variable ``t_k >= |residual_k|``
    per pattern so the summed absolute value stays linear. Solved with
    ``scipy.optimize.linprog(method="highs")``.

    Unlike ``min_notional``, ``min_residual`` is a box-constrained LP
    over ``alpha`` alone and is therefore always feasible (``alpha=0``
    trivially satisfies every bound) — it never raises for
    infeasibility, it just reports a large ``total_residual`` if
    ``position_cap`` is set too tight to hedge well.

    Parameters
    ----------
    book_exposure : Series of length ``n_patterns`` (from
                    ``book_pattern_exposure``).
    betas         : ``(n_cells, n_patterns)`` DataFrame — full beta
                    panel; restricted to ``candidates`` internally.
    epsilon       : Series of length ``n_patterns``, aligned to
                    ``book_exposure``/``betas.columns`` (from
                    ``pattern_epsilon`` or supplied directly). Required
                    for ``method="min_notional"``, ignored otherwise.
    position_cap  : per-instrument notional cap — a single float
                    (applied uniformly) or a Series indexed like
                    ``candidates``. Required for
                    ``method="min_residual"``, ignored otherwise.
    cost          : optional per-candidate unit cost/weight, indexed
                    like ``candidates``. Defaults to all-ones (pure
                    notional minimisation) — a real price/liquidity
                    weight can be dropped in later without an API
                    change.
    candidates    : ``(expiry, tenor)`` index of tradeable hedge
                    points. Defaults to
                    ``liquid_hedge_candidates(betas.index)``.
    tol           : threshold on ``|alpha_i|`` for counting a position
                    as active in ``n_active``.
    method        : ``"min_notional"`` or ``"min_residual"``.

    Returns
    -------
    dict with keys ``alpha`` (signed positions, Series over
    ``candidates``), ``hedge_exposure``, ``book_exposure``,
    ``residual_exposure``, ``total_notional`` (``sum cost_i*|alpha_i|``,
    always computed regardless of ``method``), ``total_residual``
    (``sum_k |residual_k|``, likewise always computed), ``n_active``,
    ``method``, ``raw_result`` (the ``scipy`` ``OptimizeResult``), plus
    ``epsilon`` (``method="min_notional"``) or ``position_cap``
    (``method="min_residual"``).

    Raises
    ------
    ValueError if ``method="min_notional"`` and the LP is infeasible
    (no combination of candidates can bring every pattern within its
    epsilon band) — widen ``epsilon`` or the candidate set rather than
    trusting a garbage solution. Also raised if the parameter required
    by the chosen ``method`` is missing.
    """
    if method not in ("min_notional", "min_residual"):
        raise ValueError(
            f"method must be 'min_notional' or 'min_residual', got {method!r}"
        )
    if method == "min_notional" and epsilon is None:
        raise ValueError("epsilon is required for method='min_notional'")
    if method == "min_residual" and position_cap is None:
        raise ValueError("position_cap is required for method='min_residual'")

    if candidates is None:
        candidates = liquid_hedge_candidates(betas.index)

    patterns = list(book_exposure.index)
    B_df = betas.loc[betas.index.intersection(candidates), patterns].dropna()
    if B_df.empty:
        raise ValueError("no candidate cells with complete betas")
    cand = B_df.index
    n = len(cand)

    if cost is None:
        cost_arr = np.ones(n)
    else:
        cost_arr = cost.reindex(cand).fillna(1.0).values

    B = B_df.values                      # (n, k)
    book = book_exposure.loc[patterns].values
    k = len(patterns)

    if method == "min_notional":
        eps = epsilon.loc[patterns].values
        c = np.concatenate([cost_arr, cost_arr])          # (2n,)
        A_ub = np.zeros((2 * k, 2 * n))
        b_ub = np.zeros(2 * k)
        for j in range(k):
            # book_j - hedge_j <= eps_j  =>  -B_j·(a+) + B_j·(a-) <= eps_j - book_j
            A_ub[2 * j, :n] = -B[:, j]
            A_ub[2 * j, n:] = B[:, j]
            b_ub[2 * j] = eps[j] - book[j]
            # hedge_j - book_j <= eps_j  =>  B_j·(a+) - B_j·(a-) <= eps_j + book_j
            A_ub[2 * j + 1, :n] = B[:, j]
            A_ub[2 * j + 1, n:] = -B[:, j]
            b_ub[2 * j + 1] = eps[j] + book[j]
        bounds = [(0.0, None)] * (2 * n)
    else:
        if np.isscalar(position_cap):
            cap_series = pd.Series(float(position_cap), index=cand)
        else:
            cap_series = position_cap.reindex(cand)
        cap_arr = cap_series.values / cost_arr            # per-candidate bound on |alpha_i|

        c = np.concatenate([np.zeros(2 * n), np.ones(k)])  # (2n+k,) — minimise sum(t)
        A_ub = np.zeros((2 * k, 2 * n + k))
        b_ub = np.zeros(2 * k)
        for j in range(k):
            # t_j >= book_j - hedge_j  =>  -B_j·(a+) + B_j·(a-) - t_j <= -book_j
            A_ub[2 * j, :n] = -B[:, j]
            A_ub[2 * j, n:2 * n] = B[:, j]
            A_ub[2 * j, 2 * n + j] = -1.0
            b_ub[2 * j] = -book[j]
            # t_j >= hedge_j - book_j  =>  B_j·(a+) - B_j·(a-) - t_j <= book_j
            A_ub[2 * j + 1, :n] = B[:, j]
            A_ub[2 * j + 1, n:2 * n] = -B[:, j]
            A_ub[2 * j + 1, 2 * n + j] = -1.0
            b_ub[2 * j + 1] = book[j]
        bounds = [(0.0, cap_arr[i]) for i in range(n)] * 2 + [(0.0, None)] * k

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise ValueError(
            f"sparse_hedge LP infeasible ({method}): {res.message} — widen "
            f"epsilon/position_cap or the candidate set"
        )

    alpha_arr = res.x[:n] - res.x[n:2 * n]
    alpha = pd.Series(alpha_arr, index=cand, name="alpha")

    hedge_exposure = pd.Series(alpha_arr @ B, index=patterns, name="hedge_exposure")
    book_exposure_out = book_exposure.loc[patterns]
    residual_exposure = book_exposure_out - hedge_exposure

    total_notional = float((cost_arr * np.abs(alpha_arr)).sum())
    total_residual = float(residual_exposure.abs().sum())
    n_active = int((np.abs(alpha_arr) > tol).sum())

    out = {
        "alpha": alpha,
        "hedge_exposure": hedge_exposure,
        "book_exposure": book_exposure_out,
        "residual_exposure": residual_exposure,
        "total_notional": total_notional,
        "total_residual": total_residual,
        "n_active": n_active,
        "method": method,
        "raw_result": res,
    }
    if method == "min_notional":
        out["epsilon"] = epsilon.loc[patterns]
    else:
        out["position_cap"] = cap_series
    return out
