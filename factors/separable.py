"""Separable / functional factor models for the ``(expiry, tenor)`` cube.

Track 6 of the factor-construction lineup (after vanilla / varimax /
block / sparse-warm / constrained PCA). Every routine here assumes the
cube has *separable* second-order structure — covariance behaving like
``C_expiry ⊗ C_tenor`` — and extracts factor patterns that respect it.

* ``marginal_kronecker_cov`` — naive moment estimator: reshape each
  daily move to an ``(n_expiry × n_tenor)`` matrix ``X_t`` and average
  ``X_t X_tᵀ`` for the expiry covariance, ``X_tᵀ X_t`` for the tenor
  covariance. Divides by the *other* axis dimension so the two
  marginals are on comparable scales. One pass, no iteration. Biased
  whenever the across-axis structure isn't separable, but a useful
  baseline / warm start.

* ``kronecker_cov_mle`` — flip-flop maximum-likelihood estimator for
  the matrix-normal model ``vec(X_t) ~ N(0, C_e ⊗ C_τ)``. Alternates
  between updating ``C_e`` given ``C_τ`` and vice versa until the
  per-iteration change falls below ``tol``. Individual scales of
  ``C_e`` / ``C_τ`` are unidentifiable (``kron(αC_e, C_τ/α) = kron(C_e, C_τ)``);
  resolved by fixing ``trace(C_e) = n_expiry`` after each update.

* ``kronecker_separability_residual`` — relative Frobenius residual of
  ``Σ_full - kron(C_e, C_τ)`` against the full empirical covariance.
  Diagnostic only — values close to 0 say the separability assumption
  is tenable on this data. Sensitive to the scale convention used for
  ``C_e`` / ``C_τ`` (kron is invariant but the bare residual isn't);
  use estimators from this module to stay self-consistent.

* ``roughness_penalty_1d`` / ``roughness_penalty_2d`` — finite-difference
  penalty matrices ``P = DᵀD``. Order-2 penalises curvature, leaving
  constants and linear trends in the null space. The 2D version is the
  Kronecker sum on the two axes (independently smooth along expiry and
  along tenor).

* ``functional_pca`` — solves ``eigh(Σ̂ - λP)`` on the flat
  ``(expiry × tenor)`` covariance, biasing the leading eigenvectors
  toward smooth loadings as ``λ`` grows. At ``λ = 0`` it reduces
  exactly to vanilla PCA on ``Σ̂`` (asserted in the tests).

* ``marginal_eigen_patterns`` — outer-product patterns from the top
  eigenvectors of ``C_e`` and ``C_τ``, packaged in the same
  ``{"name", "grid", "version"}`` list-of-dicts as
  ``pattern_basis.preset_separable_poly``. Drops straight into
  ``factors.sparse_pca_warm`` via
  ``pattern_basis.patterns_to_prior_df`` for a fully data-driven
  multi-factor warm start.

The wide panel passed in must have a full ``MultiIndex(expiry, tenor)``
rectangle of columns (typically the output of ``pca.to_wide`` after
``.diff().dropna()``). Rows with any NaN are dropped before estimation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from pattern_basis import degree_pair_grid, tensor_product_name


def _stack_panels(wide: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """Reshape a wide panel into ``(T, n_expiry, n_tenor)`` per-day matrices.

    Picks the expiry / tenor label sets out of the column MultiIndex in
    appearance order (matching the canonical ``to_wide`` ordering) and
    reindexes to the full rectangle before reshaping, so the array
    layout is ``X[t, i, j] == wide[t, (expiry_i, tenor_j)]`` regardless
    of how the caller sorted the columns. NaN rows are dropped.
    """
    if not isinstance(wide.columns, pd.MultiIndex):
        raise ValueError("wide.columns must be a (expiry, tenor) MultiIndex")
    exp_labels = list(dict.fromkeys(wide.columns.get_level_values("expiry")))
    ten_labels = list(dict.fromkeys(wide.columns.get_level_values("tenor")))
    n_e, n_t = len(exp_labels), len(ten_labels)
    expected = n_e * n_t
    if wide.shape[1] != expected:
        raise ValueError(
            f"wide has {wide.shape[1]} columns but the (expiry, tenor) "
            f"rectangle is {n_e} × {n_t} = {expected}; the Kronecker / "
            "functional decomposition needs a full grid."
        )
    target_cols = pd.MultiIndex.from_product(
        [exp_labels, ten_labels], names=["expiry", "tenor"]
    )
    clean = wide.reindex(columns=target_cols).dropna(how="any")
    if clean.empty:
        raise ValueError("no fully-observed rows in wide after dropna")
    panel = clean.values.reshape(-1, n_e, n_t)
    return panel, exp_labels, ten_labels


def marginal_kronecker_cov(
    wide: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Naive one-pass moment estimator of the matrix-normal marginals.

    For each fully-observed day ``t`` we reshape the row into
    ``X_t`` of shape ``(n_expiry × n_tenor)`` and average::

        C_expiry = mean_t(X_t @ X_tᵀ) / n_tenor
        C_tenor  = mean_t(X_tᵀ @ X_t) / n_expiry

    Dividing by the *other* axis dimension is the natural scale
    correction — under the matrix-normal model
    ``E[X X^T] = tr(C_tenor) · C_expiry``, so dividing by ``n_tenor``
    cancels the ``tr(C_tenor) = n_tenor`` normalisation convention
    (and analogously for the tenor side). Result: the two marginals
    are roughly on the same per-cell variance scale. Individual scales
    are still unidentifiable; ``kron(C_e, C_τ)`` is the only invariant
    quantity.

    Returns ``(C_expiry, C_tenor)`` as DataFrames indexed and columned
    by the canonical expiry / tenor labels recovered from the input.
    """
    panel, exp_labels, ten_labels = _stack_panels(wide)
    T, n_e, n_t = panel.shape
    C_e = np.zeros((n_e, n_e))
    C_t = np.zeros((n_t, n_t))
    for t in range(T):
        X = panel[t]
        C_e += X @ X.T
        C_t += X.T @ X
    C_e /= (T * n_t)
    C_t /= (T * n_e)
    return (
        pd.DataFrame(C_e, index=exp_labels, columns=exp_labels),
        pd.DataFrame(C_t, index=ten_labels, columns=ten_labels),
    )


def kronecker_separability_residual(
    wide: pd.DataFrame,
    C_e: pd.DataFrame,
    C_t: pd.DataFrame,
) -> float:
    """Relative Frobenius residual of ``Σ_full - kron(C_e, C_t)``.

    Reconstructs the full ``(n_e · n_t) × (n_e · n_t)`` sample
    covariance, subtracts ``kron(C_e, C_t)``, and returns the relative
    Frobenius norm ``||Σ - K|| / ||Σ||``. Values close to ``0`` mean
    the data really is matrix-normal-shaped; values approaching ``1``
    say a single Kronecker factor cannot describe the second-order
    structure.

    Strict reading of the spec — no automatic re-scaling. The bare
    residual is sensitive to whether ``C_e`` / ``C_t`` use a
    compatible normalisation; estimators from this module already
    pick one (``marginal_kronecker_cov`` via ``E[XX^T] = tr(C_t) C_e``,
    ``kronecker_cov_mle`` via ``trace(C_e) = n_e``), so callers that
    stay inside this module's estimators don't need to think about it.

    Layout: ``Σ_full`` is the covariance of ``vec_row(X_t)`` (i.e.
    expiry-major reshape, ``X[i, j] = wide[t, (e_i, t_j)]``), so the
    matching Kronecker order is ``C_e ⊗ C_t`` — *not* the
    matrix-normal-convention ``C_t ⊗ C_e``.
    """
    panel, _, _ = _stack_panels(wide)
    T, n_e, n_t = panel.shape
    flat = panel.reshape(T, n_e * n_t)
    sigma_full = (flat.T @ flat) / T
    kron = np.kron(C_e.values, C_t.values)
    diff_norm = float(np.linalg.norm(sigma_full - kron))
    full_norm = float(np.linalg.norm(sigma_full))
    return diff_norm / full_norm if full_norm > 0 else 0.0


def kronecker_cov_mle(
    wide: pd.DataFrame,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Flip-flop MLE for the matrix-normal model.

    Iterates the closed-form per-axis updates::

        C_e ← (1 / (T · n_t)) Σ_t X_t inv(C_t) X_tᵀ
        C_t ← (1 / (T · n_e)) Σ_t X_tᵀ inv(C_e) X_t

    starting from the naive ``marginal_kronecker_cov`` estimate, until
    ``||ΔC_e|| + ||ΔC_t||`` drops below ``tol`` or ``max_iter`` runs
    out. Individual scales of ``C_e`` / ``C_t`` are unidentifiable
    (``kron`` is invariant under ``(αC_e, C_t/α)``); resolved by
    renormalising ``trace(C_e) = n_expiry`` after each update and
    pushing the scale factor into ``C_t``.

    Returns ``(C_expiry, C_tenor, info)`` where ``info`` reports
    ``n_iter`` and ``converged`` so callers can flag a bad fit.
    """
    panel, exp_labels, ten_labels = _stack_panels(wide)
    T, n_e, n_t = panel.shape

    C_e_init, C_t_init = marginal_kronecker_cov(wide)
    C_e = C_e_init.values.copy()
    C_t = C_t_init.values.copy()
    scale = np.trace(C_e) / n_e
    if scale > 0:
        C_e /= scale
        C_t *= scale

    converged = False
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        inv_C_t = np.linalg.pinv(C_t)
        C_e_new = np.zeros_like(C_e)
        for t in range(T):
            X = panel[t]
            C_e_new += X @ inv_C_t @ X.T
        C_e_new /= (T * n_t)

        inv_C_e = np.linalg.pinv(C_e_new)
        C_t_new = np.zeros_like(C_t)
        for t in range(T):
            X = panel[t]
            C_t_new += X.T @ inv_C_e @ X
        C_t_new /= (T * n_e)

        scale = np.trace(C_e_new) / n_e
        if scale > 0:
            C_e_new /= scale
            C_t_new *= scale

        delta = (float(np.linalg.norm(C_e_new - C_e)) +
                 float(np.linalg.norm(C_t_new - C_t)))
        C_e = C_e_new
        C_t = C_t_new
        if delta < tol:
            converged = True
            break

    return (
        pd.DataFrame(C_e, index=exp_labels, columns=exp_labels),
        pd.DataFrame(C_t, index=ten_labels, columns=ten_labels),
        {"n_iter": n_iter, "converged": converged},
    )


def roughness_penalty_1d(n: int, order: int = 2) -> np.ndarray:
    """``n × n`` smoothness penalty ``P = DᵀD`` from an order-``k`` difference.

    ``D`` is the ``k``-th finite-difference operator on a uniform grid:
    ``order=1`` penalises slope (constants live in the null space),
    ``order=2`` (default) penalises curvature (constants and linear
    trends both live in the null space). Returns the zero matrix when
    ``order >= n`` (penalty is undefined for too-short vectors).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order >= n:
        return np.zeros((n, n))
    D = np.eye(n)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D.T @ D


def roughness_penalty_2d(n_e: int, n_t: int, order: int = 2) -> np.ndarray:
    """2D smoothness penalty as a Kronecker sum of the two 1D penalties.

    ``P_2D = kron(P_e, I_t) + kron(I_e, P_t)``. Applied to a flat
    expiry-major vector, ``vᵀ P_2D v`` is the sum of the row-wise and
    column-wise 1D penalties — i.e. the loading must be smooth both
    along expiry (for each fixed tenor) and along tenor (for each
    fixed expiry).
    """
    P_e = roughness_penalty_1d(n_e, order)
    P_t = roughness_penalty_1d(n_t, order)
    return np.kron(P_e, np.eye(n_t)) + np.kron(np.eye(n_e), P_t)


def functional_pca(
    wide: pd.DataFrame,
    lam: float,
    k: int = 4,
    penalty_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Roughness-penalised PCA on the flat ``(expiry · tenor)`` covariance.

    Solves ``eigh(Σ̂ - λ · P)`` and returns the top-``k``
    eigenvalues / eigenvectors in descending order. ``Σ̂`` is the
    (uncentred, divided by ``T``) sample cross-product of the flat
    daily moves; ``P`` is :func:`roughness_penalty_2d` of order
    ``penalty_order``. At ``λ = 0`` the result is identical (up to
    eigenvector sign) to vanilla PCA on ``Σ̂`` — the tests assert
    this.

    Returns numpy arrays (``eigvals (k,)``, ``eigvecs (n_e · n_t, k)``);
    the caller reshapes columns back to the ``(n_e, n_t)`` grid as
    needed.
    """
    panel, _, _ = _stack_panels(wide)
    T, n_e, n_t = panel.shape
    flat = panel.reshape(T, n_e * n_t)
    Sigma = (flat.T @ flat) / T
    M = Sigma - lam * roughness_penalty_2d(n_e, n_t, penalty_order) if lam != 0 else Sigma
    eigvals_asc, eigvecs_asc = np.linalg.eigh(M)
    eigvals = eigvals_asc[::-1][:k]
    eigvecs = eigvecs_asc[:, ::-1][:, :k]
    return eigvals.copy(), eigvecs.copy()


def _sign_orient(vec: np.ndarray) -> np.ndarray:
    """Sign convention: largest absolute entry is positive. Stable
    cross-run comparison without depending on the eigensolver's sign
    choice."""
    if vec.size == 0:
        return vec
    if abs(vec.max()) < abs(vec.min()):
        return -vec
    return vec


def marginal_eigen_patterns(
    C_e: pd.DataFrame,
    C_t: pd.DataFrame,
    n_modes_e: int = 3,
    n_modes_t: int = 3,
    total_degree: int | None = None,
) -> list[dict]:
    """Outer-product patterns from the top eigenvectors of each marginal.

    For each ``(i, j)`` in the tensor product of top-``n_modes_e``
    expiry eigenvectors and top-``n_modes_t`` tenor eigenvectors,
    builds ``v_e ⊗ v_τ`` (outer product) and packages it as one
    ``{"name", "grid", "version"}`` dict. ``total_degree``, if set,
    drops pairs with ``i + j > total_degree`` (triangular truncation
    via the shared :func:`pattern_basis.degree_pair_grid`).

    Naming is the same ``tensor_product_name`` used by
    :func:`pattern_basis.preset_separable_poly`, so for typical
    rates / vol data — where the first marginal eigenvector is roughly
    a level shape, the second a slope, etc. — names like
    ``exp_level_x_ten_slope`` line up between the two families and the
    comparison sheet can put them side by side.

    Sign convention: each eigenvector is oriented so its largest
    absolute entry is positive (so the level pattern is positive
    everywhere, not arbitrarily flipped). Returns ready-to-feed-into
    ``pattern_basis.patterns_to_prior_df`` →
    ``factors.sparse_pca_warm`` format.
    """
    if n_modes_e > C_e.shape[0] or n_modes_t > C_t.shape[0]:
        raise ValueError(
            f"requested {n_modes_e}/{n_modes_t} modes but C_e/C_t have "
            f"{C_e.shape[0]}/{C_t.shape[0]} rows"
        )
    _, Ve = np.linalg.eigh(C_e.values)
    _, Vt = np.linalg.eigh(C_t.values)
    Ve_top = Ve[:, ::-1][:, :n_modes_e]
    Vt_top = Vt[:, ::-1][:, :n_modes_t]

    out: list[dict] = []
    pairs: Iterable[tuple[int, int]] = degree_pair_grid(
        n_modes_e - 1, n_modes_t - 1,
        include_zero_zero=True, total_degree_cutoff=total_degree,
    )
    for i, j in pairs:
        e_vec = _sign_orient(Ve_top[:, i].copy())
        t_vec = _sign_orient(Vt_top[:, j].copy())
        grid_arr = np.outer(e_vec, t_vec)
        grid = pd.DataFrame(grid_arr, index=C_e.index, columns=C_t.index)
        grid.index.name = "expiry"
        grid.columns.name = "tenor"
        out.append({
            "name": tensor_product_name(i, j),
            "grid": grid, "version": 0,
        })
    return out
