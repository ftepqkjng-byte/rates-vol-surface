"""Tests for ``factors/separable.py``.

Three groups:

1. **Kronecker recovery** — generate matrix-normal data with known
   ``C_e_true`` / ``C_t_true``, verify both the naive moment estimator
   and the flip-flop MLE recover the true marginals (and that the
   separability residual is small).
2. **Non-separable contrast** — inject a non-Kronecker rank-1 mode
   into otherwise Kronecker data and verify the separability residual
   grows materially; without this the diagnostic would just be a
   constant.
3. **Penalty / functional PCA sanity** — ``functional_pca(lam=0)``
   matches a direct ``eigh`` of the sample covariance, and the
   roughness penalty actually has the constant + linear null space
   it's supposed to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from factors.separable import (  # noqa: E402
    marginal_kronecker_cov,
    kronecker_separability_residual,
    kronecker_cov_mle,
    roughness_penalty_1d,
    roughness_penalty_2d,
    functional_pca,
    marginal_eigen_patterns,
)


def _random_cov(n: int, trace_target: float, seed: int) -> np.ndarray:
    """Symmetric positive-definite matrix with controlled trace.

    Trace is fixed up-front so that the matrix-normal trace conventions
    used by the estimators (``tr(C_e) = n_e``, ``tr(C_t) = n_t``) hold
    on the synthetic data and the bias term in the moment estimator
    cancels exactly.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    C = A @ A.T + 0.1 * np.eye(n)
    return C * (trace_target / np.trace(C))


def _matrix_normal_panel(
    T: int, C_e: np.ndarray, C_t: np.ndarray, seed: int,
) -> pd.DataFrame:
    """Sample ``T`` iid matrix-normal panels and pack into a wide
    DataFrame matching the layout of ``pca.to_wide`` output."""
    rng = np.random.default_rng(seed)
    n_e, n_t = C_e.shape[0], C_t.shape[0]
    L_e = np.linalg.cholesky(C_e)
    L_t = np.linalg.cholesky(C_t)
    Z = rng.normal(size=(T, n_e, n_t))
    # X[t] = L_e @ Z[t] @ L_t.T  →  vec_row(X[t]) ~ N(0, C_e ⊗ C_t)
    X = np.einsum("ij,tjk,lk->til", L_e, Z, L_t)
    exp_labels = [f"e{i}" for i in range(n_e)]
    ten_labels = [f"t{j}" for j in range(n_t)]
    cols = pd.MultiIndex.from_product(
        [exp_labels, ten_labels], names=["expiry", "tenor"]
    )
    flat = X.reshape(T, n_e * n_t)
    return pd.DataFrame(
        flat, columns=cols,
        index=pd.date_range("2020-01-01", periods=T, freq="B"),
    )


# --- 1. Kronecker recovery ---------------------------------------------------

@pytest.fixture(scope="module")
def kron_panel():
    n_e, n_t, T = 5, 4, 4000
    C_e_true = _random_cov(n_e, trace_target=n_e, seed=1)
    C_t_true = _random_cov(n_t, trace_target=n_t, seed=2)
    wide = _matrix_normal_panel(T, C_e_true, C_t_true, seed=3)
    return wide, C_e_true, C_t_true


def test_naive_marginal_recovers_kron_structure(kron_panel):
    wide, C_e_true, C_t_true = kron_panel
    C_e_est, C_t_est = marginal_kronecker_cov(wide)
    np.testing.assert_allclose(C_e_est.values, C_e_true, atol=0.15)
    np.testing.assert_allclose(C_t_est.values, C_t_true, atol=0.15)


def test_mle_recovers_kron_structure(kron_panel):
    wide, C_e_true, C_t_true = kron_panel
    C_e_est, C_t_est, info = kronecker_cov_mle(wide, max_iter=30)
    assert info["converged"], (
        f"MLE did not converge in {info['n_iter']} iterations"
    )
    np.testing.assert_allclose(C_e_est.values, C_e_true, atol=0.15)
    np.testing.assert_allclose(C_t_est.values, C_t_true, atol=0.15)


def test_separability_residual_small_on_kron_data(kron_panel):
    wide, _, _ = kron_panel
    C_e, C_t, _ = kronecker_cov_mle(wide)
    resid = kronecker_separability_residual(wide, C_e, C_t)
    assert resid < 0.10, (
        f"separability residual {resid:.3f} should be small on truly "
        "matrix-normal data"
    )


# --- 2. Non-separable contrast -----------------------------------------------

def test_separability_residual_distinguishes_nonseparable():
    """Inject a rank-1 non-Kronecker direction on top of clean
    matrix-normal data and verify the residual jumps materially — i.e.
    the diagnostic actually has signal, isn't a constant."""
    n_e, n_t, T = 5, 4, 4000
    C_e_true = _random_cov(n_e, trace_target=n_e, seed=1)
    C_t_true = _random_cov(n_t, trace_target=n_t, seed=2)
    wide_good = _matrix_normal_panel(T, C_e_true, C_t_true, seed=3)

    rng = np.random.default_rng(99)
    # Random unit direction in the flat (n_e · n_t) space; a generic
    # Gaussian vector reshapes to a full-rank matrix, so the resulting
    # rank-1 covariance contribution u u^T is *not* a single Kronecker
    # factor.
    contam_dir = rng.normal(size=n_e * n_t)
    contam_dir /= np.linalg.norm(contam_dir)
    contam_scale = rng.normal(size=(wide_good.shape[0], 1)) * 3.0
    wide_bad = wide_good + contam_scale * contam_dir

    C_e_g, C_t_g, _ = kronecker_cov_mle(wide_good)
    C_e_b, C_t_b, _ = kronecker_cov_mle(wide_bad)
    resid_good = kronecker_separability_residual(wide_good, C_e_g, C_t_g)
    resid_bad = kronecker_separability_residual(wide_bad, C_e_b, C_t_b)
    assert resid_bad > resid_good + 0.1, (
        f"residual should jump on non-separable data: "
        f"good={resid_good:.3f}, bad={resid_bad:.3f}"
    )


# --- 3. Penalty / functional PCA sanity --------------------------------------

def test_functional_pca_lam_zero_matches_standard_pca():
    n_e, n_t, T = 4, 3, 600
    C_e_true = _random_cov(n_e, trace_target=n_e, seed=1)
    C_t_true = _random_cov(n_t, trace_target=n_t, seed=2)
    wide = _matrix_normal_panel(T, C_e_true, C_t_true, seed=3)

    k = 5
    fpca_vals, fpca_vecs = functional_pca(wide, lam=0.0, k=k)

    flat = wide.dropna(how="any").values
    Sigma = (flat.T @ flat) / flat.shape[0]
    eigvals_asc, eigvecs_asc = np.linalg.eigh(Sigma)
    direct_vals = eigvals_asc[::-1][:k]
    direct_vecs = eigvecs_asc[:, ::-1][:, :k]

    np.testing.assert_allclose(fpca_vals, direct_vals, atol=1e-9)
    # Subspace equality up to per-column sign: |<f_i, d_i>| ≈ 1, ≈ 0 off-diag
    overlap = np.abs(fpca_vecs.T @ direct_vecs)
    np.testing.assert_allclose(overlap, np.eye(k), atol=1e-8)


def test_roughness_penalty_zero_space_and_oscillation():
    n = 10
    P = roughness_penalty_1d(n, order=2)
    constant = np.ones(n)
    linear = np.arange(n, dtype=float)
    oscillating = np.array(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    )
    assert constant @ P @ constant < 1e-12, (
        "order-2 penalty should annihilate constants"
    )
    assert linear @ P @ linear < 1e-12, (
        "order-2 penalty should annihilate linear trends"
    )
    assert oscillating @ P @ oscillating > 10.0, (
        "alternating ±1 should incur a clearly non-trivial curvature penalty"
    )


def test_roughness_penalty_2d_shape_and_null_space():
    n_e, n_t = 5, 4
    P = roughness_penalty_2d(n_e, n_t, order=2)
    assert P.shape == (n_e * n_t, n_e * n_t)
    # The bivariate constant lives in both 1D null spaces, so the sum
    # also annihilates it.
    constant = np.ones(n_e * n_t)
    assert constant @ P @ constant < 1e-10


def test_marginal_eigen_patterns_shape_and_naming(kron_panel):
    wide, _, _ = kron_panel
    C_e, C_t = marginal_kronecker_cov(wide)
    patterns = marginal_eigen_patterns(C_e, C_t, n_modes_e=2, n_modes_t=2,
                                       total_degree=2)
    # (0,0), (0,1), (1,0) — (1,1) cut by total_degree=2 is also kept
    # (1+1 = 2 ≤ 2). Triangular check: (1,1) IS kept, only nothing > 2.
    names = [p["name"] for p in patterns]
    assert set(names) == {
        "exp_level_x_ten_level",
        "exp_level_x_ten_slope",
        "exp_slope_x_ten_level",
        "exp_slope_x_ten_slope",
    }
    for p in patterns:
        assert p["grid"].shape == (C_e.shape[0], C_t.shape[0])
        assert p["grid"].index.tolist() == C_e.index.tolist()
        assert p["grid"].columns.tolist() == C_t.index.tolist()
