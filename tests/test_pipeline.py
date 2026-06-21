"""Tests for ``data/pipeline.strip_parallel_shift``.

Covers backwards compatibility of the original ``beta_mode="sigma"`` /
``agg="mean"`` defaults, the leakage-free guarantee under both
``beta_mode`` options, the degenerate-common-factor sanity check, and
the qualitative claim that the regression mode lowers an idiosyncratic
cell's residual-vs-shift correlation versus the sigma mode.
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

from data.pipeline import strip_parallel_shift  # noqa: E402


def _mock_diff(seed: int = 0, n_days: int = 200, n_cells: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_days, n_cells)),
        index=pd.date_range("2020-01-01", periods=n_days, freq="B"),
        columns=[f"cell_{i}" for i in range(n_cells)],
    )


def _original_strip(diff: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Verbatim copy of the pre-refactor implementation, used as the
    backwards-compat oracle so the regression test doesn't depend on a
    binary fixture on disk."""
    sigma = diff.rolling(window=window).std().shift(1)
    shift = (diff / sigma).mean(axis=1)
    residual = diff.sub(sigma.mul(shift, axis=0))
    return residual.dropna(how="all")


def test_default_args_match_original_behavior():
    diff = _mock_diff()
    expected = _original_strip(diff)
    actual = strip_parallel_shift(diff)
    pd.testing.assert_frame_equal(actual, expected)


def test_degenerate_common_factor_residual_is_zero():
    """If ``diff_{i,t} = c_t · s_i`` with strictly positive ``s_i``, the
    sigma-mode residual collapses to zero analytically — every cell sees
    the same z-score ``c_t / σ_c(t)`` and the rescaled shift cancels the
    raw move exactly."""
    rng = np.random.default_rng(42)
    n_days, n_cells = 200, 8
    c = rng.normal(size=n_days)
    s = rng.uniform(0.5, 2.0, size=n_cells)
    diff = pd.DataFrame(
        np.outer(c, s),
        index=pd.date_range("2020-01-01", periods=n_days, freq="B"),
        columns=[f"cell_{i}" for i in range(n_cells)],
    )
    residual = strip_parallel_shift(diff)
    np.testing.assert_allclose(residual.values, 0.0, atol=1e-9)


@pytest.mark.parametrize("beta_mode", ["sigma", "regression"])
def test_leakage_free(beta_mode):
    """Perturbing only the last day's diff must not change any earlier
    residual value — proves the rolling stats truly use strictly past
    data (i.e. the ``.shift(1)`` guard is doing its job)."""
    diff = _mock_diff(seed=1)
    baseline = strip_parallel_shift(diff, beta_mode=beta_mode)

    perturbed = diff.copy()
    T = perturbed.index[-1]
    perturbed.loc[T] = perturbed.loc[T] + 100.0
    perturbed_res = strip_parallel_shift(perturbed, beta_mode=beta_mode)

    common = baseline.index.intersection(perturbed_res.index)
    earlier = common[common < T]
    assert len(earlier) > 0, "test setup needs more than one residual row"
    pd.testing.assert_frame_equal(
        baseline.loc[earlier], perturbed_res.loc[earlier]
    )


def test_regression_beta_lowers_idiosyncratic_residual_corr():
    """A pure-noise cell (independent of every other cell) should end up
    nearly uncorrelated with the shift series under ``beta_mode="regression"``.
    The sigma mode bakes a ``-σ·shift`` term into its residual, so even
    for an idiosyncratic cell the residual carries a structural negative
    correlation with ``shift``."""
    rng = np.random.default_rng(7)
    n_days, n_common = 400, 8
    c = rng.normal(size=n_days)
    s = rng.uniform(0.8, 1.5, size=n_common)
    common_block = np.outer(c, s)
    noise = rng.normal(size=(n_days, 1)) * 2.0
    diff = pd.DataFrame(
        np.hstack([common_block, noise]),
        index=pd.date_range("2020-01-01", periods=n_days, freq="B"),
        columns=[f"cell_{i}" for i in range(n_common)] + ["noise_cell"],
    )

    res_sigma = strip_parallel_shift(diff, beta_mode="sigma")
    res_reg = strip_parallel_shift(diff, beta_mode="regression")

    sigma = diff.rolling(window=60).std().shift(1)
    shift = (diff / sigma).mean(axis=1)

    idx = res_reg.index  # regression mode has the longer burn-in
    corr_sigma = res_sigma.loc[idx, "noise_cell"].corr(shift.loc[idx])
    corr_reg = res_reg["noise_cell"].corr(shift.loc[idx])
    assert abs(corr_reg) < abs(corr_sigma), (
        f"regression should lower noise-cell residual-vs-shift corr: "
        f"sigma={corr_sigma:.3f}, regression={corr_reg:.3f}"
    )
