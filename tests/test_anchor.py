"""Tests for ``factors.anchor`` — anchor-point regression factor model.

Covers OLS recovery on a known linear DGP, ridge shrinkage when the
anchor Gram matrix is singular, greedy selection ordering when one
cell dominates the variance, the strict-past leakage-free guarantee
of ``fit_ridge_rolling``, and the multicollinearity diagnostic when
two anchors are perfectly correlated.
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

from factors.anchor import (  # noqa: E402
    anchor_diagnostics,
    anomaly_report,
    beta_allocation_table,
    convex_hull_flags,
    fit_nnls,
    fit_nnls_rolling,
    fit_ridge,
    fit_ridge_rolling,
    fit_simplex,
    fit_simplex_rolling,
    get_anchor_slice,
    greedy_anchor_select,
    r2_heatmap,
    residual_panel,
)


def _make_columns(
    expiries: list[str], tenors: list[str]
) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [expiries, tenors], names=["expiry", "tenor"]
    )


def _make_diff(
    arr: np.ndarray, expiries: list[str], tenors: list[str]
) -> pd.DataFrame:
    cols = _make_columns(expiries, tenors)
    dates = pd.date_range("2020-01-01", periods=arr.shape[0], freq="B")
    return pd.DataFrame(arr, index=dates, columns=cols)


# ---------------------------------------------------------------------------
# 1. OLS recovery on a known linear DGP
# ---------------------------------------------------------------------------

def test_ols_recovery_two_anchors():
    """When every non-anchor cell is exactly ``a·anchor1 + b·anchor2 +
    tiny_noise``, ``fit_ridge(lam=0)`` should recover ``(a, b)`` to
    within ~5e-3 and ``r2_heatmap`` values should all exceed 0.95."""
    rng = np.random.default_rng(0)
    n_days = 400
    expiries = ["1Y", "5Y", "10Y", "30Y"]
    tenors = ["1Y", "5Y", "10Y", "30Y"]
    cols = _make_columns(expiries, tenors)
    n_cells = len(cols)

    f1 = rng.normal(size=n_days)
    f2 = rng.normal(size=n_days)

    anchors = [("1Y", "1Y"), ("10Y", "10Y")]
    arr = np.zeros((n_days, n_cells))
    truth: dict[tuple[str, str], tuple[float, float]] = {}
    for j, cell in enumerate(cols):
        if cell == anchors[0]:
            arr[:, j] = f1
            truth[cell] = (1.0, 0.0)
        elif cell == anchors[1]:
            arr[:, j] = f2
            truth[cell] = (0.0, 1.0)
        else:
            a = float(rng.uniform(-1.5, 1.5))
            b = float(rng.uniform(-1.5, 1.5))
            arr[:, j] = a * f1 + b * f2 + 0.005 * rng.normal(size=n_days)
            truth[cell] = (a, b)

    diff = _make_diff(arr, expiries, tenors)
    betas = fit_ridge(diff, anchors, lam=0.0)

    for cell, (a, b) in truth.items():
        beta_a = float(betas.loc[cell, "1Y|1Y"])
        beta_b = float(betas.loc[cell, "10Y|10Y"])
        assert abs(beta_a - a) < 5e-3, f"{cell}: beta_a={beta_a}, expected {a}"
        assert abs(beta_b - b) < 5e-3, f"{cell}: beta_b={beta_b}, expected {b}"

    r2 = r2_heatmap(diff, anchors, lam=0.0)
    assert (r2.values > 0.95).all(), f"min R²={r2.values.min():.4f}"


# ---------------------------------------------------------------------------
# 2. Ridge shrinkage on a singular Gram matrix
# ---------------------------------------------------------------------------

def test_ridge_required_when_anchors_are_collinear():
    """Two perfectly correlated anchors: ``lam=0`` must raise
    ``ValueError`` (singular Gram); ``lam>0`` must return finite betas."""
    rng = np.random.default_rng(1)
    n_days = 300
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)
    n_cells = len(cols)

    base = rng.normal(size=n_days)
    arr = rng.normal(size=(n_days, n_cells))
    arr[:, 0] = base
    arr[:, 4] = 2.0 * base  # exact linear combo of arr[:, 0]

    diff = _make_diff(arr, expiries, tenors)
    cell_list = list(cols)
    anchors = [cell_list[0], cell_list[4]]

    with pytest.raises(ValueError, match="singular"):
        fit_ridge(diff, anchors, lam=0.0)

    betas = fit_ridge(diff, anchors, lam=1.0)
    assert np.isfinite(betas.values).all()


# ---------------------------------------------------------------------------
# 3. Greedy selection ordering
# ---------------------------------------------------------------------------

def test_greedy_picks_dominant_cells_in_order():
    """One cell carries the loud factor F1 (~89% of cube variance), a
    second carries F2 (~11%); the rest is tiny iid noise. Greedy must
    pick them in that order and report descending marginal R²."""
    rng = np.random.default_rng(2)
    n_days = 500
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)
    n_cells = len(cols)

    f1 = rng.normal(scale=np.sqrt(8.0), size=n_days)
    f2 = rng.normal(scale=1.0, size=n_days)

    arr = rng.normal(scale=1e-2, size=(n_days, n_cells))
    cell_list = list(cols)
    arr[:, 0] = f1
    arr[:, 1] = f2
    f1_cell = cell_list[0]
    f2_cell = cell_list[1]

    diff = _make_diff(arr, expiries, tenors)
    history = greedy_anchor_select(diff, k=2, lam=0.0)

    assert len(history) == 2
    assert history[0]["anchor"] == f1_cell
    assert history[1]["anchor"] == f2_cell
    assert history[0]["marginal_r2"] > history[1]["marginal_r2"] > 0
    assert history[1]["cumulative_r2"] > history[0]["cumulative_r2"]


# ---------------------------------------------------------------------------
# 4. No look-ahead in fit_ridge_rolling
# ---------------------------------------------------------------------------

def test_no_look_ahead_in_rolling_betas():
    """Perturbing ``diff`` at date ``T`` (and beyond) must leave every
    rolling-fit beta at dates ``≤ T`` byte-identical — proves the
    window ``[t-window, t-1]`` truly excludes ``t``."""
    rng = np.random.default_rng(3)
    n_days = 300
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)
    n_cells = len(cols)

    arr = rng.normal(size=(n_days, n_cells))
    diff = _make_diff(arr, expiries, tenors)
    cell_list = list(cols)
    anchors = [cell_list[0], cell_list[4]]
    window = 60

    baseline = fit_ridge_rolling(diff, anchors, lam=0.0, window=window)

    t_idx = 200
    t_date = diff.index[t_idx]
    perturbed_arr = arr.copy()
    perturbed_arr[t_idx:] += 100.0 * rng.normal(
        size=(n_days - t_idx, n_cells)
    )
    diff_pert = _make_diff(perturbed_arr, expiries, tenors)
    perturbed = fit_ridge_rolling(
        diff_pert, anchors, lam=0.0, window=window
    )

    common = baseline.index.intersection(perturbed.index)
    before = common[common <= t_date]
    assert len(before) > 0, "test needs at least one beta date ≤ t"
    pd.testing.assert_frame_equal(
        baseline.loc[before], perturbed.loc[before]
    )


# ---------------------------------------------------------------------------
# 5. Multicollinearity diagnostic flags collinear anchors
# ---------------------------------------------------------------------------

def test_anchor_diagnostics_flag_collinear_pair():
    """Two perfectly correlated anchors should yield infinite (or
    near-infinite) VIF and a huge condition number — the signal to
    bump ``lam`` or drop one of the pair."""
    rng = np.random.default_rng(4)
    n_days = 200
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)
    n_cells = len(cols)

    base = rng.normal(size=n_days)
    arr = rng.normal(size=(n_days, n_cells))
    arr[:, 0] = base
    arr[:, 4] = 2.0 * base

    diff = _make_diff(arr, expiries, tenors)
    cell_list = list(cols)
    anchors = [cell_list[0], cell_list[4]]

    diag = anchor_diagnostics(diff, anchors)
    assert diag["condition_number"] > 1e6
    assert not np.isfinite(diag["vif"].values).all()


# ---------------------------------------------------------------------------
# Shared helpers for the constrained-regression tests below
# ---------------------------------------------------------------------------

def _cell_r2(diff, anchors, betas):
    """Per-cell centered R² (mirrors the private helper in factors.anchor —
    duplicated here so the tests don't reach into private API)."""
    am = get_anchor_slice(diff, anchors)
    resid = residual_panel(diff, am, betas)
    aligned = diff.loc[resid.index, resid.columns]
    ss_resid = (resid ** 2).sum(axis=0)
    ss_total = ((aligned - aligned.mean()) ** 2).sum(axis=0)
    safe_total = ss_total.where(ss_total > 0, 1.0)
    return 1.0 - ss_resid / safe_total


# ---------------------------------------------------------------------------
# 6. NNLS non-negativity and parity with regularised OLS
# ---------------------------------------------------------------------------

def test_nnls_non_negative_and_beats_ridge_when_betas_positive():
    """fit_nnls must return all-non-negative betas. When the true
    coefficients are strictly positive, the inherent NNLS solution
    should match or beat ridge(lam=1.0) (which shrinks toward zero)
    on mean per-cell R²."""
    rng = np.random.default_rng(10)
    n_days = 400
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)
    n_cells = len(cols)

    f1 = rng.normal(size=n_days)
    f2 = rng.normal(size=n_days)
    anchors = [("1Y", "1Y"), ("10Y", "10Y")]

    arr = np.zeros((n_days, n_cells))
    cell_to_pos = {c: i for i, c in enumerate(cols)}
    arr[:, cell_to_pos[anchors[0]]] = f1
    arr[:, cell_to_pos[anchors[1]]] = f2
    for cell in cols:
        if cell in anchors:
            continue
        a = float(rng.uniform(0.1, 1.5))  # POSITIVE
        b = float(rng.uniform(0.1, 1.5))  # POSITIVE
        arr[:, cell_to_pos[cell]] = a * f1 + b * f2 + 0.01 * rng.normal(size=n_days)

    diff = _make_diff(arr, expiries, tenors)
    nnls_betas = fit_nnls(diff, anchors)
    ridge_betas = fit_ridge(diff, anchors, lam=1.0)

    assert (nnls_betas.values >= -1e-10).all(), (
        f"NNLS produced negative beta: min={nnls_betas.values.min()}"
    )

    r2_nnls = float(_cell_r2(diff, anchors, nnls_betas).mean())
    r2_ridge = float(_cell_r2(diff, anchors, ridge_betas).mean())
    assert r2_nnls >= r2_ridge - 1e-6, (
        f"NNLS R²={r2_nnls:.4f} should not be worse than "
        f"ridge(lam=1) R²={r2_ridge:.4f} when true betas are positive"
    )


# ---------------------------------------------------------------------------
# 7. Simplex sum constraint
# ---------------------------------------------------------------------------

def test_simplex_betas_sum_to_one_and_nonneg():
    """Every non-anchor row of fit_simplex must lie in the simplex:
    non-negative and summing to 1 within numerical tolerance."""
    rng = np.random.default_rng(11)
    n_days = 250
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)

    arr = rng.normal(size=(n_days, len(cols)))
    diff = _make_diff(arr, expiries, tenors)
    cell_list = list(cols)
    anchors = [cell_list[0], cell_list[4]]

    betas = fit_simplex(diff, anchors)
    assert (betas.values >= -1e-10).all(), (
        f"simplex produced negative beta: min={betas.values.min()}"
    )
    row_sums = betas.sum(axis=1).values
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. Convex-hull diagnostic — interior case
# ---------------------------------------------------------------------------

def test_simplex_recovers_known_convex_weights_interior():
    """When every cell is exactly a convex combination of two anchor
    moves with known positive weights summing to 1, fit_simplex should
    recover the weights to within 1e-4 and convex_hull_flags should
    return all-False."""
    rng = np.random.default_rng(12)
    n_days = 400
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)
    n_cells = len(cols)

    a1 = rng.normal(size=n_days)
    a2 = rng.normal(size=n_days)
    anchors = [("1Y", "1Y"), ("10Y", "10Y")]

    weights = np.linspace(0.05, 0.95, n_cells)  # all strictly positive
    arr = np.zeros((n_days, n_cells))
    truth: dict[tuple[str, str], tuple[float, float]] = {}
    for j, cell in enumerate(cols):
        if cell == anchors[0]:
            arr[:, j] = a1
            truth[cell] = (1.0, 0.0)
        elif cell == anchors[1]:
            arr[:, j] = a2
            truth[cell] = (0.0, 1.0)
        else:
            w = float(weights[j])
            arr[:, j] = w * a1 + (1.0 - w) * a2
            truth[cell] = (w, 1.0 - w)

    diff = _make_diff(arr, expiries, tenors)
    betas = fit_simplex(diff, anchors)

    for cell, (w1, w2) in truth.items():
        b1 = float(betas.loc[cell, "1Y|1Y"])
        b2 = float(betas.loc[cell, "10Y|10Y"])
        assert abs(b1 - w1) < 1e-4, (
            f"{cell}: simplex b1={b1}, expected {w1}"
        )
        assert abs(b2 - w2) < 1e-4, (
            f"{cell}: simplex b2={b2}, expected {w2}"
        )

    flags = convex_hull_flags(diff, anchors)
    assert not flags.values.any(), (
        "no cell should be flagged when every cell is a convex combo"
    )


# ---------------------------------------------------------------------------
# 9. Convex-hull diagnostic — breach case
# ---------------------------------------------------------------------------

def test_convex_hull_flags_detect_amplitude_breach():
    """A cell with amplitude 2× the anchor moves (i.e. outside the
    anchor convex hull in magnitude) should be flagged True; its
    simplex R² should fall more than 5 percentage points below its
    NNLS R²."""
    rng = np.random.default_rng(13)
    n_days = 400
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)

    a1 = rng.normal(size=n_days)
    a2 = rng.normal(size=n_days)
    anchors = [("1Y", "1Y"), ("10Y", "10Y")]
    breach_cell = ("5Y", "5Y")

    arr = rng.normal(size=(n_days, len(cols))) * 0.01
    cell_to_pos = {c: i for i, c in enumerate(cols)}
    arr[:, cell_to_pos[anchors[0]]] = a1
    arr[:, cell_to_pos[anchors[1]]] = a2
    # 2× amplitude in the direction of anchor 1 — clearly outside the
    # convex hull (which has max amplitude 1× any anchor).
    arr[:, cell_to_pos[breach_cell]] = 2.0 * a1

    diff = _make_diff(arr, expiries, tenors)
    flags = convex_hull_flags(diff, anchors)
    assert bool(flags.loc[breach_cell[0], breach_cell[1]]), (
        "breach cell should be flagged"
    )

    nnls_betas = fit_nnls(diff, anchors)
    simplex_betas = fit_simplex(diff, anchors)
    r2_nnls = float(_cell_r2(diff, anchors, nnls_betas).loc[breach_cell])
    r2_simplex = float(_cell_r2(diff, anchors, simplex_betas).loc[breach_cell])
    assert r2_simplex < r2_nnls - 0.05, (
        f"breach cell: r2_simplex={r2_simplex:.3f}, r2_nnls={r2_nnls:.3f}"
    )


# ---------------------------------------------------------------------------
# 10. No look-ahead in NNLS rolling fits
# ---------------------------------------------------------------------------

def test_no_look_ahead_nnls_rolling():
    """Same strict-past guarantee as fit_ridge_rolling, applied to
    fit_nnls_rolling: perturbations at date T or later must leave
    betas at dates ≤ T byte-identical."""
    rng = np.random.default_rng(14)
    n_days = 200
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y"]
    cols = _make_columns(expiries, tenors)

    arr = rng.normal(size=(n_days, len(cols)))
    diff = _make_diff(arr, expiries, tenors)
    cell_list = list(cols)
    anchors = [cell_list[0], cell_list[3]]
    window = 60

    baseline = fit_nnls_rolling(diff, anchors, window=window)

    t_idx = 150
    t_date = diff.index[t_idx]
    perturbed_arr = arr.copy()
    perturbed_arr[t_idx:] += 100.0 * rng.normal(
        size=(n_days - t_idx, len(cols))
    )
    diff_pert = _make_diff(perturbed_arr, expiries, tenors)
    perturbed = fit_nnls_rolling(diff_pert, anchors, window=window)

    common = baseline.index.intersection(perturbed.index)
    before = common[common <= t_date]
    assert len(before) > 0
    pd.testing.assert_frame_equal(
        baseline.loc[before], perturbed.loc[before]
    )


# ---------------------------------------------------------------------------
# 11. No look-ahead in simplex rolling fits (smaller fixture for speed)
# ---------------------------------------------------------------------------

def test_no_look_ahead_simplex_rolling():
    """Same as the NNLS rolling test but for fit_simplex_rolling.
    Uses a smaller fixture since SLSQP per cell per window is slow."""
    rng = np.random.default_rng(15)
    n_days = 150
    expiries = ["1Y", "5Y"]
    tenors = ["1Y", "5Y"]
    cols = _make_columns(expiries, tenors)

    arr = rng.normal(size=(n_days, len(cols)))
    diff = _make_diff(arr, expiries, tenors)
    cell_list = list(cols)
    anchors = [cell_list[0], cell_list[3]]
    window = 40

    baseline = fit_simplex_rolling(diff, anchors, window=window)

    t_idx = 100
    t_date = diff.index[t_idx]
    perturbed_arr = arr.copy()
    perturbed_arr[t_idx:] += 100.0 * rng.normal(
        size=(n_days - t_idx, len(cols))
    )
    diff_pert = _make_diff(perturbed_arr, expiries, tenors)
    perturbed = fit_simplex_rolling(diff_pert, anchors, window=window)

    common = baseline.index.intersection(perturbed.index)
    before = common[common <= t_date]
    assert len(before) > 0
    pd.testing.assert_frame_equal(
        baseline.loc[before], perturbed.loc[before]
    )


# ---------------------------------------------------------------------------
# 12. beta_allocation_table — simplex rows still sum to 1 after reordering
# ---------------------------------------------------------------------------

def test_beta_allocation_table_rows_sum_to_one():
    """beta_allocation_table is a pure reindex; for simplex betas every
    row (including the one-hot anchor self-rows) must still sum to 1."""
    rng = np.random.default_rng(16)
    n_days = 200
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)

    arr = rng.normal(size=(n_days, len(cols)))
    diff = _make_diff(arr, expiries, tenors)
    cell_list = list(cols)
    anchors = [cell_list[0], cell_list[4]]

    betas = fit_simplex(diff, anchors)
    table = beta_allocation_table(betas, anchors)
    np.testing.assert_allclose(table.sum(axis=1).values, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 13. Anomaly report — injected decoupling event must fire all three flags
# ---------------------------------------------------------------------------

def test_anomaly_report_flags_injected_decoupling():
    """Build a panel where every cell tracks a single latent factor F
    (so the regression is near-perfect), then inject a one-off spike
    into the anchor cell *only* at date T. The anchor's z must blow
    past the threshold, the residual model must miss it (residual z
    blows past too), and the combined `decoupling` flag must fire on
    that date — and ideally not on the surrounding dates."""
    rng = np.random.default_rng(20)
    n_days = 300
    expiries = ["1Y", "5Y", "10Y"]
    tenors = ["1Y", "5Y", "10Y"]
    cols = _make_columns(expiries, tenors)
    n_cells = len(cols)

    f = rng.normal(size=n_days)                  # latent factor
    loadings = rng.uniform(0.8, 1.2, size=n_cells)
    arr = f[:, None] * loadings[None, :]
    arr += 0.05 * rng.normal(size=arr.shape)     # small idiosyncratic noise

    anchors = [("1Y", "1Y")]
    cell_to_pos = {c: i for i, c in enumerate(cols)}
    anchor_pos = cell_to_pos[anchors[0]]

    # Inject a ~20-sigma anchor spike at T; leave every other cell alone
    # — so the regression model (which has learned that all cells move
    # together with the anchor) predicts a big move at every cell, but
    # actually only the anchor moves → huge residual.
    t_idx = 200
    arr[t_idx, anchor_pos] += 20.0

    diff = _make_diff(arr, expiries, tenors)
    report = anomaly_report(diff, anchors, z_threshold=3.0, window=63)

    t_date = diff.index[t_idx]
    row = report.loc[t_date]
    assert bool(row["anchor_anomaly"]), (
        f"anchor_anomaly should fire at T={t_date} "
        f"(max_anchor_z={row['max_anchor_z']:.2f})"
    )
    assert bool(row["residual_anomaly"]), (
        f"residual_anomaly should fire at T={t_date} "
        f"(residual_z={row['residual_z']:.2f})"
    )
    assert bool(row["decoupling"]), (
        f"decoupling should fire at T={t_date}"
    )

    # Sanity check: decoupling shouldn't fire on most other dates
    # (false-positive rate should be small on this synthetic setup).
    n_decoupling = int(report["decoupling"].sum())
    assert n_decoupling <= 5, (
        f"too many decoupling flags ({n_decoupling}) — the test setup "
        f"was supposed to produce exactly one"
    )
