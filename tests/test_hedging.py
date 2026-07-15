"""Tests for ``factors.hedging`` — book-vega hedging against trained
patterns.

Covers exact recovery when a single candidate's exposure direction
matches the book exactly, residual-within-epsilon on a random
well-posed system, cost-based tie-breaking between two candidates with
identical exposure but different cost, the liquid-universe filter, the
infeasible-LP raise path, and the ``method="min_residual"``
(position-cap) dual formulation.
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

from config import EXPIRY_LABELS, TENOR_LABELS  # noqa: E402
from factors.hedging import (  # noqa: E402
    book_pattern_exposure,
    liquid_hedge_candidates,
    pattern_epsilon,
    sparse_hedge,
)


def _make_cand_index(labels: list[tuple[str, str]]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(labels, names=["expiry", "tenor"])


# ---------------------------------------------------------------------------
# 1. Exact recovery when one candidate matches the book direction
# ---------------------------------------------------------------------------

def test_exact_recovery_single_candidate():
    """Candidate A's beta row is exactly proportional to book_exposure; B
    and C are decoys that could only replicate it at far higher notional.
    With epsilon ~ 0 the LP should hedge entirely through A."""
    cand = _make_cand_index([("2Y", "2Y"), ("5Y", "5Y"), ("10Y", "10Y")])
    betas = pd.DataFrame(
        {
            "F1": [1.0, 0.5, 0.2],
            "F2": [-0.4, 0.5, -0.9],
        },
        index=cand,
    )
    book_exposure = pd.Series({"F1": 10.0, "F2": -4.0})
    epsilon = pd.Series({"F1": 1e-6, "F2": 1e-6})

    result = sparse_hedge(book_exposure, betas, epsilon, candidates=cand)

    assert result["n_active"] == 1
    assert result["alpha"].loc[("2Y", "2Y")] == pytest.approx(10.0, abs=1e-4)
    assert result["alpha"].loc[("5Y", "5Y")] == pytest.approx(0.0, abs=1e-4)
    assert result["alpha"].loc[("10Y", "10Y")] == pytest.approx(0.0, abs=1e-4)
    assert result["total_notional"] == pytest.approx(10.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Residual stays within epsilon on a random well-posed system
# ---------------------------------------------------------------------------

def test_residual_within_epsilon_random():
    rng = np.random.default_rng(0)
    n_cand = 20
    patterns = ["F1", "F2", "F3"]
    cand = _make_cand_index([(f"E{i}", f"T{i}") for i in range(n_cand)])
    betas = pd.DataFrame(
        rng.normal(size=(n_cand, len(patterns))), index=cand, columns=patterns
    )
    book_exposure = pd.Series(rng.normal(size=len(patterns)) * 5, index=patterns)
    epsilon = pd.Series(np.abs(book_exposure.values) * 0.2 + 0.1, index=patterns)

    result = sparse_hedge(book_exposure, betas, epsilon, candidates=cand)

    resid = result["residual_exposure"]
    assert (resid.abs() <= epsilon + 1e-6).all()


# ---------------------------------------------------------------------------
# 3. Cost-based tie-breaking
# ---------------------------------------------------------------------------

def test_prefers_cheaper_candidate_when_interchangeable():
    """Two candidates with identical exposure to a single pattern but
    different cost: the LP should route the whole hedge through the
    cheaper one."""
    cand = _make_cand_index([("2Y", "2Y"), ("5Y", "5Y")])
    betas = pd.DataFrame({"F1": [2.0, 2.0]}, index=cand)
    book_exposure = pd.Series({"F1": 6.0})
    epsilon = pd.Series({"F1": 1e-6})
    cost = pd.Series({("2Y", "2Y"): 1.0, ("5Y", "5Y"): 5.0})

    result = sparse_hedge(book_exposure, betas, epsilon, cost=cost, candidates=cand)

    assert result["alpha"].loc[("2Y", "2Y")] == pytest.approx(3.0, abs=1e-4)
    assert result["alpha"].loc[("5Y", "5Y")] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Liquid universe filter
# ---------------------------------------------------------------------------

def test_liquid_hedge_candidates_excludes_short_end_and_7y_12y():
    full = pd.MultiIndex.from_product(
        [EXPIRY_LABELS, TENOR_LABELS], names=["expiry", "tenor"]
    )
    liquid = liquid_hedge_candidates(full)

    excluded_expiry = {"1M", "2M", "3M", "6M", "9M", "1Y", "7Y", "12Y"}
    excluded_tenor = {"7Y", "12Y"}
    assert not any(e in excluded_expiry for e in liquid.get_level_values("expiry"))
    assert not any(t in excluded_tenor for t in liquid.get_level_values("tenor"))
    assert len(liquid) == 9 * 10


# ---------------------------------------------------------------------------
# 5. Infeasible LP raises
# ---------------------------------------------------------------------------

def test_infeasible_raises():
    cand = _make_cand_index([("2Y", "2Y"), ("5Y", "5Y")])
    betas = pd.DataFrame({"F1": [0.0, 0.0]}, index=cand)
    book_exposure = pd.Series({"F1": 5.0})
    epsilon = pd.Series({"F1": 0.0})

    with pytest.raises(ValueError):
        sparse_hedge(book_exposure, betas, epsilon, candidates=cand)


# ---------------------------------------------------------------------------
# 6. method="min_residual" — position-cap dual formulation
# ---------------------------------------------------------------------------

def test_min_residual_hits_position_cap_when_binding():
    """A single candidate whose beta is proportional to the book: fully
    hedging would need alpha=10, but position_cap=3 forces the LP to
    stop at the cap, leaving a nonzero residual."""
    cand = _make_cand_index([("2Y", "2Y")])
    betas = pd.DataFrame({"F1": [1.0], "F2": [-0.4]}, index=cand)
    book_exposure = pd.Series({"F1": 10.0, "F2": -4.0})

    result = sparse_hedge(
        book_exposure, betas, position_cap=3.0, candidates=cand,
        method="min_residual",
    )

    assert result["alpha"].loc[("2Y", "2Y")] == pytest.approx(3.0, abs=1e-4)
    assert result["total_residual"] == pytest.approx(7.0 + 2.8, abs=1e-4)
    assert (result["alpha"].abs() <= 3.0 + 1e-6).all()


def test_min_residual_matches_min_notional_recovery_when_cap_generous():
    """Same exact-recovery setup as test_exact_recovery_single_candidate:
    with a position_cap well above what's needed, min_residual should
    reach ~zero residual too. NOTE: unlike min_notional, min_residual has
    no preference among alpha combinations that all hit zero residual, so
    the specific alpha values are *not* asserted here — see
    test_min_residual_can_be_non_sparse_at_zero_residual below, which
    documents that this degeneracy is real and can produce large
    offsetting positions even when the residual objective is fully met."""
    cand = _make_cand_index([("2Y", "2Y"), ("5Y", "5Y"), ("10Y", "10Y")])
    betas = pd.DataFrame(
        {"F1": [1.0, 0.5, 0.2], "F2": [-0.4, 0.5, -0.9]}, index=cand
    )
    book_exposure = pd.Series({"F1": 10.0, "F2": -4.0})

    result = sparse_hedge(
        book_exposure, betas, position_cap=100.0, candidates=cand,
        method="min_residual",
    )

    assert result["total_residual"] == pytest.approx(0.0, abs=1e-4)
    assert result["hedge_exposure"]["F1"] == pytest.approx(10.0, abs=1e-4)
    assert result["hedge_exposure"]["F2"] == pytest.approx(-4.0, abs=1e-4)


def test_min_residual_can_be_non_sparse_at_zero_residual():
    """Documents a real gap: with >=3 candidates spanning only 2 patterns,
    once total_residual hits its minimum the LP is indifferent among all
    alpha combinations achieving it — nothing in the min_residual
    objective prefers the small/sparse one, so it can return large
    offsetting positions (total_notional far above the cheapest
    alternative) even though the fit is perfect. Left as a known
    limitation rather than silently "fixed" — see conversation notes on
    the lexicographic two-stage follow-up."""
    cand = _make_cand_index([("2Y", "2Y"), ("5Y", "5Y"), ("10Y", "10Y")])
    betas = pd.DataFrame(
        {"F1": [1.0, 0.5, 0.2], "F2": [-0.4, 0.5, -0.9]}, index=cand
    )
    book_exposure = pd.Series({"F1": 10.0, "F2": -4.0})

    result = sparse_hedge(
        book_exposure, betas, position_cap=100.0, candidates=cand,
        method="min_residual",
    )

    assert result["total_residual"] == pytest.approx(0.0, abs=1e-4)
    # The sparse (min_notional-equivalent) solution costs 10.0 notional;
    # min_residual alone is free to land far above that.
    assert result["total_notional"] > 50.0


def test_sparse_hedge_requires_method_specific_param():
    cand = _make_cand_index([("2Y", "2Y")])
    betas = pd.DataFrame({"F1": [1.0]}, index=cand)
    book_exposure = pd.Series({"F1": 5.0})

    with pytest.raises(ValueError):
        sparse_hedge(book_exposure, betas, candidates=cand, method="min_notional")
    with pytest.raises(ValueError):
        sparse_hedge(book_exposure, betas, candidates=cand, method="min_residual")


# ---------------------------------------------------------------------------
# book_pattern_exposure
# ---------------------------------------------------------------------------

def test_book_pattern_exposure_matches_matrix_product():
    cand = _make_cand_index([("2Y", "2Y"), ("5Y", "5Y"), ("10Y", "10Y")])
    betas = pd.DataFrame(
        {"F1": [1.0, 2.0, 3.0], "F2": [0.5, -1.0, 0.0]}, index=cand
    )
    vega = pd.Series([10.0, 20.0, 30.0], index=cand)

    exposure = book_pattern_exposure(vega, betas)

    expected_f1 = 10.0 * 1.0 + 20.0 * 2.0 + 30.0 * 3.0
    expected_f2 = 10.0 * 0.5 + 20.0 * -1.0 + 30.0 * 0.0
    assert exposure["F1"] == pytest.approx(expected_f1)
    assert exposure["F2"] == pytest.approx(expected_f2)


def test_pattern_epsilon_sqrt_diagonal():
    cov = pd.DataFrame(
        [[4.0, 0.5], [0.5, 9.0]], index=["F1", "F2"], columns=["F1", "F2"]
    )
    eps = pattern_epsilon(cov, z=2.0)
    assert eps["F1"] == pytest.approx(2.0 * 2.0)
    assert eps["F2"] == pytest.approx(2.0 * 3.0)
