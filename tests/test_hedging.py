"""Tests for ``factors.hedging`` — book-vega hedging against trained
patterns.

Covers exact recovery when a single candidate's exposure direction
matches the book exactly, residual-within-epsilon on a random
well-posed system, cost-based tie-breaking between two candidates with
identical exposure but different cost, the liquid-universe filter, and
the infeasible-LP raise path.
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
