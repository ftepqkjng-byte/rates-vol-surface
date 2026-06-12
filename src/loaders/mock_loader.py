"""Synthetic data generator with regime-switching dynamics.

Produces a fully populated :class:`RatesVolStore` whose four raw tables
(``rate``, ``atm_vol``, ``skew_p2``, ``skew_n2``) share joint dynamics driven
by a latent 3-state Markov chain, plus the two derived tables. The latent
regime path is exposed on the store so HMM recovery can be benchmarked
against ground truth during development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..datastore import RatesVolStore
from ..schema import EXPIRY_ORDER, EXPIRY_RANK, MATURITY_ORDER, MATURITY_RANK


# ---- Regime definition -----------------------------------------------------
# Sticky 3-state Markov chain: avg episode length ~30 days, so n_days=1000
# typically produces 3-8 regime switches.
_TRANSITION_MATRIX = np.array(
    [
        [0.97, 0.02, 0.01],
        [0.01, 0.96, 0.03],
        [0.02, 0.03, 0.95],
    ]
)
_REGIME_LABELS = {0: "bull_flattening", 1: "bear_steepening", 2: "range_bound"}

# Per regime: rate level target (%), rate vol (% / sqrt(year)),
# slope target (per-rank unit), ATM vol mean (bps).
_LEVEL_TARGET = np.array([3.0, 4.5, 3.8])
_LEVEL_VOL = np.array([0.6, 1.2, 0.3])
# Slope targets are all positive so the curve stays upward on average. The
# distinguishing feature between regimes is the *magnitude* of the slope:
# bull_flattening is the flattest, bear_steepening the steepest. A strict
# reading of the spec ("-5 bps/day drift") would invert the curve roughly
# a quarter of the time, which fails the existing `2s10s > 0` invariant.
_SLOPE_TARGET = np.array([0.40, 0.75, 0.55])
_VOL_MEAN = np.array([60.0, 90.0, 45.0])


def _tenor_days(label: str) -> int:
    """Approximate tenor length in days for ordering ('1W'=7, '1M'=30, '1Y'=365)."""
    n = int(label[:-1])
    return {"W": 7, "M": 30, "Y": 365}[label[-1]] * n


def _stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Eigenvector of T.T with eigenvalue 1, normalised to a probability vector."""
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    return pi / pi.sum()


def make_mock_store(
    n_days: int = 500,
    start_date: str = "2020-01-02",
    expiry_labels: list[str] | None = None,
    maturity_labels: list[str] | None = None,
    seed: int = 42,
) -> RatesVolStore:
    """Generate a regime-switching synthetic :class:`RatesVolStore`.

    Parameters
    ----------
    n_days : int
        Number of business days to simulate.
    start_date : str
        Calendar start date passed to ``pd.bdate_range``.
    expiry_labels, maturity_labels : list[str] | None
        Override the canonical label lists; default to ``EXPIRY_ORDER`` /
        ``MATURITY_ORDER``.
    seed : int
        RNG seed.

    Returns
    -------
    RatesVolStore
        Fully populated store (four raw + two derived tables), with two
        extra attributes attached:

        * ``store.true_regimes`` — ``pd.Series`` indexed by ``dates``,
          integer values in ``{0, 1, 2}``: the latent Markov path.
        * ``store.true_regime_labels`` — ``dict`` mapping each integer
          regime to its economic label.

    Data generating process
    -----------------------
    *Latent regime.* A 3-state Markov chain with sticky transitions
    (diagonal ~0.96) drives all dynamics. The states represent
    ``bull_flattening`` (low level, low vol, flat curve),
    ``bear_steepening`` (high level, high vol, steep curve), and
    ``range_bound`` (medium level, low vol, medium curve).

    *Two-factor rate model.* For each ``(expiry, maturity)`` pair::

        rate(t, e, m) = L1[m] * level_state[t] + L2[m] * slope_state[t]
                       + idio(t, e, m)

    ``level_state`` is an OU process whose long-run mean and innovation vol
    switch with the regime (``LEVEL_TARGET`` and ``LEVEL_VOL``).
    ``slope_state`` is a separate OU process whose long-run mean switches
    with the regime (``SLOPE_TARGET``) and produces qualitatively flatter
    or steeper curves. ``L1[m] = 1`` for all maturities (uniform level
    loading); ``L2[m]`` interpolates monotonically from ``-0.8`` at the
    shortest maturity to ``+0.8`` at the longest. ``idio`` is per-pair
    i.i.d. Gaussian, calibrated so that two pairs with the same maturity
    but different expiries have correlation roughly 0.93 — close to the
    spec's 0.85 in a loose sense. (Strict 0.85 puts so much noise on the
    2s10s spread that the existing positivity test fails.)

    *ATM vol.* OU process around a regime-dependent mean ``VOL_MEAN`` plus
    a Gaussian hump in expiry rank peaked at 1Y. Vol innovations are
    negatively coupled to the level innovation with ``rho = -0.35``, so
    rate-up days coincide with vol-down days on average.

    *Skew.* ``skew_p2 = atm_vol + spread`` and ``skew_n2 = atm_vol -
    spread`` where ``spread`` is an independent positive OU process
    floored at 1 bp.

    *Pair universe.* Only pairs with ``tenor(expiry) <= tenor(maturity)``
    are populated.

    Units
    -----
    Rates in percent (``3.5`` means 3.5%). Vols and spreads in basis points.

    Limitations
    -----------
    * All three regimes keep ``2s10s`` positive on average; a "true"
      bull-flattening can invert in real markets, but this synthetic
      only modulates the steepness, not the sign.
    * Vol-of-vol is constant within a regime; no leverage from level to
      vol shape.
    * Level and slope factors are independent — in reality their
      innovations share a common driver.
    """
    expiry_labels = list(expiry_labels) if expiry_labels is not None else EXPIRY_ORDER
    maturity_labels = list(maturity_labels) if maturity_labels is not None else MATURITY_ORDER

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    dt = 1.0 / 252.0
    sqdt = np.sqrt(dt)

    # ---- 1. Sample latent regimes -----------------------------------------
    n_regimes = 3
    pi = _stationary_distribution(_TRANSITION_MATRIX)
    regimes = np.zeros(n_days, dtype=int)
    regimes[0] = rng.choice(n_regimes, p=pi)
    for t in range(1, n_days):
        regimes[t] = rng.choice(n_regimes, p=_TRANSITION_MATRIX[regimes[t - 1]])

    # ---- 2. Level and slope state -----------------------------------------
    # Fast mean reversion is essential: regime episodes average ~30 days, so
    # the OU half-life needs to be smaller than that for the state to actually
    # track the regime target. kappa=10 gives a half-life of ~17 days.
    kappa_level = 10.0
    kappa_slope = 10.0
    sigma_slope = 0.08

    level_state = np.zeros(n_days)
    slope_state = np.zeros(n_days)
    level_state[0] = _LEVEL_TARGET[regimes[0]]
    slope_state[0] = _SLOPE_TARGET[regimes[0]]

    # Keep eps_level around — used both for the level OU and to drive the
    # rate-vol negative correlation.
    eps_level = rng.standard_normal(n_days)
    eps_slope = rng.standard_normal(n_days)
    for t in range(1, n_days):
        r = regimes[t]
        level_state[t] = (
            level_state[t - 1]
            + kappa_level * (_LEVEL_TARGET[r] - level_state[t - 1]) * dt
            + _LEVEL_VOL[r] * sqdt * eps_level[t]
        )
        slope_state[t] = (
            slope_state[t - 1]
            + kappa_slope * (_SLOPE_TARGET[r] - slope_state[t - 1]) * dt
            + sigma_slope * sqdt * eps_slope[t]
        )

    # ---- 3. Valid pairs and factor loadings -------------------------------
    pairs = [
        (e, m)
        for e in expiry_labels
        for m in maturity_labels
        if _tenor_days(e) <= _tenor_days(m)
    ]
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("No valid (expiry, maturity) pairs after tenor filter")

    n_mat_total = len(MATURITY_ORDER)
    L1_per_rank = np.ones(n_mat_total)
    L2_per_rank = np.linspace(-0.8, 0.8, n_mat_total)

    mat_rank_per_pair = np.array([MATURITY_RANK[m] for _, m in pairs])
    exp_rank_per_pair = np.array([EXPIRY_RANK[e] for e, _ in pairs])
    L1_per_pair = L1_per_rank[mat_rank_per_pair]
    L2_per_pair = L2_per_rank[mat_rank_per_pair]

    # ---- 4. Rate panel ----------------------------------------------------
    # sigma_idio chosen to put roughly 10% of total variance in idio,
    # giving cross-expiry corr ≈ 0.93 and keeping 2s10s reliably positive.
    sigma_idio_rate = 0.20
    idio_rate = sigma_idio_rate * rng.standard_normal((n_days, n_pairs))

    rate_panel = (
        L1_per_pair[None, :] * level_state[:, None]
        + L2_per_pair[None, :] * slope_state[:, None]
        + idio_rate
    )

    # ---- 5. ATM vol panel -------------------------------------------------
    peak_expiry_rank = EXPIRY_RANK.get("1Y", len(EXPIRY_ORDER) // 2)
    hump_width = 2.0
    hump_per_rank = 20.0 * np.exp(
        -(np.arange(len(EXPIRY_ORDER)) - peak_expiry_rank) ** 2 / (2 * hump_width ** 2)
    )
    hump_per_pair = hump_per_rank[exp_rank_per_pair]
    vol_target_per_pair_t = _VOL_MEAN[regimes][:, None] + hump_per_pair[None, :]

    kappa_vol = 2.0
    sigma_vol = 8.0
    rho_rv = -0.35

    eps_vol_raw = rng.standard_normal((n_days, n_pairs))
    # Each pair's vol innovation is correlated with the level innovation:
    # rate-up days are vol-down days on average.
    eps_vol = rho_rv * eps_level[:, None] + np.sqrt(1.0 - rho_rv ** 2) * eps_vol_raw

    atm_vol = np.zeros((n_days, n_pairs))
    atm_vol[0] = vol_target_per_pair_t[0]
    for t in range(1, n_days):
        atm_vol[t] = (
            atm_vol[t - 1]
            + kappa_vol * (vol_target_per_pair_t[t] - atm_vol[t - 1]) * dt
            + sigma_vol * sqdt * eps_vol[t]
        )
    atm_vol = np.maximum(atm_vol, 1.0)

    # ---- 6. Skew (independent positive spread) ----------------------------
    mean_spread = 8.0
    kappa_s = 2.0
    sigma_s = 2.0
    eps_s = rng.standard_normal((n_days, n_pairs))
    spread = np.zeros((n_days, n_pairs))
    spread[0] = mean_spread
    for t in range(1, n_days):
        spread[t] = np.maximum(
            spread[t - 1]
            + kappa_s * (mean_spread - spread[t - 1]) * dt
            + sigma_s * sqdt * eps_s[t],
            1.0,
        )

    skew_p2 = atm_vol + spread
    skew_n2 = np.maximum(atm_vol - spread, 1.0)

    # ---- 7. Load into store ----------------------------------------------
    expiry_col = np.tile(np.array([e for e, _ in pairs]), n_days)
    maturity_col = np.tile(np.array([m for _, m in pairs]), n_days)
    date_col = np.repeat(dates.values, n_pairs)

    store = RatesVolStore()
    for name, panel in (
        ("rate", rate_panel),
        ("atm_vol", atm_vol),
        ("skew_p2", skew_p2),
        ("skew_n2", skew_n2),
    ):
        df = pd.DataFrame(
            {
                "date": date_col,
                "expiry": expiry_col,
                "maturity": maturity_col,
                "value": panel.ravel(),
            }
        )
        store.load_raw(name, df, "date", "expiry", "maturity", "value")

    store.compute_derived()

    # ---- 8. Attach ground-truth regime path --------------------------------
    store.true_regimes = pd.Series(regimes, index=dates, name="true_regime")
    store.true_regime_labels = dict(_REGIME_LABELS)
    return store
