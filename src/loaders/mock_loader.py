"""Synthetic data generator for local development and tests.

Produces a fully populated :class:`RatesVolStore` whose four raw tables
(``rate``, ``atm_vol``, ``skew_p2``, ``skew_n2``) share joint dynamics chosen
to be financially plausible, and then computes the two derived tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..datastore import RatesVolStore
from ..schema import EXPIRY_ORDER, MATURITY_ORDER


def _tenor_days(label: str) -> int:
    """Approximate tenor length in days for ordering ('1W'=7, '1M'=30, '1Y'=365)."""
    n = int(label[:-1])
    return {"W": 7, "M": 30, "Y": 365}[label[-1]] * n


def make_mock_store(
    n_days: int = 500,
    start_date: str = "2020-01-02",
    expiry_labels: list[str] | None = None,
    maturity_labels: list[str] | None = None,
    seed: int = 42,
) -> RatesVolStore:
    """Generate a synthetic :class:`RatesVolStore` with joint rate/vol dynamics.

    Parameters
    ----------
    n_days : int
        Number of business days to simulate.
    start_date : str
        Calendar start date (parsed by ``pd.bdate_range``).
    expiry_labels, maturity_labels : list[str] | None
        Override label lists; default to ``EXPIRY_ORDER`` / ``MATURITY_ORDER``.
    seed : int
        Seed for the RNG.

    Data generating process
    -----------------------
    *Pair universe.* For each table, only ``(expiry, maturity)`` pairs with
    ``tenor(expiry) <= tenor(maturity)`` are populated — e.g. there is no
    ``(10Y, 1Y)`` because a 10Y option on a 1Y swap is unrealistic.

    *Rates.* One Ornstein–Uhlenbeck process per maturity, Euler-Maruyama
    discretized with ``dt = 1/252``:

        ``dr_t = kappa * (theta_m - r_t) dt + sigma_r * sqrt(dt) * eps_r_t``

    Long-run means ``theta_m`` interpolate from ~3.0 (short end) to ~4.5
    (long end), in percent. Brownian increments are correlated across
    maturities via ``corr(i,j) = exp(-decay * |rank_i - rank_j|)`` so neighbouring
    tenors co-move more strongly than distant ones. The same swap-rate sample
    is broadcast across every valid expiry — i.e. forward-starting rates are
    approximated by spot rates of the same maturity.

    *ATM vol.* Per ``(expiry, maturity)``, an OU process around a mean that
    has a "vol hump": base level ~55bps plus a Gaussian bump peaked at the
    ``1Y`` expiry rank (intermediate expiries reach ~75bps). Each vol
    innovation is negatively correlated (``rho = -0.30``) with the same-maturity
    rate innovation, so rates-up days tend to coincide with vol-down days.

    *Skew.* ``skew_p2 = atm_vol + spread``, ``skew_n2 = atm_vol - spread``,
    where ``spread`` is its own (small, positive) OU process around ~8bps,
    floored at 1bp so the +2 strike always sits above the -2 strike.

    *Units.* Rates in percent (``3.5`` means 3.5%). Vols and spreads in basis
    points (``65.0`` means 65 bps of normal vol).

    Limitations
    -----------
    * No regime switching, no jumps, constant vol-of-vol — realised paths are
      smoother than real markets.
    * Curve shape is monotonically upward; no inversion regimes.
    * Skew is symmetric around ATM (``skew_p2 - atm = atm - skew_n2``) by
      construction, so the derived ``skew_spread`` reflects only the spread
      process and not any smile asymmetry.
    * Forward-starting rates are approximated by their spot counterparts;
      there is no genuine forward-rate calculation across the expiry axis.
    """
    expiry_labels = list(expiry_labels) if expiry_labels is not None else EXPIRY_ORDER
    maturity_labels = list(maturity_labels) if maturity_labels is not None else MATURITY_ORDER

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    dt = 1.0 / 252.0

    # ---- Rates: one OU process per maturity, with cross-maturity correlation
    n_mat = len(maturity_labels)
    theta = np.linspace(3.0, 4.5, n_mat)
    kappa_r, sigma_r = 1.0, 0.50

    decay = 0.15
    idx = np.arange(n_mat)
    corr_r = np.exp(-decay * np.abs(idx[:, None] - idx[None, :]))
    L_r = np.linalg.cholesky(corr_r)

    z_r = rng.standard_normal((n_days, n_mat))
    eps_r = z_r @ L_r.T  # rows have covariance corr_r

    rates = np.zeros((n_days, n_mat))
    rates[0] = theta
    sqdt = np.sqrt(dt)
    for t in range(1, n_days):
        rates[t] = (
            rates[t - 1]
            + kappa_r * (theta - rates[t - 1]) * dt
            + sigma_r * sqdt * eps_r[t]
        )

    # ---- Valid (expiry, maturity) pairs
    pairs = [
        (e, m)
        for e in expiry_labels
        for m in maturity_labels
        if _tenor_days(e) <= _tenor_days(m)
    ]
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("No valid (expiry, maturity) pairs after tenor filter")

    expiry_rank_in_list = {e: i for i, e in enumerate(expiry_labels)}
    maturity_rank_in_list = {m: i for i, m in enumerate(maturity_labels)}
    pair_mat_idx = np.array([maturity_rank_in_list[m] for _, m in pairs])

    # ---- ATM vol: OU around a vol-hump mean, negatively coupled to rates
    peak_expiry_rank = expiry_rank_in_list.get("1Y", len(expiry_labels) // 2)
    hump_width = 2.0
    base_vol, hump_amp = 55.0, 20.0
    expiry_ranks_per_pair = np.array(
        [expiry_rank_in_list[e] for e, _ in pairs], dtype=float
    )
    means_v = base_vol + hump_amp * np.exp(
        -((expiry_ranks_per_pair - peak_expiry_rank) ** 2) / (2 * hump_width ** 2)
    )

    kappa_v, sigma_v, rho_rv = 1.5, 8.0, -0.30
    eps_v_raw = rng.standard_normal((n_days, n_pairs))
    # Couple each vol innovation to its maturity's rate innovation.
    eps_r_for_pair = eps_r[:, pair_mat_idx]
    eps_v = rho_rv * eps_r_for_pair + np.sqrt(1.0 - rho_rv ** 2) * eps_v_raw

    atm_vol = np.zeros((n_days, n_pairs))
    atm_vol[0] = means_v
    for t in range(1, n_days):
        atm_vol[t] = (
            atm_vol[t - 1]
            + kappa_v * (means_v - atm_vol[t - 1]) * dt
            + sigma_v * sqdt * eps_v[t]
        )

    # ---- Skew spread: positive OU, 1bp floor
    mean_spread, kappa_s, sigma_s = 8.0, 2.0, 2.0
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

    # ---- Broadcast rate(t, m) across valid expiries per pair
    rate_panel = rates[:, pair_mat_idx]  # shape (n_days, n_pairs)

    # ---- Materialize long-format records and load
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
    return store
