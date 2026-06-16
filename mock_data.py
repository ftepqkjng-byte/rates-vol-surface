"""Synthetic rate / vol surface data with regime-switching dynamics.

Produces the four long-format tables used by ``data/mock/*.pkl``:
``rate``, ``atm_vol``, ``skew_p2``, ``skew_n2``. Each is a DataFrame with
columns ``[date, expiry, tenor, value]``.

* ``expiry`` is the forward-starting time of the option / forward.
* ``tenor`` is the length of the underlying rate.

``make_mock_data`` returns a dict of these plus the latent ``true_regimes``
Series; ``save_mock_data`` writes them as pickles. Running this file as a
script regenerates ``data/mock/`` end-to-end.

A sticky 3-state Markov chain drives the long-run mean / vol of a
two-factor rate model (level + slope) and the mean of an ATM-vol OU
process.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import EXPIRY_LABELS, TENOR_LABELS, EXPIRY_RANK, TENOR_RANK

# ---- Regime parameters ------------------------------------------------------
_TRANSITION_MATRIX = np.array(
    [
        [0.97, 0.02, 0.01],
        [0.01, 0.96, 0.03],
        [0.02, 0.03, 0.95],
    ]
)
REGIME_LABELS = {0: "bull_flattening", 1: "bear_steepening", 2: "range_bound"}
_LEVEL_TARGET = np.array([3.0, 4.5, 3.8])   # rate level (%) per regime
_LEVEL_VOL = np.array([0.6, 1.2, 0.3])      # rate vol (% / sqrt(year))
_SLOPE_TARGET = np.array([0.40, 0.75, 0.55])
_VOL_MEAN = np.array([60.0, 90.0, 45.0])    # ATM vol (bps)


def _stationary_distribution(T: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    return pi / pi.sum()


def make_mock_data(
    n_days: int = 500,
    start_date: str = "2020-01-02",
    seed: int = 42,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Return four long-format DataFrames + the ground-truth regime series.

    Output keys:
        ``rate``, ``atm_vol``, ``skew_p2``, ``skew_n2`` — each is a
        DataFrame with columns ``[date, expiry, tenor, value]``.
        ``true_regimes`` — Series of integer regime ids indexed by date.

    Units: rates in percent (3.5 means 3.5%); vols and spreads in bps.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    dt = 1.0 / 252.0
    sqdt = np.sqrt(dt)

    # 1. Sample latent regimes.
    pi = _stationary_distribution(_TRANSITION_MATRIX)
    regimes = np.zeros(n_days, dtype=int)
    regimes[0] = rng.choice(3, p=pi)
    for t in range(1, n_days):
        regimes[t] = rng.choice(3, p=_TRANSITION_MATRIX[regimes[t - 1]])

    # 2. Level and slope OU states (kappa=10 -> ~17-day half-life,
    #    shorter than the ~30-day regime episodes so the state tracks).
    kappa = 10.0
    sigma_slope = 0.08

    level_state = np.zeros(n_days)
    slope_state = np.zeros(n_days)
    level_state[0] = _LEVEL_TARGET[regimes[0]]
    slope_state[0] = _SLOPE_TARGET[regimes[0]]
    eps_level = rng.standard_normal(n_days)
    eps_slope = rng.standard_normal(n_days)
    for t in range(1, n_days):
        r = regimes[t]
        level_state[t] = (
            level_state[t - 1]
            + kappa * (_LEVEL_TARGET[r] - level_state[t - 1]) * dt
            + _LEVEL_VOL[r] * sqdt * eps_level[t]
        )
        slope_state[t] = (
            slope_state[t - 1]
            + kappa * (_SLOPE_TARGET[r] - slope_state[t - 1]) * dt
            + sigma_slope * sqdt * eps_slope[t]
        )

    # 3. Pair universe — full expiry × tenor cross-product, no length filter.
    pairs = [(e, t) for e in EXPIRY_LABELS for t in TENOR_LABELS]
    n_pairs = len(pairs)
    tenor_rank_per_pair = np.array([TENOR_RANK[t] for _, t in pairs])
    expiry_rank_per_pair = np.array([EXPIRY_RANK[e] for e, _ in pairs])

    # 4. Factor loadings — L1 uniform (level), L2 monotone over tenor rank (slope).
    L1_per_pair = np.ones(n_pairs)
    L2_per_pair = np.linspace(-0.8, 0.8, len(TENOR_LABELS))[tenor_rank_per_pair]

    # 5. Rate panel — sigma_idio gives ~10% per-pair idio variance share.
    sigma_idio_rate = 0.20
    idio_rate = sigma_idio_rate * rng.standard_normal((n_days, n_pairs))
    rate_panel = (
        L1_per_pair[None, :] * level_state[:, None]
        + L2_per_pair[None, :] * slope_state[:, None]
        + idio_rate
    )

    # 6. ATM vol — OU around regime-dependent mean + Gaussian hump in expiry
    #    rank peaked at 1Y. Vol innovations anti-correlated with level.
    peak_expiry_rank = EXPIRY_RANK["1Y"]
    hump_per_rank = 20.0 * np.exp(
        -(np.arange(len(EXPIRY_LABELS)) - peak_expiry_rank) ** 2 / (2 * 2.0 ** 2)
    )
    hump_per_pair = hump_per_rank[expiry_rank_per_pair]
    vol_target_t = _VOL_MEAN[regimes][:, None] + hump_per_pair[None, :]

    rho_rv = -0.35
    eps_vol_raw = rng.standard_normal((n_days, n_pairs))
    eps_vol = rho_rv * eps_level[:, None] + np.sqrt(1.0 - rho_rv ** 2) * eps_vol_raw

    kappa_vol, sigma_vol = 2.0, 8.0
    atm_vol = np.zeros((n_days, n_pairs))
    atm_vol[0] = vol_target_t[0]
    for t in range(1, n_days):
        atm_vol[t] = (
            atm_vol[t - 1]
            + kappa_vol * (vol_target_t[t] - atm_vol[t - 1]) * dt
            + sigma_vol * sqdt * eps_vol[t]
        )
    atm_vol = np.maximum(atm_vol, 1.0)

    # 7. Skew spread (positive OU floored at 1 bp).
    kappa_s, sigma_s, mean_s = 2.0, 2.0, 8.0
    eps_s = rng.standard_normal((n_days, n_pairs))
    spread = np.zeros((n_days, n_pairs))
    spread[0] = mean_s
    for t in range(1, n_days):
        spread[t] = np.maximum(
            spread[t - 1]
            + kappa_s * (mean_s - spread[t - 1]) * dt
            + sigma_s * sqdt * eps_s[t],
            1.0,
        )
    skew_p2 = atm_vol + spread
    skew_n2 = np.maximum(atm_vol - spread, 1.0)

    # 8. Materialise as long-format DataFrames.
    expiry_col = np.tile([e for e, _ in pairs], n_days)
    tenor_col = np.tile([t for _, t in pairs], n_days)
    date_col = np.repeat(dates.values, n_pairs)

    def _long(panel: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": date_col,
                "expiry": expiry_col,
                "tenor": tenor_col,
                "value": panel.ravel(),
            }
        )

    return {
        "rate": _long(rate_panel),
        "atm_vol": _long(atm_vol),
        "skew_p2": _long(skew_p2),
        "skew_n2": _long(skew_n2),
        "true_regimes": pd.Series(regimes, index=dates, name="true_regime"),
    }


def save_mock_data(
    output_dir: str | Path = "data/mock",
    n_days: int = 500,
    start_date: str = "2020-01-02",
    seed: int = 42,
) -> None:
    """Generate and pickle the four tables (and regime series) under ``output_dir``."""
    tables = make_mock_data(n_days=n_days, start_date=start_date, seed=seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("rate", "atm_vol", "skew_p2", "skew_n2"):
        tables[name].to_pickle(out / f"{name}.pkl")
    tables["true_regimes"].to_pickle(out / "true_regimes.pkl")
    print(f"wrote 4 tables + true_regimes to {out}/")


if __name__ == "__main__":
    save_mock_data()
