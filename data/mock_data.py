"""Synthetic rate / vol surface data.

Produces the four long-format tables under ``data/mock/*.pkl``:
``rate``, ``atm_vol``, ``skew_p2``, ``skew_n2``. Each is a DataFrame with
columns ``[date, expiry, tenor, value]``.

* ``expiry`` is the forward-starting time of the option / forward.
* ``tenor`` is the length of the underlying rate.

After writing the raw tables, ``save_mock_data`` invokes
``pipeline.build_all`` (sibling file in this folder) to materialise the
daily-diff and parallel-shift-stripped residual panels next to them.

DGP summary
-----------
* Rate panel = level OU + slope OU (loadings vary linearly across tenor
  rank) + per-cell idiosyncratic noise.
* ATM vol = OU around a Gaussian hump in expiry (peaked at 1Y),
  anti-correlated with the level innovation.
* Skew = ATM vol ± a positive OU spread (floored at 1 bp).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project-root helpers (config.py) importable when this script
# is run from either the repo root or the data folder.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EXPIRY_LABELS, TENOR_LABELS, EXPIRY_RANK, TENOR_RANK  # noqa: E402

# ---- DGP parameters --------------------------------------------------------
_LEVEL_TARGET = 4.0      # rate level (%)
_LEVEL_VOL = 0.8         # rate vol (% / sqrt(year))
_SLOPE_TARGET = 0.55
_VOL_MEAN = 65.0         # ATM vol (bps)

_DEFAULT_DIR = Path(__file__).resolve().parent / "mock"


def make_mock_data(
    n_days: int = 500,
    start_date: str = "2020-01-02",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return the four long-format DataFrames.

    Output keys: ``rate``, ``atm_vol``, ``skew_p2``, ``skew_n2``. Each is
    a DataFrame with columns ``[date, expiry, tenor, value]``.

    Units: rates in percent (3.5 means 3.5%); vols and spreads in bps.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    dt = 1.0 / 252.0
    sqdt = np.sqrt(dt)

    # 1. Level and slope OU states. kappa=10 -> ~17-day half-life.
    kappa = 10.0
    sigma_slope = 0.08
    level_state = np.zeros(n_days)
    slope_state = np.zeros(n_days)
    level_state[0] = _LEVEL_TARGET
    slope_state[0] = _SLOPE_TARGET
    eps_level = rng.standard_normal(n_days)
    eps_slope = rng.standard_normal(n_days)
    for t in range(1, n_days):
        level_state[t] = (
            level_state[t - 1]
            + kappa * (_LEVEL_TARGET - level_state[t - 1]) * dt
            + _LEVEL_VOL * sqdt * eps_level[t]
        )
        slope_state[t] = (
            slope_state[t - 1]
            + kappa * (_SLOPE_TARGET - slope_state[t - 1]) * dt
            + sigma_slope * sqdt * eps_slope[t]
        )

    # 2. Pair universe — full expiry × tenor cross-product, no length filter.
    pairs = [(e, t) for e in EXPIRY_LABELS for t in TENOR_LABELS]
    n_pairs = len(pairs)
    tenor_rank_per_pair = np.array([TENOR_RANK[t] for _, t in pairs])
    expiry_rank_per_pair = np.array([EXPIRY_RANK[e] for e, _ in pairs])

    # 3. Factor loadings — L1 uniform (level), L2 monotone over tenor rank (slope).
    L1_per_pair = np.ones(n_pairs)
    L2_per_pair = np.linspace(-0.8, 0.8, len(TENOR_LABELS))[tenor_rank_per_pair]

    # 4. Rate panel — sigma_idio gives ~10% per-pair idio variance share.
    sigma_idio_rate = 0.20
    idio_rate = sigma_idio_rate * rng.standard_normal((n_days, n_pairs))
    rate_panel = (
        L1_per_pair[None, :] * level_state[:, None]
        + L2_per_pair[None, :] * slope_state[:, None]
        + idio_rate
    )

    # 5. ATM vol — OU around a Gaussian expiry hump (peaked at 1Y). Vol
    #    innovations are anti-correlated with the rate level innovation.
    peak_expiry_rank = EXPIRY_RANK["1Y"]
    hump_per_rank = 20.0 * np.exp(
        -(np.arange(len(EXPIRY_LABELS)) - peak_expiry_rank) ** 2 / (2 * 2.0 ** 2)
    )
    hump_per_pair = hump_per_rank[expiry_rank_per_pair]
    vol_target = _VOL_MEAN + hump_per_pair       # (n_pairs,) — time-invariant.

    rho_rv = -0.35
    eps_vol_raw = rng.standard_normal((n_days, n_pairs))
    eps_vol = rho_rv * eps_level[:, None] + np.sqrt(1.0 - rho_rv ** 2) * eps_vol_raw

    kappa_vol, sigma_vol = 2.0, 8.0
    atm_vol = np.zeros((n_days, n_pairs))
    atm_vol[0] = vol_target
    for t in range(1, n_days):
        atm_vol[t] = (
            atm_vol[t - 1]
            + kappa_vol * (vol_target - atm_vol[t - 1]) * dt
            + sigma_vol * sqdt * eps_vol[t]
        )
    atm_vol = np.maximum(atm_vol, 1.0)

    # 6. Skew spread (positive OU floored at 1 bp).
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

    # 7. Materialise as long-format DataFrames.
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
    }


def save_mock_data(
    output_dir: str | Path = _DEFAULT_DIR,
    n_days: int = 500,
    start_date: str = "2020-01-02",
    seed: int = 42,
    run_pipeline: bool = True,
) -> None:
    """Write the four raw tables to ``output_dir``; if ``run_pipeline`` is
    set, also invoke ``pipeline.build_all`` so the diff and residual pkls
    land in the same folder. Default ``output_dir`` resolves to
    ``data/mock`` relative to this script's location."""
    tables = make_mock_data(n_days=n_days, start_date=start_date, seed=seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("rate", "atm_vol", "skew_p2", "skew_n2"):
        tables[name].to_pickle(out / f"{name}.pkl")
    print(f"wrote 4 raw tables to {out}/")
    if run_pipeline:
        from pipeline import build_all
        build_all(input_dir=out, output_dir=out)


if __name__ == "__main__":
    save_mock_data()
