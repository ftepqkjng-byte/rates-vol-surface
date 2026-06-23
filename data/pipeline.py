"""Daily-diff and parallel-shift-stripped panels from raw surface pkls.

For each canonical surface (``rate``, ``atm_vol``, ``skew_p2``,
``skew_n2``) this reads the long-format raw pkl and writes two derived
pkls alongside it, **in the same long format**
``[date, expiry, tenor, value]``:

* ``{name}_diff.pkl``     — the daily move (``wide.diff()`` pivoted back).
* ``{name}_residual.pkl`` — the diff with the realised-std-weighted
                            parallel shift subtracted, pivoted back.

Aligning the derived schema with the raw schema means any consumer
reads all three the same way::

    panel = to_wide(load_long(path))

The parallel shift on day ``t`` is the cross-sectional aggregate of diffs
*normalised* by each cell's trailing realised std::

    σ_{i,t} = std(diff_i) over [t - window, t - 1]      (strict past)
    shift_t = agg_i ( diff_{i,t} / σ_{i,t} )
    residual_{i,t} = diff_{i,t} - β_{i,t} · shift_t

Default behaviour (``beta_mode="sigma"``, ``agg="mean"``,
``sigma_floor_pct=None``) keeps β_{i,t} = σ_{i,t} so a single scalar
``shift_t`` describes a "parallel" move across cells whose raw scales
differ wildly. Optional knobs (regression-estimated β, robust
aggregators, σ flooring) are described on ``strip_parallel_shift`` and
diagnosed in ``notebooks/residual_diagnostics.ipynb``. Every mode is
leakage-free: σ_{i,t} and the rolling β_{i,t} are both built from data
strictly before ``t``.

Run as a script to materialise all four surfaces under ``data/mock/``::

    python data/pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

# Make the project-root helpers importable when this script is run from
# either the repo root or the data folder.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pca import load_long, to_wide  # noqa: E402

SURFACES = ("rate", "atm_vol", "skew_p2", "skew_n2")
_DEFAULT_DIR = Path(__file__).resolve().parent / "mock"


def compute_diff(wide: pd.DataFrame) -> pd.DataFrame:
    """First daily diff of a wide panel; the all-NaN first row is dropped."""
    return wide.diff().dropna(how="all")


def _wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Pivot a wide ``(date × MultiIndex(expiry, tenor))`` panel back to
    the canonical long format ``[date, expiry, tenor, value]``. NaN cells
    are preserved (no implicit dropna) so the long table stays aligned
    with the diff / residual mask."""
    # ``future_stack=True`` adopts the pandas-2.1 stack behaviour, which
    # keeps NaN cells by default (matching the ``dropna=False`` we want).
    s = wide.stack(level=["expiry", "tenor"], future_stack=True)
    return s.rename("value").reset_index()[["date", "expiry", "tenor", "value"]]


def _aggregate_shift(
    z: pd.DataFrame,
    agg: Literal["mean", "median", "trimmed_mean"],
    trim_pct: float,
) -> pd.Series:
    if agg == "mean":
        return z.mean(axis=1)
    if agg == "median":
        return z.median(axis=1)
    if agg == "trimmed_mean":
        return z.apply(
            lambda row: trim_mean(row.dropna().values, trim_pct)
            if row.notna().any() else np.nan,
            axis=1,
        )
    raise ValueError(
        f"agg must be 'mean'/'median'/'trimmed_mean', got {agg!r}"
    )


def strip_parallel_shift(
    diff: pd.DataFrame,
    window: int = 60,
    beta_mode: Literal["sigma", "regression"] = "sigma",
    agg: Literal["mean", "median", "trimmed_mean"] = "mean",
    trim_pct: float = 0.1,
    sigma_floor_pct: float | None = None,
) -> pd.DataFrame:
    """Subtract the (realised-std-normalised) parallel shift from a diff panel.

    Default args reproduce the original behaviour:
    ``shift_t = mean_i(diff_{i,t}/σ_{i,t})`` and
    ``residual = diff − σ · shift``.

    Parameters
    ----------
    diff
        Wide daily-diff panel (``date × cell``).
    window
        Rolling lookback for σ and (in regression mode) for the rolling
        OLS β. ~3 months of business days by default.
    beta_mode
        How each cell's loading on the common shift is set.

        * ``"sigma"`` *(default, original behaviour)* — ``β_{i,t} = σ_{i,t}``.
        * ``"regression"`` — ``β_{i,t}`` from a rolling OLS regression of
          ``diff_i`` on the sigma-mode shift series, computed strictly
          over past ``(shift, diff)`` pairs of length ``window``.
          Adds a second ``window``-row burn-in before any residual is
          produced.
    agg
        Cross-sectional aggregator of normalised diffs to ``shift_t``.

        * ``"mean"`` *(default)*, ``"median"``, ``"trimmed_mean"``
          (drops ``trim_pct`` from each tail per row).
    trim_pct
        Tail fraction used only when ``agg="trimmed_mean"``.
    sigma_floor_pct
        If set, per-row σ values below the row's ``sigma_floor_pct``
        quantile are lifted up to that quantile before normalising —
        keeps illiquid cells from blowing up into outlier z-scores.
        ``None`` (default) leaves σ as-is.

    Notes
    -----
    Leakage-free: σ_{i,t} and (regression mode) β_{i,t} are computed via
    ``.shift(1)`` on rolling statistics, so day ``t``'s residual uses
    only information available at ``t``.
    """
    sigma = diff.rolling(window=window).std().shift(1)

    if sigma_floor_pct is not None:
        floor = sigma.quantile(sigma_floor_pct, axis=1)
        sigma = sigma.clip(lower=floor, axis=0)

    z = diff / sigma
    shift = _aggregate_shift(z, agg, trim_pct)

    if beta_mode == "sigma":
        residual = diff.sub(sigma.mul(shift, axis=0))
    elif beta_mode == "regression":
        cov = diff.rolling(window=window).cov(shift)
        var_shift = shift.rolling(window=window).var()
        beta = cov.div(var_shift, axis=0).shift(1)
        residual = diff.sub(beta.mul(shift, axis=0))
    else:
        raise ValueError(
            f"beta_mode must be 'sigma'/'regression', got {beta_mode!r}"
        )

    return residual.dropna(how="all")


def build_all(
    input_dir: str | Path = _DEFAULT_DIR,
    output_dir: str | Path = _DEFAULT_DIR,
    window: int = 60,
) -> None:
    """Read every surface in ``SURFACES`` from ``input_dir`` and write the
    diff and residual pkls to ``output_dir``. Default paths resolve
    relative to this script's location, so the script can be invoked from
    any working directory."""
    inp, out = Path(input_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in SURFACES:
        wide = to_wide(load_long(inp / f"{name}.pkl"))
        diff = compute_diff(wide)
        residual = strip_parallel_shift(diff, window=window)
        _wide_to_long(diff).to_pickle(out / f"{name}_diff.pkl")
        _wide_to_long(residual).to_pickle(out / f"{name}_residual.pkl")
        print(f"{name}: diff {diff.shape[0]}d × {diff.shape[1]} cells, "
              f"residual {residual.shape[0]}d × {residual.shape[1]} cells")


if __name__ == "__main__":
    build_all()
