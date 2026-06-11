"""Curve-level and cross-pair derived features built on top of a RatesVolStore.

Each public function takes a :class:`RatesVolStore` and returns a
``pd.DataFrame`` indexed by date. Missing pairs are logged and skipped rather
than raised so a partially populated store still produces what it can.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ..schema import EXPIRY_ORDER, MATURITY_RANK

if TYPE_CHECKING:
    from ..datastore import RatesVolStore

_LOG = logging.getLogger(__name__)

_DEFAULT_SPREAD_PAIRS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("6M", "2Y"), ("6M", "10Y")),
    (("6M", "2Y"), ("6M", "5Y")),
    (("6M", "5Y"), ("6M", "30Y")),
    (("6M", "1Y"), ("6M", "10Y")),
]


def _label_num(label: str) -> str:
    """Strip the trailing unit char: '2Y' -> '2', '10Y' -> '10', '6M' -> '6'."""
    return label[:-1]


def _spread_col(short_pair: tuple[str, str], long_pair: tuple[str, str]) -> str:
    return f"{_label_num(short_pair[1])}s{_label_num(long_pair[1])}s"


def _detect_bps_scale(panel: pd.DataFrame) -> float:
    """Return a multiplier that brings rate values to basis points.

    Heuristic on the panel's maximum absolute value:

    * < 1     -> decimal  (e.g. 0.045), multiply by 10000
    * < 100   -> percent  (e.g. 4.5),   multiply by 100
    * otherwise -> already in bps, multiply by 1
    """
    if panel.empty:
        return 1.0
    abs_max = float(panel.abs().max(skipna=True).max(skipna=True))
    if abs_max < 1.0:
        return 10_000.0
    if abs_max < 100.0:
        return 100.0
    return 1.0


def _expiry_buckets() -> tuple[set[str], set[str], set[str]]:
    """Split ``EXPIRY_ORDER`` into roughly equal short / mid / long thirds."""
    n = len(EXPIRY_ORDER)
    return (
        set(EXPIRY_ORDER[: n // 3]),
        set(EXPIRY_ORDER[n // 3 : 2 * n // 3]),
        set(EXPIRY_ORDER[2 * n // 3 :]),
    )


def curve_spreads(
    store: "RatesVolStore",
    pairs: list[tuple[tuple[str, str], tuple[str, str]]] | None = None,
) -> pd.DataFrame:
    """Compute long-minus-short rate spreads between ``(expiry, maturity)`` points.

    Each entry in ``pairs`` is ``((short_expiry, short_maturity),
    (long_expiry, long_maturity))``; the spread is ``rate[long] - rate[short]``.
    The column name is ``"{short_mat}s{long_mat}s"`` (e.g. ``"2s10s"``).

    Defaults (all using ``6M`` expiry):
    ``2s10s``, ``2s5s``, ``5s30s``, ``1s10s``.

    Output is in basis points; the input unit is detected from the rate panel's
    value range (see :func:`_detect_bps_scale`).

    Examples
    --------
    >>> from src.loaders.mock_loader import make_mock_store
    >>> store = make_mock_store(n_days=10, seed=0)
    >>> df = curve_spreads(store)
    >>> sorted(df.columns.tolist())
    ['1s10s', '2s10s', '2s5s', '5s30s']
    >>> df.shape == (10, 4)
    True
    >>> bool((df['2s10s'] > 0).all())   # upward-sloping curve in mock data
    True
    """
    pairs = pairs if pairs is not None else _DEFAULT_SPREAD_PAIRS
    default_cols = [_spread_col(s, l) for s, l in pairs]

    if "rate" not in store:
        _LOG.warning("curve_spreads: 'rate' table not in store")
        return pd.DataFrame(columns=default_cols)

    panel = store.as_panel("rate", dropna_threshold=1.0)
    if panel.empty:
        return pd.DataFrame(columns=default_cols)

    scale = _detect_bps_scale(panel)

    cols: dict[str, pd.Series] = {}
    for short_pair, long_pair in pairs:
        name = _spread_col(short_pair, long_pair)
        if short_pair not in panel.columns or long_pair not in panel.columns:
            _LOG.warning(
                "curve_spreads: skipping %s - missing pair %s or %s",
                name, short_pair, long_pair,
            )
            continue
        cols[name] = (panel[long_pair] - panel[short_pair]) * scale

    if not cols:
        return pd.DataFrame(columns=default_cols, index=panel.index)
    return pd.DataFrame(cols, index=panel.index)


def vol_surface_metrics(store: "RatesVolStore") -> pd.DataFrame:
    """Summarise the ATM vol surface and skew, one row per date.

    Columns:
        * ``vol_level`` — cross-sectional mean of all ATM vols.
        * ``vol_slope`` — mean(long-expiry vols) − mean(short-expiry vols).
        * ``vol_hump``  — mean(mid-expiry vols) − 0.5 · (mean(short) + mean(long)).
        * ``skew_level`` — mean of ``skew_spread`` across all pairs.
        * ``skew_slope`` — long-expiry skew minus short-expiry skew.

    Short / mid / long expiry buckets are thirds of :data:`EXPIRY_ORDER`
    (short = {1W,2W,1M,2M}, mid = {3M,6M,9M,1Y}, long = {2Y,3Y,5Y,7Y,10Y}).

    If ``atm_vol`` is missing the three vol metrics are omitted; if
    ``skew_spread`` is missing the two skew metrics are omitted; either way
    the returned frame always carries all five column names (filled with NaN
    where data was unavailable).

    Examples
    --------
    >>> from src.loaders.mock_loader import make_mock_store
    >>> store = make_mock_store(n_days=10, seed=0)
    >>> df = vol_surface_metrics(store)
    >>> list(df.columns)
    ['vol_level', 'vol_slope', 'vol_hump', 'skew_level', 'skew_slope']
    >>> df.shape == (10, 5)
    True
    >>> bool((df['vol_hump'] > 0).mean() > 0.5)   # mock data has a vol hump
    True
    """
    columns = ["vol_level", "vol_slope", "vol_hump", "skew_level", "skew_slope"]
    has_vol = "atm_vol" in store
    has_skew = "skew_spread" in store

    if not has_vol:
        _LOG.warning("vol_surface_metrics: 'atm_vol' not in store; vol_* will be NaN")
    if not has_skew:
        _LOG.warning("vol_surface_metrics: 'skew_spread' not in store; skew_* will be NaN")

    if not has_vol and not has_skew:
        return pd.DataFrame(columns=columns)

    short_exps, mid_exps, long_exps = _expiry_buckets()
    parts: dict[str, pd.Series] = {}

    if has_vol:
        vol = store.as_panel("atm_vol", dropna_threshold=1.0)
        if not vol.empty:
            exp_level = vol.columns.get_level_values("expiry")
            short_m = exp_level.isin(short_exps)
            mid_m = exp_level.isin(mid_exps)
            long_m = exp_level.isin(long_exps)
            parts["vol_level"] = vol.mean(axis=1)
            parts["vol_slope"] = vol.loc[:, long_m].mean(axis=1) - vol.loc[:, short_m].mean(axis=1)
            parts["vol_hump"] = vol.loc[:, mid_m].mean(axis=1) - 0.5 * (
                vol.loc[:, short_m].mean(axis=1) + vol.loc[:, long_m].mean(axis=1)
            )

    if has_skew:
        sk = store.as_panel("skew_spread", dropna_threshold=1.0)
        if not sk.empty:
            exp_level = sk.columns.get_level_values("expiry")
            short_m = exp_level.isin(short_exps)
            long_m = exp_level.isin(long_exps)
            parts["skew_level"] = sk.mean(axis=1)
            parts["skew_slope"] = sk.loc[:, long_m].mean(axis=1) - sk.loc[:, short_m].mean(axis=1)

    if not parts:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(parts)
    for c in columns:
        if c not in df.columns:
            df[c] = float("nan")
    return df[columns]


def term_structure_features(
    store: "RatesVolStore",
    name: str = "rate",
) -> pd.DataFrame:
    """For each expiry, the long-minus-short maturity slope; for each maturity,
    the mean across expiries.

    Column names: ``"{expiry}_slope"`` and ``"{maturity}_level"``. Within each
    expiry, the slope uses the shortest and longest *available* maturities
    ordered by :data:`MATURITY_RANK`; expiries with fewer than two maturities
    are skipped with a log warning.

    For the ``rate`` table — where the mock loader broadcasts swap rates
    across expiries — ``{maturity}_level`` reduces to the maturity's rate
    series; for ``atm_vol`` it gives the average across option expiries.

    Examples
    --------
    >>> from src.loaders.mock_loader import make_mock_store
    >>> store = make_mock_store(n_days=10, seed=0)
    >>> df = term_structure_features(store, name='rate')
    >>> any(c.endswith('_slope') for c in df.columns)
    True
    >>> any(c.endswith('_level') for c in df.columns)
    True
    >>> df.shape[0]
    10
    """
    if name not in store:
        _LOG.warning("term_structure_features: %r not in store", name)
        return pd.DataFrame()

    panel = store.as_panel(name, dropna_threshold=1.0)
    if panel.empty:
        return pd.DataFrame()

    cols: dict[str, pd.Series] = {}

    expiries = list(dict.fromkeys(panel.columns.get_level_values("expiry")))
    for expiry in expiries:
        sub = panel.xs(expiry, axis=1, level="expiry")
        if sub.shape[1] < 2:
            _LOG.warning(
                "term_structure_features: expiry %s has < 2 maturities; skipping slope",
                expiry,
            )
            continue
        ordered = sorted(sub.columns, key=lambda m: MATURITY_RANK[m])
        short_m, long_m = ordered[0], ordered[-1]
        cols[f"{expiry}_slope"] = sub[long_m] - sub[short_m]

    maturities = list(dict.fromkeys(panel.columns.get_level_values("maturity")))
    for maturity in maturities:
        sub = panel.xs(maturity, axis=1, level="maturity")
        cols[f"{maturity}_level"] = sub.mean(axis=1)

    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols, index=panel.index)
