"""RatesVolStore: central in-memory container for rate / vol panels.

Each table is a ``pd.Series`` with a three-level MultiIndex
(``date``, ``expiry``, ``maturity``) and ``float64`` values; the Series name
is the table name (e.g. ``"rate"``).
"""

from __future__ import annotations

import pickle

import pandas as pd

from .schema import (
    DATA_TYPES,
    EXPIRY_RANK,
    MATURITY_RANK,
    sort_pairs,
)

_INDEX_NAMES = ["date", "expiry", "maturity"]


class RatesVolStore:
    """In-memory store for up to six named (date, expiry, maturity) -> float panels."""

    def __init__(self) -> None:
        self._tables: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------ helpers

    def _require(self, name: str) -> pd.Series:
        if name not in self._tables:
            raise KeyError(f"Table {name!r} is not loaded")
        return self._tables[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tables

    def __iter__(self):
        return iter(self._tables)

    # --------------------------------------------------- loading / registration

    def register(self, name: str, series: pd.Series) -> None:
        if name not in DATA_TYPES:
            raise ValueError(f"Unknown table name {name!r}; expected one of {DATA_TYPES}")
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name}: expected pd.Series, got {type(series).__name__}")
        if list(series.index.names) != _INDEX_NAMES:
            raise ValueError(
                f"{name}: index level names must be {_INDEX_NAMES}, "
                f"got {list(series.index.names)}"
            )
        if series.dtype != "float64":
            raise TypeError(f"{name}: values must be float64, got {series.dtype}")
        if series.index.has_duplicates:
            raise ValueError(f"{name}: duplicate index entries are not allowed")

        expiries = series.index.get_level_values("expiry")
        maturities = series.index.get_level_values("maturity")
        bad_e = set(expiries) - set(EXPIRY_RANK)
        bad_m = set(maturities) - set(MATURITY_RANK)
        if bad_e:
            raise ValueError(f"{name}: unknown expiry labels {sorted(bad_e)}")
        if bad_m:
            raise ValueError(f"{name}: unknown maturity labels {sorted(bad_m)}")

        # Sort by (date, expiry_rank, maturity_rank).
        sort_frame = pd.DataFrame(
            {
                "d": series.index.get_level_values("date"),
                "e": expiries.map(EXPIRY_RANK),
                "m": maturities.map(MATURITY_RANK),
            }
        )
        order = sort_frame.sort_values(["d", "e", "m"], kind="stable").index.to_numpy()
        sorted_series = series.iloc[order].copy()
        sorted_series.name = name
        self._tables[name] = sorted_series

    def load_raw(
        self,
        name: str,
        df_raw: pd.DataFrame,
        date_col: str,
        expiry_col: str,
        maturity_col: str,
        value_col: str,
    ) -> None:
        df = df_raw.loc[:, [date_col, expiry_col, maturity_col, value_col]].copy()
        df = df.rename(
            columns={
                date_col: "date",
                expiry_col: "expiry",
                maturity_col: "maturity",
                value_col: "value",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df["expiry"] = df["expiry"].astype(str)
        df["maturity"] = df["maturity"].astype(str)
        df["value"] = df["value"].astype("float64")
        series = df.set_index(_INDEX_NAMES)["value"]
        self.register(name, series)

    # --------------------------------------------------------------- retrieval

    def get(
        self,
        name: str,
        expiry: str | list[str] | None = None,
        maturity: str | list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        series = self._require(name)
        if start is not None:
            series = series[series.index.get_level_values("date") >= pd.Timestamp(start)]
        if end is not None:
            series = series[series.index.get_level_values("date") <= pd.Timestamp(end)]
        if expiry is not None:
            exp_list = [expiry] if isinstance(expiry, str) else list(expiry)
            series = series[series.index.get_level_values("expiry").isin(exp_list)]
        if maturity is not None:
            mat_list = [maturity] if isinstance(maturity, str) else list(maturity)
            series = series[series.index.get_level_values("maturity").isin(mat_list)]
        return series

    def as_panel(self, name: str, dropna_threshold: float = 0.0) -> pd.DataFrame:
        series = self._require(name)
        panel = series.unstack(["expiry", "maturity"])
        cols_sorted = sort_pairs(list(panel.columns))
        panel = panel.reindex(
            columns=pd.MultiIndex.from_tuples(cols_sorted, names=["expiry", "maturity"])
        )
        if len(panel.columns) > 0:
            nan_frac = panel.isna().mean(axis=0)
            panel = panel.loc[:, nan_frac <= dropna_threshold]
        return panel

    def available_pairs(self, name: str) -> list[tuple[str, str]]:
        series = self._require(name)
        pair_idx = series.index.droplevel("date").unique()
        return sort_pairs([tuple(p) for p in pair_idx])

    def date_range(self, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        series = self._require(name)
        dates = series.index.get_level_values("date")
        return dates.min(), dates.max()

    # ---------------------------------------------------------------- alignment

    def align(
        self, name_a: str, name_b: str, how: str = "inner"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if how not in ("inner", "outer"):
            raise ValueError(f"how must be 'inner' or 'outer', got {how!r}")
        # Pass threshold=1.0 so alignment isn't pre-pruned by column NaN counts.
        panel_a = self.as_panel(name_a, dropna_threshold=1.0)
        panel_b = self.as_panel(name_b, dropna_threshold=1.0)

        if how == "inner":
            dates = panel_a.index.intersection(panel_b.index)
            cols = panel_a.columns.intersection(panel_b.columns)
        else:
            dates = panel_a.index.union(panel_b.index)
            cols = panel_a.columns.union(panel_b.columns)

        cols_sorted = pd.MultiIndex.from_tuples(
            sort_pairs(list(cols)), names=["expiry", "maturity"]
        )
        return (
            panel_a.reindex(index=dates, columns=cols_sorted),
            panel_b.reindex(index=dates, columns=cols_sorted),
        )

    # ----------------------------------------------------------------- derived

    def compute_derived(self) -> None:
        if "skew_p2" not in self._tables or "skew_n2" not in self._tables:
            raise ValueError(
                "compute_derived requires both skew_p2 and skew_n2 to be loaded"
            )

        p2 = self._tables["skew_p2"]
        n2 = self._tables["skew_n2"]
        common = p2.index.intersection(n2.index)
        p2c = p2.reindex(common)
        n2c = n2.reindex(common)
        valid = p2c.notna() & n2c.notna()
        p2c = p2c[valid]
        n2c = n2c[valid]

        spread = (p2c - n2c).astype("float64")
        mid = ((p2c + n2c) / 2.0).astype("float64")
        self.register("skew_spread", spread)
        self.register("skew_mid", mid)

    # ------------------------------------------------------------- persistence

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_disk(cls, path: str) -> "RatesVolStore":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected RatesVolStore on disk, got {type(obj).__name__}")
        return obj

    def summary(self) -> pd.DataFrame:
        rows = []
        # Iterate in canonical DATA_TYPES order so the summary is deterministic.
        for name in DATA_TYPES:
            if name not in self._tables:
                continue
            series = self._tables[name]
            dates = series.index.get_level_values("date")
            n_pairs = len(self.available_pairs(name))
            n_dates = int(dates.nunique())
            n_obs = int(len(series))
            expected = n_dates * n_pairs
            n_present = int(series.notna().sum())
            n_missing = max(expected - n_present, 0)
            n_missing_pct = (100.0 * n_missing / expected) if expected else 0.0
            rows.append(
                {
                    "table_name": name,
                    "n_pairs": n_pairs,
                    "date_start": dates.min(),
                    "date_end": dates.max(),
                    "n_obs": n_obs,
                    "n_missing_pct": n_missing_pct,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "table_name",
                "n_pairs",
                "date_start",
                "date_end",
                "n_obs",
                "n_missing_pct",
            ],
        )
