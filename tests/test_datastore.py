"""Tests for src.datastore.RatesVolStore.

All test data is generated inline via make_mock_store().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datastore import RatesVolStore
from src.loaders.mock_loader import make_mock_store
from src.schema import EXPIRY_RANK, MATURITY_RANK


@pytest.fixture(scope="module")
def store() -> RatesVolStore:
    return make_mock_store(n_days=120, seed=0)


def test_register_validates_index_names(store: RatesVolStore) -> None:
    series = store.get("rate").copy()
    series.index = series.index.set_names(["bad", "expiry", "maturity"])
    new_store = RatesVolStore()
    with pytest.raises(ValueError, match="index level names"):
        new_store.register("rate", series)


def test_register_sorts_by_schema_order(store: RatesVolStore) -> None:
    series = store.get("rate")
    shuffled = series.sample(frac=1.0, random_state=0)
    assert not shuffled.index.equals(series.index)  # really scrambled

    new_store = RatesVolStore()
    new_store.register("rate", shuffled)
    panel = new_store.as_panel("rate", dropna_threshold=1.0)
    ranks = [(EXPIRY_RANK[e], MATURITY_RANK[m]) for e, m in panel.columns]
    assert ranks == sorted(ranks)


def test_get_filters_by_expiry(store: RatesVolStore) -> None:
    result = store.get("rate", expiry="1Y")
    assert set(result.index.get_level_values("expiry").unique()) == {"1Y"}


def test_get_filters_by_date_range(store: RatesVolStore) -> None:
    all_dates = store.get("rate").index.get_level_values("date").unique().sort_values()
    start, end = all_dates[10], all_dates[20]
    result = store.get("rate", start=start, end=end)
    res_dates = result.index.get_level_values("date").unique()
    assert res_dates.min() == start
    assert res_dates.max() == end
    assert len(res_dates) == 11  # inclusive both ends


def test_as_panel_shape(store: RatesVolStore) -> None:
    panel = store.as_panel("rate", dropna_threshold=1.0)
    pairs = store.available_pairs("rate")
    assert panel.shape[1] == len(pairs)


def test_align_inner_intersection(store: RatesVolStore) -> None:
    a, b = store.align("rate", "atm_vol", how="inner")
    assert list(a.columns) == list(b.columns)
    assert list(a.index) == list(b.index)


def test_compute_derived_skew_spread(store: RatesVolStore) -> None:
    p2 = store.get("skew_p2")
    n2 = store.get("skew_n2")
    spread = store.get("skew_spread")

    common = spread.index.intersection(p2.index).intersection(n2.index)
    expected = (p2.loc[common] - n2.loc[common]).astype("float64")
    np.testing.assert_allclose(spread.loc[common].values, expected.values, atol=1e-10)


def test_save_load_roundtrip(store: RatesVolStore, tmp_path) -> None:
    path = tmp_path / "store.pkl"
    store.save(str(path))
    loaded = RatesVolStore.from_disk(str(path))
    pd.testing.assert_frame_equal(loaded.summary(), store.summary())


def test_summary_completeness(store: RatesVolStore) -> None:
    summary = store.summary()
    expected_cols = [
        "table_name",
        "n_pairs",
        "date_start",
        "date_end",
        "n_obs",
        "n_missing_pct",
    ]
    assert list(summary.columns) == expected_cols
    assert set(summary["table_name"]) == {
        "rate",
        "atm_vol",
        "skew_p2",
        "skew_n2",
        "skew_spread",
        "skew_mid",
    }
    assert len(summary) == 6
