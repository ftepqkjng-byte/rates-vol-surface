"""Tests for src.features.derived and src.features.surface_pca.

All test data is generated inline via make_mock_store().
"""

from __future__ import annotations

import numpy as np
import pytest

from src.datastore import RatesVolStore
from src.features.derived import curve_spreads
from src.features.surface_pca import SurfacePCA
from src.loaders.mock_loader import make_mock_store


@pytest.fixture(scope="module")
def store() -> RatesVolStore:
    return make_mock_store(n_days=300, seed=42)


# --------------------------------------------------------------- curve_spreads


def test_curve_spreads_2s10s(store: RatesVolStore) -> None:
    spreads = curve_spreads(store)
    panel = store.as_panel("rate", dropna_threshold=1.0)
    expected = panel[("6M", "10Y")] - panel[("6M", "2Y")]
    # Sign must match exactly: same operands, same direction.
    np.testing.assert_array_equal(
        np.sign(spreads["2s10s"].values), np.sign(expected.values)
    )
    # Upward-sloping mock curve -> majority of days are positive.
    assert (spreads["2s10s"] > 0).mean() > 0.8


def test_curve_spreads_missing_pair(store: RatesVolStore, caplog) -> None:
    bogus_pairs = [
        (("1Y", "5Y"), ("1Y", "10Y")),
        (("99Y", "99Y"), ("1Y", "10Y")),  # bogus, should be skipped
    ]
    with caplog.at_level("WARNING", logger="src.features.derived"):
        result = curve_spreads(store, pairs=bogus_pairs)
    assert "5s10s" in result.columns
    assert "99s10s" not in result.columns
    assert "skipping" in caplog.text.lower()


# ----------------------------------------------------------------- SurfacePCA


def test_surface_pca_explained_variance(store: RatesVolStore) -> None:
    # Rate panel: rates broadcast across expiries -> low effective rank,
    # strong cross-maturity correlation -> top 3 PCs dominate.
    panel = store.as_panel("rate", dropna_threshold=1.0)
    spca = SurfacePCA(n_components=3, standardize=True).fit(panel)
    assert spca.explained_variance_ratio_.sum() > 0.80


def test_surface_pca_reconstruct_approx(store: RatesVolStore) -> None:
    panel = store.as_panel("rate", dropna_threshold=1.0)
    n_comp = min(panel.shape[1], 20)
    spca = SurfacePCA(n_components=n_comp, standardize=True).fit(panel)
    scores = spca.transform(panel)
    recon = spca.reconstruct(scores)
    aligned = panel[spca.feature_cols_]
    rel_err = (aligned - recon).abs().mean().mean() / aligned.abs().mean().mean()
    assert rel_err < 0.05


def test_surface_pca_fit_transform_shape(store: RatesVolStore) -> None:
    panel = store.as_panel("atm_vol", dropna_threshold=1.0)
    n_comp = 5
    spca = SurfacePCA(n_components=n_comp).fit(panel)
    scores = spca.transform(panel)
    assert scores.shape == (panel.shape[0], n_comp)
    assert list(scores.columns) == [f"PC{i + 1}" for i in range(n_comp)]


def test_loading_heatmap_data_shape(store: RatesVolStore) -> None:
    panel = store.as_panel("atm_vol", dropna_threshold=1.0)
    spca = SurfacePCA(n_components=3).fit(panel)
    hm = spca.loading_heatmap_data("PC1")
    assert hm.index.name == "expiry"
    assert hm.columns.name == "maturity"
    assert len(hm.index) > 1
    assert len(hm.columns) > 1


def test_incremental_transform_matches_batch(store: RatesVolStore) -> None:
    panel = store.as_panel("rate", dropna_threshold=1.0)
    spca = SurfacePCA(n_components=5, standardize=True).fit(panel)
    idx = 100
    row = panel.iloc[idx]
    batch = spca.transform(panel.iloc[[idx]]).iloc[0]
    inc = spca.incremental_transform(row)
    np.testing.assert_allclose(inc.values, batch.values, atol=1e-8)


def test_rolling_transform_shape(store: RatesVolStore) -> None:
    panel = store.as_panel("rate", dropna_threshold=1.0)
    n_comp = 3
    spca = SurfacePCA(n_components=n_comp, standardize=True).fit(panel)
    window = 100
    rolling = spca.rolling_transform(panel, window=window, step=20)

    # One row per qualifying date (window-1 .. end).
    assert len(rolling) == len(panel) - window + 1
    assert len(rolling) < len(panel)

    pc_cols = [c for c in rolling.columns if c.startswith("PC")]
    assert len(pc_cols) == n_comp
    assert "refit_date" in rolling.columns


def test_compare_windows_sorted(store: RatesVolStore) -> None:
    panel = store.as_panel("rate", dropna_threshold=1.0)
    spca = SurfacePCA(n_components=3, standardize=True).fit(panel)

    mid = panel.index[len(panel) // 2]
    window_a = (str(panel.index[0].date()), str(mid.date()))
    window_b = (str(mid.date()), str(panel.index[-1].date()))

    diff_df = spca.compare_windows(panel, window_a, window_b)
    assert list(diff_df.columns) == [
        "expiry", "maturity", "loading_a", "loading_b", "diff", "abs_diff"
    ]
    diffs = diff_df["abs_diff"].values
    assert (diffs[:-1] >= diffs[1:]).all()
