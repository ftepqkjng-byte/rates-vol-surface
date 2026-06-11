"""PCA decomposition of rate and vol panels.

The :class:`SurfacePCA` class wraps an sklearn ``PCA`` so callers always work
with named ``(expiry, maturity)`` columns and ``PC{i}``-named scores rather
than raw numpy arrays. Missing data are handled at two layers: columns with
too many NaNs are dropped at fit time, and surviving holes are imputed with
fit-time column means at transform time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..schema import EXPIRY_RANK, MATURITY_RANK

if TYPE_CHECKING:
    from ..datastore import RatesVolStore


class SurfacePCA:
    """PCA decomposition of a wide rate or vol panel.

    Inputs come from :meth:`RatesVolStore.as_panel` — rows are dates, columns
    are an ``(expiry, maturity)`` MultiIndex. After :meth:`fit`, the standard
    sklearn attributes are exposed as labelled pandas objects:

    Attributes
    ----------
    components_ : pd.DataFrame
        Shape ``(n_components, n_features)``. Rows ``PC1..PCk``; columns are
        the surviving ``(expiry, maturity)`` MultiIndex.
    explained_variance_ratio_ : pd.Series
        Per-PC variance ratio.
    feature_cols_ : list[tuple[str, str]]
        ``(expiry, maturity)`` pairs that survived the NaN filter.
    loadings_ : pd.DataFrame
        ``components_`` scaled by ``sqrt(explained_variance_)`` — the
        classical factor-analysis loadings, interpretable (under
        ``standardize=True``) as the correlation between each PC and each
        standardized input feature.
    """

    def __init__(
        self,
        n_components: int = 5,
        dropna_threshold: float = 0.05,
        standardize: bool = True,
    ) -> None:
        self.n_components = n_components
        self.dropna_threshold = dropna_threshold
        self.standardize = standardize

        self.feature_cols_: list[tuple[str, str]] = []
        self.column_means_: pd.Series | None = None
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.components_: pd.DataFrame | None = None
        self.explained_variance_ratio_: pd.Series | None = None
        self.loadings_: pd.DataFrame | None = None
        self._col_index: pd.MultiIndex | None = None

    # ------------------------------------------------------------------ helpers

    def _check_fitted(self) -> None:
        if self.pca_ is None:
            raise RuntimeError("SurfacePCA has not been fit yet")

    def _prepare(self, panel: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        if fitting:
            nan_frac = panel.isna().mean(axis=0)
            keep = nan_frac <= self.dropna_threshold
            sub = panel.loc[:, keep]
            self.feature_cols_ = [tuple(c) for c in sub.columns]
            self._col_index = pd.MultiIndex.from_tuples(
                self.feature_cols_, names=["expiry", "maturity"]
            )
            self.column_means_ = sub.mean(axis=0)
            return sub.fillna(self.column_means_)

        self._check_fitted()
        sub = panel.reindex(columns=self._col_index)
        return sub.fillna(self.column_means_)

    # ----------------------------------------------------------------- public

    def fit(self, panel: pd.DataFrame) -> "SurfacePCA":
        X = self._prepare(panel, fitting=True)
        if X.empty or X.shape[1] == 0:
            raise ValueError("No feature columns survived the NaN-fraction filter")

        if self.standardize:
            self.scaler_ = StandardScaler()
            X_arr = self.scaler_.fit_transform(X.values)
        else:
            self.scaler_ = None
            X_arr = X.values

        n_comp = min(self.n_components, X_arr.shape[0], X_arr.shape[1])
        self.pca_ = PCA(n_components=n_comp)
        self.pca_.fit(X_arr)

        pc_names = [f"PC{i + 1}" for i in range(n_comp)]
        self.components_ = pd.DataFrame(
            self.pca_.components_, index=pc_names, columns=self._col_index
        )
        self.explained_variance_ratio_ = pd.Series(
            self.pca_.explained_variance_ratio_,
            index=pc_names,
            name="explained_variance_ratio",
        )
        sqrt_var = np.sqrt(self.pca_.explained_variance_)
        self.loadings_ = self.components_.mul(sqrt_var, axis=0)
        return self

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X = self._prepare(panel, fitting=False)
        X_arr = X.values
        if self.scaler_ is not None:
            X_arr = self.scaler_.transform(X_arr)
        scores = self.pca_.transform(X_arr)
        return pd.DataFrame(scores, index=X.index, columns=self.components_.index)

    def fit_transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        return self.fit(panel).transform(panel)

    def reconstruct(
        self, scores: pd.DataFrame, n_components: int | None = None
    ) -> pd.DataFrame:
        """Inverse-transform PC scores back to the original surface units.

        ``n_components`` truncates to the first ``k`` PCs for partial
        reconstruction (e.g. ``k=2`` for a level+slope approximation). Defaults
        to all fitted components.
        """
        self._check_fitted()
        total = self.components_.shape[0]
        n_use = total if n_components is None else min(n_components, total)

        s_arr = scores.iloc[:, :n_use].values
        comp_arr = self.pca_.components_[:n_use]
        X_centered = s_arr @ comp_arr + self.pca_.mean_
        if self.scaler_ is not None:
            X_arr = self.scaler_.inverse_transform(X_centered)
        else:
            X_arr = X_centered
        return pd.DataFrame(X_arr, index=scores.index, columns=self._col_index)

    def explained_variance_plot_data(self) -> pd.DataFrame:
        """Return ``(pc, explained, cumulative)`` rows for a scree-style plot."""
        if self.explained_variance_ratio_ is None:
            return pd.DataFrame(columns=["pc", "explained", "cumulative"])
        return pd.DataFrame(
            {
                "pc": list(self.explained_variance_ratio_.index),
                "explained": self.explained_variance_ratio_.values,
                "cumulative": self.explained_variance_ratio_.cumsum().values,
            }
        )

    def loading_heatmap_data(self, pc: str = "PC1") -> pd.DataFrame:
        """Return loadings for ``pc`` as an expiry×maturity grid.

        Rows are sorted by canonical ``EXPIRY_RANK``, columns by
        ``MATURITY_RANK``. ``(expiry, maturity)`` combinations not present in
        ``feature_cols_`` become NaN. Values are signed loadings; take
        ``.abs()`` on the result for a magnitude view.
        """
        if self.loadings_ is None:
            return pd.DataFrame()
        if pc not in self.loadings_.index:
            raise KeyError(
                f"{pc!r} not in loadings (available: {list(self.loadings_.index)})"
            )
        row = self.loadings_.loc[pc]
        hm = row.unstack("maturity")
        present_exp = sorted(set(hm.index), key=EXPIRY_RANK.get)
        present_mat = sorted(set(hm.columns), key=MATURITY_RANK.get)
        return hm.reindex(index=present_exp, columns=present_mat)


def joint_pca(
    store: "RatesVolStore",
    rate_weight: float = 0.5,
    n_components: int = 8,
) -> tuple[SurfacePCA, pd.DataFrame]:
    """Fit independent PCAs on the rate and ATM-vol panels and join scores by date.

    ``n_components`` is split between the two surfaces (``n_components // 2``
    for rate, the remainder for vol). Each surface is standardized
    independently before its own PCA, so both contribute on equal footing.

    Returns ``(rate_pca, scores)``. ``scores`` has columns
    ``["rate_PC1", ..., "rate_PCk", "vol_PC1", ..., "vol_PCm"]`` indexed by
    the dates common to both panels (inner join). The vol PCA is attached to
    the returned rate PCA as ``.vol_pca_`` so its loadings remain inspectable.

    ``rate_weight`` is accepted for API stability but currently ignored —
    independent standardization makes per-surface weighting moot. The
    parameter is reserved for a future shared-eigenproblem formulation.
    """
    if "rate" not in store or "atm_vol" not in store:
        raise ValueError("joint_pca requires both 'rate' and 'atm_vol' tables")

    n_rate = max(n_components // 2, 1)
    n_vol = max(n_components - n_rate, 1)

    rate_panel = store.as_panel("rate", dropna_threshold=1.0)
    vol_panel = store.as_panel("atm_vol", dropna_threshold=1.0)

    rate_pca = SurfacePCA(n_components=n_rate, standardize=True).fit(rate_panel)
    vol_pca = SurfacePCA(n_components=n_vol, standardize=True).fit(vol_panel)

    rate_scores = rate_pca.transform(rate_panel)
    vol_scores = vol_pca.transform(vol_panel)
    rate_scores.columns = [f"rate_{c}" for c in rate_scores.columns]
    vol_scores.columns = [f"vol_{c}" for c in vol_scores.columns]

    combined = rate_scores.join(vol_scores, how="inner")
    rate_pca.vol_pca_ = vol_pca  # type: ignore[attr-defined]
    return rate_pca, combined
