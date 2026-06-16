"""Tiny PCA helper for the long-format pkl tables under data/mock/.

Each pkl is a DataFrame with columns ``[date, expiry, tenor, value]``.

* ``expiry`` — forward-starting time of the option / forward.
* ``tenor``  — length of the underlying rate.

``load_long`` filters whatever labels arrive in the pkl down to the
canonical ``EXPIRY_LABELS`` / ``TENOR_LABELS`` sets and (for older pkls)
renames a ``maturity`` column to ``tenor``. ``to_wide`` pivots to a
``(date) × (expiry, tenor)`` panel; ``run_pca`` runs sklearn's PCA.
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import EXPIRY_LABELS, TENOR_LABELS, EXPIRY_RANK, TENOR_RANK

_EXPIRY_RANK = EXPIRY_RANK
_TENOR_RANK = TENOR_RANK


def load_long(path: str) -> pd.DataFrame:
    """Load a long-format pkl, normalise the column name, and filter to
    the canonical expiry / tenor label sets.

    Accepts pkls that use either ``tenor`` (preferred) or ``maturity`` as
    the rate-length column name; the legacy name is renamed on load.
    """
    df = pd.read_pickle(path)
    df["date"] = pd.to_datetime(df["date"])
    if "maturity" in df.columns and "tenor" not in df.columns:
        df = df.rename(columns={"maturity": "tenor"})

    df = df[df["expiry"].isin(EXPIRY_LABELS) & df["tenor"].isin(TENOR_LABELS)]
    return df.reset_index(drop=True)


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long ``[date, expiry, tenor, value]`` to a wide panel
    with columns sorted by canonical (expiry, tenor) rank."""
    wide = long_df.pivot(index="date", columns=["expiry", "tenor"], values="value")
    sorted_cols = sorted(
        wide.columns, key=lambda c: (_EXPIRY_RANK[c[0]], _TENOR_RANK[c[1]])
    )
    wide = wide.reindex(
        columns=pd.MultiIndex.from_tuples(sorted_cols, names=["expiry", "tenor"])
    )
    return wide.sort_index()


def run_pca(
    wide: pd.DataFrame,
    n_components: int = 5,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run PCA on a wide panel; returns (scores, loadings, explained_variance_ratio).

    Columns with any NaN are dropped before fitting.
    """
    X = wide.dropna(axis=1)
    if standardize:
        scaler = StandardScaler()
        arr = scaler.fit_transform(X.values)
    else:
        arr = X.values

    pca = PCA(n_components=n_components)
    scores_arr = pca.fit_transform(arr)
    pc_names = [f"PC{i + 1}" for i in range(pca.n_components_)]

    scores = pd.DataFrame(scores_arr, index=X.index, columns=pc_names)
    loadings = pd.DataFrame(pca.components_, index=pc_names, columns=X.columns)
    explained = pd.Series(pca.explained_variance_ratio_, index=pc_names)
    return scores, loadings, explained


def reconstruct(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    wide: pd.DataFrame,
    n_components: int | None = None,
) -> pd.DataFrame:
    """Reconstruct the wide panel from PC scores.

    Assumes the PCA was fit on ``wide`` with ``standardize=True``.
    ``n_components`` truncates to the first ``k`` PCs (e.g. ``k=1`` for
    a level-only approximation). Defaults to all available PCs.
    """
    k = scores.shape[1] if n_components is None else min(n_components, scores.shape[1])
    cols = loadings.columns
    scaled_recon = scores.iloc[:, :k].values @ loadings.iloc[:k].values
    means = wide[cols].mean().values
    stds = wide[cols].std().values
    return pd.DataFrame(
        scaled_recon * stds + means, index=scores.index, columns=cols
    )
