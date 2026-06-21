"""Joint cross-surface structure between two score panels.

* ``cross_surface_cca`` — CCA between two surfaces' scores (typically
  rate PCs vs vol PCs, or rate block scores vs vol block scores).
* ``lagged_corr`` — quick sanity-check cross-correlation at integer
  lags before reaching for the full CCA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA


def cross_surface_cca(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    n_components: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Canonical correlation between two score panels (any pair —
    typically rate PCs vs vol PCs, or rate block scores vs vol block
    scores). Returns (canonical_a, canonical_b, canonical_correlations).
    """
    common = scores_a.index.intersection(scores_b.index)
    A = scores_a.loc[common].values
    B = scores_b.loc[common].values
    k = min(n_components, A.shape[1], B.shape[1])
    cca = CCA(n_components=k, max_iter=1000)
    Ac, Bc = cca.fit_transform(A, B)
    names = [f"CC{i + 1}" for i in range(k)]
    canonical_a = pd.DataFrame(Ac, index=common, columns=names)
    canonical_b = pd.DataFrame(Bc, index=common, columns=names)
    corrs = pd.Series(
        [np.corrcoef(Ac[:, i], Bc[:, i])[0, 1] for i in range(k)],
        index=names, name="canonical_corr",
    )
    return canonical_a, canonical_b, corrs


def lagged_corr(
    a: pd.Series,
    b: pd.Series,
    lags: range = range(-5, 6),
) -> pd.Series:
    """Cross-correlation between ``a`` and ``b`` at integer lags. Positive
    lag = ``a`` leads ``b``. Useful as a quick check before full CCA.
    """
    common = a.index.intersection(b.index)
    a = a.loc[common]; b = b.loc[common]
    return pd.Series(
        {k: a.corr(b.shift(-k)) for k in lags}, name="lagged_corr"
    )
