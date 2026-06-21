"""Varimax-style orthogonal rotation of PCA loadings for per-factor sparsity.

Rotation is orthogonal so total variance is preserved — useful when the
vanilla PCs are dense but you want loadings concentrated on a few cube
cells per factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def varimax(
    loadings: pd.DataFrame,
    gamma: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> pd.DataFrame:
    """Kaiser varimax rotation. Input ``loadings`` is (k, p) — rows are PCs,
    columns the cube cells. Returns rotated loadings with the same shape /
    index / columns. Rotation is orthogonal so total variance is preserved.
    """
    L = loadings.values.T          # (p, k)
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        Lambda = L @ R
        B = Lambda ** 3 - (gamma / p) * Lambda @ np.diag(
            (Lambda ** 2).sum(axis=0)
        )
        U, S, Vt = np.linalg.svd(L.T @ B, full_matrices=False)
        R = U @ Vt
        d = S.sum()
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            break
    rotated = (L @ R).T            # (k, p)
    return pd.DataFrame(rotated, index=loadings.index, columns=loadings.columns)


def rotate_scores(scores: pd.DataFrame, loadings: pd.DataFrame,
                  rotated_loadings: pd.DataFrame) -> pd.DataFrame:
    """Apply the same rotation to the score series so the score basis
    matches the rotated loadings. Solves ``rotated = R.T @ loadings`` for
    ``R`` and right-multiplies the scores by ``R``.
    """
    R = np.linalg.lstsq(loadings.values.T, rotated_loadings.values.T, rcond=None)[0]
    rotated = scores.values @ R
    return pd.DataFrame(rotated, index=scores.index, columns=rotated_loadings.index)
