"""Separable Legendre polynomial bases for the pattern creator.

Builds tensor-product loading patterns on the ``(expiry, tenor)`` grid:
each pattern is the outer product of a 1D Legendre basis vector on the
expiry axis and one on the tenor axis. The 1D bases are evaluated at
equally spaced points in ``[-1, 1]`` and normalised so their max absolute
value equals 1, matching the magnitude scale of the existing block-based
``±1`` presets.

Kept separate from ``streamlit_apps/pattern_creator.py`` so the pure
numerical core has no streamlit dependency.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from numpy.polynomial import Legendre

from config import EXPIRY_LABELS, TENOR_LABELS


_DEGREE_NAMES = {0: "level", 1: "slope", 2: "curvature"}


def degree_name(deg: int) -> str:
    """``0 -> "level"``, ``1 -> "slope"``, ``2 -> "curvature"``, else ``"degN"``."""
    return _DEGREE_NAMES.get(deg, f"deg{deg}")


def tensor_product_name(
    i: int,
    j: int,
    first_axis: str = "exp",
    second_axis: str = "ten",
) -> str:
    """Canonical name for an ``(i, j)`` tensor-product pattern.

    Returns ``f"{first_axis}_{degree_name(i)}_x_{second_axis}_{degree_name(j)}"``,
    e.g. ``"exp_level_x_ten_slope"``. Shared by the Legendre preset and
    by ``factors.separable.marginal_eigen_patterns`` so both basis
    families use the same cell labels in any downstream comparison.
    """
    return f"{first_axis}_{degree_name(i)}_x_{second_axis}_{degree_name(j)}"


def degree_pair_grid(
    max_degree_first: int,
    max_degree_second: int,
    include_zero_zero: bool = True,
    total_degree_cutoff: int | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield ``(i, j)`` degree pairs for a tensor-product basis on the
    rectangle ``[0..max_degree_first] x [0..max_degree_second]``.

    Two filters can prune the rectangle:

    * ``include_zero_zero=False`` skips the constant ``(0, 0)`` pair.
    * ``total_degree_cutoff=k`` keeps only pairs with ``i + j <= k``
      (triangular truncation — common for cross-term-light bases).

    Iteration order is first-axis-then-second-axis. Shared by the
    Legendre preset and ``factors.separable.marginal_eigen_patterns``
    so each pruning rule lives in one place.
    """
    for i in range(max_degree_first + 1):
        for j in range(max_degree_second + 1):
            if total_degree_cutoff is not None and i + j > total_degree_cutoff:
                continue
            if not include_zero_zero and i == 0 and j == 0:
                continue
            yield (i, j)


def patterns_to_prior_df(patterns: list[dict]) -> pd.DataFrame:
    """Flatten the list-of-dicts pattern format into the row-per-factor
    DataFrame that ``factors.sparse_pca_warm`` (and the other anchored
    factor helpers) accept as a multi-factor prior.

    Each entry's ``grid`` (an ``(expiry × tenor)`` DataFrame) is ravelled
    into a Series indexed by the canonical ``(expiry, tenor)``
    MultiIndex produced by ``pca.to_wide``. Stacked rows form the
    returned DataFrame, ready to drop straight into
    ``sparse_pca_warm(wide, prior=...)``.
    """
    series_map: dict[str, pd.Series] = {}
    for p in patterns:
        grid = p["grid"]
        idx = pd.MultiIndex.from_product(
            [grid.index.tolist(), grid.columns.tolist()],
            names=["expiry", "tenor"],
        )
        series_map[p["name"]] = pd.Series(
            grid.values.ravel(), index=idx, name=p["name"]
        )
    return pd.DataFrame(series_map).T


def legendre_basis(
    n_points: int,
    max_degree: int,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """1D Legendre basis sampled along ``[-1, 1]``.

    Returns an array of shape ``(max_degree + 1, n_points)``. Each row is
    one degree, normalised so ``max(|row|) == 1`` (degree-0 stays as the
    constant 1).

    ``positions`` lets a caller supply pre-mapped coordinates in
    ``[-1, 1]`` (e.g. derived from year fractions); when ``None`` the
    points are equally spaced.
    """
    if positions is None:
        x = np.linspace(-1.0, 1.0, n_points)
    else:
        x = np.asarray(positions, dtype=float)
        if x.shape != (n_points,):
            raise ValueError(f"positions shape {x.shape} != ({n_points},)")
    out = np.empty((max_degree + 1, n_points))
    for d in range(max_degree + 1):
        vec = Legendre.basis(d)(x)
        amax = max(abs(float(vec.min())), abs(float(vec.max())))
        out[d] = vec / amax if amax > 0 else vec
    return out


def preset_separable_poly(
    max_degree_expiry: int = 1,
    max_degree_tenor: int = 1,
    include_level: bool = True,
    degree_cutoff: int | None = None,
) -> list[dict]:
    """Tensor-product Legendre patterns on the ``(expiry, tenor)`` grid.

    Iterates over ``(i, j)`` with ``i in 0..max_degree_expiry`` and
    ``j in 0..max_degree_tenor``, building an outer-product grid of the
    two 1D bases (see :func:`legendre_basis`) and naming it
    ``f"exp_{degree_name(i)}_x_ten_{degree_name(j)}"``.

    Two filters can prune the rectangular grid of patterns:

    * ``include_level`` — drop the pure constant ``(0, 0)`` pattern when
      ``False`` (default ``True`` keeps it).
    * ``degree_cutoff`` — if set, keep only patterns whose combined
      degree ``i + j`` is at most the cutoff (triangular truncation).
      With ``max_degree_expiry=max_degree_tenor=2`` and
      ``degree_cutoff=3`` the high-cross-term ``(2, 2)`` is dropped,
      leaving 8 of the 9 rectangular patterns. ``None`` (default) keeps
      the full rectangle.

    Sampling fallback: ``config.py`` does not currently expose a numeric
    year-fraction map for ``EXPIRY_LABELS`` / ``TENOR_LABELS``, so the
    Legendre polynomials are evaluated at uniform rank positions (each
    label gets equal weight along ``[-1, 1]``). If a year-fraction map
    is added later, pass it in via the ``positions`` argument of
    :func:`legendre_basis` and lift it into this function the same way.

    Returns a list of ``{"name", "grid", "version"}`` dicts matching the
    shape produced by the existing block-based presets. Can be empty if
    the filters exclude everything (e.g. ``include_level=False`` with
    ``degree_cutoff=0``).
    """
    exp_basis = legendre_basis(len(EXPIRY_LABELS), max_degree_expiry)
    ten_basis = legendre_basis(len(TENOR_LABELS), max_degree_tenor)
    out: list[dict] = []
    for i, j in degree_pair_grid(
        max_degree_expiry, max_degree_tenor,
        include_zero_zero=include_level, total_degree_cutoff=degree_cutoff,
    ):
        grid_arr = np.outer(exp_basis[i], ten_basis[j])
        grid = pd.DataFrame(
            grid_arr, index=EXPIRY_LABELS, columns=TENOR_LABELS
        )
        grid.index.name = "expiry"
        grid.columns.name = "tenor"
        out.append({
            "name": tensor_product_name(i, j),
            "grid": grid, "version": 0,
        })
    return out
