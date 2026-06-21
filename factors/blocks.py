"""Block PCA — partition the (expiry, tenor) grid into disjoint blocks,
fit PCA inside each block independently, stitch per-block reconstructions
back into the full grid.

Each block's factors are structurally local to a region of the cube, so
they map directly to a tradeable hedge instrument. Grid size is the only
tuning knob (finer ⇒ more factors but smaller per-block residual).
"""

from __future__ import annotations

import pandas as pd

from pca import run_pca


# Default 3x3 partition. Edit (or pass your own) for finer / coarser grids.
DEFAULT_EXPIRY_BLOCKS: list[tuple[str, list[str]]] = [
    ("short_exp", ["1M", "2M", "3M", "6M", "9M"]),
    ("mid_exp",   ["1Y", "2Y", "3Y", "4Y", "5Y"]),
    ("long_exp",  ["7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]),
]
DEFAULT_TENOR_BLOCKS: list[tuple[str, list[str]]] = [
    ("short_ten", ["1Y", "2Y", "3Y"]),
    ("mid_ten",   ["4Y", "5Y", "7Y", "10Y"]),
    ("long_ten",  ["12Y", "15Y", "20Y", "25Y", "30Y"]),
]


def make_blocks(
    expiry_groups: list[tuple[str, list[str]]] = DEFAULT_EXPIRY_BLOCKS,
    tenor_groups:  list[tuple[str, list[str]]] = DEFAULT_TENOR_BLOCKS,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Cartesian product of expiry × tenor groups. Returns a dict mapping
    ``(expiry_block_name, tenor_block_name)`` to the list of
    ``(expiry, tenor)`` cells in that block. Default 3×3 grid gives 9
    blocks covering the canonical cube.
    """
    return {
        (e_name, t_name): [(e, t) for e in e_list for t in t_list]
        for e_name, e_list in expiry_groups
        for t_name, t_list in tenor_groups
    }


def block_pca(
    wide: pd.DataFrame,
    blocks: dict[tuple[str, str], list[tuple[str, str]]] | None = None,
    n_components: int = 2,
    standardize: bool = True,
) -> dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]]:
    """Run PCA independently inside each block of the partition.

    Returns ``{block_key: {"scores": s, "loadings": L, "explained": e}}``.
    Each block's PCA is fit only on the cells in that block, so factors
    are local to that region of the cube. Per-block component count is
    capped at the block's column count.
    """
    if blocks is None:
        blocks = make_blocks()
    out: dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]] = {}
    for key, cells in blocks.items():
        cols = [c for c in cells if c in wide.columns]
        if not cols:
            continue
        sub = wide[cols].dropna(axis=1)
        if sub.shape[1] == 0:
            continue
        k = min(n_components, sub.shape[1])
        s, L, e = run_pca(sub, n_components=k, standardize=standardize)
        out[key] = {"scores": s, "loadings": L, "explained": e}
    return out


def reconstruct_block(
    wide: pd.DataFrame,
    block_results: dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]],
    n_components: int | None = None,
) -> pd.DataFrame:
    """Per-block reconstruction stitched back into the full grid.

    ``n_components`` truncates each block to its first ``k`` factors
    (same ``k`` across blocks). ``None`` uses every fitted PC. The result
    has the same row index as the per-block scores and only the columns
    that were actually fit (cells outside the partition are dropped).
    """
    parts = []
    for res in block_results.values():
        s, L = res["scores"], res["loadings"]
        k = s.shape[1] if n_components is None else min(n_components, s.shape[1])
        cols = L.columns
        scaled = s.iloc[:, :k].values @ L.iloc[:k].values
        means = wide[cols].mean().values
        stds = wide[cols].std().values
        parts.append(pd.DataFrame(scaled * stds + means,
                                  index=s.index, columns=cols))
    return pd.concat(parts, axis=1)


def stack_block_scores(
    block_results: dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]],
    top_k: int | None = None,
) -> pd.DataFrame:
    """Flatten per-block scores into one (date × factor) DataFrame.
    Column names are ``{expiry_block}|{tenor_block}|PC{i}``. Convenient
    for feeding the block factors into a downstream model (e.g. CCA
    against vol-surface factors).
    """
    parts = []
    for (e_name, t_name), res in block_results.items():
        s = res["scores"]
        k = s.shape[1] if top_k is None else min(top_k, s.shape[1])
        sub = s.iloc[:, :k].copy()
        sub.columns = [f"{e_name}|{t_name}|{c}" for c in sub.columns]
        parts.append(sub)
    return pd.concat(parts, axis=1)


def block_summary(
    block_results: dict[tuple[str, str], dict[str, pd.DataFrame | pd.Series]],
) -> pd.DataFrame:
    """One row per block: shape and cumulative variance at each retained
    PC. Use to decide where the partition is too coarse (PC1 dominant) or
    too fine (variance spread thinly across local PCs)."""
    rows = []
    for (e_name, t_name), res in block_results.items():
        e = res["explained"]
        row = {
            "expiry_block": e_name,
            "tenor_block":  t_name,
            "n_cells":      res["loadings"].shape[1],
            "n_pcs":        len(e),
        }
        for i in range(len(e)):
            row[f"cum_var_at_{i + 1}"] = float(e.iloc[: i + 1].sum())
        rows.append(row)
    return pd.DataFrame(rows)
