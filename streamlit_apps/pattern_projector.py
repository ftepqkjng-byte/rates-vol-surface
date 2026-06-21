"""Streamlit tool for the trading desk: project a surface move onto a
saved set of pattern weights, and (placeholder) read book exposures off
the same basis.

Run with::

    streamlit run streamlit_apps/pattern_projector.py

Inputs
------
* Weights pkl — a DataFrame whose rows are pattern names and whose
  columns are a MultiIndex of ``(expiry, tenor)`` matching the canonical
  universe in ``config.py``. Typically the loadings returned by
  ``factors.sparse_pca_warm`` after the desk's optimisation step.
* Surface pkl — long-format ``[date, expiry, tenor, value]`` as
  produced by ``mock_data.py`` (or the real-data equivalent in prod).

The projection coefficients are obtained by OLS — patterns from
``sparse_pca_warm`` are unit-norm but not orthogonal, so a plain dot
product would mis-attribute the move.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make the project-root helpers importable when this app is launched from
# either the repo root or the streamlit_apps folder.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EXPIRY_LABELS, TENOR_LABELS  # noqa: E402
from pca import load_long, to_wide  # noqa: E402


# --- Helpers ---------------------------------------------------------------
def to_grid(series: pd.Series) -> np.ndarray:
    """Reshape a ``(expiry, tenor)``-MultiIndex Series to a
    ``(n_expiry × n_tenor)`` array, NaN for any cell the Series omits."""
    g = pd.DataFrame(np.nan, index=EXPIRY_LABELS, columns=TENOR_LABELS,
                     dtype=float)
    for (e, t), v in series.items():
        if e in g.index and t in g.columns:
            g.loc[e, t] = float(v)
    return g.values


def heatmap(arr: np.ndarray, title: str, vmax: float):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", origin="upper")
    ax.set_xticks(range(len(TENOR_LABELS)))
    ax.set_xticklabels(TENOR_LABELS, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(EXPIRY_LABELS)))
    ax.set_yticklabels(EXPIRY_LABELS, fontsize=8)
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def project(weights: pd.DataFrame, delta: pd.Series):
    """OLS-project ``delta`` onto the rows of ``weights``.

    Returns ``(alpha, recon, resid, r2, common)`` — ``common`` is the
    MultiIndex of cells used (intersection of both sides, NaNs dropped).
    """
    delta = delta.dropna()
    common = weights.columns.intersection(delta.index)
    if len(common) == 0:
        raise ValueError("No overlap between weights columns and surface cells.")
    W = weights[common].values                            # (k, p)
    d = delta.loc[common].values.astype(float)            # (p,)
    alpha, *_ = np.linalg.lstsq(W.T, d, rcond=None)       # (k,)
    recon = W.T @ alpha
    resid = d - recon
    ss_tot = float(d @ d)
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return alpha, recon, resid, r2, common


def snap_to_index(d, idx: pd.DatetimeIndex) -> pd.Timestamp:
    """Snap a picked date to the nearest trading day in the index."""
    pos = idx.searchsorted(pd.Timestamp(d))
    if pos >= len(idx):
        pos = len(idx) - 1
    return idx[pos]


# --- App -------------------------------------------------------------------
st.set_page_config(page_title="Pattern Projector", layout="wide")
st.title("Pattern Projector")
st.caption(
    "Decompose a surface move between two dates into a saved pattern "
    "basis. Section 2 is a placeholder for reading book exposures off "
    "the same patterns."
)

with st.sidebar:
    st.header("Inputs")
    weights_path = st.text_input(
        "Weights pkl", "data/priors.pkl",
        help="rows = pattern names, columns = MultiIndex(expiry, tenor).",
    )
    surface_path = st.text_input(
        "Surface pkl", "data/mock/rate.pkl",
        help="Long-format [date, expiry, tenor, value].",
    )

    try:
        weights = pd.read_pickle(weights_path)
    except Exception as e:
        st.error(f"Failed to load weights: {e}")
        st.stop()
    if not isinstance(weights.columns, pd.MultiIndex):
        st.error("Weights columns must be a MultiIndex (expiry, tenor).")
        st.stop()
    st.success(f"{weights.shape[0]} pattern(s) × {weights.shape[1]} cells")

    try:
        wide = to_wide(load_long(surface_path))
    except Exception as e:
        st.error(f"Failed to load surface: {e}")
        st.stop()
    st.success(f"{len(wide)} date(s) × {wide.shape[1]} cells")

# --- Section 1: period projection -----------------------------------------
st.markdown("---")
st.subheader("1 — Project a period change onto the patterns")

idx = wide.index
min_d, max_d = idx.min().date(), idx.max().date()
d1_col, d2_col = st.columns(2)
with d1_col:
    d1 = st.date_input("Start date", value=min_d,
                       min_value=min_d, max_value=max_d)
with d2_col:
    d2 = st.date_input("End date", value=max_d,
                       min_value=min_d, max_value=max_d)

d1_ts = snap_to_index(d1, idx)
d2_ts = snap_to_index(d2, idx)
st.caption(f"Snapped to trading dates: **{d1_ts.date()} → {d2_ts.date()}**")

if d1_ts == d2_ts:
    st.warning("Start and end dates snap to the same trading day; Δ is zero.")
    st.stop()

delta = wide.loc[d2_ts] - wide.loc[d1_ts]
alpha, recon, resid, r2, common = project(weights, delta)
alpha_s = pd.Series(alpha, index=weights.index, name="exposure")

m1, m2, m3 = st.columns(3)
m1.metric("R² explained", f"{r2:.1%}")
m2.metric("‖Δ‖", f"{np.linalg.norm(delta.dropna().values):.4f}")
m3.metric("‖residual‖", f"{np.linalg.norm(resid):.4f}")

st.markdown("**Pattern exposures** — OLS projection coefficients α")
bar_col, tbl_col = st.columns([2, 1])
with bar_col:
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in alpha_s.values]
    ax.bar(alpha_s.index.astype(str), alpha_s.values, color=colors)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("α")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
with tbl_col:
    st.dataframe(alpha_s.to_frame().style.format("{:+.4f}"),
                 use_container_width=True)

delta_grid = to_grid(delta.loc[common])
recon_grid = to_grid(pd.Series(recon, index=common))
resid_grid = to_grid(pd.Series(resid, index=common))
vmax = float(np.nanmax(np.abs(np.concatenate(
    [delta_grid.ravel(), recon_grid.ravel(), resid_grid.ravel()]
))))
vmax = max(vmax, 1e-9)

c1, c2, c3 = st.columns(3)
with c1:
    st.pyplot(heatmap(delta_grid,
                      f"Actual Δ  ({d1_ts.date()} → {d2_ts.date()})", vmax))
with c2:
    st.pyplot(heatmap(recon_grid, "Reconstructed (Σ αᵢ·wᵢ)", vmax))
with c3:
    st.pyplot(heatmap(resid_grid, "Residual", vmax))

with st.expander("Per-pattern contribution surfaces (αᵢ · wᵢ)"):
    n_cols = min(3, len(alpha_s))
    cols = st.columns(n_cols)
    for i, (name, a) in enumerate(alpha_s.items()):
        contrib = pd.Series(
            a * weights.loc[name, common].values.astype(float),
            index=common,
        )
        with cols[i % n_cols]:
            st.pyplot(heatmap(to_grid(contrib),
                              f"{name}   (α = {a:+.4f})", vmax))

# --- Section 2: book exposure (placeholder) -------------------------------
st.markdown("---")
st.subheader("2 — Book exposure on the patterns  *(placeholder)*")
st.info(
    "**TODO** — define the book input format and the cell-risk → "
    "pattern-space projection.\n\n"
    "Likely flow: upload (or paste) a `(expiry, tenor) → risk` table "
    "(DV01 for rates, vega for vol), align it to the same MultiIndex as "
    "the weights, then project the risk vector onto the pattern basis "
    "the same way Section 1 projects a price move. Output: one number "
    "per pattern = the desk's exposure to that factor."
)
