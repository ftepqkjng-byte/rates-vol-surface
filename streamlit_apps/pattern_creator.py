"""Streamlit app for hand-drawing sparse-PCA prior patterns.

Pick how many patterns you want, click cells in the (expiry, tenor)
grid to set them to ±1 (or any number), optionally smooth, preview as
a 3D surface, and save to a local pkl.

Run with::

    streamlit run streamlit_apps/pattern_creator.py

The saved pkl is a ``pd.DataFrame`` whose index is pattern names and
whose columns are a MultiIndex of ``(expiry, tenor)`` — exactly the
multi-factor prior shape expected by ``factors.sparse_pca_warm``.
Load and use::

    priors = pd.read_pickle("data/priors.pkl")
    sparse_pca_warm(rate_diff, priors, anchor=1.0)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter

# Make the project-root helpers importable when this app is launched from
# either the repo root or the streamlit_apps folder.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EXPIRY_LABELS, TENOR_LABELS  # noqa: E402
from pattern_basis import (  # noqa: E402
    degree_name,
    legendre_basis,
    preset_separable_poly,
)


def smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """2D Gaussian blur on the (expiry, tenor) grid. ``sigma=0`` is identity."""
    return gaussian_filter(arr, sigma=sigma) if sigma > 0 else arr.copy()


def plot_3d(arr: np.ndarray, title: str):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(np.arange(len(TENOR_LABELS)), np.arange(len(EXPIRY_LABELS)))
    amax = max(abs(float(arr.min())), abs(float(arr.max())), 1e-9)
    surf = ax.plot_surface(X, Y, arr, cmap="RdBu_r", edgecolor="grey",
                           linewidth=0.3, vmin=-amax, vmax=amax,
                           antialiased=True)
    ax.set_xticks(range(len(TENOR_LABELS)))
    ax.set_xticklabels(TENOR_LABELS, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(EXPIRY_LABELS)))
    ax.set_yticklabels(EXPIRY_LABELS, fontsize=8)
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry"); ax.set_zlabel("loading")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, pad=0.1)
    return fig


def to_series(arr: np.ndarray, name: str) -> pd.Series:
    """Convert a (n_expiry × n_tenor) array to a Series with the canonical
    MultiIndex — matches the column layout produced by ``pca.to_wide``."""
    return pd.Series(
        arr.ravel(),
        index=pd.MultiIndex.from_product(
            [EXPIRY_LABELS, TENOR_LABELS], names=["expiry", "tenor"]
        ),
        name=name,
    )


def blank_grid() -> pd.DataFrame:
    g = pd.DataFrame(0.0, index=EXPIRY_LABELS, columns=TENOR_LABELS)
    g.index.name = "expiry"
    g.columns.name = "tenor"
    return g


# --- Presets -----------------------------------------------------------------
# Each preset returns a list of {"name", "grid", "version"} dicts ready to
# slot into st.session_state.patterns. All cells in the preset region are
# filled with 1.0; the user can adjust signs / values after loading.

def _patterns_from_partition(
    exp_groups: list[tuple[str, list[str]]],
    ten_groups: list[tuple[str, list[str]]],
) -> list[dict]:
    """One pattern per (expiry group × tenor group). Each pattern is 1.0 on
    its block and 0 elsewhere."""
    out: list[dict] = []
    for e_name, exps in exp_groups:
        for t_name, tens in ten_groups:
            grid = blank_grid()
            grid.loc[exps, tens] = 1.0
            out.append({
                "name":    f"{e_name}|{t_name}",
                "grid":    grid,
                "version": 0,
            })
    return out


def preset_quadrants_2x2() -> list[dict]:
    """4 patterns — the (short/long expiry) × (short/long tenor) quadrants."""
    mid_e = len(EXPIRY_LABELS) // 2
    mid_t = len(TENOR_LABELS) // 2
    return _patterns_from_partition(
        [("short_exp", EXPIRY_LABELS[:mid_e]),
         ("long_exp",  EXPIRY_LABELS[mid_e:])],
        [("short_ten", TENOR_LABELS[:mid_t]),
         ("long_ten",  TENOR_LABELS[mid_t:])],
    )


def preset_blocks_3x3() -> list[dict]:
    """9 patterns — the canonical 3×3 partition shared with block PCA."""
    # Import lazily so the app still runs if factors.py is in flux.
    from factors import DEFAULT_EXPIRY_BLOCKS, DEFAULT_TENOR_BLOCKS
    return _patterns_from_partition(DEFAULT_EXPIRY_BLOCKS, DEFAULT_TENOR_BLOCKS)


def preset_diagonals_3() -> list[dict]:
    """3 patterns — anti-diagonal bands by ``expiry_rank + tenor_rank``.

    Band 1 is the top-left corner (short expiry × short tenor), band 3 is
    the bottom-right corner (long × long), band 2 is everything in between.
    """
    n_e, n_t = len(EXPIRY_LABELS), len(TENOR_LABELS)
    max_sum = (n_e - 1) + (n_t - 1)
    boundaries = [(0, max_sum / 3),
                  (max_sum / 3, 2 * max_sum / 3),
                  (2 * max_sum / 3, max_sum + 1)]  # +1 so the corner cell is included
    out: list[dict] = []
    for k, (lo, hi) in enumerate(boundaries):
        grid = blank_grid()
        for ei, e in enumerate(EXPIRY_LABELS):
            for ti, t in enumerate(TENOR_LABELS):
                if lo <= (ei + ti) < hi:
                    grid.loc[e, t] = 1.0
        out.append({
            "name":    f"diag_band_{k + 1}",
            "grid":    grid,
            "version": 0,
        })
    return out


POLY_PRESET_NAME = "Separable polynomial basis (Level/Slope/Curvature)"

PRESETS: dict[str, callable] = {
    "(none)":             None,
    "4 quadrants (2×2)":  preset_quadrants_2x2,
    "9 blocks (3×3)":     preset_blocks_3x3,
    "3 diagonal bands":   preset_diagonals_3,
    POLY_PRESET_NAME:     preset_separable_poly,
}


# --- App ---------------------------------------------------------------------
st.set_page_config(page_title="Sparse-PCA Prior Creator", layout="wide")
st.title("Sparse-PCA Prior Pattern Creator")
st.caption(
    "Build one or more loading patterns as warm-start priors for "
    "`factors.sparse_pca_warm`. Type values into the grid (typically `0`, "
    "`+1`, `-1`); the smoothed version is what gets saved and previewed."
)

with st.sidebar:
    st.header("Settings")

    # Preset loader — must render before the n_patterns widget so we can
    # update n_patterns_input session-state value before that widget
    # instantiates in the same run.
    preset_choice = st.selectbox(
        "Load preset", list(PRESETS.keys()),
        help="Replace current patterns with a preset. Block-based presets "
             "fill 1.0 on their region and 0 elsewhere; the separable "
             "polynomial preset fills each grid with a tensor-product "
             "Legendre basis (level/slope/curvature) — edit signs / "
             "values after loading.",
    )
    poly_kwargs: dict = {}
    if preset_choice == POLY_PRESET_NAME:
        poly_kwargs["max_degree_expiry"] = int(st.number_input(
            "Max degree (expiry)", min_value=0, max_value=3, value=1,
            step=1, key="poly_deg_e",
            help="Highest Legendre degree along the expiry axis. "
                 "0=level, 1=slope, 2=curvature.",
        ))
        poly_kwargs["max_degree_tenor"] = int(st.number_input(
            "Max degree (tenor)", min_value=0, max_value=3, value=1,
            step=1, key="poly_deg_t",
            help="Highest Legendre degree along the tenor axis. "
                 "0=level, 1=slope, 2=curvature.",
        ))
        poly_kwargs["include_level"] = st.checkbox(
            "Include (level × level) constant pattern",
            value=True, key="poly_include_level",
            help="The (0, 0) tensor product is the constant '1 everywhere' "
                 "pattern. Uncheck to keep only patterns that vary across "
                 "the cube.",
        )
        max_sum = poly_kwargs["max_degree_expiry"] + poly_kwargs["max_degree_tenor"]
        use_cutoff = st.checkbox(
            "Apply total-degree cutoff (i + j ≤ N)",
            value=False, key="poly_cutoff_on",
            help="Triangular truncation: drop patterns whose combined "
                 "degree exceeds the cutoff, so very high-order cross "
                 "terms like (2, 2) are skipped.",
        )
        if use_cutoff:
            poly_kwargs["degree_cutoff"] = int(st.number_input(
                "Max total degree (i + j)",
                min_value=0, max_value=max(max_sum, 1),
                value=min(max_sum, 2), step=1, key="poly_cutoff",
            ))
        n_planned = sum(
            1
            for i in range(poly_kwargs["max_degree_expiry"] + 1)
            for j in range(poly_kwargs["max_degree_tenor"] + 1)
            if (poly_kwargs.get("degree_cutoff") is None
                or i + j <= poly_kwargs["degree_cutoff"])
            and (poly_kwargs["include_level"] or not (i == 0 and j == 0))
        )
        st.caption(f"Current settings would generate **{n_planned}** pattern(s).")
    if st.button("Load preset", use_container_width=True):
        loader = PRESETS[preset_choice]
        if loader is not None:
            new_patterns = loader(**poly_kwargs)
            if not new_patterns:
                st.toast("Preset returned 0 patterns — relax the cutoff "
                         "or re-enable the constant pattern.", icon="⚠️")
            else:
                st.session_state.patterns = new_patterns
                st.session_state.n_patterns_input = len(new_patterns)
                st.session_state.epoch += 1
                st.toast(f"Loaded {len(new_patterns)} patterns from "
                         f"'{preset_choice}'.", icon="✅")

    n_patterns = int(st.number_input(
        "Number of patterns", min_value=1, max_value=16,
        value=1, step=1, key="n_patterns_input",
    ))
    sigma = float(st.slider("Smoothing sigma (Gaussian)", 0.0, 3.0, 0.0,
                            step=0.1,
                            help="Apply a 2D Gaussian blur to the raw "
                                 "0/±1 grid. 0 = hard boundary."))
    save_path = st.text_input("Save path", "data/priors.pkl")
    st.markdown("---")
    st.markdown(
        f"Grid is **{len(EXPIRY_LABELS)} expiries × {len(TENOR_LABELS)} "
        f"tenors** (from `config.py`)."
    )

if preset_choice == POLY_PRESET_NAME:
    max_e = int(st.session_state.get("poly_deg_e", 1))
    max_t = int(st.session_state.get("poly_deg_t", 1))
    exp_b = legendre_basis(len(EXPIRY_LABELS), max_e)
    ten_b = legendre_basis(len(TENOR_LABELS), max_t)
    fig_b, axes_b = plt.subplots(1, 2, figsize=(11, 2.6))
    for d, vec in enumerate(exp_b):
        axes_b[0].plot(range(len(EXPIRY_LABELS)), vec, marker="o",
                       label=degree_name(d), linewidth=1.4, markersize=3)
    axes_b[0].set_xticks(range(len(EXPIRY_LABELS)))
    axes_b[0].set_xticklabels(EXPIRY_LABELS, rotation=60, ha="right", fontsize=7)
    axes_b[0].axhline(0, color="grey", lw=0.5)
    axes_b[0].set_title("Expiry-axis Legendre basis")
    axes_b[0].legend(fontsize=8, loc="best")
    for d, vec in enumerate(ten_b):
        axes_b[1].plot(range(len(TENOR_LABELS)), vec, marker="o",
                       label=degree_name(d), linewidth=1.4, markersize=3)
    axes_b[1].set_xticks(range(len(TENOR_LABELS)))
    axes_b[1].set_xticklabels(TENOR_LABELS, rotation=60, ha="right", fontsize=7)
    axes_b[1].axhline(0, color="grey", lw=0.5)
    axes_b[1].set_title("Tenor-axis Legendre basis")
    axes_b[1].legend(fontsize=8, loc="best")
    fig_b.tight_layout()
    st.markdown("**1D basis preview** — what the preset will outer-product "
                "into each pattern. Reflects current sidebar degrees.")
    st.pyplot(fig_b)

if "patterns" not in st.session_state:
    st.session_state.patterns = []
# Bumped whenever patterns are structurally replaced (preset load). Threaded
# into per-tab widget keys so stale state from a prior layout — most visibly
# a previous tab's name — can't bleed through onto the new patterns.
if "epoch" not in st.session_state:
    st.session_state.epoch = 0

while len(st.session_state.patterns) < n_patterns:
    i = len(st.session_state.patterns) + 1
    st.session_state.patterns.append(
        {"name": f"pattern_{i}", "grid": blank_grid(), "version": 0}
    )
while len(st.session_state.patterns) > n_patterns:
    st.session_state.patterns.pop()
# Back-compat: older session_state entries may lack the "version" key.
for p in st.session_state.patterns:
    p.setdefault("version", 0)

tab_labels = [f"#{i + 1}: {p['name']}" for i, p in enumerate(st.session_state.patterns)]
tabs = st.tabs(tab_labels)

smoothed_patterns: list[tuple[str, np.ndarray]] = []
for i, tab in enumerate(tabs):
    with tab:
        pdict = st.session_state.patterns[i]

        name_col, _spacer = st.columns([1, 3])
        with name_col:
            pdict["name"] = st.text_input(
                "Pattern name", pdict["name"],
                key=f"name_{i}_e{st.session_state.epoch}",
            )

        # Bulk-fill: set every cell in (expiries × tenors) to one value.
        # Versioned editor key forces re-init after each fill so the data
        # editor reflects the update.
        with st.expander("Bulk fill — set many cells at once", expanded=True):
            bf_e, bf_t, bf_v, bf_btns = st.columns([3, 3, 1, 2])
            with bf_e:
                fill_expiries = st.multiselect(
                    "Expiries", EXPIRY_LABELS, default=EXPIRY_LABELS,
                    key=f"fe_{i}",
                )
            with bf_t:
                fill_tenors = st.multiselect(
                    "Tenors", TENOR_LABELS, default=[],
                    key=f"ft_{i}",
                )
            with bf_v:
                fill_value = st.number_input(
                    "Value", value=1.0, step=1.0, key=f"fv_{i}",
                )
            with bf_btns:
                st.write("")
                st.write("")
                apply_btn = st.button("Apply fill", key=f"fa_{i}",
                                      use_container_width=True)
                reset_btn = st.button("Reset to 0", key=f"fr_{i}",
                                      use_container_width=True)
            if apply_btn and fill_expiries and fill_tenors:
                pdict["grid"].loc[fill_expiries, fill_tenors] = float(fill_value)
                pdict["version"] += 1
                st.rerun()
            if reset_btn:
                pdict["grid"] = blank_grid()
                pdict["version"] += 1
                st.rerun()

        edit_col, plot_col = st.columns(2)
        with edit_col:
            st.markdown("**Raw grid** (rows = expiry, cols = tenor; "
                        "typical entries `0`, `+1`, `-1`)")
            pdict["grid"] = st.data_editor(
                pdict["grid"],
                key=f"editor_{i}_e{st.session_state.epoch}_v{pdict['version']}",
                use_container_width=True, height=600,
            )
        with plot_col:
            raw = pdict["grid"].reindex(
                index=EXPIRY_LABELS, columns=TENOR_LABELS
            ).fillna(0.0).values.astype(float)
            smoothed = smooth(raw, sigma)
            st.markdown(f"**Smoothed 3D surface** (sigma = {sigma:.1f})")
            st.pyplot(plot_3d(smoothed, pdict["name"]))

        smoothed_patterns.append((pdict["name"], smoothed))

st.markdown("---")
save_col, info_col = st.columns([1, 4])
with save_col:
    save_btn = st.button("Save patterns", type="primary",
                         use_container_width=True)
with info_col:
    if save_btn:
        out = pd.DataFrame(
            {name: to_series(arr, name) for name, arr in smoothed_patterns}
        ).T
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_pickle(path)
        st.success(f"Wrote {len(out)} pattern(s) to `{path}`.")
        st.caption(
            "Load with `pd.read_pickle(...)` and pass directly to "
            "`factors.sparse_pca_warm(rate_diff, priors, anchor=...)`."
        )
        st.dataframe(out, use_container_width=True)
