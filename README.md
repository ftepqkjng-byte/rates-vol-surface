# rates-vol-surface

PCA / factor research on a rate-and-vol cube indexed by `(expiry, tenor)`.
The goal is to decompose daily cube moves into **a few interpretable
factors** that can serve as hedging reference (factor exposures →
tradeable hedge instruments).

* `expiry` — forward-starting time of the option / forward
* `tenor`  — length of the underlying rate

Local research runs entirely on the four mock pkls under `data/mock/`.
The company environment has a separate script that pulls real data and
emits pkls in the **exact same shape**, so the analysis code is
identical across environments — just drop in different pkls.

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook notebooks/pca.ipynb
```

Regenerate the mock pkls:

```bash
python data/mock_data.py
```

Hand-draw sparse-PCA priors in a Streamlit app:

```bash
streamlit run streamlit_apps/pattern_creator.py
```

## Layout

```
.
├── data/
│   ├── mock/                       # raw + derived pkls per surface
│   │   ├── {name}.pkl              # raw long-format
│   │   ├── {name}_diff.pkl         # daily diff (wide)
│   │   └── {name}_residual.pkl     # diff with std-weighted parallel shift stripped
│   ├── mock_data.py                # synthetic generator + auto-runs pipeline
│   └── pipeline.py                 # builds *_diff.pkl and *_residual.pkl from raw
├── notebooks/
│   ├── pca.ipynb                   # PCA on parallel-shift-stripped residual panels
│   ├── factors.ipynb               # extensions: varimax / block PCA / regression / CCA on diffs
│   ├── sparse_pca.ipynb            # warm-started sparse PCA on hand-drawn priors
│   ├── soft_constrained_pca.ipynb  # joint-ALS variant on the same priors
│   └── pattern_projection.ipynb    # decompose surface moves onto hand-drawn patterns
├── streamlit_apps/
│   ├── pattern_creator.py          # hand-draw sparse-PCA priors, save to pkl
│   └── pattern_projector.py        # decompose a period's surface move onto a saved pattern basis
├── config.py                       # canonical (expiry, tenor) universe — single source of truth
├── pca.py                          # PCA helpers: load_long, to_wide, run_pca, reconstruct
├── factors/                        # extensions split by family (rotation, blocks,
│                                   #   sparse, regression, cca, metrics);
│                                   #   __init__.py re-exports everything
├── requirements.txt
├── README.md
└── CLAUDE.md                       # project context for AI assistants
```

Each pkl is a long-format DataFrame with columns `[date, expiry, tenor, value]`.

## Research direction

All extension work is on **daily diffs** of the panels — we hedge
moves, not levels. Vanilla PCA on the diff cube has two problems for
hedging use:

* Dominant PCs are dense linear combinations of all `(expiry, tenor)`
  cells — hard to map to a tradeable hedge instrument.
* Local structure (e.g. short-expiry × short-tenor moves) gets averaged
  into global factors and is lost.

The priority research tracks (in order):

1. **Varimax-rotated PCA** — rotate PC loadings so each factor is
   supported on a sparse region of the cube. Same cumulative variance,
   redistributed onto interpretable axes.
2. **Block PCA** — partition the cube into a grid of disjoint blocks
   (start with 3×3: short/mid/long expiry × short/mid/long tenor), fit
   PCA inside each block independently, then stitch per-block
   reconstructions back into the full grid. Each block's factors are
   structurally local — they only load on cells inside that block,
   which makes them directly hedge-mappable. Grid size is the only
   tuning knob; finer grids → more factors but smaller per-block residual.
3. **Sparse PCA with warm start** — initialise loadings at an
   artificial pattern (the desk's hand-drawn factor — e.g. a 2s10s
   steepener mask) and iterate `sparse_pca_warm` with a Tikhonov
   anchor on `||w − w_prior||²` so the result fits the market better
   while staying recognisable. Single `anchor` knob trades fit vs.
   prior similarity; optional `l1` adds explicit sparsification.
4. **Regression on bucket / block factors** — generic OLS interface
   (`factors.regress`) for testing any factor design against any target
   panel. First use is "cube cells ~ block PCs"; same call handles
   varimax PCs, hand-built spreads, lagged factors, or any future
   pattern. Returns betas, fitted, residuals, and per-target R² in one
   dict.
5. **Joint rate-vol structure** — once single-surface factors are
   characterised, check whether rate and vol cubes share latent
   drivers (CCA on stacked block scores, or lagged correlation between
   the corresponding rate and vol blocks).

**Out of scope for now**: parametric overlays (Nelson-Siegel / SABR),
tensor decomposition, autoencoders, regime detection.

**Comparison metrics** for every method: variance retained at K
factors, loading sparsity (Gini), factor stability under rolling-window
refit, and hedge replication error (residual variance after K-factor
reconstruction).
