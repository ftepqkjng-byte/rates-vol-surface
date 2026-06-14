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
python mock_data.py
```

## Layout

```
.
├── data/mock/        # 4 pkl tables: rate, atm_vol, skew_p2, skew_n2
├── notebooks/pca.ipynb
├── mock_data.py      # synthetic generator (regime-switching DGP)
├── pca.py            # PCA helpers: load_long, to_wide, run_pca, reconstruct
├── requirements.txt
├── README.md
└── CLAUDE.md         # project context for AI assistants
```

Each pkl is a long-format DataFrame with columns `[date, expiry, tenor, value]`.

## Research direction

Vanilla PCA on the cube has two problems for hedging use:

* PC1 absorbs the bulk of variance (level mode dominates) — leaving the
  remaining PCs to fight over residuals.
* Loadings are dense linear combinations of all `(expiry, tenor)` cells
  — hard to map to a tradeable hedge instrument.

The priority research tracks (in order):

1. **Varimax-rotated PCA** — rotate PC loadings so each factor is
   supported on a sparse region of the cube. Same cumulative variance,
   redistributed onto interpretable axes.
2. **Hierarchical PCA** — extract the level mode (cube mean) first,
   then run PCA on the residual. PC1 stops being a level absorber, and
   slope / butterfly / region-specific modes show up cleanly.
3. **Pre-defined bucket factors + residual PCA** — define a small set
   of economically meaningful regions (short-end level, 2s10s slope,
   vol skew level, etc.), regress them out, run PCA on the residual.
   This is the most directly hedge-mappable: each bucket corresponds to
   a vega / DV01 bucket trader already thinks in.
4. **Joint rate-vol structure** — once single-surface factors are
   characterised, check whether rate and vol cubes share latent
   drivers (CCA or lagged correlations).

**Out of scope for now**: parametric overlays (Nelson-Siegel / SABR),
tensor decomposition, autoencoders, regime detection.

**Comparison metrics** for every method: variance retained at K
factors, loading sparsity (Gini or `|loading| > threshold` count),
factor stability under rolling-window refit, and hedge replication
error (residual variance after K-factor reconstruction).
