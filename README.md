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

## What the project does

Per surface (`rate`, `atm_vol`, `skew_p2`, `skew_n2`) the pipeline is:

1. Load long-format pkl `[date, expiry, tenor, value]` and pivot to a
   wide `date × (expiry, tenor)` panel.
2. Diff to daily moves (`*_diff.pkl`). Optionally subtract a
   realised-std-weighted cross-sectional parallel shift to leave a
   leakage-free residual panel (`*_residual.pkl`).
3. Fit a factor model on the diff/residual panel — vanilla PCA,
   varimax-rotated PCA, block PCA on a 3×3 partition, sparse PCA
   warm-started from a hand-drawn prior, or a joint anchor +
   decorrelation constrained PCA.
4. Score every variant on the same comparison sheet: variance
   retained at K, loading sparsity (Gini), rolling-window stability,
   hedge-replication residual.

The two Streamlit apps round-trip a hand-drawn prior basis:
`pattern_creator.py` to author / version it (block presets, plus a
separable Legendre level / slope / curvature preset), and
`pattern_projector.py` for the desk to project a realised surface
move onto a saved basis.

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook notebooks/pca.ipynb
```

Regenerate the mock pkls (also auto-builds the `_diff` / `_residual`
pkls via `data/pipeline.py`):

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
│   │   ├── {name}.pkl              # raw long-format [date, expiry, tenor, value]
│   │   ├── {name}_diff.pkl         # same long format — daily diff
│   │   └── {name}_residual.pkl     # same long format — diff with parallel shift stripped
│   ├── mock_data.py                # synthetic generator + auto-runs pipeline
│   └── pipeline.py                 # builds *_diff.pkl and *_residual.pkl from raw
├── notebooks/
│   ├── pca.ipynb                   # PCA on parallel-shift-stripped residual panels
│   ├── factors.ipynb               # varimax / block PCA / regression / CCA on diffs
│   ├── sparse_pca.ipynb            # warm-started sparse PCA on hand-drawn priors
│   ├── soft_constrained_pca.ipynb  # joint-ALS anchored variant
│   ├── decorr_pca.ipynb            # autograd anchor + explicit score-decorrelation
│   ├── separable_factors.ipynb     # Kronecker-marginal + functional (roughness-penalised) PCA
│   ├── residual_diagnostics.ipynb  # audit of strip_parallel_shift defaults
│   └── pattern_projection.ipynb    # decompose surface moves onto hand-drawn patterns
├── streamlit_apps/
│   ├── pattern_creator.py          # hand-draw sparse-PCA priors, save to pkl
│   └── pattern_projector.py        # project a period's surface move onto a saved basis
├── config.py                       # canonical (expiry, tenor) universe — single source of truth
├── pca.py                          # PCA helpers: load_long, to_wide, run_pca, reconstruct
├── pattern_basis.py                # separable Legendre basis for pattern_creator presets
├── factors/                        # extensions split by family (rotation, blocks,
│                                   #   sparse, regression, cca, separable, metrics);
│                                   #   __init__.py re-exports everything
├── tests/                          # pytest suite (pipeline + separable invariants)
├── requirements.txt
├── README.md
└── CLAUDE.md                       # project context for AI assistants
```

Each raw pkl is a long-format DataFrame with columns
`[date, expiry, tenor, value]`.

## Why not just vanilla PCA on diffs

Two problems for hedging use:

* Dominant PCs are dense linear combinations of all `(expiry, tenor)`
  cells — hard to map to a tradeable hedge instrument.
* Local structure (e.g. short-expiry × short-tenor moves) gets averaged
  into global factors and is lost.

Every track below is one way to push back on one or both.

## Current progress

The seven tracks below are all implemented end-to-end, each with a
driving notebook:

1. **Vanilla PCA on residuals** — `notebooks/pca.ipynb`. Baseline.
   Confirms the dominant PC is the parallel-shift direction, which is
   why subsequent work fits on the parallel-shift-stripped residual
   (or on diffs, where the comparison is more interpretable).
2. **Varimax + block PCA** — `notebooks/factors.ipynb`. `factors.varimax`
   rotates the PCA loadings; `factors.block_pca` runs PCA inside each
   cell of a `(expiry-block × tenor-block)` partition (3×3 default,
   shared with `pattern_creator`). Block factors are structurally
   local — they only load inside one block — and so map directly to a
   hedge bucket. `factors.regress` and `factors.stack_block_scores`
   support cube-cell ~ block-PC regressions.
3. **Sparse PCA, warm-started** — `notebooks/sparse_pca.ipynb`.
   `factors.sparse_pca_warm(wide, prior, anchor, l1)` initialises a
   loading at the desk's hand-drawn pattern and iterates so it fits
   the market without drifting away from the prior. Anchor / L1 sweeps
   give a fit-vs-recognisability trade-off curve.
4. **Constrained PCA variants** — two takes on the same problem of
   keeping multiple anchored factors from collapsing onto each other:
   * `factors.soft_constrained_pca` + `lambda_search`
     (`notebooks/soft_constrained_pca.ipynb`) — joint-ALS with a
     `||V − V0||²` anchor. Works for the single-factor case; with
     multiple priors the fitted loadings end up highly correlated
     even at moderate `lam`, which motivated the next item.
   * `factors.decorr_constrained_pca` (`notebooks/decorr_pca.ipynb`) —
     PyTorch autograd fit, anchor + an explicit `Corr(F)` penalty on
     the score matrix using an oblique projection
     `F = X V (VᵀV)⁻¹`. Side-by-side comparison vs.
     soft-constrained and an orthogonal-Procrustes baseline
     (`factors.procrustes_pca_baseline`); 2D `(λ_anchor, λ_decorr)`
     Pareto sweep.
5. **Pattern projection / desk tooling** —
   `notebooks/pattern_projection.ipynb` plus the two Streamlit apps.
   `factors.project_onto_patterns` cross-sectionally decomposes a
   single surface move onto a fixed pattern basis (OLS / ridge /
   NNLS / lasso) and reports the `pattern_corr` vs `exposure_corr`
   diagnostic so multicollinearity is visible.
6. **Separable / functional factor models** —
   `notebooks/separable_factors.ipynb`. `factors.marginal_kronecker_cov`
   and `factors.kronecker_cov_mle` estimate per-axis marginals under
   a Kronecker covariance assumption (with
   `kronecker_separability_residual` flagging when that assumption
   doesn't hold); `factors.marginal_eigen_patterns` packages the
   resulting top eigenvectors into a `sparse_pca_warm`-ready prior.
   `factors.functional_pca` solves `eigh(Σ̂ - λ·P)` with a 2D
   second-difference roughness penalty for smoothness-biased
   loadings; the notebook chooses `λ` against a rolling-stability
   sweep.
7. **Book-vega pattern hedging** — `notebooks/pattern_hedging.ipynb`,
   `factors/hedging.py`. Given a book's vega and each cell's
   regression beta on a fixed set of already-trained pattern scores
   (`factors.regress(targets=diff, factors=pattern_scores)`),
   `factors.book_pattern_exposure` aggregates the book's exposure to
   each pattern:
   $$\text{Book}_k = \sum_i \text{vega}_i \cdot \beta_{i,k}.$$
   `factors.sparse_hedge` then finds the smallest-notional hedge at a
   liquid subset of grid points
   (`config.LIQUID_EXPIRY_LABELS × config.LIQUID_TENOR_LABELS`, via
   `factors.liquid_hedge_candidates`) that brings every pattern's
   residual exposure back within tolerance:
   $$\min_{\alpha} \sum_i \text{cost}_i \, |\alpha_i|
   \quad \text{s.t.} \quad
   \left| \text{Book}_k - \sum_i \alpha_i \, \beta_{i,k} \right| \le \varepsilon_k
   \ \ \forall k.$$
   This is solved as a linear program (`scipy.optimize.linprog`): the
   L1 objective and the absolute-value constraint are both linearised
   by splitting each position into non-negative parts,
   $\alpha_i = \alpha_i^+ - \alpha_i^-$ with
   $|\alpha_i| = \alpha_i^+ + \alpha_i^-$, turning both into linear
   expressions in $(\alpha^+, \alpha^-) \ge 0$. `cost_i` defaults to
   `1.0` (pure notional minimisation) until a real price/liquidity
   vector exists. `epsilon` defaults to a per-pattern tolerance from
   the diagonal of the pattern-score covariance,
   `factors.pattern_epsilon`:
   $$\varepsilon_k = z \cdot \sqrt{\mathrm{Var}(F_k)},$$
   deliberately diagonal-only (ignoring cross-pattern covariance) so
   the whole problem stays a pure LP rather than a quadratically
   constrained one.

**Comparison metrics** (`factors.metrics_table`) — variance retained
at K, loading sparsity (Gini), rolling-window stability, hedge
replication residual — are wired into the notebooks for all of the
above.

## Next steps

Roughly in priority order:

1. **Repeat the full sheet on real company pkls.** Swap the four mock
   pkls for the API-pulled ones (same shape), regenerate the
   comparison sheet, and check whether the mock-derived calls
   (3×3 is the right block grid, decorr beats soft-constrained at
   moderate λ, etc.) still hold.
2. **Joint rate-vol structure as its own notebook.** `factors/cca.py`
   already exposes `cross_surface_cca` and `lagged_corr`; what's
   missing is a dedicated notebook driving CCA on stacked block
   scores across the rate and vol surfaces and reporting the leading
   canonical pairs.
3. **Curated prior library.** Promote the ad-hoc priors used in the
   sparse / constrained notebooks into a versioned `data/priors/`
   set (parallel, slope, butterfly, 2s10s steepener, vol-cone-style
   skews) and benchmark each as a `sparse_pca_warm` warm start
   against data-only PCA.
4. **Hedge-replication backtest beyond R².** Replay daily moves
   against a fixed instrument set (e.g. swaptions on the block
   corners) and report PnL-replication residual, not just
   reconstruction R² on the cube.

**Out of scope for now**: parametric overlays (Nelson-Siegel / SABR),
tensor decomposition (Tucker / CP), autoencoders, regime detection,
real Bloomberg ingestion in this repo, production deployment.
