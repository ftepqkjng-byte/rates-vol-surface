# rates-vol-surface

## 1. Project Overview

A research toolkit for jointly modelling the USD interest-rate curve and the
swaption ATM-vol / skew surface. The pipeline turns daily panels into PCA
factors, fits a Hidden Markov Model to detect curve / vol regimes, and emits
forward-looking scenario probabilities (for example, "what is the probability
of a bull-flattening regime over the next twenty-one business days?"). The
intended consumer is exotic-rates trading and risk: regime probabilities feed
hedging decisions and event-study analytics around macro releases.

## 2. Repository Structure

```
rates-vol-surface/
├── configs/
│   └── schema.yaml             # canonical expiry / maturity labels, data-type registry
├── data/
│   ├── mock/                   # synthetic data for local dev (gitkept)
│   └── bloomberg/              # raw Bloomberg CSV exports (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb     # surfaces, time series, coverage, per-pair stats
│   ├── 02_pca_analysis.ipynb         # SurfacePCA loadings, scree, joint-PCA scores
│   ├── 03_regime_detection.ipynb     # HMM fit, regime characterisation, Viterbi overlay
│   └── 04_scenario_prediction.ipynb  # scenario probabilities, calibration plots
├── src/
│   ├── schema.py               # ordered label lists + EXPIRY_RANK / MATURITY_RANK lookups
│   ├── datastore.py            # RatesVolStore — central typed container for all panels
│   ├── loaders/
│   │   ├── mock_loader.py      # make_mock_store: joint OU dynamics for offline use
│   │   └── bloomberg_loader.py # CSV / ticker parsing for raw Bloomberg pulls
│   ├── features/
│   │   ├── derived.py          # curve spreads, vol-surface metrics, term-structure features
│   │   └── surface_pca.py      # SurfacePCA + joint_pca on rate × vol panels
│   ├── models/
│   │   ├── regime_hmm.py       # RateRegimeHMM — Gaussian HMM with scenario_probability
│   │   └── scenario_clf.py     # scenario-probability classifier (planned)
│   └── utils/
│       └── plotting.py         # shared chart styling (planned)
├── tests/
│   ├── test_datastore.py       # store contract: register, get, align, persistence
│   └── test_features.py        # curve_spreads, SurfacePCA invariants
├── .gitignore
├── requirements.txt
└── README.md
```

## 3. Quickstart

```bash
# 1. Create environment
conda create -n rates-vol python=3.11
conda activate rates-vol

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the test suite (uses only mock data, fast)
pytest tests/ -v

# 4. Launch the exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

The exploration notebook runs end-to-end on mock data — no Bloomberg files
required for local experimentation.

## 4. Data Sources

The project supports two distinct environments, both feeding the same
`RatesVolStore` contract.

**Local development.** Two loaders are available:

- `make_mock_store(n_days, seed)` builds a fully populated store with joint
  OU dynamics for rates, ATM vol, and skew. Reproducible via `seed`; the
  default of `n_days=500` is the right scale for fitting PCA / HMM.
- `bloomberg_loader.load_bloomberg_csv(path, name)` ingests wide-format CSV
  exports (dates in the first column, Bloomberg tickers as the remaining
  columns). Parsing covers the common `USSV{XX}{YY}`, `USSV{N}M{YY}`, and
  `USSO{YY}` ticker conventions; ambiguous date formats are auto-detected.
  Place files under `data/bloomberg/` (gitignored).

**Bank environment.** Production data lands via an internal market-data API.
The intended workflow is:

1. Pull raw frames from the API and reshape them in any convenient form
   (long, wide, mixed). They do not need to match the canonical schema.
2. Call `store.load_raw(name, df, date_col, expiry_col, maturity_col,
   value_col)` once per table; this normalises columns, validates the labels
   against `schema.yaml`, and sorts the index canonically.
3. `store.compute_derived()` to materialise `skew_spread` and `skew_mid`.
4. `store.save("snapshot.pkl")` to persist. All downstream notebooks /
   models read the pickle so production and research diverge only at this
   handoff point.

## 5. Module Reference

- **`src/schema.py`** loads `configs/schema.yaml` at import time and exposes
  the canonical label lists (`EXPIRY_ORDER`, `MATURITY_ORDER`), the
  positional rank dicts used everywhere as a sort key, the data-type
  registry, and `sort_pairs()` for ordering `(expiry, maturity)` tuples.

- **`src/datastore.py`** defines `RatesVolStore`, the central container. A
  store holds up to six named `pd.Series` keyed by a `(date, expiry,
  maturity)` MultiIndex with `float64` values. Primary methods: `register`
  (validate + sort + store), `load_raw` (reshape an arbitrary frame),
  `get` (filtered slice), `as_panel` (wide pivot), `align` (joint
  rate × vol panel), `compute_derived` (`skew_spread`, `skew_mid`),
  `save` / `from_disk`, `summary`.

- **`src/loaders/mock_loader.py`** — `make_mock_store` simulates correlated
  OU rates (3-4.5%), an ATM-vol surface with a hump around the 1Y expiry,
  and a small positive skew spread. Vol innovations are negatively coupled
  to rate innovations (per-pair ρ ≈ -0.30).

- **`src/loaders/bloomberg_loader.py`** — `parse_bbg_ticker` returns
  `(expiry, maturity)` for the supported formats or `None`; `load_bloomberg_csv`
  reshapes a wide CSV into the canonical Series and prints a parse summary;
  `load_bloomberg_directory` concatenates multiple files with later-file-wins
  deduplication. Includes a `__main__` block that previews a CSV's ticker
  coverage without loading it.

- **`src/features/derived.py`** — `curve_spreads` computes 2s10s / 2s5s /
  5s30s / 1s10s in bps with automatic unit detection (decimal / percent /
  bps); `vol_surface_metrics` summarises level, slope, hump, and skew
  level / slope per date; `term_structure_features` exposes per-expiry
  slopes and per-maturity levels.

- **`src/features/surface_pca.py`** — `SurfacePCA` wraps sklearn's PCA so
  callers work with named `(expiry, maturity)` columns and `PC{i}` scores.
  Handles NaNs (drop column above threshold; impute residual with fit-time
  means) and exposes `components_`, `explained_variance_ratio_`, `loadings_`,
  plot-ready helpers. `joint_pca` fits independent PCAs on rate and vol and
  joins the scores by date.

- **`src/models/regime_hmm.py`** — `RateRegimeHMM` wraps `hmmlearn.GaussianHMM`.
  Stores Viterbi states, posterior probabilities, the transition matrix, and
  per-regime feature means as named pandas objects. `scenario_probability`
  propagates each date's posterior forward by `T^h` and sums the requested
  regime columns; `label_regimes` attaches economic names idempotently.

- **`src/models/scenario_clf.py`** (planned) — supervised classifier mapping
  current features and regime posteriors to discrete forward-looking
  scenarios, with calibration and inference entry points.

- **`src/utils/plotting.py`** (planned) — shared styling so the notebooks
  and the eventual reporting layer agree on colours, axis labels, and
  regime-overlay conventions.

## 6. Adding New Data

To bring in a new table (for example, `skew_p4` for the +4 strike):

1. Add the label to `configs/schema.yaml`:

   ```yaml
   data_types:
     - rate
     - atm_vol
     - skew_p2
     - skew_n2
     - skew_p4          # new
     - skew_spread
     - skew_mid

   raw_data_types:
     - rate
     - atm_vol
     - skew_p2
     - skew_n2
     - skew_p4          # new
   ```

2. Load it like any other raw table:

   ```python
   store.load_raw("skew_p4", df, date_col="DT", expiry_col="EXP",
                  maturity_col="MAT", value_col="vol")
   ```

3. If the new table participates in a derived feature, add the recipe to
   `datastore.compute_derived` (or to `features/derived.py` if the feature
   is a curve / surface statistic rather than a per-pair transform). Adding
   `skew_p4 - skew_p2` as an extra wing-asymmetry table, for instance,
   would go in `compute_derived` and would also need a `data_types` entry.

4. If the new table needs its own loader (different file format or API),
   add a module under `src/loaders/` that returns a `pd.Series` in standard
   `(date, expiry, maturity)` form and call `store.register(...)`.

## 7. Known Limitations

- **Mock data is intentionally smooth.** The synthetic OU dynamics in
  `mock_loader` produce stationary, fat-tail-free series — real rates exhibit
  jumps around macro releases, regime switches in volatility, and
  state-dependent correlation. Treat mock results as plumbing tests, not as
  evidence about the underlying model's economic plausibility.
- **`bloomberg_loader.parse_bbg_ticker` covers a limited dictionary.** Only
  USD swaption-vol (`USSV...`) and swap-rate (`USSO...`) tickers are parsed;
  payer / receiver swaption (`USSP...`, `USRC...`), other currencies,
  caps / floors, and the legacy short-month encoding (`USSV0310`) need to be
  added as their conventions appear.
- **HMM regime labels are unsupervised.** EM returns regimes
  `0..n_regimes-1` in an arbitrary order; assigning economic meaning
  ("bull-flattening", "vol-blowout") requires inspecting
  `regime_stats_` / `regime_conditional_stats` and calling
  `label_regimes` by hand. Reproducibility is via `random_state`, not by
  any canonical labelling.
- **No-arbitrage constraints on the vol surface are not enforced.**
  Calendar arbitrage (total variance non-decreasing in expiry) and butterfly
  arbitrage (positive density) are not checked at load time or during PCA
  reconstruction. Downstream consumers that need arbitrage-free surfaces must
  add a smoothing or projection step.
