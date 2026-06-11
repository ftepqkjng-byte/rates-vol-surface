"""HMM-based regime detection for rate dynamics.

This module fits a Gaussian Hidden Markov Model (via ``hmmlearn``) to a
feature panel derived from rates / vol data — typically PC scores from
:class:`src.features.surface_pca.SurfacePCA` or curve-spread / vol metrics
from :mod:`src.features.derived`. Any DatetimeIndex DataFrame of float
features will work as input.

Modelling assumptions
---------------------
- **Discrete hidden states.** At each date the market sits in exactly one
  of ``n_regimes`` unobserved regimes. Conditional on the regime, the
  feature vector is multivariate Gaussian with state-specific mean and
  covariance (covariance shape configurable via ``covariance_type``).
- **First-order Markov dynamics.** Tomorrow's regime depends only on
  today's; the transition kernel is encoded in ``transition_matrix_``.
- **Stationary kernel.** Both the transition matrix and the emission
  Gaussians are time-invariant — no exogenous drivers, no
  regime-switching by calendar.
- **Label arbitrariness.** EM produces unordered integer state IDs
  ``0..n_regimes - 1``; call :meth:`RateRegimeHMM.label_regimes` to attach
  economic meaning (``"bull_flattening"``, ``"vol_blowout"``, ...) once the
  regimes have been inspected.

The transition matrix
---------------------
``transition_matrix_.loc[i, j]`` is the one-step probability of moving from
regime ``i`` to regime ``j``; rows sum to 1. Multi-step transitions are
matrix powers: ``T ** h`` gives the unconditional ``h``-step distribution
when applied to an initial row vector.

Scenario probability — worked example
-------------------------------------
After fitting on a feature DataFrame::

    model = RateRegimeHMM(n_regimes=4).fit(features)
    p_flat = model.scenario_probability([2], horizon_days=21)

``p_flat`` is a Series indexed by every historical date; each value is the
probability of being in regime 2 (say "bull flattening") twenty-one
business days after that date. Equivalently, by hand for the most recent
date::

    import numpy as np
    T_h = np.linalg.matrix_power(model.transition_matrix_.values, 21)
    posterior_today = model.regime_probs_.iloc[-1].values
    p21_flat = (posterior_today @ T_h)[2]

Returning the full time series (rather than only the most recent date) is
deliberate: it gives a free hindsight backtest. Plot the series against the
realised regime indicator to diagnose whether the model would actually have
warned about a flattening event ``horizon_days`` ahead of time.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RateRegimeHMM:
    """Hidden Markov Model for interest rate regime detection.

    Fits a Gaussian HMM on a feature matrix (typically PC scores from
    :class:`SurfacePCA` or curve-spread features from
    :mod:`src.features.derived`). Exposes regime labelling, the transition
    matrix, posterior probabilities, and forward-looking scenario
    probabilities.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.hmm_: GaussianHMM | None = None
        self.feature_cols_: list[str] = []
        self._regime_ids_: list = []
        self._viterbi_states_: pd.Series | None = None
        self.regime_labels_: pd.Series | None = None
        self.regime_probs_: pd.DataFrame | None = None
        self.transition_matrix_: pd.DataFrame | None = None
        self.regime_stats_: pd.DataFrame | None = None

    # ------------------------------------------------------------------ helpers

    def _check_fitted(self) -> None:
        if self.hmm_ is None:
            raise RuntimeError("RateRegimeHMM has not been fit yet")

    def _regime_to_position(self, r) -> int:
        """Map a user-supplied regime identifier to its positional index 0..n-1.

        Accepts the current label (whatever its type) or a raw positional int.
        """
        if r in self._regime_ids_:
            return self._regime_ids_.index(r)
        if isinstance(r, (int, np.integer)) and 0 <= int(r) < self.n_regimes:
            return int(r)
        raise KeyError(
            f"Unknown regime {r!r} (available: {self._regime_ids_})"
        )

    # ---------------------------------------------------------------- fit / use

    def fit(self, features: pd.DataFrame) -> "RateRegimeHMM":
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pd.DataFrame")
        clean = features.dropna(how="any")
        if len(clean) < self.n_regimes:
            raise ValueError(
                f"Need at least n_regimes={self.n_regimes} clean rows, "
                f"got {len(clean)}"
            )
        self.feature_cols_ = list(clean.columns)

        self.hmm_ = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        X = clean.values
        self.hmm_.fit(X)

        ids: list = list(range(self.n_regimes))
        self._regime_ids_ = ids

        viterbi = self.hmm_.predict(X)
        posterior = self.hmm_.predict_proba(X)

        self._viterbi_states_ = pd.Series(viterbi, index=clean.index, name="state")
        self.regime_labels_ = pd.Series(viterbi, index=clean.index, name="regime")
        self.regime_probs_ = pd.DataFrame(posterior, index=clean.index, columns=ids)
        self.transition_matrix_ = pd.DataFrame(
            self.hmm_.transmat_, index=ids, columns=ids
        )
        self.regime_stats_ = pd.DataFrame(
            self.hmm_.means_, index=ids, columns=self.feature_cols_
        )
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Posterior regime probabilities for an out-of-sample feature panel.

        Computed by hmmlearn's forward-backward (smoothing) pass over the
        supplied sequence — i.e. each row's posterior conditions on the
        whole supplied chunk, not only on its past.
        """
        self._check_fitted()
        clean = features.dropna(how="any")
        missing = set(self.feature_cols_) - set(clean.columns)
        if missing:
            raise ValueError(f"features is missing columns: {sorted(missing)}")
        X = clean[self.feature_cols_].values
        probs = self.hmm_.predict_proba(X)
        return pd.DataFrame(probs, index=clean.index, columns=self._regime_ids_)

    def regime_conditional_stats(self, target: pd.Series) -> pd.DataFrame:
        """Mean / std / count of a target series within each fitted regime."""
        self._check_fitted()
        common = self.regime_labels_.index.intersection(target.index)
        labels = self.regime_labels_.loc[common]
        values = target.loc[common]
        valid = values.notna()
        result = (
            values[valid]
            .groupby(labels[valid])
            .agg(["mean", "std", "count"])
        )
        result.index.name = "regime"
        return result

    def scenario_probability(
        self,
        scenario_regimes: list,
        horizon_days: int = 21,
    ) -> pd.Series:
        """Probability of being in ``scenario_regimes`` ``horizon_days`` ahead.

        For each date in the training set, propagates that date's posterior
        regime distribution forward by ``horizon_days`` business steps via
        the fitted transition matrix and sums the components corresponding
        to ``scenario_regimes``. Regime identifiers may be positional ints
        or current labels (mix and match is fine).
        """
        self._check_fitted()
        if horizon_days < 0:
            raise ValueError("horizon_days must be non-negative")

        positions = [self._regime_to_position(r) for r in scenario_regimes]
        T = self.hmm_.transmat_
        T_h = (
            np.eye(self.n_regimes)
            if horizon_days == 0
            else np.linalg.matrix_power(T, horizon_days)
        )
        forward = self.regime_probs_.values @ T_h
        scenario = forward[:, positions].sum(axis=1)
        return pd.Series(
            scenario, index=self.regime_probs_.index, name="scenario_prob"
        )

    # ------------------------------------------------------------------ labels

    def label_regimes(self, labels: dict[int, str]) -> None:
        """Attach human-readable labels (or any unique hashable values) to
        regime ids, rebuilding ``regime_labels_`` / ``regime_probs_`` /
        ``transition_matrix_`` / ``regime_stats_`` to use the new names.

        Keys are positional regime ids (``0..n_regimes-1``); calling this
        again with a different mapping is fine — relabelling always works
        off the underlying Viterbi positional sequence, so it is idempotent.
        """
        self._check_fitted()
        for k in labels:
            if not isinstance(k, (int, np.integer)):
                raise TypeError(
                    "label_regimes keys must be positional regime ids (int)"
                )
            if not 0 <= int(k) < self.n_regimes:
                raise ValueError(f"regime id out of range: {k}")

        new_ids: list = [
            labels.get(i, self._regime_ids_[i]) for i in range(self.n_regimes)
        ]
        if len(set(new_ids)) != len(new_ids):
            raise ValueError("Resulting labels are not unique")

        self._regime_ids_ = new_ids
        self.regime_labels_ = pd.Series(
            [new_ids[i] for i in self._viterbi_states_.values],
            index=self._viterbi_states_.index,
            name="regime",
        )
        self.regime_probs_.columns = new_ids
        self.transition_matrix_.index = new_ids
        self.transition_matrix_.columns = new_ids
        self.regime_stats_.index = new_ids

    # ------------------------------------------------------------- persistence

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_disk(cls, path: str) -> "RateRegimeHMM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected RateRegimeHMM, got {type(obj).__name__}")
        return obj
