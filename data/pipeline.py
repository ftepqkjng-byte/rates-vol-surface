"""Daily-diff and parallel-shift-stripped panels from raw surface pkls.

For each canonical surface (``rate``, ``atm_vol``, ``skew_p2``,
``skew_n2``) this reads the long-format raw pkl and writes two wide-format
derived pkls alongside it:

* ``{name}_diff.pkl``     — ``wide.diff()`` (the daily move panel).
* ``{name}_residual.pkl`` — the diff with the realised-std-weighted
                            parallel shift subtracted.

The parallel shift on day ``t`` is the cross-sectional mean of diffs
*normalised* by each cell's trailing realised std::

    σ_{i,t} = std(diff_i) over [t - window, t - 1]      (strict past)
    shift_t = mean_i ( diff_{i,t} / σ_{i,t} )
    residual_{i,t} = diff_{i,t} - σ_{i,t} · shift_t

Normalising lets a single scalar ``shift_t`` describe a "parallel" move
across cells whose raw scales differ wildly (1M×1Y vs 30Y×30Y rates,
front-end vs long-end vols). Rescaling the shift back by σ_{i,t} means
each cell absorbs a share of the shift proportional to its typical
magnitude. Because σ_{i,t} is computed on data strictly before ``t``
(``.rolling(window).std().shift(1)``), the residual on day ``t`` uses
only information available at ``t`` — no look-ahead.

Run as a script to materialise all four surfaces under ``data/mock/``::

    python data/pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the project-root helpers importable when this script is run from
# either the repo root or the data folder.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pca import load_long, to_wide  # noqa: E402

SURFACES = ("rate", "atm_vol", "skew_p2", "skew_n2")
_DEFAULT_DIR = Path(__file__).resolve().parent / "mock"


def compute_diff(wide: pd.DataFrame) -> pd.DataFrame:
    """First daily diff of a wide panel; the all-NaN first row is dropped."""
    return wide.diff().dropna(how="all")


def strip_parallel_shift(
    diff: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Subtract the realised-std-weighted parallel shift from a diff panel.

    ``window`` is the rolling lookback used for the per-cell realised std
    (defaults to ~3 months of business days). The first ``window`` rows
    of ``diff`` have no σ yet and are dropped from the output.
    """
    sigma = diff.rolling(window=window).std().shift(1)
    shift = (diff / sigma).mean(axis=1)
    residual = diff.sub(sigma.mul(shift, axis=0))
    return residual.dropna(how="all")


def build_all(
    input_dir: str | Path = _DEFAULT_DIR,
    output_dir: str | Path = _DEFAULT_DIR,
    window: int = 60,
) -> None:
    """Read every surface in ``SURFACES`` from ``input_dir`` and write the
    diff and residual pkls to ``output_dir``. Default paths resolve
    relative to this script's location, so the script can be invoked from
    any working directory."""
    inp, out = Path(input_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in SURFACES:
        wide = to_wide(load_long(inp / f"{name}.pkl"))
        diff = compute_diff(wide)
        residual = strip_parallel_shift(diff, window=window)
        diff.to_pickle(out / f"{name}_diff.pkl")
        residual.to_pickle(out / f"{name}_residual.pkl")
        print(f"{name}: diff {diff.shape}, residual {residual.shape}")


if __name__ == "__main__":
    build_all()
