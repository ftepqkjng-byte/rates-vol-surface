"""Loader for raw Bloomberg CSV exports living under ``data/bloomberg/``.

Wide-format CSVs (dates in the first column, one ticker per remaining column)
are reshaped into a long :class:`pd.Series` keyed by the canonical
``(date, expiry, maturity)`` MultiIndex.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import warnings

import pandas as pd

from ..schema import EXPIRY_ORDER, EXPIRY_RANK, MATURITY_RANK

# Optional Bloomberg suffix: "Curncy", "Index", "Comdty". Accept tickers with or without it.
_BBG_SUFFIX = r"(?:\s+(?:Curncy|Index|Comdty))?\s*$"

# USSV with explicit month expiry: USSV3M10
_RE_SWAPTION_MONTH = re.compile(
    r"^USSV(?P<exp>\d{1,2})M(?P<mat>\d{1,2})" + _BBG_SUFFIX, re.IGNORECASE
)
# USSV with explicit year expiry: USSV1Y10
_RE_SWAPTION_YEAR_EXPLICIT = re.compile(
    r"^USSV(?P<exp>\d{1,2})Y(?P<mat>\d{1,2})" + _BBG_SUFFIX, re.IGNORECASE
)
# USSV compact 4-digit form: USSV0110 -> (01Y, 10Y).
_RE_SWAPTION_YEAR_COMPACT = re.compile(
    r"^USSV(?P<exp>\d{2})(?P<mat>\d{1,2})" + _BBG_SUFFIX, re.IGNORECASE
)
# USSO swap rate (maturity only): USSO10
_RE_SWAP_RATE = re.compile(
    r"^USSO(?P<mat>\d{1,2})" + _BBG_SUFFIX, re.IGNORECASE
)


def parse_bbg_ticker(ticker: str) -> tuple[str, str] | None:
    """Parse a Bloomberg swaption-vol or swap-rate ticker into ``(expiry, maturity)``.

    Returns ``None`` when the ticker does not match any supported pattern or
    when the decoded expiry/maturity is not present in the canonical lists
    from :mod:`src.schema`.

    Supported formats
    -----------------
    **USD swaption normal-vol tickers** (``USSV...``)

    - ``USSV{N}M{YY} [Curncy]`` — month expiry on a year-tail swap.
      ``USSV3M10`` → ``("3M", "10Y")``. ``N`` is one or two digits; ``YY`` is
      one or two digits interpreted as years.
    - ``USSV{N}Y{YY} [Curncy]`` — explicit year expiry.
      ``USSV1Y10`` → ``("1Y", "10Y")``.
    - ``USSV{XX}{YY} [Curncy]`` — compact 4-digit form, both halves in years
      (``XX`` is exactly two digits, ``YY`` is one or two digits).
      ``USSV0110`` → ``("1Y", "10Y")``; ``USSV0530`` → ``("5Y", "30Y")``.

    **USD swap-rate tickers** (``USSO...``)

    - ``USSO{YY} [Curncy]`` — par swap rate; ``YY`` is years.
      ``USSO10`` → ``("1W", "10Y")``.

    Conventions and edge cases
    --------------------------
    - The Bloomberg suffix (``Curncy``, ``Index``, ``Comdty``) is optional,
      case-insensitive, and may carry trailing whitespace.
    - Swap rates have no option expiry. Because the project schema requires
      a three-level ``(date, expiry, maturity)`` index for every table, this
      function tags swap rates with ``EXPIRY_ORDER[0]`` (``"1W"`` by default)
      as a sentinel meaning "spot-starting".
    - The textual short-month encoding ``01=1M, 03=3M, 06=6M`` sometimes seen
      in legacy Bloomberg exports is **not** supported because it collides
      with the compact year form ``USSV{XX}{YY}``. Use the explicit ``{N}M``
      suffix to disambiguate (``USSV3M10`` rather than ``USSV0310``).
    - The compact form is tried last so that explicit-suffix forms win when
      both could match.
    - Decoded labels that are not in :data:`schema.EXPIRY_ORDER` or
      :data:`schema.MATURITY_ORDER` (e.g. ``"11Y"``, ``"30Y"`` as an expiry)
      yield ``None`` — the parser never invents canonical labels.
    """
    if not isinstance(ticker, str):
        return None
    t = ticker.strip()

    # Try in the order: USSO, USSV explicit-M, USSV explicit-Y, USSV compact.
    m = _RE_SWAP_RATE.match(t)
    if m:
        mat = f"{int(m.group('mat'))}Y"
        if mat in MATURITY_RANK:
            return (EXPIRY_ORDER[0], mat)
        return None

    m = _RE_SWAPTION_MONTH.match(t)
    if m:
        exp = f"{int(m.group('exp'))}M"
        mat = f"{int(m.group('mat'))}Y"
        if exp in EXPIRY_RANK and mat in MATURITY_RANK:
            return (exp, mat)
        return None

    m = _RE_SWAPTION_YEAR_EXPLICIT.match(t)
    if m:
        exp = f"{int(m.group('exp'))}Y"
        mat = f"{int(m.group('mat'))}Y"
        if exp in EXPIRY_RANK and mat in MATURITY_RANK:
            return (exp, mat)
        return None

    m = _RE_SWAPTION_YEAR_COMPACT.match(t)
    if m:
        exp = f"{int(m.group('exp'))}Y"
        mat = f"{int(m.group('mat'))}Y"
        if exp in EXPIRY_RANK and mat in MATURITY_RANK:
            return (exp, mat)
        return None

    return None


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Try ISO/American first; fall back to dayfirst if >50% fail."""
    # Pandas warns when DD/MM/YYYY values get parsed under the default
    # month-first heuristic. Silence it: we intentionally try a guess and
    # only fall back when >50% become NaT.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.isna().mean() > 0.5:
            parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce")
    return parsed


def load_bloomberg_csv(path: str, name: str) -> pd.Series:
    """Load a wide Bloomberg CSV into a canonical ``(date, expiry, maturity)`` Series.

    Layout assumption: first column = dates; remaining columns = Bloomberg
    tickers (one per series). Date parsing tries the default formats first
    and falls back to ``dayfirst=True`` if more than half the values fail —
    this covers both ``YYYY-MM-DD`` and European ``DD/MM/YYYY`` exports.

    Unparseable tickers are dropped; a one-line summary is printed.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path}: expected at least 2 columns (date + tickers)")

    date_col = df.columns[0]
    parsed_dates = _parse_dates(df[date_col].astype(str))
    df = (
        df.assign(__date=parsed_dates)
        .dropna(subset=["__date"])
        .drop(columns=[date_col])
        .set_index("__date")
        .rename_axis("date")
    )

    parsed_map: dict[str, tuple[str, str]] = {}
    skipped: list[str] = []
    for col in df.columns:
        result = parse_bbg_ticker(col)
        if result is None:
            skipped.append(col)
        else:
            parsed_map[col] = result

    print(
        f"[load_bloomberg_csv] {path}: parsed {len(parsed_map)}/{len(df.columns)} "
        f"tickers ({len(skipped)} skipped)"
    )
    if skipped:
        print(f"  Skipped examples: {skipped[:5]}")

    if not parsed_map:
        empty_idx = pd.MultiIndex.from_arrays(
            [
                pd.DatetimeIndex([], name="date"),
                pd.Index([], name="expiry", dtype=object),
                pd.Index([], name="maturity", dtype=object),
            ]
        )
        return pd.Series([], dtype="float64", index=empty_idx, name=name)

    keep_cols = list(parsed_map)
    wide = df[keep_cols].apply(pd.to_numeric, errors="coerce")
    wide.columns.name = "ticker"
    long = (
        wide.stack(future_stack=True)
        .rename("value")
        .reset_index()
        .dropna(subset=["value"])
    )
    long["expiry"] = long["ticker"].map(lambda t: parsed_map[t][0])
    long["maturity"] = long["ticker"].map(lambda t: parsed_map[t][1])
    long = long.drop(columns=["ticker"])
    long["value"] = long["value"].astype("float64")

    series = long.set_index(["date", "expiry", "maturity"])["value"]
    # Multiple tickers could in principle decode to the same pair — keep last.
    series = series[~series.index.duplicated(keep="last")]
    series.name = name
    return series


def load_bloomberg_directory(directory: str, name: str) -> pd.Series:
    """Load every ``*.csv`` in ``directory`` and concatenate, deduplicating overlap.

    When two files contain the same ``(date, expiry, maturity)`` triple, the
    later file in lexicographic order wins.
    """
    csvs = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {directory!r}")

    parts = [load_bloomberg_csv(p, name) for p in csvs]
    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.name = name
    return combined.sort_index()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.loaders.bloomberg_loader <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    head = pd.read_csv(csv_path, nrows=5)
    print(f"File:    {csv_path}")
    print(f"Columns: {len(head.columns)} (first 6: {list(head.columns[:6])})")
    print(f"First rows:\n{head.head()}\n")

    parsed: list[tuple[str, tuple[str, str]]] = []
    skipped: list[str] = []
    for col in head.columns[1:]:
        result = parse_bbg_ticker(col)
        if result is None:
            skipped.append(col)
        else:
            parsed.append((col, result))

    n_tickers = len(head.columns) - 1
    print(f"Ticker parse: {len(parsed)}/{n_tickers} OK, {len(skipped)} skipped")
    if parsed:
        print(f"  Parsed examples: {parsed[:5]}")
    if skipped:
        print(f"  Skipped examples: {skipped[:5]}")
