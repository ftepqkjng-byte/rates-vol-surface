"""Canonical label lists, ordering ranks, and the data-type registry.

Loaded once at import time from ``configs/schema.yaml`` via a path relative to
this file, so behavior is independent of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "configs" / "schema.yaml"

with _SCHEMA_PATH.open("r", encoding="utf-8") as _f:
    _schema = yaml.safe_load(_f)

EXPIRY_ORDER: list[str] = list(_schema["expiry_labels"])
MATURITY_ORDER: list[str] = list(_schema["maturity_labels"])

EXPIRY_RANK: dict[str, int] = {label: i for i, label in enumerate(EXPIRY_ORDER)}
MATURITY_RANK: dict[str, int] = {label: i for i, label in enumerate(MATURITY_ORDER)}

DATA_TYPES: list[str] = list(_schema["data_types"])
RAW_DATA_TYPES: list[str] = list(_schema["raw_data_types"])


def sort_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Sort (expiry, maturity) pairs by canonical ordering."""
    return sorted(pairs, key=lambda p: (EXPIRY_RANK[p[0]], MATURITY_RANK[p[1]]))
