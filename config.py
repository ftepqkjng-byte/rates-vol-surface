"""Canonical universe of the rate-and-vol cube.

Single source of truth for the ``(expiry, tenor)`` label sets. Every
other module (``mock_data.py``, ``pca.py``, ``factors.py`` via ``pca``,
the Streamlit pattern creator) imports from here. Edit these lists and
rerun ``python mock_data.py`` if the universe ever needs to change.

* ``expiry`` — forward-starting time of the option / forward.
* ``tenor``  — length of the underlying rate.
"""

EXPIRY_LABELS = [
    "1M", "2M", "3M", "6M", "9M",
    "1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y",
    "12Y", "15Y", "20Y", "25Y", "30Y",
]

TENOR_LABELS = [
    "1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y",
    "12Y", "15Y", "20Y", "25Y", "30Y",
]

EXPIRY_RANK = {label: i for i, label in enumerate(EXPIRY_LABELS)}
TENOR_RANK = {label: i for i, label in enumerate(TENOR_LABELS)}
