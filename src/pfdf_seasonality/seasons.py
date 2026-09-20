"""Season definitions and the small numeric helpers the analysis shares.

These were copy-pasted across the scripts before consolidation: ``SEASONS`` in
12 files, ``normalize`` in 4 (as four textually different but mathematically
identical versions), ``to_xy`` in 3, ``circular`` in 2. Consolidating them was
verified by re-running every script and diffing its outputs byte-for-byte
against the pre-consolidation results.

The variants differed only in ``np.asarray`` wrapping, comments, ``axis=1``
versus ``axis=-1`` on inputs that were always 2-D, and an algebraically
equivalent final line in ``normalize``. ``EPS`` was 1e-4 and ``LAT0`` was 39 deg
in every copy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["SEASONS", "SEASON_NAMES", "MONTH_TO_SEASON", "EPS", "LAT0",
           "season_columns", "normalize", "season_of", "season_totals",
           "circular", "circular_mean_doy", "to_xy", "doy_to_date",
           "circ_diff_days"]

# Meteorological seasons, uppercase keys. `build_station_seasonality` needs
# lowercase for its column names; use `season_columns` rather than a second dict.
SEASONS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
SEASON_NAMES = np.array(list(SEASONS))          # order matches raster coding 0..3
MONTH_TO_SEASON = {m: s for s, ms in SEASONS.items() for m in ms}

EPS = 1e-4          # floor on model probabilities, so one odd event can't veto a model
LAT0 = np.deg2rad(39.0)     # reference latitude for the equirectangular projection


def season_columns(suffix: str = "_frac", lower: bool = True) -> dict[str, str]:
    """Map season name -> column name, e.g. ``{"DJF": "djf_frac", ...}``."""
    return {s: f"{s.lower() if lower else s}{suffix}" for s in SEASONS}


def normalize(P, eps: float = EPS):
    """Rows -> probability vectors with a small floor.

    Works on any trailing-axis-12 array; rows that sum to zero become NaN.
    """
    P = np.clip(np.nan_to_num(np.asarray(P, float), nan=0.0), 0, None)
    s = P.sum(axis=-1, keepdims=True)
    P = np.divide(P, s, out=np.full_like(P, np.nan), where=s > 0)
    P = np.clip(P, eps, None)
    return P / P.sum(axis=-1, keepdims=True)


def season_totals(P):
    """(..., 12) monthly weights -> (..., 4) seasonal sums, in SEASONS order."""
    P = np.asarray(P, float)
    return np.stack([P[..., [m - 1 for m in ms]].sum(axis=-1)
                     for ms in SEASONS.values()], axis=-1)


def season_of(P):
    """(..., 12) monthly weights -> (dominant season name, its share)."""
    seas = season_totals(P)
    return SEASON_NAMES[seas.argmax(axis=-1)], seas.max(axis=-1)


def circular(doy):
    """Mean day-of-year, concentration R, and the Rayleigh p-value."""
    ang = 2 * np.pi * np.asarray(doy) / 365.25
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    R = float(np.hypot(c, s))
    mean_doy = float((np.arctan2(s, c) % (2 * np.pi)) * 365.25 / (2 * np.pi))
    n = len(doy)
    # Rayleigh test for departure from a uniform distribution around the year
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n)))
    return mean_doy, R, p


def circular_mean_doy(P: np.ndarray) -> np.ndarray:
    """Mean day-of-year of an (n, 12) monthly probability array.

    Note the 365.0 here against 365.25 in :func:`circular`. That inconsistency
    predates the package and is preserved deliberately: the seasonal-clock
    results (clock_doy, pred_days, residual_d, and the forecast errors built
    on them) were produced with 365.0, and changing it shifts every one of
    them by a quarter day. Expects P already normalized per row.
    """
    theta = 2 * np.pi * (np.arange(12) + 0.5) / 12
    s = (P * np.sin(theta)).sum(axis=1)
    c = (P * np.cos(theta)).sum(axis=1)
    return (np.arctan2(s, c) % (2 * np.pi)) / (2 * np.pi) * 365.0


def circ_diff_days(d1, d2):
    """Absolute difference between two days-of-year, wrapped on the circle."""
    return np.abs((np.asarray(d1) - np.asarray(d2) + 182.625) % 365.25 - 182.625)


def to_xy(lat, lon):
    """Equirectangular km about LAT0 - fine over the few-hundred-km search radius."""
    r = 6371.0
    return np.column_stack([r * np.deg2rad(np.asarray(lon)) * np.cos(LAT0),
                            r * np.deg2rad(np.asarray(lat))])


def doy_to_date(doy) -> str:
    """Day-of-year -> '27 Dec' style label."""
    return (pd.Timestamp("2001-01-01")
            + pd.Timedelta(days=float(doy) - 1)).strftime("%d %b")
