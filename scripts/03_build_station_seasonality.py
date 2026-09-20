"""
Station-level seasonality of intense SHORT-duration rainfall, from the NOAA
Atlas 14 annual-maximum series.

Why not just use the PFDS seasonality endpoint? Because its shortest duration is
60-min. The published station AMS files go down to 15-min and carry the date of
every annual maximum, which is what this project actually needs.

For each station and duration (15m / 30m / 60m) this computes:

  * month_frac_01..12  fraction of years whose annual maximum fell in that month
  * djf/mam/jja/son_frac and dominant_season
  * exc<T>_m01..12     % of years whose annual maximum exceeded the station's
                       empirical T-year annual-maximum quantile in that month
                       (T = 2, 5, 10). This is the AMS analogue of the Atlas 14
                       "Section V" seasonality bars and of the "exceedances of
                       1yr I15" maps built from CONUS404.
  * circular_mean_doy  mean day-of-year of annual maxima, computed on the circle
  * circular_R         concentration, 0 = no preferred season, 1 = all one date
  * seasonality_class  DJF / MAM / JJA / SON / bimodal (see THRESHOLDS below)

Run:
  /opt/anaconda3/envs/SciMAN/bin/python build_station_seasonality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pfdf_seasonality import atlas14 as a
from pfdf_seasonality.paths import PROCESSED
from pfdf_seasonality.seasons import SEASONS as _SEASONS

OUT = PROCESSED / "seasonality"
OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

MIN_YEARS = 15          # stations with shorter records are flagged, not dropped
RETURN_PERIODS = [2, 5, 10]
SEASONS = {k.lower(): v for k, v in _SEASONS.items()}
# A station is called "bimodal" when the second-ranked season is within this
# fraction of the first; otherwise it takes the leading season's name.
BIMODAL_RATIO = 0.75
WEST_LON_MAX = -102.0


def circular_stats(doy: np.ndarray) -> tuple[float, float]:
    """Mean day-of-year and concentration R for dates on a 365.25-day circle."""
    ang = 2 * np.pi * doy / 365.25
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    R = float(np.hypot(c, s))
    mean_doy = float((np.arctan2(s, c) % (2 * np.pi)) * 365.25 / (2 * np.pi))
    return mean_doy, R


def station_seasonality(values: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sid, dur), g in values.groupby(["station_id", "duration"], sort=False):
        g = g.dropna(subset=["date"])
        n = len(g)
        if n == 0:
            continue
        rec = dict(station_id=sid, duration=dur, n_years=n,
                   first_year=int(g.year.min()), last_year=int(g.year.max()))

        counts = g.month.value_counts().reindex(range(1, 13), fill_value=0)
        for m in range(1, 13):
            rec[f"month_frac_{m:02d}"] = counts[m] / n
        for s, months in SEASONS.items():
            rec[f"{s}_frac"] = counts[months].sum() / n

        order = sorted(SEASONS, key=lambda s: rec[f"{s}_frac"], reverse=True)
        top, second = order[0], order[1]
        rec["dominant_season"] = top.upper()
        rec["second_season"] = second.upper()
        rec["season_ratio"] = (
            rec[f"{second}_frac"] / rec[f"{top}_frac"] if rec[f"{top}_frac"] > 0 else np.nan
        )
        rec["seasonality_class"] = (
            "bimodal" if rec["season_ratio"] >= BIMODAL_RATIO else top.upper()
        )

        rec["circular_mean_doy"], rec["circular_R"] = circular_stats(
            g.date.dt.dayofyear.to_numpy()
        )

        # empirical T-year quantiles of the annual-maximum series
        d = g.depth_in.to_numpy()
        for T in RETURN_PERIODS:
            thr = float(np.quantile(d, 1 - 1 / T))
            rec[f"q{T}yr_in"] = thr
            exc = g[g.depth_in > thr]
            ec = exc.month.value_counts().reindex(range(1, 13), fill_value=0)
            for m in range(1, 13):
                rec[f"exc{T}_m{m:02d}"] = 100.0 * ec[m] / n
            rec[f"exc{T}_n"] = int(len(exc))
            if len(exc):
                ss = {s: ec[months].sum() for s, months in SEASONS.items()}
                rec[f"exc{T}_season"] = max(ss, key=ss.get).upper()
            else:
                rec[f"exc{T}_season"] = None

        rec["short_record"] = n < MIN_YEARS
        rows.append(rec)

    df = pd.DataFrame(rows)
    meta = stations.drop_duplicates(["station_id", "duration"])[
        ["station_id", "duration", "name", "state", "lat", "lon", "elev_ft", "volume"]
    ]
    df = meta.merge(df, on=["station_id", "duration"], how="right")
    df["western_us"] = df.lon <= WEST_LON_MAX
    return df


def main():
    print("Parsing Atlas 14 annual-maximum series ...")
    stations, values = a.read_all_ams()

    print("\nComputing per-station seasonality ...")
    df = station_seasonality(values, stations)

    out = OUT / "atlas14_station_seasonality.csv"
    df.to_csv(out, index=False)
    values.to_csv(OUT / "atlas14_annual_maxima.csv", index=False)
    print(f"\nwrote {out}  ({len(df)} station-duration records)")

    w = df[df.western_us]
    print(f"\nWestern US (lon <= {WEST_LON_MAX}): {len(w)} station-duration records, "
          f"{w.station_id.nunique()} stations")
    for dur in ["15m", "30m", "60m"]:
        d = w[w.duration == dur]
        if not len(d):
            continue
        print(f"\n--- {dur} : {len(d)} stations, median record {d.n_years.median():.0f} yr ---")
        print("  states  :", d.state.value_counts().to_dict())
        print("  class   :", d.seasonality_class.value_counts().to_dict())
        print("  exc2yr  :", d.exc2_season.value_counts().to_dict())
    print("\nStations with < %d years (flagged, not dropped): %d"
          % (MIN_YEARS, int(df.short_record.sum())))


if __name__ == "__main__":
    main()
