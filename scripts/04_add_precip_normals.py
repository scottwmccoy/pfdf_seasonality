"""
Attach 1991-2020 monthly precipitation normals to the Atlas 14 stations.

This gives a purely gauge-based "wettest season" to set against the gauge-based
"season of most intense short-duration rainfall" — the same contrast as the
CONUS404 "wettest season same as most intense?" panel, but from observations.

Atlas 14 station ids are COOP ids ("04-0161" = COOP 040161), which map onto
GHCN-Daily ids as USC00 + the 6-digit COOP id. Normals come from the NCEI bulk
file `mly-prcp-normal-metric-30yr.csv` (units: tenths of mm).

  /opt/anaconda3/envs/SciMAN/bin/python add_precip_normals.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pfdf_seasonality.paths import GHCN_NORMALS, PROCESSED
from pfdf_seasonality.seasons import SEASONS

NORMALS = GHCN_NORMALS / "mly-prcp-normal-metric-30yr.csv"
STATIONS = PROCESSED / "seasonality" / "atlas14_station_seasonality.csv"


def coop_to_ghcn(sid: str) -> str | None:
    """'04-0161' -> 'USC00040161'. Returns None for ids that are not COOP-shaped."""
    m = re.fullmatch(r"(\d{2})-(\w+)", str(sid).strip())
    if not m:
        return None
    state, num = m.groups()
    if not num.isdigit():
        return None
    return f"USC00{state}{num.zfill(4)}"


#: Columns this script adds to `03_`'s output. Dropped before the merge so a
#: second run rebuilds them instead of colliding with itself.
_ADDED = ([f"prcp_norm_{m:02d}" for m in range(1, 13)]
          + ["ghcn_id", "wettest_season", "wettest_month", "annual_prcp_mm",
             "has_normals"])


def main():
    if not STATIONS.exists():
        raise SystemExit(
            f"{STATIONS} is missing. Run 03_build_station_seasonality.py first; "
            "this script adds the NCEI normals to that file in place.")
    st = pd.read_csv(STATIONS)

    # This script writes back to the file it reads. Without this, a second run
    # merges the normals onto columns that already hold them, producing
    # `prcp_norm_01_x` / `_y` and failing on the next line that names one.
    # Rerunning 03_ afterwards is also safe: it rewrites the file without these
    # columns, and rerunning 04_ puts them back.
    stale = [c for c in _ADDED if c in st.columns]
    if stale:
        print(f"  rebuilding {len(stale)} columns left by an earlier run")
        st = st.drop(columns=stale)

    st["ghcn_id"] = st.station_id.map(coop_to_ghcn)

    nm = pd.read_csv(NORMALS, usecols=["GHCN_ID", "month", "MLY-PRCP-NORMAL"])
    nm.columns = ["ghcn_id", "month", "prcp"]
    nm["prcp"] = pd.to_numeric(nm.prcp, errors="coerce")
    nm = nm[nm.prcp > -9000]
    nm["prcp_mm"] = nm.prcp / 10.0          # tenths of mm -> mm
    wide = nm.pivot_table(index="ghcn_id", columns="month", values="prcp_mm")
    wide = wide.reindex(columns=range(1, 13))
    wide.columns = [f"prcp_norm_{m:02d}" for m in range(1, 13)]

    st = st.merge(wide, left_on="ghcn_id", right_index=True, how="left")

    P = st[[f"prcp_norm_{m:02d}" for m in range(1, 13)]].to_numpy(float)
    tot = np.nansum(P, axis=1)
    have = np.isfinite(P).all(axis=1) & (tot > 0)

    for s, months in SEASONS.items():
        st[f"prcp_{s.lower()}_frac"] = np.where(
            have, P[:, [m - 1 for m in months]].sum(axis=1) / np.where(tot == 0, np.nan, tot),
            np.nan,
        )
    seas = st[[f"prcp_{s.lower()}_frac" for s in SEASONS]].to_numpy(float)
    lead = np.array(list(SEASONS))[np.nan_to_num(seas, nan=-1).argmax(axis=1)]
    st["wettest_season"] = np.where(have, lead, None)
    st["wettest_month"] = np.where(have, np.nan_to_num(P, nan=-1).argmax(axis=1) + 1, np.nan)
    st["annual_prcp_mm"] = np.where(have, tot, np.nan)
    st["has_normals"] = have

    st.to_csv(STATIONS, index=False)

    w = st[st.western_us & st.has_normals & ~st.short_record]
    print(f"normals matched: {int(st.has_normals.sum())}/{len(st)} station-duration records")
    print(f"western US, usable: {len(w)}\n")
    for dur in ["15m", "30m", "60m"]:
        d = w[w.duration == dur]
        if not len(d):
            continue
        same = (d.wettest_season == d.dominant_season).mean() * 100
        print(f"--- {dur}: {len(d)} stations ---")
        print(f"  wettest season == season of annual max {dur} rainfall: {same:.0f}%")
        ct = pd.crosstab(d.wettest_season, d.dominant_season)
        ct = ct.reindex(index=list(SEASONS), columns=list(SEASONS), fill_value=0)
        print("  rows = wettest season, cols = season of annual maximum")
        print(ct.to_string())
        print()


if __name__ == "__main__":
    main()
