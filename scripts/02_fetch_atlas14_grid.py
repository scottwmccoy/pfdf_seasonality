"""
Probe the Atlas 14 seasonality endpoint on a regular grid over the western US.

Purpose: establish the *actual* spatial resolution of the Atlas 14 seasonality
product (it is piecewise-constant over regional-frequency sub-regions, not a
grid) and map where it exists at all.

Writes raw/atlas14_grid_<dur>_<step>.csv, resumable via the shared JSON cache.

  /opt/anaconda3/envs/SciMAN/bin/python build_seasonality_grid.py 60m 0.5
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pfdf_seasonality import atlas14 as a

LAT0, LAT1 = 31.0, 49.5
LON0, LON1 = -125.0, -102.0


def main(dur="60m", step=0.5):
    lats = np.round(np.arange(LAT0, LAT1 + 1e-9, step), 4)
    lons = np.round(np.arange(LON0, LON1 + 1e-9, step), 4)
    grid = pd.DataFrame(
        [(la, lo) for la in lats for lo in lons], columns=["latitude", "longitude"]
    )
    print(f"{dur} @ {step} deg -> {len(grid)} points "
          f"({len(lats)} lat x {len(lons)} lon)", flush=True)
    df = a.seasonality_table(grid, dur=dur)
    out = a.RAW / f"atlas14_grid_{dur}_{step}.csv"
    df.to_csv(out, index=False)
    cov = df.covered.mean() * 100
    print(f"\nwrote {out}")
    print(f"covered: {df.covered.sum()}/{len(df)} points ({cov:.1f}%)")
    print(f"distinct seasonality sub-regions: {df.loc[df.covered, 'region_id'].nunique()}")
    print(df.loc[df.covered].groupby('region').region_id.nunique().to_string())


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "60m"
    s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    main(d, s)
