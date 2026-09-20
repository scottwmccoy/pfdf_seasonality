"""
Does the choice of rainfall duration change the SEASONALITY answer?

The cloud-optimized CONUS404 stores stop at hourly. The native 15-minute
precipitation (auxhist24) exists only as raw monthly tar bundles in the 815 TB
NCAR GDEX archive, so it is not reachable in a working session. The gridded
analysis therefore uses RAINNCVMAX — the daily maximum precipitation *rate* at
model timestep, which is a shorter-duration quantity than 15-min accumulation.

This script tests whether that substitution matters for the only thing we use it
for: the MONTH in which the annual maximum falls. It compares, cell by cell over
a region spanning both the winter and monsoon regimes:

    month of annual max RAINNCVMAX  (instantaneous rate, already extracted)
    month of annual max PREC_ACC_NC (true 1-hour accumulation, hourly store)

If these agree, the duration substitution is immaterial for seasonality and the
gridded result can be read as a short-duration intensity climatology.

  /opt/anaconda3/envs/PointMan/bin/python validate_duration_sensitivity.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from pfdf_seasonality.paths import CONUS404, PROCESSED, REPORTS
from pfdf_seasonality.seasons import SEASONS
from pfdf_seasonality.report import tee

warnings.filterwarnings("ignore")

C404 = CONUS404
OUT = PROCESSED / "seasonality"
ENDPOINT = "https://usgs.osn.mghpcc.org/"
HOURLY = "s3://hytest/conus404/conus404_hourly.zarr"

# One 175x175 chunk-aligned tile chosen to span the DJF/JJA regime boundary
# (southern California through western Arizona). Hourly chunks are
# (time=144, y=175, x=175), so aligning to that keeps the read efficient.
TILE_Y, TILE_X = 175, 175
YEARS = range(2010, 2023)                 # 13 water years: enough to fix the mode


def main():
    g = np.load(C404 / "grid.npz")
    lat_w, lon_w, (y0, y1, x0, x1) = g["lat"], g["lon"], g["box"]

    # locate a chunk-aligned tile centred near 34N, -115E (Mojave / lower Colorado),
    # which contains both winter-dominated and monsoon-dominated cells
    d = (lat_w - 34.0) ** 2 + (lon_w + 115.0) ** 2
    cy, cx = np.unravel_index(d.argmin(), d.shape)
    gy0 = int((y0 + cy) // TILE_Y * TILE_Y)
    gx0 = int((x0 + cx) // TILE_X * TILE_X)
    gy1, gx1 = gy0 + TILE_Y, gx0 + TILE_X
    print(f"validation tile (global idx): y[{gy0}:{gy1}] x[{gx0}:{gx1}]", flush=True)

    fs = fsspec.filesystem("s3", anon=True, client_kwargs={"endpoint_url": ENDPOINT})
    ds = xr.open_dataset(fs.get_mapper(HOURLY), engine="zarr", consolidated=True,
                         chunks={"time": 144})
    lat_t = ds.lat.isel(y=slice(gy0, gy1), x=slice(gx0, gx1)).values
    lon_t = ds.lon.isel(y=slice(gy0, gy1), x=slice(gx0, gx1)).values

    hourly_month = np.zeros((len(list(YEARS)), TILE_Y, TILE_X), dtype="int8")
    for i, wy in enumerate(YEARS):
        t = time.time()
        sub = ds.PREC_ACC_NC.isel(y=slice(gy0, gy1), x=slice(gx0, gx1)).sel(
            time=slice(f"{wy - 1}-10-01", f"{wy}-09-30"))
        arr = sub.values
        k = np.nanargmax(arr, axis=0)
        months = pd.DatetimeIndex(sub.time.values).month.to_numpy()
        hourly_month[i] = months[k]
        print(f"  WY{wy} hourly {time.time() - t:5.0f} s  "
              f"peak {np.nanmax(arr):.0f} mm/h", flush=True)
        del arr

    # matching RAINNCVMAX months for the same cells and years
    ry0, ry1 = gy0 - y0, gy1 - y0
    rx0, rx1 = gx0 - x0, gx1 - x0
    rate_month = np.stack([
        np.load(C404 / f"amax_wy{wy}.npz")["month"][ry0:ry1, rx0:rx1] for wy in YEARS
    ])

    def frac(monthstack):
        return np.stack([(monthstack == m).mean(axis=0) for m in range(1, 13)])

    def dom(F):
        s = np.stack([F[[m - 1 for m in ms]].sum(axis=0) for ms in SEASONS.values()])
        return s.argmax(axis=0)

    Fh, Fr = frac(hourly_month), frac(rate_month)
    dh, dr = dom(Fh), dom(Fr)

    same_year_month = (hourly_month == rate_month).mean() * 100
    same_season = (dh == dr).mean() * 100
    tv = 0.5 * np.abs(Fh - Fr).sum(axis=0)

    print("\n" + "=" * 70)
    print("DURATION SENSITIVITY: 1-hour accumulation vs instantaneous rate")
    print("=" * 70)
    print(f"  tile: {TILE_Y}x{TILE_X} cells, {len(list(YEARS))} water years, "
          f"lat {lat_t.min():.1f}-{lat_t.max():.1f}, lon {lon_t.min():.1f}-{lon_t.max():.1f}")
    print(f"  identical month of annual max, same cell & year : {same_year_month:5.1f}%")
    print(f"  identical dominant SEASON of the climatology    : {same_season:5.1f}%")
    print(f"  median total-variation distance between the two "
          f"monthly distributions: {np.median(tv):.3f}")
    names = list(SEASONS)
    ct = pd.crosstab(pd.Series([names[i] for i in dh.ravel()], name="hourly"),
                     pd.Series([names[i] for i in dr.ravel()], name="rate"))
    print(ct.reindex(index=names, columns=names, fill_value=0).to_string())

    np.savez_compressed(OUT / "duration_sensitivity.npz",
                        hourly_month=hourly_month, rate_month=rate_month,
                        lat=lat_t, lon=lon_t, years=np.array(list(YEARS)))
    print(f"\nwrote {OUT / 'duration_sensitivity.npz'}")


if __name__ == "__main__":
    with tee("duration_sensitivity_report.txt"):
        main()
