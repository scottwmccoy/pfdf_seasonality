"""
Extract the seasonality ingredients from CONUS404 over the western US.

Two quantities, mirroring the two competing models in the gauge analysis:

  intensity  RAINNCVMAX, the daily maximum grid-scale precipitation flux
             (kg m-2 s-1 -> x3600 = mm/h). CONUS404 is convection-permitting,
             so the cumulus term RAINCVMAX is identically zero and RAINNCVMAX
             is the total precipitation rate. It is an *instantaneous* rate at
             model timestep, so it is a shorter-duration quantity than the
             Atlas 14 60-min annual maxima - closer in spirit to I15.
             For each water year we keep the annual maximum and the month in
             which it occurred.

  amount     PREC_ACC_NC from the monthly store -> monthly precipitation
             climatology, the "wettest season" model.

Source: HyTEST Open Storage Network pod, anonymous S3, no egress fees.
        s3://hytest/conus404/{conus404_xtrm_daily,conus404_monthly}.zarr
        CONUS404 v3.0 covers water years 1980-2024.

Resumable: each water year is written separately and skipped if present.

  /opt/anaconda3/envs/PointMan/bin/python extract_conus404.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from pfdf_seasonality.paths import CONUS404, LOGS

warnings.filterwarnings("ignore")

OUT = CONUS404
OUT.mkdir(parents=True, exist_ok=True)

ENDPOINT = "https://usgs.osn.mghpcc.org/"
XTRM = "s3://hytest/conus404/conus404_xtrm_daily.zarr"
MONTHLY = "s3://hytest/conus404/conus404_monthly.zarr"

WEST = dict(lon=(-125.5, -101.5), lat=(30.5, 49.6))
WATER_YEARS = range(1980, 2025)          # WY1980 = 1979-10-01 .. 1980-09-30


def store(url):
    fs = fsspec.filesystem("s3", anon=True, client_kwargs={"endpoint_url": ENDPOINT})
    return xr.open_dataset(fs.get_mapper(url), engine="zarr", consolidated=True,
                           chunks={"time": 24})


def west_box(ds):
    lat, lon = ds.lat.values, ds.lon.values
    m = ((lon >= WEST["lon"][0]) & (lon <= WEST["lon"][1])
         & (lat >= WEST["lat"][0]) & (lat <= WEST["lat"][1]))
    yy, xx = np.where(m)
    return int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1


def main():
    ds = store(XTRM)
    y0, y1, x0, x1 = west_box(ds)
    print(f"western US box  y[{y0}:{y1}]  x[{x0}:{x1}]  "
          f"({y1 - y0} x {x1 - x0} cells at 4 km)", flush=True)

    grid = OUT / "grid.npz"
    if not grid.exists():
        g = ds.isel(y=slice(y0, y1), x=slice(x0, x1))
        np.savez_compressed(grid, lat=g.lat.values.astype("float32"),
                            lon=g.lon.values.astype("float32"),
                            box=np.array([y0, y1, x0, x1]))
        print(f"wrote {grid}", flush=True)

    # ---------------------------------------------------- intensity, per water year
    todo = [wy for wy in WATER_YEARS if not (OUT / f"amax_wy{wy}.npz").exists()]
    print(f"{len(todo)} water years to extract", flush=True)
    for i, wy in enumerate(todo):
        t = time.time()
        sub = ds.RAINNCVMAX.isel(y=slice(y0, y1), x=slice(x0, x1)).sel(
            time=slice(f"{wy - 1}-10-01", f"{wy}-09-30"))
        arr = sub.values                                  # (days, y, x), mm/s
        if arr.shape[0] == 0:
            print(f"  WY{wy}: no data, skipping", flush=True)
            continue
        months = pd.DatetimeIndex(sub.time.values).month.to_numpy()
        # Monthly maxima are a strict superset of the annual maximum (the annual
        # max is the largest of the twelve) and cost nothing extra, since the
        # year is already in memory. They also support monthly exceedance counts.
        mmax = np.full((12, arr.shape[1], arr.shape[2]), np.nan, dtype="float32")
        for mi, m in enumerate(range(1, 13)):
            sel = months == m
            if sel.any():
                mmax[mi] = np.nanmax(arr[sel], axis=0) * 3600.0       # -> mm/h
        k = np.nanargmax(np.nan_to_num(mmax, nan=-1.0), axis=0)
        amax = np.take_along_axis(mmax, k[None], axis=0)[0]
        np.savez_compressed(OUT / f"amax_wy{wy}.npz",
                            amax=amax.astype("float32"),
                            month=(k + 1).astype("int8"),
                            monthly_max=mmax,
                            ndays=np.int32(arr.shape[0]))
        print(f"  WY{wy}  {time.time() - t:5.0f} s   peak {np.nanmax(amax):5.0f} mm/h  "
              f"({i + 1}/{len(todo)})", flush=True)
        del arr

    # -------------------------------------------------- amount, monthly climatology
    mfile = OUT / "monthly_climatology.npz"
    if not mfile.exists():
        t = time.time()
        dm = store(MONTHLY)
        sub = dm.PREC_ACC_NC.isel(y=slice(y0, y1), x=slice(x0, x1))
        clim = sub.groupby("time.month").mean("time").values.astype("float32")
        np.savez_compressed(mfile, clim=clim)             # (12, y, x) mm/month
        print(f"monthly climatology {time.time() - t:.0f} s -> {mfile}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
