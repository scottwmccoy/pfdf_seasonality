"""
Extract TRUE 1-hour precipitation seasonality from CONUS404.

Why this exists: `validate_duration_sensitivity.py` showed that the daily maximum
precipitation *rate* (RAINNCVMAX, instantaneous at model timestep) and the 1-hour
accumulation (PREC_ACC_NC) put the annual maximum in the same season for only
~70% of cells. The duration matters. Atlas 14's annual maxima are 15/30/60-minute
accumulations, so the apples-to-apples CONUS404 quantity is the 1-hour
accumulation, not the instantaneous rate.

The native 15-minute CONUS404 output (auxhist24) would be better still, but it
exists only as raw monthly tar bundles inside the 815 TB NCAR GDEX archive with
no cloud-optimized copy, so it is out of reach here.

Reads the hourly store tile by tile (chunks are time=144, y=175, x=175) and
stores, per water year and tile, the monthly maximum 1-hour precipitation and the
month of the annual maximum. Resumable: finished tiles are skipped.

Roughly 770 GB of reads over the full western US and 45 water years; free egress
from the HyTEST OSN pod, ~2 hours at observed throughput.

  /opt/anaconda3/envs/PointMan/bin/python extract_conus404_hourly.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from pfdf_seasonality.paths import CONUS404, CONUS404_HOURLY, EVENTS

warnings.filterwarnings("ignore")

C404 = CONUS404
OUT = CONUS404_HOURLY
OUT.mkdir(parents=True, exist_ok=True)

ENDPOINT = "https://usgs.osn.mghpcc.org/"
HOURLY = "s3://hytest/conus404/conus404_hourly.zarr"
TILE = 175                                   # matches the zarr spatial chunking
WATER_YEARS = list(range(1980, 2025))


def main():
    g = np.load(C404 / "grid.npz")
    y0, y1, x0, x1 = [int(v) for v in g["box"]]
    lat_w, lon_w = g["lat"], g["lon"]

    fs = fsspec.filesystem("s3", anon=True, client_kwargs={"endpoint_url": ENDPOINT})
    ds = xr.open_dataset(fs.get_mapper(HOURLY), engine="zarr", consolidated=True,
                         chunks={"time": 144})

    # chunk-aligned tiles covering the western box
    gy = list(range(y0 // TILE * TILE, y1, TILE))
    gx = list(range(x0 // TILE * TILE, x1, TILE))
    tiles = [(a, b) for a in gy for b in gx]

    # do tiles containing debris flows first, so partial results are useful
    ev = pd.read_csv(EVENTS)
    has_ev = []
    for (ty, tx) in tiles:
        la = lat_w[ty - y0:ty - y0 + TILE, tx - x0:tx - x0 + TILE]
        lo = lon_w[ty - y0:ty - y0 + TILE, tx - x0:tx - x0 + TILE]
        if la.size == 0:
            has_ev.append(0)
            continue
        inside = ((ev.latitude.between(la.min(), la.max()))
                  & (ev.longitude.between(lo.min(), lo.max()))).sum()
        has_ev.append(int(inside))
    tiles = [t for _, t in sorted(zip(has_ev, tiles), key=lambda p: -p[0])]
    print(f"{len(tiles)} tiles of {TILE}x{TILE}; "
          f"{sum(1 for h in has_ev if h)} contain debris flows", flush=True)

    for ti, (ty, tx) in enumerate(tiles):
        dest = OUT / f"tile_y{ty}_x{tx}.npz"
        if dest.exists():
            continue
        t0 = time.time()
        mmax = np.full((len(WATER_YEARS), 12, TILE, TILE), np.nan, dtype="float32")
        amonth = np.zeros((len(WATER_YEARS), TILE, TILE), dtype="int8")
        for wi, wy in enumerate(WATER_YEARS):
            sub = ds.PREC_ACC_NC.isel(y=slice(ty, ty + TILE), x=slice(tx, tx + TILE)).sel(
                time=slice(f"{wy - 1}-10-01", f"{wy}-09-30"))
            arr = sub.values                                   # (hours, y, x), mm
            if arr.size == 0:
                continue
            months = pd.DatetimeIndex(sub.time.values).month.to_numpy()
            for mi, m in enumerate(range(1, 13)):
                sel = months == m
                if sel.any():
                    mmax[wi, mi, :arr.shape[1], :arr.shape[2]] = np.nanmax(arr[sel], axis=0)
            k = np.nanargmax(np.nan_to_num(mmax[wi], nan=-1.0), axis=0)
            amonth[wi] = (k + 1).astype("int8")
            del arr
        np.savez_compressed(dest, monthly_max=mmax, month=amonth,
                            ty=np.int32(ty), tx=np.int32(tx),
                            years=np.array(WATER_YEARS))
        print(f"  tile {ti + 1}/{len(tiles)}  y{ty} x{tx}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
