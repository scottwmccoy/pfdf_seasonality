"""
NOAA Atlas 14 access for the post-fire debris-flow seasonality project.

Two independent products are provided here, because they answer the question at
different resolutions and different durations:

1. `seasonality(lat, lon, dur)` — the PFDS "Section V. Seasonality analysis"
   values, i.e. the percentage of precipitation totals of a given duration that
   exceeded the precipitation-frequency estimate for that duration and AEP, by
   month. This is the exact quantity plotted on the PFDS supplementary page.

   Undocumented endpoint, reverse-engineered from `pfds/code/code_map_16.js`
   (function `SsnPlotSrc`):

       https://hdsc.nws.noaa.gov/cgi-bin/new/cgi_ssnBarPlots.py
           ?lat=<deg>&lon=<deg>&dur=<60m|24h|2d|10d>&output=mem

   `output=mem` returns the numbers as JS assignments; `output=file` renders a
   PNG and returns its path. Values are % of station-years, one per month, for
   AEPs 1/2, 1/5, 1/10, 1/25, 1/50, 1/100.

   LIMITS: shortest duration is 60-min; values are constant within an Atlas 14
   regional-frequency sub-region (not a grid); OR and WA are not covered.

2. `read_ams(...)` — the underlying station annual-maximum series published with
   each Atlas 14 volume, at 15-min, 30-min and 60-min, WITH the date of each
   annual maximum. This is the route to genuine short-duration seasonality:

       https://hdsc.nws.noaa.gov/pub/hdsc/data/<vol>/dur<DUR>_ams_na14v<N>.txt

   Volumes relevant to the western US:
       sw  = na14v6  (CA, NV, AZ, NM, UT ...)
       inw = na14v12 (ID, MT, WY ...)
       mw  = na14v8  (CO ...)
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from .paths import ATLAS14

RAW = ATLAS14
CACHE = ATLAS14 / "atlas14_seasonality_cache.json"

SSN_URL = (
    "https://hdsc.nws.noaa.gov/cgi-bin/new/cgi_ssnBarPlots.py"
    "?lat={lat:.4f}&lon={lon:.4f}&dur={dur}&output=mem"
)
PUB = "https://hdsc.nws.noaa.gov/pub/hdsc/data"

SSN_DURATIONS = ["60m", "24h", "2d", "10d"]
AEPS = ["1_2", "1_5", "1_10", "1_25", "1_50", "1_100"]
MONTHS = list(range(1, 13))

# Atlas 14 volumes covering the western US, and the AMS durations each publishes.
VOLUMES = {
    "sw": dict(tag="na14v6", name="California (Vol. 6)", durs=("15m", "30m", "60m")),
    "inw": dict(tag="na14v12", name="Interior Northwest (Vol. 12)", durs=("15m", "30m", "60m")),
    "mw": dict(tag="na14v8", name="Midwestern States (Vol. 8)", durs=("15m", "30m", "60m")),
    # Volume 1 publishes only 60-min and up, and in a different file layout.
    "sa": dict(tag="na14v1", name="Semiarid Southwest (Vol. 1)", durs=("60m",)),
}
AMS_DURATIONS = {"15m": "dur15m", "30m": "dur30m", "60m": "dur01h"}


# ---------------------------------------------------------------------------
# 1. seasonality endpoint
# ---------------------------------------------------------------------------

_FIELDS = ["result", "file", "region", "reg", "Number_of_stations",
           "Cumulative_years_of_record"]


def _load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _save_cache(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c))


def seasonality(lat: float, lon: float, dur: str = "60m", cache: dict | None = None,
                pause: float = 0.35, timeout: int = 30) -> dict:
    """Return the Atlas 14 seasonality record for one point.

    Keys: region, reg, n_stations, cum_years, and `aep_1_2` ... `aep_1_100`,
    each a list of 12 monthly percentages (Jan..Dec). Returns
    {'covered': False} where Atlas 14 has no seasonality (e.g. OR, WA).
    """
    if dur not in SSN_DURATIONS:
        raise ValueError(f"dur must be one of {SSN_DURATIONS}")
    key = f"{lat:.4f},{lon:.4f},{dur}"
    own_cache = cache is None
    cache = _load_cache() if own_cache else cache
    if key in cache:
        return cache[key]

    url = SSN_URL.format(lat=lat, lon=lon, dur=dur)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        txt = r.read().decode("utf-8", "replace")
    time.sleep(pause)

    out: dict = {"lat": lat, "lon": lon, "dur": dur}
    for f in _FIELDS:
        m = re.search(rf"{f}\s*=\s*'([^']*)'", txt)
        if m:
            out[f.lower().replace("number_of_stations", "n_stations")
                .replace("cumulative_years_of_record", "cum_years")] = m.group(1)
    for aep in AEPS:
        m = re.search(rf"AEP_{aep}\s*=\s*\[([^\]]*)\]", txt)
        if m:
            out["aep_" + aep] = [float(v.strip().strip("'")) for v in m.group(1).split(",")]
    out["covered"] = out.get("region", "none") not in ("none", "", None)
    if out["covered"]:
        out["n_stations"] = int(out.get("n_stations", 0))
        out["cum_years"] = int(out.get("cum_years", 0))

    cache[key] = out
    if own_cache:
        _save_cache(cache)
    return out


def seasonality_table(points: pd.DataFrame, dur: str = "60m",
                      lat_col: str = "latitude", lon_col: str = "longitude",
                      progress: bool = True, workers: int = 2, pause: float = 0.5,
                      cache_file: Path | None = None) -> pd.DataFrame:
    """Query the seasonality endpoint for every row of `points`.

    Because Atlas 14 seasonality is constant within a sub-region, identical
    (region, n_stations, cum_years) signatures are collapsed in the returned
    `region_id` column — that is the true spatial resolution of the product.

    Requests run on a small thread pool (default 4) because each server-side
    call takes ~0.9 s; the cache is checkpointed every 200 completions so a run
    can be interrupted and resumed. `cache_file` lets concurrent runs keep
    separate caches.
    """
    global CACHE
    if cache_file is not None:
        CACHE = Path(cache_file)
    cache = _load_cache()
    lock = threading.Lock()
    todo = [(float(r[lat_col]), float(r[lon_col])) for _, r in points.iterrows()]

    def one(pt):
        la, lo = pt
        key = f"{la:.4f},{lo:.4f},{dur}"
        with lock:
            hit = cache.get(key)
        if hit is not None:
            return hit
        try:
            s = seasonality(la, lo, dur, cache={}, pause=pause)
        except Exception as e:
            # A failed request is NOT the same as "Atlas 14 has no coverage
            # here". Flag it as query_ok=False and do not cache it, so a retry
            # can pick it up; conflating the two silently turns server 503s
            # into fake coverage gaps.
            return {"lat": la, "lon": lo, "dur": dur, "covered": None,
                    "query_ok": False, "region": None,
                    "error": f"{type(e).__name__}: {str(e)[:60]}"}
        s["query_ok"] = True
        with lock:
            cache[key] = s
        return s

    recs = [None] * len(todo)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, p): i for i, p in enumerate(todo)}
        for f in cf.as_completed(futs):
            recs[futs[f]] = f.result()
            done += 1
            if progress and done % 100 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
            if done % 200 == 0:
                with lock:
                    _save_cache(cache)
    _save_cache(cache)

    rows = []
    for s in recs:
        row = {
            "lat": s["lat"], "lon": s["lon"], "dur": s["dur"],
            "covered": s["covered"], "query_ok": s.get("query_ok", True),
            "region": s.get("region"), "error": s.get("error"),
            "n_stations": s.get("n_stations"), "cum_years": s.get("cum_years"),
        }
        for aep in AEPS:
            v = s.get("aep_" + aep)
            for m in MONTHS:
                row[f"aep{aep}_m{m:02d}"] = v[m - 1] if v else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    sig = df.region.astype(str) + "|" + df.n_stations.astype(str) + "|" + df.cum_years.astype(str)
    df["region_id"] = pd.factorize(sig)[0]
    df.loc[df.covered != True, "region_id"] = -1
    nbad = int((~df.query_ok).sum())
    if nbad:
        print(f"  WARNING: {nbad}/{len(df)} queries failed (server error, not a "
              f"coverage gap) - rerun to fill them in", flush=True)
    return df


# ---------------------------------------------------------------------------
# 2. station annual-maximum series
# ---------------------------------------------------------------------------

# Header lines differ between volumes:
#   sw : "04-0161 ALTURAS\t, CA\t, 41.4931\t -120.5528\t 4400"
#   inw: "10-9158, TOPAZ           , ID,  42.6250, -112.0881, 4918"
#   mw : "05-2354 DRAKE            , CO,   40.4333     -105.3394     6170"
_HDR = re.compile(
    r"^(?P<sid>\d{2}-\w+)[,\s]+(?P<name>.+?)[,\s]+(?P<state>[A-Z]{2})[,\s]+"
    r"(?P<lat>-?\d+\.\d+)[,\s]+(?P<lon>-?\d+\.\d+)[,\s]+(?P<elev>-?\d+)\s*$"
)
_VAL = re.compile(r"^\s*(?P<mm>-?\d+)/(?P<dd>-?\d+)/(?P<yyyy>-?\d+)\s+(?P<depth>-?\d+\.\d+)\s*$")


def download_ams(vol: str, dur: str, force: bool = False) -> Path:
    """Fetch one volume/duration AMS file into raw/atlas14_ams/."""
    tag = VOLUMES[vol]["tag"]
    dest = RAW / "atlas14_ams" / f"{vol}_{AMS_DURATIONS[dur]}_ams.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    url = f"{PUB}/{vol}/{AMS_DURATIONS[dur]}_ams_{tag}.txt"
    urllib.request.urlretrieve(url, dest)
    return dest


# Volume 1 uses its own layout, with depth BEFORE date and the longitude sign
# dropped:  "(011948) 04-5356 MARKLEEVILLE   , CA,  38.6919  119.7803  5530, ANMAX"
_HDR_V1 = re.compile(
    r"^\((?P<seq>\d+)\)\s+(?P<sid>\S+)\s+(?P<name>.+?)\s*,\s*(?P<state>[A-Z]{2})\s*,\s*"
    r"(?P<lat>-?\d+\.\d+)\s+(?P<lon>-?\d+\.\d+)\s+(?P<elev>-?\d+)\s*,"
)
_VAL_V1 = re.compile(r"^\s*(?P<depth>-?\d+\.\d+)\s+(?P<mm>-?\d+)/(?P<dd>-?\d+)/(?P<yyyy>-?\d+)\s*$")


def read_ams_v1(dur: str = "60m") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the Atlas 14 Volume 1 (semiarid southwest) AMS file."""
    path = RAW / "atlas14_ams" / f"sa_{AMS_DURATIONS[dur]}_ams.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(
            f"{PUB}/sa/{AMS_DURATIONS[dur]}_ams_na14v1.txt", path
        )
    stations, values, sid = {}, [], None
    for line in path.read_text(encoding="latin-1").splitlines():
        h = _HDR_V1.match(line)
        if h:
            g = h.groupdict()
            sid = g["sid"].strip()
            lon = float(g["lon"])
            stations.setdefault(sid, dict(
                station_id=sid, name=g["name"].strip(), state=g["state"],
                lat=float(g["lat"]), lon=-abs(lon), elev_ft=int(g["elev"]),
                volume="sa", duration=dur,
            ))
            continue
        v = _VAL_V1.match(line)
        if v and sid:
            g = v.groupdict()
            yyyy, mm, dd, depth = (int(g["yyyy"]), int(g["mm"]), int(g["dd"]),
                                   float(g["depth"]))
            if yyyy < 0 or mm < 1 or depth < 0:
                continue
            values.append(dict(station_id=sid, year=yyyy, month=mm, day=dd,
                               depth_in=depth))
    st = pd.DataFrame(stations.values())
    va = pd.DataFrame(values).drop_duplicates()
    va["date"] = pd.to_datetime(
        dict(year=va.year, month=va.month, day=va.day.clip(1, 28)), errors="coerce"
    )
    va["volume"], va["duration"] = "sa", dur
    return st, va


def read_ams(vol: str, dur: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one AMS file. Returns (stations, values).

    stations: station_id, name, state, lat, lon, elev_ft, volume, duration
    values  : station_id, date, depth_in  (missing years, coded -9/-9/-999
              with depth -9.99, are dropped)
    """
    path = download_ams(vol, dur)
    stations, values = [], []
    sid = None
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        h = _HDR.match(line)
        if h:
            g = h.groupdict()
            sid = g["sid"].strip()
            if not any(s["station_id"] == sid for s in stations[-1:]):
                stations.append(dict(
                    station_id=sid, name=g["name"].strip().rstrip(","),
                    state=g["state"], lat=float(g["lat"]), lon=float(g["lon"]),
                    elev_ft=int(g["elev"]), volume=vol, duration=dur,
                ))
            continue
        v = _VAL.match(line)
        if v and sid:
            g = v.groupdict()
            yyyy, mm, dd = int(g["yyyy"]), int(g["mm"]), int(g["dd"])
            depth = float(g["depth"])
            if yyyy < 0 or mm < 1 or depth < 0:
                continue  # missing water year
            values.append(dict(station_id=sid, year=yyyy, month=mm, day=dd,
                               depth_in=depth))
    st = pd.DataFrame(stations).drop_duplicates("station_id").reset_index(drop=True)
    va = pd.DataFrame(values)
    va["date"] = pd.to_datetime(
        dict(year=va.year, month=va.month, day=va.day.clip(1, 28)), errors="coerce"
    )
    va["volume"] = vol
    va["duration"] = dur
    return st, va


def read_all_ams(vols=("sw", "inw", "mw", "sa"), durs=("15m", "30m", "60m")):
    """Parse every western-US volume/duration combination that exists."""
    sts, vas = [], []
    for v in vols:
        for d in durs:
            if d not in VOLUMES[v]["durs"]:
                continue
            s, a = (read_ams_v1(d) if v == "sa" else read_ams(v, d))
            sts.append(s)
            vas.append(a)
            print(f"  {v:4s} {d:4s}  {len(s):5d} stations  {len(a):7d} annual maxima")
    stations = pd.concat(sts, ignore_index=True)
    values = pd.concat(vas, ignore_index=True)
    return stations, values
