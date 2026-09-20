"""
Which rainfall product should this project use to set debris-flow seasonality?

Five candidate products classify "season of most intense short-duration
rainfall" at a location. They differ in temporal resolution, spatial support,
coverage, and whether they are observations or a model:

  A14-region   Atlas 14 seasonality endpoint, 60-min, piecewise-constant over
               87 regional sub-regions, no OR/WA          (observation)
  A14-AMS-15   Atlas 14 station annual maxima, 15-min, IDW k=5 <=150 km
  A14-AMS-30   same, 30-min
  A14-AMS-60   same, 60-min, densest network (1253 stations), no OR/WA
  C404-rate    CONUS404 RAINNCVMAX, instantaneous rate, 4 km, full coverage
  C404-1hour   CONUS404 PREC_ACC_NC, 1-hour accumulation, 4 km, full coverage
  R15-exc1yr   15-min reanalysis, season with most exceedances of the local
               1-yr I15 level, ~500 m raster, full coverage   (Dave's TIFs)
  R15-exc24    same, but exceedances of a fixed 24 mm/h threshold

This script answers two questions with the same machinery:

  1. Does the native 15-minute product actually differ from the 1-hour and
     instantaneous-rate products we already have? Yesterday's
     `compare_conus404_durations.py` inferred the answer by BRACKETING 15 min
     between those two. Dave's rasters let us measure it directly, and test
     whether the bracket logic held.

  2. Is any reanalysis product worth its cost relative to the gauge-based
     Atlas 14 metrics? The decisive test is not agreement between products,
     it is skill at the thing the project predicts: the observed month of
     real debris flows.

Note on periods: the 15-min rasters cover 1980-2021; CONUS404 here is
1980-2024; Atlas 14 station records average 28-37 years ending ~2000-2015.
Nothing is period-matched, and that is a real (small) caveat on every
cross-product comparison below.

  /opt/anaconda3/envs/PointMan/bin/python evaluate_seasonality_products.py

Outputs: compiled/seasonality_product_evaluation_report.txt
         compiled/seasonality_product_evaluation.png
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from pfdf_seasonality.paths import CONUS404, PROCESSED, REPORTS, FIGURES, STATES_ZIP, REANALYSIS_15MIN
from pfdf_seasonality.seasons import SEASONS, MONTH_TO_SEASON
from pfdf_seasonality import style
from pfdf_seasonality.style import SEASON_COLOR, OKABE_ITO, WEST_EXTENT as EXTENT

style.apply()

C404 = CONUS404
OUT = PROCESSED / "seasonality"
TIF = REANALYSIS_15MIN
WEST_STATES = ["WA", "OR", "CA", "NV", "ID", "MT", "WY", "UT", "CO", "AZ", "NM"]

SEASON_NAMES = np.array(list(SEASONS))          # index 0..3 == raster coding

REPORT = []


def say(msg=""):
    print(msg, flush=True)
    REPORT.append(msg)


def sample_tif(name, lat, lon):
    """Nearest-cell season index at each (lat, lon); -1 where nodata/outside.

    The rasters are 8000x8000 float64 (512 MB each), so read only the window
    covering the query points rather than the whole array.
    """
    lat = np.asarray(lat, float).ravel()
    lon = np.asarray(lon, float).ravel()
    with rasterio.open(TIF / f"{name}.tif") as src:
        inv = ~src.transform
        col, row = inv * (lon, lat)
        col = np.floor(col).astype(int)
        row = np.floor(row).astype(int)
        good = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        out = np.full(lat.shape, -1, dtype="int8")
        if not good.any():
            return out
        r0, r1 = int(row[good].min()), int(row[good].max()) + 1
        c0, c1 = int(col[good].min()), int(col[good].max()) + 1
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        arr = src.read(1, window=win)
        nod = src.nodata
        v = arr[row[good] - r0, col[good] - c0]
        v = np.where(np.isfinite(v) & (v != nod) & (v >= 0), v, -1)
        out[good] = v.astype("int8")
    return out


def season_of_frac(P):
    """P (..., 12) monthly weights -> season name array."""
    P = np.nan_to_num(np.asarray(P, float), nan=0.0)
    seas = np.stack([P[..., [m - 1 for m in ms]].sum(axis=-1) for ms in SEASONS.values()],
                    axis=-1)
    idx = seas.argmax(axis=-1)
    return np.where(seas.sum(axis=-1) > 0, SEASON_NAMES[idx], None)


def agreement(a, b, mask=None):
    a, b = np.asarray(a), np.asarray(b)
    ok = (a != -1) & (b != -1) if a.dtype.kind == "i" else pd.notna(a) & pd.notna(b)
    if mask is not None:
        ok = ok & mask
    return (a[ok] == b[ok]).mean() * 100, int(ok.sum())


def confusion(a, b, labs=SEASON_NAMES, rowname="", colname=""):
    ok = pd.notna(a) & pd.notna(b)
    ct = pd.crosstab(pd.Series(np.asarray(a)[ok], name=rowname),
                     pd.Series(np.asarray(b)[ok], name=colname))
    return ct.reindex(index=list(labs), columns=list(labs), fill_value=0)


# ============================================================ 1. grid level
def grid_section():
    g = np.load(C404 / "grid.npz")
    lat, lon = g["lat"], g["lon"]
    y0, y1, x0, x1 = [int(v) for v in g["box"]]
    ny, nx = lat.shape

    states = gpd.read_file(STATES_ZIP).to_crs(4326)
    west = states[states.STUSPS.isin(WEST_STATES)]
    try:
        geom = west.geometry.union_all()
    except AttributeError:
        geom = west.geometry.unary_union
    shapely.prepare(geom)
    land = shapely.contains_xy(geom, lon.ravel(), lat.ravel()).reshape(lat.shape)

    # our two products, dominant season per cell
    months = [np.load(f)["month"] for f in sorted(C404.glob("amax_wy*.npz"))]
    month_r = np.stack(months)
    frac_r = np.stack([(month_r == m).mean(axis=0) for m in range(1, 13)])
    dom_rate = season_of_frac(np.moveaxis(frac_r, 0, -1))

    frac_h = np.full((12, ny, nx), np.nan, dtype="float32")
    for f in sorted((C404 / "hourly").glob("tile_y*_x*.npz")):
        d = np.load(f)
        ty, tx, mon = int(d["ty"]), int(d["tx"]), d["month"]
        T = mon.shape[1]
        dy0, dx0 = max(ty - y0, 0), max(tx - x0, 0)
        dy1, dx1 = min(ty - y0 + T, ny), min(tx - x0 + T, nx)
        if dy1 <= dy0 or dx1 <= dx0:
            continue
        sy0, sx0 = dy0 - (ty - y0), dx0 - (tx - x0)
        sub = mon[:, sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        for mi, m in enumerate(range(1, 13)):
            frac_h[mi, dy0:dy1, dx0:dx1] = (sub == m).mean(axis=0)
    dom_hour = season_of_frac(np.moveaxis(frac_h, 0, -1))

    clim = np.load(C404 / "monthly_climatology.npz")["clim"]
    dom_wet_c404 = season_of_frac(np.moveaxis(clim, 0, -1))

    # 15-min rasters sampled onto the same cell centres
    def named(name):
        v = sample_tif(name, lat, lon)
        return np.where(v >= 0, SEASON_NAMES[np.clip(v, 0, 3)], None).reshape(lat.shape)

    d15, d24, dwet = named("season_of_most_exc1yr"), named("season_of_most_exc24"), \
        named("wettest_season")

    ok = land & pd.notna(d15) & pd.notna(dom_rate) & pd.notna(dom_hour)

    say("=" * 74)
    say("1. THE 15-MINUTE PRODUCT, MEASURED DIRECTLY")
    say("=" * 74)
    say(f"  {land.sum():,} CONUS404 cells inside the 11 western states; "
        f"{int(ok.sum()):,} with all three products defined")
    say()
    a, n = agreement(d15, dom_hour, land)
    say(f"  15-min vs CONUS404 1-hour : {a:5.1f}% same dominant season (n={n:,})")
    a, n = agreement(d15, dom_rate, land)
    say(f"  15-min vs CONUS404 rate   : {a:5.1f}% same dominant season (n={n:,})")
    a, n = agreement(dom_rate, dom_hour, land)
    say(f"  rate   vs 1-hour          : {a:5.1f}%  (yesterday's bracket width)")
    say()
    say("  15-min vs 1-hour, by season:")
    say(confusion(d15, dom_hour, rowname="15min", colname="1hour").to_string())

    # --- the bracket test
    say()
    say("-" * 74)
    say("  BRACKET TEST: yesterday's claim was that where rate and 1-hour agree,")
    say("  the 15-min answer is pinned to that value.")
    say("-" * 74)
    same_bracket = ok & (dom_rate == dom_hour)
    diff_bracket = ok & (dom_rate != dom_hour)
    pinned = (d15[same_bracket] == dom_hour[same_bracket]).mean() * 100
    say(f"    where rate == 1-hour (n={int(same_bracket.sum()):,}): "
        f"15-min agrees {pinned:.1f}% of the time")
    if diff_bracket.sum():
        sid_h = (d15[diff_bracket] == dom_hour[diff_bracket]).mean() * 100
        sid_r = (d15[diff_bracket] == dom_rate[diff_bracket]).mean() * 100
        say(f"    where they differ    (n={int(diff_bracket.sum()):,}): "
            f"15-min sides with 1-hour {sid_h:.1f}%, with rate {sid_r:.1f}%, "
            f"neither {100 - sid_h - sid_r:.1f}%")
    say()
    a, n = agreement(d15, d24, land)
    say(f"  15-min exc1yr vs exc24 (local vs fixed threshold): {a:.1f}% (n={n:,})")
    a, n = agreement(dwet, dom_wet_c404, land)
    say(f"  15-min wettest vs CONUS404 wettest season        : {a:.1f}% (n={n:,})")

    # --- CONTROL: is the gap duration, or is it the metric?
    # Dave's rasters count EXCEEDANCES of a local 1-yr level; ours take the
    # MONTH OF THE ANNUAL MAXIMUM. Recompute ours on Dave's metric so the only
    # remaining difference is the duration.
    say()
    say("-" * 74)
    say("  CONTROL: duration effect or metric effect?")
    say("  Dave's rasters count exceedances of the local 1-yr level; the")
    say("  comparisons above used our month-of-annual-max season. Recomputing")
    say("  ours as exceedance counts isolates duration from metric:")
    say("-" * 74)

    def exceedance_season(monthly_max):
        nyr, _, yy, xx = monthly_max.shape
        flat = np.nan_to_num(monthly_max.reshape(nyr * 12, yy, xx), nan=0.0)
        k = nyr + 1
        thr = np.partition(flat, -k, axis=0)[-k]
        exc = (flat > thr).reshape(nyr, 12, yy, xx)
        counts = np.stack([exc[:, [m - 1 for m in ms]].sum(axis=(0, 1))
                           for ms in SEASONS.values()])
        dom = np.where(thr > 0, SEASON_NAMES[counts.argmax(axis=0)], None)
        return dom

    mm_r = np.stack([np.load(f)["monthly_max"].astype("float32")
                     for f in sorted(C404.glob("amax_wy*.npz"))])
    exc_rate = exceedance_season(mm_r)
    del mm_r
    mm_h = np.full((45, 12, ny, nx), np.nan, dtype="float32")
    for f in sorted((C404 / "hourly").glob("tile_y*_x*.npz")):
        d = np.load(f)
        ty, tx, mmax = int(d["ty"]), int(d["tx"]), d["monthly_max"]
        T = mmax.shape[2]
        dy0, dx0 = max(ty - y0, 0), max(tx - x0, 0)
        dy1, dx1 = min(ty - y0 + T, ny), min(tx - x0 + T, nx)
        if dy1 <= dy0 or dx1 <= dx0:
            continue
        sy0, sx0 = dy0 - (ty - y0), dx0 - (tx - x0)
        mm_h[:, :, dy0:dy1, dx0:dx1] = \
            mmax[:, :, sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    exc_hour = exceedance_season(mm_h)
    del mm_h

    for lab, ours_max, ours_exc in [("1-hour", dom_hour, exc_hour),
                                    ("rate", dom_rate, exc_rate)]:
        a_max, _ = agreement(d15, ours_max, land)
        a_exc, n = agreement(d15, ours_exc, land)
        say(f"    15-min vs CONUS404 {lab:6s}: {a_max:5.1f}% (month-of-max metric) "
            f"-> {a_exc:5.1f}% (exceedance metric, n={n:,})")
    a, n = agreement(exc_rate, exc_hour, land)
    say(f"    rate vs 1-hour on the exceedance metric: {a:.1f}%")

    # --- where does the 15-min product actually move the map?
    say()
    say("-" * 74)
    say("  WHERE THE PRODUCTS DIVERGE, BY STATE")
    say("-" * 74)
    state_label = np.full(lat.shape, -1, dtype="int8")
    for i, geom_s in enumerate(west.geometry):
        shapely.prepare(geom_s)
        state_label[shapely.contains_xy(geom_s, lon.ravel(),
                                        lat.ravel()).reshape(lat.shape)] = i
    names_st = west.STUSPS.to_numpy()
    say(f"    {'':4s} {'vs 1-hour':>10s} {'vs rate':>9s}   "
        f"{'15-min DJF%':>12s} {'1-hr DJF%':>10s} {'rate DJF%':>10s}")
    rows = []
    for i, nm in enumerate(names_st):
        m = land & (state_label == i)
        if m.sum() < 500:
            continue
        a_h, _ = agreement(d15, dom_hour, m)
        a_r, _ = agreement(d15, dom_rate, m)
        djf = [(x[m] == "DJF").mean() * 100 for x in (d15, dom_hour, dom_rate)]
        rows.append((nm, a_h, a_r, *djf))
    for nm, a_h, a_r, d1, d2, d3 in sorted(rows, key=lambda r: r[1]):
        say(f"    {nm:4s} {a_h:9.1f}% {a_r:8.1f}%   {d1:11.1f}% {d2:9.1f}% {d3:9.1f}%")
    say()
    say("  'DJF%' is the share of each state's land the product calls")
    say("  winter-dominant. A product that shrinks DJF in California is")
    say("  moving the winter/summer regime boundary where most events are.")

    # California is where 54% of the events are, and where the products split
    # hardest. Check whether the metric (exceedance vs month-of-max) explains
    # the 15-min product falling OUTSIDE the rate-to-1-hour range.
    ca = land & (state_label == int(np.where(names_st == "CA")[0][0]))
    say()
    say("  CALIFORNIA, winter-dominant share of land under every variant:")
    for lab, arr in [("CONUS404 1-hour, month-of-max ", dom_hour),
                     ("CONUS404 1-hour, exceedance   ", exc_hour),
                     ("CONUS404 rate,   month-of-max ", dom_rate),
                     ("CONUS404 rate,   exceedance   ", exc_rate),
                     ("15-min reanalysis, exceedance ", d15)]:
        say(f"    {lab}: {(arr[ca] == 'DJF').mean() * 100:5.1f}%")
    say()
    say("  If the 15-min value sits outside the span of our four, the")
    say("  difference is not duration alone - it is a different dataset")
    say("  (period 1980-2021 vs 1980-2024, and a different model run).")

    return dict(lat=lat, lon=lon, land=land, states=states, d15=d15, dwet=dwet,
                dom_rate=dom_rate, dom_hour=dom_hour, dom_wet_c404=dom_wet_c404)


# ====================================================== 2. against the gauges
def gauge_section():
    say()
    say("=" * 74)
    say("2. AGAINST THE GAUGES — does finer model resolution match observations")
    say("   better? (Atlas 14 station AMS is the only true 15-min observation)")
    say("=" * 74)
    st = pd.read_csv(OUT / "atlas14_station_seasonality.csv")
    st = st[st.western_us & ~st.short_record].copy()

    g = np.load(C404 / "grid.npz")
    lat, lon = g["lat"], g["lon"]
    from scipy.spatial import cKDTree
    LAT0 = np.deg2rad(39.0)

    def to_xy(la, lo):
        r = 6371.0
        return np.column_stack([r * np.deg2rad(np.asarray(lo)) * np.cos(LAT0),
                                r * np.deg2rad(np.asarray(la))])

    tree = cKDTree(to_xy(lat.ravel(), lon.ravel()))

    # CONUS404 dominant season fields, flattened, computed once
    months = [np.load(f)["month"] for f in sorted(C404.glob("amax_wy*.npz"))]
    month_r = np.stack(months)
    frac_r = np.stack([(month_r == m).mean(axis=0) for m in range(1, 13)])
    dom_rate = season_of_frac(np.moveaxis(frac_r, 0, -1)).ravel()
    ny, nx = lat.shape
    y0, y1, x0, x1 = [int(v) for v in g["box"]]
    frac_h = np.full((12, ny, nx), np.nan, dtype="float32")
    for f in sorted((C404 / "hourly").glob("tile_y*_x*.npz")):
        d = np.load(f)
        ty, tx, mon = int(d["ty"]), int(d["tx"]), d["month"]
        T = mon.shape[1]
        dy0, dx0 = max(ty - y0, 0), max(tx - x0, 0)
        dy1, dx1 = min(ty - y0 + T, ny), min(tx - x0 + T, nx)
        if dy1 <= dy0 or dx1 <= dx0:
            continue
        sy0, sx0 = dy0 - (ty - y0), dx0 - (tx - x0)
        sub = mon[:, sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        for mi, m in enumerate(range(1, 13)):
            frac_h[mi, dy0:dy1, dx0:dx1] = (sub == m).mean(axis=0)
    dom_hour = season_of_frac(np.moveaxis(frac_h, 0, -1)).ravel()

    rows = []
    for dur in ["15m", "30m", "60m"]:
        s = st[st.duration == dur]
        if not len(s):
            continue
        _, idx = tree.query(to_xy(s.lat.to_numpy(), s.lon.to_numpy()))
        d15 = sample_tif("season_of_most_exc1yr", s.lat.to_numpy(), s.lon.to_numpy())
        d15 = np.where(d15 >= 0, SEASON_NAMES[np.clip(d15, 0, 3)], None)
        obs = s.dominant_season.to_numpy()
        for lab, pred in [("15-min reanalysis", d15),
                          ("CONUS404 1-hour", dom_hour[idx]),
                          ("CONUS404 rate", dom_rate[idx])]:
            a, n = agreement(pred, obs)
            rows.append(dict(gauge=dur, product=lab, pct=a, n=n))
    tab = pd.DataFrame(rows)
    say()
    piv = tab.pivot(index="product", columns="gauge", values="pct").round(1)
    piv = piv.reindex(["15-min reanalysis", "CONUS404 1-hour", "CONUS404 rate"])
    say("  % of stations where the gridded product matches the gauge's own")
    say("  dominant season (columns = gauge AMS duration):")
    say(piv.to_string())
    say()
    say("  n stations: " + ", ".join(
        f"{d} {tab[tab.gauge == d].n.iloc[0]:,}" for d in ["15m", "30m", "60m"]))
    say()
    best = piv["15m"].idxmax()
    say(f"  Against TRUE 15-min observations, the best gridded product is: {best}")
    return tab


# ============================================ 3. skill on real debris flows
def event_section():
    say()
    say("=" * 74)
    say("3. THE TEST THAT MATTERS — skill at the observed month of real")
    say("   debris flows")
    say("=" * 74)

    ev = pd.read_csv(OUT / "pfdf_events_gauge_seasonality.csv", parse_dates=["event_date"])
    a14 = pd.read_csv(OUT / "pfdf_events_with_rainfall_seasonality.csv")
    cr = pd.read_csv(OUT / "pfdf_events_conus404_seasonality.csv")
    ch = pd.read_csv(OUT / "pfdf_events_conus404_seasonality_hourly.csv")

    df = ev[["event_id", "latitude", "longitude", "df_season",
             "int15m_season", "int30m_season", "int60m_season", "wet_season"]].copy()
    df = df.merge(a14[["event_id", "a14_dominant_month", "a14_covered"]], on="event_id",
                  how="left")
    df["a14_region_season"] = df.a14_dominant_month.map(
        lambda m: MONTH_TO_SEASON.get(int(m)) if pd.notna(m) else None)
    df = df.merge(cr[["event_id", "c404_int_season", "c404_wet_season"]],
                  on="event_id", how="left")
    df = df.merge(ch[["event_id", "c404_int_season", "c404_wet_season"]],
                  on="event_id", how="left", suffixes=("_rate", "_hour"))

    for key, name in [("r15_int", "season_of_most_exc1yr"),
                      ("r15_int24", "season_of_most_exc24"),
                      ("r15_wet", "wettest_season")]:
        v = sample_tif(name, df.latitude.to_numpy(), df.longitude.to_numpy())
        df[key] = np.where(v >= 0, SEASON_NAMES[np.clip(v, 0, 3)], None)

    # product -> (intense column, paired wettest column, provenance)
    products = [
        ("A14-region  60-min",  "a14_region_season", "wet_season",      "gauge"),
        ("A14-AMS     15-min",  "int15m_season",     "wet_season",      "gauge"),
        ("A14-AMS     30-min",  "int30m_season",     "wet_season",      "gauge"),
        ("A14-AMS     60-min",  "int60m_season",     "wet_season",      "gauge"),
        ("C404        rate",    "c404_int_season_rate", "c404_wet_season_rate", "model"),
        ("C404        1-hour",  "c404_int_season_hour", "c404_wet_season_hour", "model"),
        ("R15 exc1yr  15-min",  "r15_int",           "r15_wet",         "model"),
        ("R15 exc24   15-min",  "r15_int24",         "r15_wet",         "model"),
    ]

    say()
    say("  COVERAGE — how many of the 349 events each product can classify:")
    for lab, icol, _, prov in products:
        n = df[icol].notna().sum()
        say(f"    {lab:22s} {prov:6s} {n:3d} events  ({n / len(df) * 100:.0f}%)")
    say()
    say("  The Atlas 14 shortfall is OR and WA, which Atlas 14 never published.")

    # common subset so the comparison is like for like
    icols = [p[1] for p in products]
    common = df[icols].notna().all(axis=1) & df.df_season.notna()
    say()
    say(f"  Common subset where every product is defined: n = {int(common.sum())}")

    def score(sub, icol, wcol):
        hit = (sub[icol] == sub.df_season).mean() * 100
        dec = sub[sub[icol] != sub[wcol]]
        if len(dec) < 5:
            return hit, len(sub), np.nan, np.nan, len(dec)
        return (hit, len(sub),
                (dec[icol] == dec.df_season).mean() * 100,
                (dec[wcol] == dec.df_season).mean() * 100, len(dec))

    # Each product pairs with its own wettest map above, so a product that
    # calls more events "decisive" is answering a different question. Scoring
    # every product against ONE fixed amount reference (the NCEI 1991-2020
    # normals, which is the abstract's M1) removes that confound: only the
    # intensity side varies, which is the thing being compared.
    for title, sub_mask, fixed_wet in [
            ("ALL EVENTS EACH PRODUCT COVERS", None, None),
            ("COMMON SUBSET (identical events, each product's own wettest map)",
             common, None),
            ("COMMON SUBSET, FIXED AMOUNT REFERENCE (NCEI normals wettest season)",
             common, "wet_season")]:
        say()
        say("-" * 74)
        say(f"  {title}")
        say("-" * 74)
        say(f"    {'product':22s} {'n':>4s} {'hit%':>6s} | decisive subset: "
            f"{'n':>4s} {'intense%':>9s} {'wettest%':>9s}")
        res = []
        for lab, icol, wcol, prov in products:
            sub = df[common] if sub_mask is not None else df[df[icol].notna()
                                                             & df.df_season.notna()]
            hit, n, di, dw, dn = score(sub, icol, fixed_wet or wcol)
            say(f"    {lab:22s} {n:4d} {hit:6.1f} | {'':17s}{dn:4d} "
                f"{di:9.1f} {dw:9.1f}")
            res.append(dict(product=lab, prov=prov, n=n, hit=hit,
                            dec_n=dn, dec_int=di, dec_wet=dw))
        if sub_mask is not None and fixed_wet:
            common_res = pd.DataFrame(res)
        # uniform baseline
        say(f"    {'(uniform baseline)':22s} {'':4s} {25.0:6.1f} |")

    say()
    say("  'hit%' = share of debris flows whose observed season equals the")
    say("  product's predicted most-intense season. 'decisive subset' keeps")
    say("  only events where that product's own wettest and most-intense")
    say("  seasons differ, which is where intensity and amount make opposite")
    say("  predictions and the comparison is diagnostic.")
    return df, common_res, products


# ================================================================== figure
def make_figure(G, gauge_tab, cres):
    states = G["states"]
    lat, lon, land = G["lat"], G["lon"], G["land"]

    def frame(ax, title):
        states.boundary.plot(ax=ax, linewidth=0.4, color="0.25", zorder=5)
        ax.set_xlim(*EXTENT[:2]); ax.set_ylim(*EXTENT[2:])
        ax.set_aspect(1 / 0.75)
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)

    def cat(ax, dom, title):
        idx = np.full(lat.shape, np.nan)
        for i, s in enumerate(SEASON_NAMES):
            idx[(dom == s) & land] = i
        cmap = ListedColormap([SEASON_COLOR[s] for s in SEASONS]); cmap.set_bad("white")
        ax.pcolormesh(lon, lat, idx, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
                      shading="nearest", rasterized=True)
        frame(ax, title)

    fig = plt.figure(figsize=(13.6, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.92], hspace=0.20, wspace=0.36)

    cat(fig.add_subplot(gs[0, 0]), G["d15"], "a  15-min reanalysis (Dave)")
    cat(fig.add_subplot(gs[0, 1]), G["dom_hour"], "b  CONUS404 1-hour")
    cat(fig.add_subplot(gs[0, 2]), G["dom_rate"], "c  CONUS404 instantaneous rate")

    ax = fig.add_subplot(gs[1, 0])
    z = np.full(lat.shape, np.nan)
    ok = land & pd.notna(G["d15"]) & pd.notna(G["dom_hour"])
    z[ok & (G["d15"] == G["dom_hour"])] = 0
    z[ok & (G["d15"] != G["dom_hour"])] = 1
    cmap = ListedColormap(["#DDD8CE", "#D55E00"]); cmap.set_bad("white")
    ax.pcolormesh(lon, lat, z, cmap=cmap, vmin=0, vmax=1, shading="nearest",
                  rasterized=True)
    a, _ = agreement(G["d15"], G["dom_hour"], land)
    frame(ax, f"d  15-min vs 1-hour differ ({100 - a:.0f}% of land)")
    ax.legend(handles=[
        Line2D([], [], marker="s", linestyle="none", color="#D55E00", markersize=7,
               label="differ"),
        Line2D([], [], marker="s", linestyle="none", color="#DDD8CE", markersize=7,
               label="same")],
        loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)

    ax = fig.add_subplot(gs[1, 1])
    piv = gauge_tab.pivot(index="product", columns="gauge", values="pct").reindex(
        ["15-min reanalysis", "CONUS404 1-hour", "CONUS404 rate"])
    xx = np.arange(len(piv))
    w = 0.26
    for i, dur in enumerate(["15m", "30m", "60m"]):
        ax.bar(xx + (i - 1) * w, piv[dur], w, label=f"{dur} gauges",
               color=[OKABE_ITO[4], OKABE_ITO[0], OKABE_ITO[2]][i])
        for j, v in enumerate(piv[dur]):
            ax.text(j + (i - 1) * w, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(xx)
    ax.set_xticklabels(["15-min\nreanalysis", "CONUS404\n1-hour", "CONUS404\nrate"],
                       fontsize=7.5)
    ax.set_ylim(0, 108); ax.set_ylabel("% matching gauge season")
    ax.set_title("e  agreement with Atlas 14 gauge observations", fontsize=9,
                 loc="left", fontweight="bold")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center")

    ax = fig.add_subplot(gs[1, 2])
    c = cres.sort_values("dec_int", ascending=True)
    colors = ["#0072B2" if p == "gauge" else "#D55E00" for p in c.prov]
    yy = np.arange(len(c))
    ax.barh(yy, c.dec_int, 0.7, color=colors)
    for i, (_, r) in enumerate(c.iterrows()):
        ax.text(r.dec_int + 1.5, i, f"{r.dec_int:.0f}%", va="center", fontsize=7.5)
    ax.set_yticks(yy)
    ax.set_yticklabels([" ".join(p.split()) for p in c["product"]], fontsize=7.5)
    # provenance is carried by the tick-label color, so no legend is needed
    for tick, col in zip(ax.get_yticklabels(), colors):
        tick.set_color(col)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of decisive-subset flows in the predicted intense season\n"
                  "blue = gauge-based, orange = reanalysis", fontsize=8)
    ax.set_title("f  skill on real debris flows (common subset)", fontsize=9,
                 loc="left", fontweight="bold")

    handles = [Line2D([], [], marker="s", linestyle="none", color=SEASON_COLOR[s],
                      markersize=8, label=s) for s in SEASONS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.925), fontsize=9)
    fig.suptitle("Which rainfall product sets debris-flow seasonality best?\n"
                 "native 15-min reanalysis vs CONUS404 vs Atlas 14 gauges",
                 fontsize=11, y=1.005)
    out = FIGURES / "seasonality_product_evaluation.png"
    fig.savefig(out, bbox_inches="tight")
    say()
    say(f"wrote {out}")


def main():
    G = grid_section()
    gauge_tab = gauge_section()
    _, cres, _ = event_section()
    make_figure(G, gauge_tab, cres)
    (REPORTS / "seasonality_product_evaluation_report.txt").write_text("\n".join(REPORT) + "\n")
    print("wrote", REPORTS / "seasonality_product_evaluation_report.txt")


if __name__ == "__main__":
    main()
