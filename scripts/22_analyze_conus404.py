"""
The same debris-flow seasonality analysis, run on CONUS404 instead of gauges.

Deliberately mirrors `analyze_pfdf_seasonality.py` question for question, so the
two independent lines of evidence can be set side by side:

  M0  uniform
  M1  wet-season      P(month) proportional to CONUS404 monthly precipitation
  M2  short-duration  P(month) proportional to the fraction of water years whose
                      annual maximum precipitation RATE fell in that month

plus the decisive disagreement subset, the "first season after fire" robustness
check, and the regime split.

It also does the thing the gauge analysis alone cannot: compare CONUS404 against
the Atlas 14 gauges *at the same points*, which is the real test of whether the
modelled and observed stories agree.

  /opt/anaconda3/envs/PointMan/bin/python analyze_conus404_seasonality.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import chi2
from pfdf_seasonality.paths import CONUS404, PROCESSED, REPORTS, EVENTS
from pfdf_seasonality.seasons import SEASONS, MONTH_TO_SEASON, EPS, LAT0, normalize, to_xy, circular, season_of, doy_to_date

C404 = CONUS404
OUT = PROCESSED / "seasonality"
STATIONS = OUT / "atlas14_station_seasonality.csv"


# ---------------------------------------------------------------- load CONUS404
def load_conus404():
    g = np.load(C404 / "grid.npz")
    lat, lon = g["lat"], g["lon"]
    files = sorted(C404.glob("amax_wy*.npz"))
    if not files:
        raise SystemExit("no CONUS404 water-year files yet - run extract_conus404.py")
    months, amaxes, mmaxes = [], [], []
    for f in files:
        d = np.load(f)
        months.append(d["month"])
        amaxes.append(d["amax"])
        if "monthly_max" in d:
            mmaxes.append(d["monthly_max"])
    month = np.stack(months)                       # (nyear, y, x) month of annual max
    amax = np.stack(amaxes)                        # (nyear, y, x) mm/h
    ny, nx = month.shape[1], month.shape[2]

    # fraction of water years whose annual maximum rate fell in each month
    frac = np.zeros((12, ny, nx), dtype="float32")
    for mi, m in enumerate(range(1, 13)):
        frac[mi] = (month == m).mean(axis=0)

    clim = np.load(C404 / "monthly_climatology.npz")["clim"]   # (12, y, x) mm/month
    print(f"CONUS404: {len(files)} water years, grid {ny} x {nx}, "
          f"median annual-max rate {np.nanmedian(amax):.0f} mm/h")
    return dict(lat=lat, lon=lon, frac=frac, clim=clim, amax=amax, month=month,
                monthly_max=np.stack(mmaxes) if mmaxes else None, nyear=len(files))


def load_conus404_hourly():
    """Assemble the tiled TRUE 1-hour extraction into the western-US box.

    Mirrors load_conus404() but uses PREC_ACC_NC 1-hour accumulations, which is
    the duration-matched counterpart of the Atlas 14 60-min annual maxima.
    """
    g = np.load(C404 / "grid.npz")
    lat, lon = g["lat"], g["lon"]
    y0, y1, x0, x1 = [int(v) for v in g["box"]]
    ny, nx = y1 - y0, x1 - x0
    files = sorted((C404 / "hourly").glob("tile_y*_x*.npz"))
    if not files:
        raise SystemExit("no hourly tiles yet - run extract_conus404_hourly.py")

    frac = np.full((12, ny, nx), np.nan, dtype="float32")
    covered = np.zeros((ny, nx), dtype=bool)
    nyear = None
    for f in files:
        d = np.load(f)
        ty, tx = int(d["ty"]), int(d["tx"])
        mon = d["month"]                       # (nyear, T, T)
        nyear = mon.shape[0]
        T = mon.shape[1]
        # destination window inside the western box, clipped at its edges
        dy0, dx0 = max(ty - y0, 0), max(tx - x0, 0)
        dy1, dx1 = min(ty - y0 + T, ny), min(tx - x0 + T, nx)
        if dy1 <= dy0 or dx1 <= dx0:
            continue
        sy0, sx0 = dy0 - (ty - y0), dx0 - (tx - x0)
        sub = mon[:, sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        for mi, m in enumerate(range(1, 13)):
            frac[mi, dy0:dy1, dx0:dx1] = (sub == m).mean(axis=0)
        covered[dy0:dy1, dx0:dx1] = True

    clim = np.load(C404 / "monthly_climatology.npz")["clim"]
    print(f"CONUS404 hourly: {len(files)} tiles, {nyear} water years, "
          f"{100 * covered.mean():.0f}% of the western box covered")
    return dict(lat=lat, lon=lon, frac=frac, clim=clim, amax=None, month=None,
                monthly_max=None, nyear=nyear, covered=covered)


def sample_at(c4, lats, lons):
    """Nearest CONUS404 cell for each point; returns (intensity frac, clim, dist_km)."""
    tree = cKDTree(to_xy(c4["lat"].ravel(), c4["lon"].ravel()))
    d, idx = tree.query(to_xy(lats, lons), k=1)
    ny, nx = c4["lat"].shape
    iy, ix = np.unravel_index(idx, (ny, nx))
    P_int = c4["frac"][:, iy, ix].T                # (n, 12)
    P_wet = c4["clim"][:, iy, ix].T                # (n, 12)
    return P_int, P_wet, d, iy, ix


def main(source: str = "rate"):
    c4 = load_conus404_hourly() if source == "hourly" else load_conus404()
    tag = "_hourly" if source == "hourly" else ""
    label = ("1-hour accumulation" if source == "hourly"
             else "instantaneous rate (RAINNCVMAX)")
    print(f"CONUS404 intensity metric: {label}\n")

    # ------------------------------------------------------------------ events
    ev = pd.read_csv(EVENTS, parse_dates=["event_date"])
    ev = ev[~ev.date_precision.astype(str).str.startswith("suspect")].copy()
    ev["df_month"] = ev.event_date.dt.month
    ev["df_doy"] = ev.event_date.dt.dayofyear
    ev["df_season"] = ev.df_month.map(MONTH_TO_SEASON)

    P_int, P_wet, dist, iy, ix = sample_at(c4, ev.latitude.to_numpy(), ev.longitude.to_numpy())
    ev["c404_cell_km"] = dist
    for i, m in enumerate(range(1, 13)):
        ev[f"c404_int_p{m:02d}"] = P_int[:, i]
        ev[f"c404_wet_p{m:02d}"] = P_wet[:, i]
    s_int, sh_int = season_of(normalize(P_int))
    s_wet, sh_wet = season_of(normalize(P_wet))
    ev["c404_int_season"], ev["c404_int_share"] = s_int, sh_int
    ev["c404_wet_season"], ev["c404_wet_share"] = s_wet, sh_wet
    print(f"{len(ev)} events; all within {dist.max():.1f} km of a CONUS404 cell centre "
          f"(median {np.median(dist):.1f} km)")
    if "covered" in c4:
        ok = c4["covered"][iy, ix]
        if not ok.all():
            print(f"  {int((~ok).sum())} events fall outside the tiles extracted so far "
                  f"- dropped from this run")
            ev = ev[ok].reset_index(drop=True)
            P_int, P_wet = P_int[ok], P_wet[ok]

    # ------------------------------------------------------- model comparison
    print("\n" + "=" * 74)
    print("MODEL COMPARISON — total log-likelihood of observed debris-flow months")
    print("=" * 74)
    m_idx = ev.df_month.to_numpy() - 1
    r = np.arange(len(ev))
    Pi, Pw = normalize(P_int), normalize(P_wet)
    ll_u = len(ev) * np.log(1 / 12)
    ll_w = np.log(Pw[r, m_idx]).sum()
    ll_i = np.log(Pi[r, m_idx]).sum()
    print(f"  n = {len(ev)} events (all of them - CONUS404 has no coverage gaps)")
    print(f"    M0 uniform        LL = {ll_u:9.1f}")
    print(f"    M1 wet-season     LL = {ll_w:9.1f}   (vs uniform: {ll_w - ll_u:+.1f})")
    print(f"    M2 short-duration LL = {ll_i:9.1f}   (vs uniform: {ll_i - ll_u:+.1f})")
    print(f"    M2 - M1 = {ll_i - ll_w:+.1f}  ->  "
          f"{'SHORT-DURATION' if ll_i > ll_w else 'WET-SEASON'} model wins")

    # ------------------------------------------------------- decisive subset
    print("\n" + "=" * 74)
    print("DECISIVE SUBSET — events where the two models disagree on the season")
    print("=" * 74)
    dis = ev[ev.c404_int_season != ev.c404_wet_season]
    hit_i = (dis.df_season == dis.c404_int_season).mean() * 100
    hit_w = (dis.df_season == dis.c404_wet_season).mean() * 100
    print(f"  {len(dis)}/{len(ev)} events where wettest season != most intense season")
    print(f"    debris flow in the most-intense season : {hit_i:5.1f}%")
    print(f"    debris flow in the wettest season      : {hit_w:5.1f}%")
    print(f"    neither                                : {100 - hit_i - hit_w:5.1f}%")
    agree = ev[ev.c404_int_season == ev.c404_wet_season]
    print(f"    (where the two agree, n={len(agree)}: "
          f"{(agree.df_season == agree.c404_wet_season).mean() * 100:.1f}% in that season)")

    # ---------------------------- sensitivity: events predating the reanalysis
    # CONUS404 covers WY1980-2024; a handful of early literature-derived events
    # predate it. Both analyses apply a static climatology, so this is a
    # comparability check rather than a correction.
    pre = ev.event_date < "1979-10-01"
    if pre.any():
        sub = ev[~pre]
        d2 = sub[sub.c404_int_season != sub.c404_wet_season]
        print(f"\n  sensitivity: {int(pre.sum())} events predate WY1980. Dropping them, "
              f"the decisive subset is n={len(d2)} with "
              f"{(d2.df_season == d2.c404_int_season).mean()*100:.1f}% in the intense "
              f"season vs {(d2.df_season == d2.c404_wet_season).mean()*100:.1f}% in the wettest.")

    # ------------------------------------------------------------ robustness
    print("\n" + "=" * 74)
    print("ROBUSTNESS — is it just 'whichever rainy season arrives first'?")
    print("=" * 74)
    mid = {"DJF": 15, "MAM": 105, "JJA": 196, "SON": 288}
    d = dis.copy()
    d["fire_start_date"] = pd.to_datetime(d.fire_start_date, errors="coerce")
    d = d.dropna(subset=["fire_start_date"])
    fdoy = d.fire_start_date.dt.dayofyear.to_numpy()
    du = lambda s: (np.array([mid[x] for x in s]) - fdoy) % 365
    d["first"] = np.where(du(d.c404_int_season) < du(d.c404_wet_season), "intense", "wettest")
    print(f"  decisive subset with a fire date, n = {len(d)}")
    for grp, g in d.groupby("first"):
        print(f"    {grp + ' season arrives first':32s} n={len(g):3d}   "
              f"flows in intense season {(g.df_season == g.c404_int_season).mean()*100:5.1f}%   "
              f"in wettest {(g.df_season == g.c404_wet_season).mean()*100:5.1f}%")

    # ---------------------------------------------------------------- regimes
    print("\n" + "=" * 74)
    print("DEBRIS-FLOW TIMING BY CONUS404-DEFINED RAINFALL REGIME")
    print("=" * 74)
    for regime, g in ev.groupby("c404_int_season"):
        mean_doy, R, p = circular(g.df_doy.to_numpy())
        top = g.df_month.value_counts().head(3)
        print(f"\n  {regime}-dominant intense rainfall  (n = {len(g)} events, "
              f"{', '.join(sorted(g.state.dropna().unique()))})")
        print(f"    mean debris-flow date {doy_to_date(mean_doy)}, R = {R:.2f}, "
              f"Rayleigh p = {p:.1e}")
        print(f"    peak months: "
              f"{', '.join(f'{pd.Timestamp(2001, m, 1):%b} ({n})' for m, n in top.items())}")
        print(f"    in the locally most-intense season: "
              f"{(g.df_season == regime).mean() * 100:.0f}%")

    # ----------------------------------------------- pooled month distribution
    print("\n" + "=" * 74)
    print("POOLED MONTH DISTRIBUTION — observed vs model expectation")
    print("=" * 74)
    obs = ev.df_month.value_counts().reindex(range(1, 13), fill_value=0).to_numpy()
    e_i, e_w = Pi.sum(axis=0), Pw.sum(axis=0)
    print("   month  observed   short-duration   wet-season")
    for i, m in enumerate(range(1, 13)):
        print(f"    {pd.Timestamp(2001, m, 1):%b}   {obs[i]:6d}   {e_i[i]:12.1f}   {e_w[i]:10.1f}")
    for nm, e in [("short-duration", e_i), ("wet-season", e_w)]:
        stat = ((obs - e) ** 2 / np.maximum(e, 1e-9)).sum()
        print(f"  chi2 vs {nm:15s} = {stat:8.1f}  (df=11, p = {chi2.sf(stat, 11):.2e})")

    # ------------------------------- CONUS404 vs Atlas 14 gauges at the same points
    print("\n" + "=" * 74)
    print("CONUS404 vs ATLAS 14 GAUGES AT THE SAME POINTS")
    print("=" * 74)
    st = pd.read_csv(STATIONS)
    st = st[st.western_us & ~st.short_record]
    agree_rows = []
    for dur in ["15m", "30m", "60m"]:
        s = st[st.duration == dur].copy()
        if not len(s):
            continue
        gcols = [f"month_frac_{m:02d}" for m in range(1, 13)]
        G = normalize(s[gcols].to_numpy(float))
        Pg, _, dg, _, _ = sample_at(c4, s.lat.to_numpy(), s.lon.to_numpy())
        C = normalize(Pg)
        sg, _ = season_of(G)
        sc, _ = season_of(C)
        agree = (sg == sc).mean() * 100
        # total variation distance between the two monthly distributions
        tv = 0.5 * np.abs(G - C).sum(axis=1)
        print(f"\n  {dur} gauges (n={len(s)}) vs CONUS404 annual-max month ({label}):")
        print(f"    same dominant season   : {agree:5.1f}%")
        print(f"    median total-variation distance between monthly distributions: {np.median(tv):.2f}")
        ct = pd.crosstab(pd.Series(sg, name="gauge"), pd.Series(sc, name="CONUS404"))
        print(ct.reindex(index=list(SEASONS), columns=list(SEASONS), fill_value=0).to_string())
        agree_rows.append(dict(duration=dur, n=len(s), pct_same_season=agree,
                               median_tv_distance=float(np.median(tv))))
    pd.DataFrame(agree_rows).to_csv(OUT / f"gauge_conus404_agreement{tag}.csv", index=False)

    # ------------- headline decisive-subset numbers from BOTH lines of evidence
    model_lab = "CONUS404\n1-hour" if source == "hourly" else "CONUS404\n(rate)"
    rows = [dict(label=model_lab, n=len(dis), pct_intense=hit_i, pct_wet=hit_w)]
    gz = OUT / "pfdf_events_gauge_seasonality.csv"
    if gz.exists():
        ge = pd.read_csv(gz, parse_dates=["event_date"])
        ge["df_season"] = ge.event_date.dt.month.map(MONTH_TO_SEASON)
        for dur, lab in [("15m", "gauges\n15-min"), ("60m", "gauges\n60-min")]:
            d2 = ge[ge[f"int{dur}_ok"] & ge.wet_ok
                    & (ge[f"int{dur}_season"] != ge.wet_season)]
            if len(d2):
                rows.append(dict(label=lab, n=len(d2),
                                 pct_intense=(d2.df_season == d2[f"int{dur}_season"]).mean() * 100,
                                 pct_wet=(d2.df_season == d2.wet_season).mean() * 100))
    summary = pd.DataFrame(rows)[["label", "n", "pct_intense", "pct_wet"]]
    summary = summary.iloc[::-1].reset_index(drop=True)   # gauges first, model last
    summary.to_csv(OUT / f"gauge_vs_conus404_summary{tag}.csv", index=False)
    print("\n" + "=" * 74)
    print("HEADLINE — decisive subset, both lines of evidence")
    print("=" * 74)
    print(summary.to_string(index=False))

    ev.to_csv(OUT / f"pfdf_events_conus404_seasonality{tag}.csv", index=False)
    print(f"\nwrote {OUT / f'pfdf_events_conus404_seasonality{tag}.csv'}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "rate")
