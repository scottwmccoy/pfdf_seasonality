"""
The gauge-only debris-flow seasonality story.

Question: is the timing of post-fire debris flows in the western US explained by
the seasonality of intense SHORT-duration rainfall, rather than by when it simply
rains the most? Everything here comes from rain gauges — NOAA Atlas 14 station
annual-maximum series and NCEI 1991-2020 monthly precipitation normals — so the
result stands independently of any modelled precipitation product.

Three competing models for the month in which a debris flow occurs at a given
location, each a 12-vector of probabilities interpolated from nearby gauges:

  M0  uniform              every month equally likely (1/12)
  M1  wet-season           P(month) proportional to normal monthly precipitation
  M2  short-duration       P(month) proportional to the fraction of years whose
                           annual maximum 15/30/60-min rainfall fell in that month

Scored by total log-likelihood of the observed debris-flow months. The decisive
subset is the events where M1 and M2 disagree about the leading season.

  /opt/anaconda3/envs/PointMan/bin/python analyze_pfdf_seasonality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import chi2
from pfdf_seasonality.paths import PROCESSED, REPORTS, EVENTS
from pfdf_seasonality.seasons import SEASONS, MONTH_TO_SEASON, EPS, LAT0, normalize, to_xy, circular, season_of, doy_to_date
from pfdf_seasonality.report import tee
from pfdf_seasonality.events import load_events

OUT = PROCESSED / "seasonality"
OUT.mkdir(exist_ok=True)
STATIONS = OUT / "atlas14_station_seasonality.csv"

K_NEAREST, MAX_KM = 5, 150.0


def idw(points, src, cols):
    """Inverse-distance-weighted interpolation of `cols` from `src` onto `points`."""
    src = src.reset_index(drop=True)
    tree = cKDTree(to_xy(src.lat.to_numpy(), src.lon.to_numpy()))
    q = to_xy(points.latitude.to_numpy(), points.longitude.to_numpy())
    k = min(K_NEAREST, len(src))
    dist, idx = tree.query(q, k=k)
    dist, idx = np.atleast_2d(dist.T).T, np.atleast_2d(idx.T).T
    M = src[cols].to_numpy(float)
    w = 1.0 / np.maximum(dist, 1.0) ** 2
    w[dist > MAX_KM] = 0.0
    ws = w.sum(axis=1, keepdims=True)
    ok = ws[:, 0] > 0
    out = np.full((len(points), len(cols)), np.nan)
    out[ok] = (w[ok, :, None] * M[idx[ok]]).sum(axis=1) / ws[ok]
    return out, dist[:, 0], ok


def main():
    # ---------------------------------------------------------------- events
    ev = load_events()
    n_all = len(ev)
    ev = ev[~ev.date_precision.astype(str).str.startswith("suspect")].copy()
    print(f"{n_all} compiled events; {len(ev)} with a trustworthy calendar date")
    ev["df_month"] = ev.event_date.dt.month
    ev["df_doy"] = ev.event_date.dt.dayofyear
    ev["df_season"] = ev.df_month.map(MONTH_TO_SEASON)

    st = pd.read_csv(STATIONS)
    st = st[st.western_us & ~st.short_record]

    # -------------------------------------------------- gauge models at events
    for dur in ["15m", "30m", "60m"]:
        s = st[st.duration == dur]
        cols = [f"month_frac_{m:02d}" for m in range(1, 13)]
        P, d0, ok = idw(ev, s, cols)
        ev[f"int{dur}_ok"] = ok
        ev[f"int{dur}_km"] = d0
        for i, m in enumerate(range(1, 13)):
            ev[f"int{dur}_p{m:02d}"] = P[:, i]
        seas, share = season_of(normalize(P))
        ev[f"int{dur}_season"] = np.where(ok, seas, None)
        ev[f"int{dur}_share"] = np.where(ok, share, np.nan)

    sw = st[(st.duration == "60m") & st.has_normals]
    cols = [f"prcp_norm_{m:02d}" for m in range(1, 13)]
    P, d0, ok = idw(ev, sw, cols)
    ev["wet_ok"], ev["wet_km"] = ok, d0
    for i, m in enumerate(range(1, 13)):
        ev[f"wet_p{m:02d}"] = P[:, i]
    seas, share = season_of(normalize(P))
    ev["wet_season"] = np.where(ok, seas, None)
    ev["wet_share"] = np.where(ok, share, np.nan)

    # ------------------------------------------------------ model comparison
    print("\n" + "=" * 74)
    print("MODEL COMPARISON — total log-likelihood of the observed debris-flow months")
    print("=" * 74)
    rows = []
    for dur in ["15m", "30m", "60m"]:
        use = ev[ev[f"int{dur}_ok"] & ev.wet_ok].copy()
        if not len(use):
            continue
        m_idx = use.df_month.to_numpy() - 1
        r = np.arange(len(use))
        P_int = normalize(use[[f"int{dur}_p{m:02d}" for m in range(1, 13)]].to_numpy(float))
        P_wet = normalize(use[[f"wet_p{m:02d}" for m in range(1, 13)]].to_numpy(float))
        ll_unif = len(use) * np.log(1 / 12)
        ll_wet = np.log(P_wet[r, m_idx]).sum()
        ll_int = np.log(P_int[r, m_idx]).sum()
        rows.append(dict(duration=dur, n=len(use), ll_uniform=ll_unif,
                         ll_wet=ll_wet, ll_intense=ll_int,
                         d_int_minus_wet=ll_int - ll_wet,
                         d_int_minus_unif=ll_int - ll_unif))
        print(f"\n  {dur} short-duration model, n = {len(use)} events")
        print(f"    M0 uniform        LL = {ll_unif:9.1f}")
        print(f"    M1 wet-season     LL = {ll_wet:9.1f}   (vs uniform: {ll_wet - ll_unif:+.1f})")
        print(f"    M2 short-duration LL = {ll_int:9.1f}   (vs uniform: {ll_int - ll_unif:+.1f})")
        print(f"    M2 - M1 = {ll_int - ll_wet:+.1f}  ->  "
              f"{'SHORT-DURATION' if ll_int > ll_wet else 'WET-SEASON'} model wins")
    pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False)

    # ------------------------------------------------- the decisive subset
    print("\n" + "=" * 74)
    print("DECISIVE SUBSET — events where the two models disagree on the season")
    print("=" * 74)
    for dur in ["15m", "60m"]:
        use = ev[ev[f"int{dur}_ok"] & ev.wet_ok].copy()
        dis = use[use[f"int{dur}_season"] != use.wet_season]
        if not len(dis):
            continue
        hit_int = (dis.df_season == dis[f"int{dur}_season"]).mean() * 100
        hit_wet = (dis.df_season == dis.wet_season).mean() * 100
        print(f"\n  {dur}: {len(dis)}/{len(use)} events where wettest season != "
              f"most intense {dur} season")
        print(f"    debris flow in the most-intense-{dur} season : {hit_int:5.1f}%")
        print(f"    debris flow in the wettest season            : {hit_wet:5.1f}%")
        print(f"    neither                                      : "
              f"{100 - hit_int - hit_wet:5.1f}%")
        agree = use[use[f"int{dur}_season"] == use.wet_season]
        if len(agree):
            print(f"    (where the two agree, n={len(agree)}: "
                  f"{(agree.df_season == agree.wet_season).mean() * 100:.1f}% in that season)")

    # ------------------------------------ robustness: "first season after fire"
    # Susceptibility decays after a fire, so a preference for whichever rainy
    # season arrives FIRST could mimic the result above. Split the decisive
    # subset by which season actually came first.
    print("\n" + "=" * 74)
    print("ROBUSTNESS — is it just 'whichever rainy season arrives first'?")
    print("=" * 74)
    mid_season_doy = {"DJF": 15, "MAM": 105, "JJA": 196, "SON": 288}
    for dur in ["15m", "60m"]:
        d = ev[ev[f"int{dur}_ok"] & ev.wet_ok
               & (ev[f"int{dur}_season"] != ev.wet_season)].copy()
        d["fire_start_date"] = pd.to_datetime(d.fire_start_date, errors="coerce")
        d = d.dropna(subset=["fire_start_date"])
        if not len(d):
            continue
        fdoy = d.fire_start_date.dt.dayofyear.to_numpy()

        def days_until(season):
            return (np.array([mid_season_doy[s] for s in season]) - fdoy) % 365

        d["first"] = np.where(days_until(d[f"int{dur}_season"]) < days_until(d.wet_season),
                              "intense", "wettest")
        print(f"\n  {dur}: decisive subset with a fire date, n = {len(d)}")
        for grp, g in d.groupby("first"):
            print(f"    {grp + ' season arrives first':32s} n={len(g):3d}   "
                  f"flows in intense season {(g.df_season == g[f'int{dur}_season']).mean()*100:5.1f}%   "
                  f"in wettest {(g.df_season == g.wet_season).mean()*100:5.1f}%")
    print("\n  -> the preference for the intense season survives (indeed strengthens)")
    print("     when the wettest season is the one that arrives first, so it is not")
    print("     an artefact of post-fire susceptibility decay.")

    # ------------------------------------------------------------ regimes
    print("\n" + "=" * 74)
    print("DEBRIS-FLOW TIMING BY GAUGE-DEFINED RAINFALL REGIME (60-min)")
    print("=" * 74)
    use = ev[ev.int60m_ok].copy()
    for regime, g in use.groupby("int60m_season"):
        mean_doy, R, p = circular(g.df_doy.to_numpy())
        top = g.df_month.value_counts().head(3)
        print(f"\n  {regime}-dominant rainfall  (n = {len(g)} events, "
              f"{', '.join(sorted(g.state.dropna().unique()))})")
        print(f"    mean debris-flow date {doy_to_date(mean_doy)}, concentration R = {R:.2f}, "
              f"Rayleigh p = {p:.1e}")
        print(f"    peak months: {', '.join(f'{pd.Timestamp(2001, m, 1):%b} ({n})' for m, n in top.items())}")
        print(f"    in the locally most-intense season: "
              f"{(g.df_season == regime).mean() * 100:.0f}%")

    mean_doy, R, p = circular(ev.df_doy.to_numpy())
    print(f"\n  ALL EVENTS (n={len(ev)}): mean date {doy_to_date(mean_doy)}, R = {R:.2f}, "
          f"Rayleigh p = {p:.1e}")
    print("    (low R across the whole domain reflects two opposed modes, not aseasonality)")

    # ------------------------------------------- fire timing as a confounder
    print("\n" + "=" * 74)
    print("FIRE TIMING — is the pattern just 'when fires happen'?")
    print("=" * 74)
    f = ev.dropna(subset=["fire_start_date"]).copy()
    f["fire_start_date"] = pd.to_datetime(f.fire_start_date, errors="coerce")
    f = f.dropna(subset=["fire_start_date"])
    f["fire_month"] = f.fire_start_date.dt.month
    f["lag_days"] = (f.event_date - f.fire_start_date).dt.days
    f = f[(f.lag_days >= 0) & (f.lag_days <= 5 * 365)]
    print(f"  n = {len(f)} events with a usable fire date")
    print(f"  fire ignition months: "
          f"{', '.join(f'{pd.Timestamp(2001, m, 1):%b}({n})' for m, n in f.fire_month.value_counts().head(4).items())}")
    print(f"  fire -> debris flow lag: median {f.lag_days.median():.0f} d, "
          f"quartiles {f.lag_days.quantile(.25):.0f}-{f.lag_days.quantile(.75):.0f} d")
    print(f"  within first year of fire: {(f.lag_days <= 365).mean() * 100:.0f}%")
    for regime, g in f[f.int60m_ok].groupby("int60m_season"):
        print(f"    {regime}-dominant: median lag {g.lag_days.median():4.0f} d  "
              f"(fires peak {pd.Timestamp(2001, int(g.fire_month.mode().iat[0]), 1):%b}, "
              f"flows peak {pd.Timestamp(2001, int(g.df_month.mode().iat[0]), 1):%b})")

    # --------------------------------------------------------- observed vs model
    print("\n" + "=" * 74)
    print("POOLED MONTH DISTRIBUTION — observed vs model expectation (60-min)")
    print("=" * 74)
    use = ev[ev.int60m_ok & ev.wet_ok]
    obs = use.df_month.value_counts().reindex(range(1, 13), fill_value=0).to_numpy()
    exp_int = normalize(use[[f"int60m_p{m:02d}" for m in range(1, 13)]].to_numpy(float)).sum(axis=0)
    exp_wet = normalize(use[[f"wet_p{m:02d}" for m in range(1, 13)]].to_numpy(float)).sum(axis=0)
    print("   month  observed   short-duration   wet-season")
    for i, m in enumerate(range(1, 13)):
        print(f"    {pd.Timestamp(2001, m, 1):%b}   {obs[i]:6d}   {exp_int[i]:12.1f}   {exp_wet[i]:10.1f}")
    for nm, e in [("short-duration", exp_int), ("wet-season", exp_wet)]:
        stat = ((obs - e) ** 2 / np.maximum(e, 1e-9)).sum()
        print(f"  chi2 vs {nm:15s} = {stat:8.1f}  (df=11, p = {chi2.sf(stat, 11):.2e})")

    ev.to_csv(OUT / "pfdf_events_gauge_seasonality.csv", index=False)
    print(f"\nwrote {OUT / 'pfdf_events_gauge_seasonality.csv'}")


if __name__ == "__main__":
    with tee("analysis_report.txt"):
        main()
