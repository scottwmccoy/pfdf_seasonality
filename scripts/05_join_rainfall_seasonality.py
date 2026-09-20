"""
Attach short-duration rainfall seasonality to every compiled post-fire
debris-flow event, from two independent angles:

  A. NOAA Atlas 14 PFDS seasonality (60-min) at the event location — the
     regional product the PFDS supplementary page plots.
  B. Nearest Atlas 14 stations' 15-min annual-maximum seasonality — true
     short-duration, at station resolution, inverse-distance weighted over the
     k nearest stations within a search radius.

Then compares the month the debris flow actually happened against the month
distribution of intense short-duration rainfall at that location.

  /opt/anaconda3/envs/PointMan/bin/python join_pfdf_rainfall_seasonality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pfdf_seasonality import atlas14 as a
from pfdf_seasonality.paths import ATLAS14, PROCESSED, EVENTS
from pfdf_seasonality.seasons import SEASONS, LAT0, to_xy

OUT = PROCESSED / "seasonality"
OUT.mkdir(exist_ok=True)

K_NEAREST = 5
MAX_KM = 150.0


# Fixed reference latitude for the equirectangular projection. It MUST be the
# same for stations and query points, or the two sets land in different
# coordinate systems and every distance is wrong.


def station_seasonality_at(points: pd.DataFrame, stn: pd.DataFrame, dur: str):
    """Inverse-distance-weighted month fractions from the k nearest stations."""
    s = stn[(stn.duration == dur) & ~stn.short_record].reset_index(drop=True)
    tree = cKDTree(to_xy(s.lat.to_numpy(), s.lon.to_numpy()))
    q = to_xy(points.latitude.to_numpy(), points.longitude.to_numpy())
    k = min(K_NEAREST, len(s))
    dist, idx = tree.query(q, k=k)
    dist = np.atleast_2d(dist.T).T
    idx = np.atleast_2d(idx.T).T

    mcols = [f"month_frac_{m:02d}" for m in range(1, 13)]
    M = s[mcols].to_numpy()
    w = 1.0 / np.maximum(dist, 1.0) ** 2
    w[dist > MAX_KM] = 0.0
    wsum = w.sum(axis=1, keepdims=True)
    ok = (wsum[:, 0] > 0)
    frac = np.full((len(points), 12), np.nan)
    frac[ok] = (w[ok, :, None] * M[idx[ok]]).sum(axis=1) / wsum[ok]

    out = pd.DataFrame(frac, columns=[f"{dur}_monthfrac_{m:02d}" for m in range(1, 13)])
    out[f"{dur}_n_stations_used"] = (w > 0).sum(axis=1)
    out[f"{dur}_nearest_station_km"] = dist[:, 0]
    for name, months in SEASONS.items():
        out[f"{dur}_{name.lower()}_frac"] = frac[:, [m - 1 for m in months]].sum(axis=1)
    seas = out[[f"{dur}_{n.lower()}_frac" for n in SEASONS]].to_numpy()
    lead = np.array(list(SEASONS))[np.argmax(seas, axis=1)]
    out[f"{dur}_dominant_season"] = np.where(np.isnan(seas).all(axis=1), None, lead)
    return out


def main():
    ev = pd.read_csv(EVENTS, parse_dates=["event_date"])
    print(f"{len(ev)} compiled debris-flow events")

    # --- A. Atlas 14 regional seasonality (60-min) at each event location
    print("\nQuerying Atlas 14 seasonality (60-min) at event locations ...")
    # own cache file so this can run alongside a grid probe without clobbering it
    ssn = a.seasonality_table(
        ev, dur="60m", workers=4,
        cache_file=ATLAS14 / "atlas14_seasonality_cache_events.json",
    )
    for c in ["covered", "region", "n_stations", "cum_years", "region_id"]:
        ev["a14_" + c] = ssn[c].values
    for m in range(1, 13):
        ev[f"a14_aep1_2_m{m:02d}"] = ssn[f"aep1_2_m{m:02d}"].values
    aep = ssn[[f"aep1_2_m{m:02d}" for m in range(1, 13)]].to_numpy(dtype=float)
    allnan = np.isnan(aep).all(axis=1)          # uncovered points (OR, WA, ...)
    safe = np.where(np.isnan(aep), -np.inf, aep)
    dom = np.full(len(ev), np.nan)
    dom[~allnan] = safe[~allnan].argmax(axis=1) + 1
    ev["a14_dominant_month"] = dom

    tot = np.nansum(aep, axis=1)
    rows = np.arange(len(ev))
    in_df_month = aep[rows, ev.event_date.dt.month.to_numpy() - 1]
    # share of the location's 60-min exceedances that fall in the month the
    # debris flow occurred
    ev["a14_share_in_df_month"] = np.where(
        (~allnan) & (tot > 0), in_df_month / np.where(tot == 0, np.nan, tot), np.nan
    )

    # --- B. nearest-station short-duration seasonality
    print("\nInterpolating station annual-maximum seasonality ...")
    stn = pd.read_csv(OUT / "atlas14_station_seasonality.csv")
    for dur in ["15m", "30m", "60m"]:
        got = station_seasonality_at(ev, stn, dur)
        for c in got.columns:
            ev[c] = got[c].values
        print(f"  {dur}: {got[f'{dur}_n_stations_used'].gt(0).sum()}/{len(ev)} events "
              f"within {MAX_KM:.0f} km of a station; median nearest "
              f"{got[f'{dur}_nearest_station_km'].median():.0f} km")

    ev.to_csv(OUT / "pfdf_events_with_rainfall_seasonality.csv", index=False)
    print(f"\nwrote {OUT / 'pfdf_events_with_rainfall_seasonality.csv'}")

    # --- summary
    print("\n" + "=" * 72)
    print("Atlas 14 60-min seasonality coverage of the debris-flow events")
    print(f"  covered      : {int(ev.a14_covered.sum())}/{len(ev)}")
    print(f"  not covered  : {int((~ev.a14_covered).sum())} "
          f"(states: {ev.loc[~ev.a14_covered, 'state'].value_counts().to_dict()})")
    print(f"  distinct Atlas 14 sub-regions hit: "
          f"{ev.loc[ev.a14_covered, 'region_id'].nunique() if 'region_id' in ev else ev.a14_region_id.nunique()}")

    print("\nDebris-flow month vs. season of intense 15-min rainfall")
    d = ev[ev["15m_dominant_season"].notna()].copy()
    d["df_season"] = d.event_date.dt.month.map(
        {m: s for s, ms in SEASONS.items() for m in ms}
    )
    ct = pd.crosstab(d["15m_dominant_season"], d.df_season)
    ct = ct.reindex(index=list(SEASONS), columns=list(SEASONS), fill_value=0)
    print("  rows = dominant season of annual max 15-min rainfall,")
    print("  cols = season the debris flow occurred")
    print(ct.to_string())
    agree = np.trace(ct.to_numpy()) / ct.to_numpy().sum() * 100
    print(f"\n  debris flow in the locally most intense 15-min rainfall season: {agree:.0f}%")


if __name__ == "__main__":
    main()
