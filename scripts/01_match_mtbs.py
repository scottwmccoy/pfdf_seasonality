"""
Match compiled PFDF events to MTBS fire perimeters for independent ignition dates.

MTBS (Monitoring Trends in Burn Severity) maps US fires >= ~1,000 acres in the
West from 1984 on, each with a perimeter polygon and an ignition date (Ig_Date).
Matching every debris-flow event to its causal fire gives a uniform,
inventory-independent ignition date; the lag analysis
(scripts/11_analyze_ignition_lags.py) uses it to quantify the interval
between ignition and the first triggering storm.

Matching rule, per event:
  candidates = fires whose perimeter contains the event point or lies within
               NEAR_KM, with Ig_Date <= event_date <= Ig_Date + WINDOW_D days
  if the inventory names the fire and any candidate matches that name, keep only
  name-matched candidates; then prefer containment over proximity, then the most
  recent ignition (reburns: the latest fire is the one that reset the clock).

Pre-1984 fires and fires below the MTBS size threshold cannot match; the lag
analysis falls back to the inventory fire_start_date for those events.

  /opt/anaconda3/envs/PointMan/bin/python match_mtbs_ignitions.py
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pfdf_seasonality.paths import MTBS, PROCESSED, EVENTS

MTBS_ZIP = MTBS / "mtbs_perimeter_data.zip"
OUT = PROCESSED / "inventory" / "pfdf_events_mtbs.csv"

NEAR_KM = 5.0            # tolerance for points just outside the mapped perimeter
WINDOW_D = 1826          # candidate fires ignited up to 5 yr before the event
ALBERS = "EPSG:5070"     # CONUS Albers, meters

STOP_TOKENS = {"FIRE", "FIRES", "COMPLEX", "THE", "RX", "WF", "PRESCRIBED"}


def norm_tokens(name) -> set[str]:
    if pd.isna(name):
        return set()
    toks = re.sub(r"[^A-Z0-9]+", " ", str(name).upper()).split()
    return {t for t in toks if t not in STOP_TOKENS and len(t) >= 3}


def main():
    ev = pd.read_csv(EVENTS, parse_dates=["event_date", "fire_start_date"])
    pts = gpd.GeoDataFrame(
        ev[["event_id", "fire_name", "event_date", "fire_start_date"]],
        geometry=gpd.points_from_xy(ev.longitude, ev.latitude), crs="EPSG:4326")

    pad = 0.5
    bbox = (ev.longitude.min() - pad, ev.latitude.min() - pad,
            ev.longitude.max() + pad, ev.latitude.max() + pad)
    print(f"loading MTBS perimeters in bbox {tuple(round(v, 1) for v in bbox)} ...",
          flush=True)
    per = gpd.read_file(f"zip://{MTBS_ZIP}", bbox=bbox,
                        columns=["event_id", "incid_name", "incid_type",
                                 "burnbndac", "ig_date"])
    per = per.rename(columns={"event_id": "mtbs_id"})   # ours is the PFDF event id
    per["ig_date"] = pd.to_datetime(per.ig_date, errors="coerce")
    per = per[per.ig_date.notna()].reset_index(drop=True)
    print(f"  {len(per)} perimeters with ignition dates "
          f"({per.ig_date.min():%Y-%m-%d} to {per.ig_date.max():%Y-%m-%d})",
          flush=True)

    per = per.to_crs(ALBERS)
    per["geometry"] = per.geometry.make_valid()
    pts = pts.to_crs(ALBERS)

    # candidate pairs within NEAR_KM (containment = distance 0)
    buf = pts.copy()
    buf["geometry"] = buf.geometry.buffer(NEAR_KM * 1000)
    pairs = gpd.sjoin(buf, per, predicate="intersects", how="inner")
    pairs = pairs.merge(per[["geometry"]], left_on="index_right",
                        right_index=True, suffixes=("", "_fire"))
    pt_geom = pts.geometry.loc[pairs.index]
    pairs["dist_km"] = gpd.GeoSeries(pairs.geometry_fire, crs=ALBERS
                                     ).distance(pt_geom, align=False) / 1000.0
    pairs["lag_days"] = (pairs.event_date - pairs.ig_date).dt.days
    pairs = pairs[(pairs.lag_days >= 0) & (pairs.lag_days <= WINDOW_D)]

    ev_tokens = pairs.fire_name.map(norm_tokens)
    mt_tokens = pairs.incid_name.map(norm_tokens)
    pairs["name_match"] = [bool(a and b and (a & b))
                           for a, b in zip(ev_tokens, mt_tokens)]

    rows = []
    for eid, g in pairs.groupby("event_id"):
        if g.name_match.any():
            g = g[g.name_match]
        contains = g[g.dist_km == 0]
        pick_from = contains if len(contains) else g
        best = pick_from.sort_values("ig_date", ascending=False).iloc[0]
        rows.append({
            "event_id": eid,
            "mtbs_event_id": best.mtbs_id,
            "mtbs_incid_name": best.incid_name,
            "mtbs_incid_type": best.incid_type,
            "mtbs_burnbnd_ac": best.burnbndac,
            "mtbs_ig_date": best.ig_date,
            "mtbs_dist_km": round(float(best.dist_km), 2),
            "mtbs_name_match": bool(best.name_match),
            "mtbs_lag_days": int(best.lag_days),
            "mtbs_n_candidates": len(g),
            "mtbs_ambiguous": len(contains) > 1,
        })
    m = pd.DataFrame(rows)
    out = ev.merge(m, on="event_id", how="left")
    out.to_csv(OUT, index=False)

    # ------------------------------------------------------------- summary
    matched = out.mtbs_event_id.notna()
    post84 = out.event_date >= "1984-01-01"
    print(f"\nmatched {matched.sum()}/{len(out)} events "
          f"({matched[post84].sum()}/{post84.sum()} of MTBS-era events)")
    print(f"  containment: {(out.mtbs_dist_km == 0).sum()}, "
          f"near (<{NEAR_KM:.0f} km): {(out.mtbs_dist_km > 0).sum()}")
    named = out.fire_name.notna() & matched
    print(f"  name agreement where inventory names the fire: "
          f"{out.loc[named, 'mtbs_name_match'].mean() * 100:.0f}% of {named.sum()}")
    print(f"  ambiguous (>1 containing fire in window): {out.mtbs_ambiguous.sum()}")

    both = out[matched & out.fire_start_date.notna()].copy()
    both["delta_d"] = (both.mtbs_ig_date - both.fire_start_date).dt.days
    print(f"\nMTBS Ig_Date vs inventory fire_start_date (n={len(both)}):")
    print(f"  median |delta| {both.delta_d.abs().median():.0f} d; "
          f"|delta| > 30 d: {(both.delta_d.abs() > 30).sum()}")
    worst = both.reindex(both.delta_d.abs().sort_values(ascending=False).index)
    cols = ["fire_name", "mtbs_incid_name", "fire_start_date", "mtbs_ig_date",
            "delta_d"]
    print(worst[cols].head(8).to_string(index=False))

    un = out[post84 & ~matched & out.fire_name.notna()]
    print(f"\nunmatched MTBS-era events with a named fire ({len(un)}):")
    print(un.groupby("fire_name").size().sort_values(ascending=False)
          .head(15).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
