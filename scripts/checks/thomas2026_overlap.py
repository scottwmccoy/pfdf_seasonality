"""What does the Thomas et al. (2026) damage register add to our compilation?

Thomas, M.A., Kostelnik, J., and Bombard, C.W., 2026, A national register of
damage and losses associated with flash floods and debris flows after wildfire:
U.S. Geological Survey data release, doi:10.5066/P14FCFRE. Described in Thomas,
M.A., Kostelnik, J., and Bombard, C.W., 2026, Mounting damage and losses in the
United States from post-wildfire flash floods and debris flows: Communications
Earth & Environment, doi:10.1038/s43247-026-03988-w.

It is a different kind of inventory from the ones we already use, in three ways
that decide how much of it we can take:

  1. Its inclusion criterion is DAMAGE, not a mapped flow. It records events
     that hit something people care about, so it reaches back to 1879 and picks
     up historical events no geomorphic inventory mapped, while omitting
     mapped flows that damaged nothing.
  2. A record is a WILDFIRE (occasionally several), not a debris flow, and each
     carries 1-6 windows of damaging activity. Exploding those windows gives
     fire x date pairs, which is exactly our event unit.
  3. Coordinates are the FIRE centroid, not the flow location. Anything adopted
     from here is therefore `location_quality = fire_scale`.

Matching is on normalized fire name plus date, using the compilation's own
`norm_fire` so the two sides are treated identically, with a spatial fallback
for records whose fire name does not match anything we hold.

  python scripts/checks/thomas2026_overlap.py

Writes compiled/thomas2026_overlap_report.txt and a candidate table of the
fire x date pairs that are unique to this source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfdf_seasonality.paths import INVENTORIES, PROCESSED, REPORTS  # noqa: E402
from pfdf_seasonality.seasons import to_xy  # noqa: E402

importlib_src = Path(__file__).resolve().parents[1] / "00_compile_inventory.py"

SRC = INVENTORIES / "thomas2026_damage" / "US_PFDF_DL_v1.csv"
ENCODING = "cp1252"          # an 'n-tilde' in a source name; not UTF-8
DAY_TOL = 3                  # date tolerance when matching to our events
NEAR_KM = 50.0               # spatial fallback: fire centroid to event location
WEST_STATES = {"California", "Colorado", "Arizona", "New Mexico", "Utah", "Nevada",
               "Idaho", "Montana", "Wyoming", "Oregon", "Washington"}

REPORT: list[str] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    REPORT.append(msg)


def load_norm_fire():
    """Borrow norm_fire + FIRE_ALIASES from the compiler, so names normalize alike."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_compile", importlib_src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.norm_fire


def split_list(value) -> list[str]:
    """Comma-delimited field -> list, dropping 'NR' and blanks."""
    if pd.isna(value):
        return []
    return [p.strip() for p in str(value).split(",")
            if p.strip() and p.strip().upper() != "NR" and p.strip().lower() != "nan"]


def explode_to_fire_dates(t: pd.DataFrame, norm_fire) -> pd.DataFrame:
    """One row per (record, fire name, damaging-activity date).

    Windows are almost always a single day (96%); where min != max we take the
    window midpoint and record the span so it can be filtered downstream.
    """
    rows = []
    for _, r in t.iterrows():
        names = split_list(r.Fire_Name) or [""]
        mins, maxs = split_list(r.Flow_Date_Min), split_list(r.Flow_Date_Max)
        for i, dmin in enumerate(mins):
            dmax = maxs[i] if i < len(maxs) else dmin
            a = pd.to_datetime(dmin, format="%Y%m%d", errors="coerce")
            b = pd.to_datetime(dmax, format="%Y%m%d", errors="coerce")
            if pd.isna(a):
                continue
            if pd.isna(b):
                b = a
            span = int((b - a).days)
            mid = a + pd.Timedelta(days=span // 2)
            for nm in names:
                rows.append(dict(
                    thomas_id=r.ID, state=r.State, fire_name=nm,
                    fire_name_norm=norm_fire(nm),
                    flow_date=mid, window_days=span,
                    fire_latitude=pd.to_numeric(r.Fire_Latitude, errors="coerce"),
                    fire_longitude=pd.to_numeric(r.Fire_Longitude, errors="coerce"),
                    fatality=r.Fatality_Indicator == "Y",
                    injury=r.Injury_Indicator == "Y",
                    dwelling=r.Dwelling_Indicator == "Y",
                    damaged_asset=r.Damaged_Asset_Type,
                ))
    out = pd.DataFrame(rows).drop_duplicates(subset=["fire_name_norm", "flow_date"])
    return out.reset_index(drop=True)


def main() -> None:
    norm_fire = load_norm_fire()

    t = pd.read_csv(SRC, dtype=str, encoding=ENCODING)
    ev = pd.read_csv(PROCESSED / "inventory" / "pfdf_events_compiled.csv",
                     low_memory=False, parse_dates=["event_date"])
    ev["fire_name_norm"] = ev.fire_name.map(norm_fire)

    say("=" * 74)
    say("THOMAS ET AL. (2026) DAMAGE REGISTER vs OUR COMPILATION")
    say("=" * 74)
    say(f"  register: {len(t)} records, {t.State.nunique()} states "
        f"({int(t.State.isin(WEST_STATES).sum())} in the 11 western states)")
    say(f"  ours    : {len(ev)} events, {ev.fire_name_norm.nunique()} distinct fire names")

    fd = explode_to_fire_dates(t, norm_fire)
    say()
    say(f"  exploded to {len(fd)} unique (fire, damaging-date) pairs")
    say(f"    single-day windows : {int((fd.window_days == 0).sum())} "
        f"({(fd.window_days == 0).mean() * 100:.0f}%)")
    say(f"    in western states  : {int(fd.state.isin(WEST_STATES).sum())}")
    say(f"    date range         : {fd.flow_date.min().date()} to {fd.flow_date.max().date()}")

    # ---------------------------------------------------------------- matching
    ours = ev.dropna(subset=["event_date"])
    by_name: dict[str, np.ndarray] = {
        k: g.event_date.to_numpy() for k, g in ours.groupby("fire_name_norm")
    }
    xy_ours = to_xy(ours.latitude.to_numpy(), ours.longitude.to_numpy())

    match_kind, match_detail = [], []
    for _, r in fd.iterrows():
        dates = by_name.get(r.fire_name_norm)
        hit = None
        if dates is not None and len(dates):
            dd = np.abs((dates - np.datetime64(r.flow_date)) / np.timedelta64(1, "D"))
            if dd.min() <= DAY_TOL:
                hit = ("name+date", f"|dt|={dd.min():.0f}d")
        if hit is None and np.isfinite(r.fire_latitude) and np.isfinite(r.fire_longitude):
            same_day = ours[(ours.event_date - r.flow_date).abs()
                            <= pd.Timedelta(days=DAY_TOL)]
            if len(same_day):
                p = to_xy([r.fire_latitude], [r.fire_longitude])[0]
                d_km = np.hypot(*(xy_ours[same_day.index] - p).T)
                if d_km.min() <= NEAR_KM:
                    hit = ("date+near", f"{d_km.min():.0f} km")
        match_kind.append(hit[0] if hit else "unique")
        match_detail.append(hit[1] if hit else "")
    fd["match"] = match_kind
    fd["match_detail"] = match_detail

    say()
    say("-" * 74)
    say(f"  OVERLAP (fire name normalized alike; +/-{DAY_TOL} d; "
        f"spatial fallback {NEAR_KM:.0f} km)")
    say("-" * 74)
    for k, n in fd.match.value_counts().items():
        say(f"    {k:12s} {n:5d}  ({n / len(fd) * 100:.0f}%)")

    uniq = fd[fd.match == "unique"].copy()
    uw = uniq[uniq.state.isin(WEST_STATES)]
    say()
    say(f"  unique pairs: {len(uniq)}  (western: {len(uw)})")
    say(f"    pre-1984 (before MTBS): {int((uniq.flow_date.dt.year < 1984).sum())}")
    say(f"    single-day windows    : {int((uniq.window_days == 0).sum())}")
    say(f"    with a fatality flag  : {int(uniq.fatality.sum())}")
    say()
    say("  unique pairs by decade:")
    dec = (uniq.flow_date.dt.year // 10 * 10).value_counts().sort_index()
    for d, n in dec.items():
        say(f"    {int(d)}s  {n:4d}  {'#' * min(n, 50)}")
    say()
    say("  unique pairs by state:")
    for s, n in uniq.state.value_counts().items():
        say(f"    {s:<16s} {n:4d}")

    say()
    say("  earliest 12 unique pairs (the historical tail this source adds):")
    for _, r in uniq.nsmallest(12, "flow_date").iterrows():
        say(f"    {r.flow_date.date()}  {r.state:<12s} {r.fire_name[:38]:<38s} "
            f"{r.thomas_id}")

    # --------------------------------------------- can these be merged at all?
    say()
    say("=" * 74)
    say("SHOULD THESE BE MERGED INTO THE DEBRIS-FLOW COMPILATION? No.")
    say("=" * 74)
    say("  The register's own ProcessSteps, step 2, states plainly:")
    say('      "The records do not distinguish flash floods from debris flows."')
    say("  Our compilation is a debris-flow database, so adopting these records")
    say("  would silently mix in an unknown number of flash floods.")
    say()
    known = set(ev.fire_name_norm) - {""}
    uniq["on_known_df_fire"] = uniq.fire_name_norm.isin(known)
    say(f"  unique pairs on a fire already known to produce debris flows: "
        f"{int(uniq.on_known_df_fire.sum())}")
    say(f"  unique pairs on a fire we have never seen                   : "
        f"{int((~uniq.on_known_df_fire).sum())}")
    say()
    say("  The decisive evidence is seasonal. Merging would change the answer:")
    mu = uniq.flow_date.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    mo = ev.event_date.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    say(f"    {'':6s} {'Thomas-unique':>14s} {'ours':>8s}")
    for season, months in [("DJF", [12, 1, 2]), ("MAM", [3, 4, 5]),
                           ("JJA", [6, 7, 8]), ("SON", [9, 10, 11])]:
        say(f"    {season:6s} {mu[months].sum() / mu.sum() * 100:13.0f}% "
            f"{mo[months].sum() / mo.sum() * 100:7.0f}%")
    say()
    say(f"  Adding {len(uniq)} pairs to {len(ev)} events would make this source "
        f"{len(uniq) / (len(uniq) + len(ev)) * 100:.0f}% of the database and")
    say("  pull the pooled distribution hard toward summer. Some of that is real")
    say("  (damage reporting is dense in the monsoon Southwest: NM, CO and AZ")
    say("  supply most unique pairs, where our database is California-weighted),")
    say("  and some is the flash-flood admixture. The two cannot be separated")
    say("  with what the register publishes, so the shift cannot be interpreted.")
    say()
    say("  Verdict: keep as an independent table, not a compilation source.")
    say("  Defensible uses that do NOT require differentiating the two processes:")
    say("    - an independent check on the regime map, since damaging events")
    say("      should cluster in the same seasons as mapped flows;")
    say("    - a consequence layer for the hazard-timing argument (fatalities,")
    say("      dwellings), which is about impact rather than process;")
    say("    - the pre-1984 historical tail, which no mapped inventory reaches.")

    out = PROCESSED / "inventory" / "thomas2026_unique_candidates.csv"
    uniq.sort_values("flow_date").to_csv(out, index=False)
    say()
    say(f"wrote {out}")
    (REPORTS / "thomas2026_overlap_report.txt").write_text("\n".join(REPORT) + "\n")
    print("wrote", REPORTS / "thomas2026_overlap_report.txt")


if __name__ == "__main__":
    main()
