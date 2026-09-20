"""
How long after ignition does the first debris flow arrive — and what sets it?

Hypothesis (SWM): in summer-intense (monsoon/thunderstorm) regions the
ignition-to-flow interval should be bimodal — days-to-months when the burn
catches its first monsoon, over a year when it misses it — while winter-intense
regions should be tight and unimodal, because a summer burn almost always gets a
triggering storm the following winter.

Test: for every event, compute the local "intense-season clock" from the gauge
intensity climatology (circular mean day-of-year of the 60-min annual-maximum
month distribution, IDW'd to the event). The first opportunity after ignition is

    pred = (clock_doy - ignition_doy) mod 365            [days]

and the number of intense seasons the site skipped before producing its first
flow is

    k = round((observed_lag - pred) / 365)

k <= 0: flow in the first intense season after ignition (k = -1: caught the tail
of the season already underway). k >= 1: skipped k seasons. The hypothesis
predicts P(k >= 1) is substantial in the JJA regime and near zero in the DJF
regime, and that the residual (observed_lag - pred - 365k) is small everywhere,
i.e. the seasonal clock — not elapsed time since fire — sets the interval.

Ignition dates: MTBS Ig_Date where matched (313/349 events); where MTBS and a
conflicting inventory fire_start_date disagree by >30 d without a name match,
the LATER ignition still preceding the flow wins (the most recent burn is what
reset susceptibility); inventory dates fill unmatched events.

  /opt/anaconda3/envs/PointMan/bin/python analyze_ignition_lags.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pfdf_seasonality.paths import PROCESSED, REPORTS, EVENTS_MTBS
from pfdf_seasonality.seasons import SEASONS as _SEASONS, circular_mean_doy
from pfdf_seasonality.report import tee

MTBS = EVENTS_MTBS
GAUGE = PROCESSED / "seasonality" / "pfdf_events_gauge_seasonality.csv"
OUT_CSV = PROCESSED / "seasonality" / "pfdf_events_ignition_lags.csv"

SEASONS = list(_SEASONS)


def main():
    mt = pd.read_csv(MTBS, parse_dates=["event_date", "fire_start_date",
                                        "mtbs_ig_date"])
    gz = pd.read_csv(GAUGE)
    cols = ["event_id", "int60m_ok", "int60m_season"] + \
           [f"int60m_p{m:02d}" for m in range(1, 13)]
    ev = mt.merge(gz[cols], on="event_id", how="left")

    # ---------------------------------------------- choose the ignition date
    inv, mtb = ev.fire_start_date, ev.mtbs_ig_date
    inv_valid = inv.notna() & (inv <= ev.event_date)
    conflict = mtb.notna() & inv_valid & ~ev.mtbs_name_match.fillna(False) \
        & ((mtb - inv).dt.days.abs() > 30)
    ign = mtb.copy()
    ign[conflict] = np.maximum(mtb[conflict], inv[conflict])   # latest burn wins
    ign[mtb.isna()] = inv[mtb.isna()]
    ev["ign_date"] = ign
    ev["ign_source"] = np.select(
        [mtb.notna() & ~conflict, conflict, mtb.isna() & inv_valid],
        ["mtbs", "conflict_latest", "inventory"], default="none")

    ok = (ev.ign_date.notna()
          & ~ev.date_precision.astype(str).str.contains("suspect")
          & ev.int60m_ok.fillna(False).astype(bool))
    dropped = len(ev) - ok.sum()
    ev = ev[ok].copy()
    ev["lag_days"] = (ev.event_date - ev.ign_date).dt.days
    ev = ev[ev.lag_days.between(0, 1826)].copy()

    # ----------------------------------------- the local intense-season clock
    P = ev[[f"int60m_p{m:02d}" for m in range(1, 13)]].to_numpy(float)
    P = P / P.sum(axis=1, keepdims=True)
    ev["clock_doy"] = circular_mean_doy(P)
    ev["ign_doy"] = ev.ign_date.dt.dayofyear.clip(upper=365)
    ev["pred_days"] = (ev.clock_doy - ev.ign_doy) % 365
    ev["k_skipped"] = np.round((ev.lag_days - ev.pred_days) / 365).astype(int)
    ev["residual_d"] = ev.lag_days - ev.pred_days - 365 * ev.k_skipped
    ev["regime"] = ev.int60m_season

    # first flow per fire = the interval to the first triggering storm
    ev["fire_id"] = ev.mtbs_event_id.fillna(ev.fire_key)
    first = ev.sort_values("event_date").groupby("fire_id", as_index=False).first()

    ev.to_csv(OUT_CSV, index=False)

    # ------------------------------------------------------------- report
    print("=" * 74)
    print("IGNITION -> FIRST DEBRIS FLOW: the interval and what sets it")
    print("=" * 74)
    print(f"usable events: {len(ev)} (excluded {dropped}: no ignition date, "
          f"suspect event date, or no gauge regime)")
    print(f"ignition source: {ev.ign_source.value_counts().to_dict()}")
    print(f"fires represented: {ev.fire_id.nunique()}  "
          f"(first-flow-per-fire n = {len(first)})")

    for name, d in [("ALL EVENTS", ev), ("FIRST FLOW PER FIRE", first)]:
        print(f"\n--- {name} ---")
        print(f"{'regime':>8} {'n':>4} {'median':>7} {'IQR':>12} "
              f"{'<=180d':>7} {'180-300':>8} {'300-730':>8} {'>730d':>6}")
        for s in SEASONS:
            g = d[d.regime == s]
            if len(g) < 5:
                continue
            q1, q3 = g.lag_days.quantile([0.25, 0.75])
            f = lambda lo, hi: (g.lag_days.between(lo, hi)).mean() * 100
            print(f"{s:>8} {len(g):>4} {g.lag_days.median():>6.0f}d "
                  f"{q1:>5.0f}-{q3:<5.0f}d {f(0,180):>6.0f}% {f(181,300):>7.0f}% "
                  f"{f(301,730):>7.0f}% {f(731,9999):>5.0f}%")

    print("\n--- INTENSE SEASONS SKIPPED BEFORE THE FIRST FLOW (per fire) ---")
    print("k <= 0: flow in the first intense season after ignition")
    tab = pd.crosstab(first.regime, first.k_skipped.clip(-1, 2))
    tab = tab.reindex(index=[s for s in SEASONS if s in tab.index], fill_value=0)
    tab.columns = [{-1: "k=-1", 0: "k=0", 1: "k=1", 2: "k>=2"}[c] for c in tab.columns]
    print(tab.to_string())
    for s in ["DJF", "JJA"]:
        g = first[first.regime == s]
        if len(g):
            print(f"  {s}: caught the first intense season {100*(g.k_skipped<=0).mean():.0f}%"
                  f"   skipped >=1 season {100*(g.k_skipped>=1).mean():.0f}%   (n={len(g)})")

    print("\n--- DOES THE SEASONAL CLOCK, NOT ELAPSED TIME, SET THE DATE? ---")
    for s in ["DJF", "JJA"]:
        g = ev[ev.regime == s]
        print(f"  {s}: median |observed - clock prediction| = "
              f"{g.residual_d.abs().median():.0f} d   (n={len(g)}; "
              f"vs 91 d expected if timing were random in the year)")

    print("\n--- JJA REGIME: WHICH BURNS CATCH THEIR FIRST MONSOON? (per fire) ---")
    j = first[first.regime == "JJA"].copy()
    j["ign_month"] = j.ign_date.dt.month
    j["branch"] = np.where(j.k_skipped <= 0, "first season", "skipped")
    tb = pd.crosstab(j.ign_month, j.branch)
    tb["% first season"] = (tb.get("first season", 0)
                            / tb.sum(axis=1) * 100).round(0).astype(int)
    print(tb.to_string())

    print("\n--- MTBS vs inventory validation (matched events with both) ---")
    b = ev[ev.mtbs_ig_date.notna() & ev.fire_start_date.notna()]
    dd = (b.mtbs_ig_date - b.fire_start_date).dt.days
    print(f"  n = {len(b)}, median |delta| = {dd.abs().median():.0f} d, "
          f"within 7 d: {100*(dd.abs()<=7).mean():.0f}%")
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    with tee("ignition_lag_report.txt"):
        main()
