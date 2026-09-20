"""Screen every source for dates that are surveys rather than storms.

The Dolan inventory was dated by Google Earth image acquisition and field
visits, which split one atmospheric river into four events and put a July
"debris flow" on the coastal-California coast. That was caught by reading a
README. This script looks for the same failure everywhere else, without
needing to read ten READMEs.

Four independent signals, none conclusive alone:

  1. DECLARED   what the loader says `date_basis` is. Ground truth where the
                source documents it; the other three signals exist to catch a
                source whose documentation we have mis-read.
  2. CONCENTRATION  a survey campaign visits on a few days and records many
                features per day. A storm-dated source spreads across many
                dates. Reported as records per distinct date, and the share of
                a source's records falling on its single busiest date.
  3. DISAGREEMENT   where two sources describe the same fire, do they agree on
                when? A survey date and a storm date for one fire disagree by
                weeks to months. This is the strongest signal because it needs
                no assumption about climate.
  4. IMPLAUSIBILITY  how likely is the observed month under the LOCAL intense-
                rainfall climatology? A field visit scheduled for good weather
                lands in the local dry season. Screening only: a source with
                genuinely off-season flows would also flag, so this ranks
                suspicion, it does not prove anything.

  python scripts/checks/date_basis_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfdf_seasonality.paths import PROCESSED, REPORTS  # noqa: E402
from pfdf_seasonality.report import tee  # noqa: E402


def main() -> None:
    rec = pd.read_csv(PROCESSED / "inventory" / "pfdf_occurrence_records_compiled.csv",
                      low_memory=False, parse_dates=["event_date", "fire_start_date"])
    ev = pd.read_csv(PROCESSED / "inventory" / "pfdf_events_compiled.csv",
                     low_memory=False, parse_dates=["event_date"])
    seas = pd.read_csv(PROCESSED / "seasonality" / "pfdf_events_with_rainfall_seasonality.csv",
                       low_memory=False)

    print("=" * 76)
    print("DATE-BASIS AUDIT — is any source dated by survey rather than by storm?")
    print("=" * 76)

    # ---------------------------------------------------------- 1 + 2
    print("\n" + "-" * 76)
    print("1-2. DECLARED BASIS AND DATE CONCENTRATION")
    print("-" * 76)
    print(f"  {'source':16s} {'basis':20s} {'records':>8s} {'dates':>6s} "
          f"{'rec/date':>9s} {'busiest date':>13s}")
    rows = []
    for key, g in rec.groupby("source_key"):
        basis = g.date_basis.dropna().unique()
        basis = basis[0] if len(basis) else "unknown"
        ndates = g.event_date.nunique()
        per = len(g) / max(ndates, 1)
        busiest = g.event_date.value_counts().iloc[0] / len(g) * 100
        rows.append(dict(source=key, basis=basis, n=len(g), ndates=ndates,
                         per=per, busiest=busiest))
        # Concentration alone cannot separate "one survey campaign" from "one
        # storm": czu2021 and the corrected dolan2020 both sit at 100% on a
        # single date and both are right. Only raise it where the basis is
        # not already established from the source's own documentation.
        unestablished = basis in ("observation", "unknown")
        flag = "  <-- survey-like" if (per > 50 and busiest > 40 and unestablished) else ""
        print(f"  {key:16s} {basis:20s} {len(g):8d} {ndates:6d} {per:9.0f} "
              f"{busiest:12.0f}%{flag}")
    conc = pd.DataFrame(rows)
    print("\n  'survey-like' = concentrated dates AND no basis documented by the")
    print("  source. Concentration alone proves nothing: czu2021 (33 records on")
    print("  one date) and the corrected dolan2020 (2,080 on one date) are both")
    print("  single real storms. Before its fix, Dolan scored 536 records per")
    print("  date across four dates, which is the pattern that should flag.")

    # ---------------------------------------------------------------- 3
    print("\n" + "-" * 76)
    print("3. CROSS-SOURCE DISAGREEMENT ON THE SAME FIRE")
    print("-" * 76)
    print("  For every fire described by two or more sources, the spread of")
    print("  dates each pair reports. Storm dates agree within days.")
    pairs = []
    for fire, g in rec.groupby("group_key"):
        if g.source_key.nunique() < 2:
            continue
        by = g.groupby("source_key").event_date.agg(["min", "max"])
        keys = list(by.index)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                gap = min(abs((by.loc[a, m1] - by.loc[b, m2]).days)
                          for m1 in ("min", "max") for m2 in ("min", "max"))
                pairs.append(dict(fire=fire, a=a, b=b, gap_days=gap))
    P = pd.DataFrame(pairs)
    if len(P):
        agg = (P.groupby(["a", "b"]).gap_days
               .agg(fires="size", median="median", worst="max")
               .reset_index().sort_values("median", ascending=False))
        print(f"\n  {'source A':16s} {'source B':16s} {'fires':>6s} "
              f"{'median gap':>11s} {'worst':>7s}")
        for _, r in agg.iterrows():
            flag = "  <-- investigate" if r["median"] > 14 else ""
            print(f"  {r.a:16s} {r.b:16s} {r.fires:6d} {r['median']:10.0f}d "
                  f"{r.worst:6.0f}d{flag}")
        print("\n  A large gap is not automatically an error: one fire can produce")
        print("  debris flows in several storms, each documented by a different")
        print("  source. It means the two sources share no common date, which is")
        print("  worth a look. cavagnaro2025 vs graber2024 (median 552 d over 2")
        print("  fires) is the strongest such case and has not been run down.")
    else:
        print("  (no fire is described by more than one source)")

    # ---------------------------------------------------------------- 4
    print("\n" + "-" * 76)
    print("4. CLIMATOLOGICAL IMPLAUSIBILITY OF THE OBSERVED MONTH")
    print("-" * 76)
    print("  P(month) under the local 60-min intense-rainfall climatology.")
    print("  Screening only — a genuinely off-season flow also scores low.")
    mcols = [f"60m_monthfrac_{m:02d}" for m in range(1, 13)]
    if all(c in seas.columns for c in mcols):
        s = seas.dropna(subset=mcols).copy()
        s["month"] = pd.to_datetime(s.event_date).dt.month
        M = s[mcols].to_numpy(float)
        M = M / np.clip(M.sum(axis=1, keepdims=True), 1e-9, None)
        s["p_month"] = M[np.arange(len(s)), s.month.to_numpy() - 1]
        src = (ev.set_index("event_id").sources.str.split("|").explode()
               .rename("source").reset_index())
        j = s.merge(src, on="event_id")
        print(f"\n  {'source':16s} {'events':>7s} {'median P(month)':>16s} "
              f"{'% below chance':>15s}")
        for key, g in j.groupby("source"):
            below = (g.p_month < 1 / 12).mean() * 100
            flag = "  <-- implausible" if below > 45 else ""
            print(f"  {key:16s} {len(g):7d} {g.p_month.median():15.3f} "
                  f"{below:14.0f}%{flag}")
        print("\n  chance = 1/12 = 0.083")
    else:
        print("  (monthly climatology columns not present; run 05_join first)")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    declared = conc[conc.basis == "observation"].source.tolist()
    suspect = conc[(conc.per > 50) & (conc.busiest > 40)
                   & conc.basis.isin(["observation", "unknown"])].source.tolist()
    print(f"  declared survey-dated : {declared or 'none'}")
    print(f"  flagged by concentration: {suspect or 'none'}")
    ev_flag = ev[ev.survey_dated == True] if "survey_dated" in ev.columns else ev.iloc[:0]
    print(f"  events carrying a survey-dated record: {len(ev_flag)} of {len(ev)}")
    if len(ev_flag):
        print(f"    states: {ev_flag.state.value_counts().to_dict()}")
    print("\n  Sources declared `observation` cannot be re-dated without an external")
    print("  storm date. Until one exists they are usable for WHERE a debris flow")
    print("  happened and unreliable for WHEN, and `survey_dated` marks the events")
    print("  they touch so timing work can exclude them.")


if __name__ == "__main__":
    with tee("date_basis_audit_report.txt"):
        main()
