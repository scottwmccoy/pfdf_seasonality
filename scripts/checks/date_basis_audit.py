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
        print("\n  READ THIS COLUMN CAREFULLY. A large gap is NOT evidence of an")
        print("  error, and this signal is the weakest of the four. It measures")
        print("  only that two sources share no common date at a fire, which is")
        print("  the normal outcome when a burn scar produces flows in several")
        print("  seasons and each source documents different ones.")
        print()
        print("  Worked example, run down 2026-09-20: cavagnaro2025 vs graber2024")
        print("  flagged at a median 552 d over two fires. Both are real.")
        print("    Cameron Peak (CO, 2020 fire): flows in Jul 2021, Jul 2022 and")
        print("      Jul 2023 - three monsoon seasons, four sources between them;")
        print("      cavagnaro holds 2021, graber2024 holds 2023, and graber2023")
        print("      and volumes227 bridge the gap.")
        print("    Dixie (CA, 2021 fire): Oct 2021, Jun 2022, Jun 2023. The Dixie")
        print("      data release names two of them in its own title.")
        print("  Use this signal to pick candidates, never to conclude.")
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

    # ---------------------------------------------------------------- 5
    print("\n" + "-" * 76)
    print("5. CHRONOLOGY — can the flow date follow the fire at all?")
    print("-" * 76)
    print("  Unambiguous, unlike the signals above: a flow cannot precede its")
    print("  own fire, and a lag of decades is a data-entry error somewhere.")
    b = rec.dropna(subset=["fire_start_date", "event_date"]).copy()
    b["lag"] = (b.event_date - b.fire_start_date).dt.days
    before = b[b.lag < 0]
    late = b[b.lag > 1826]
    print(f"\n  records with a fire date : {len(b)}")
    print(f"  flow BEFORE its own fire : {len(before)}"
          + (f"  -> {before.groupby('source_key').size().to_dict()}" if len(before) else ""))
    print(f"  flow >5 years after fire : {len(late)}"
          + (f"  -> {late.groupby('source_key').size().to_dict()}" if len(late) else ""))
    for _, r_ in before.iterrows():
        print(f"    IMPOSSIBLE  {r_.record_id}  fire {r_.fire_start_date:%Y-%m-%d} -> "
              f"flow {r_.event_date:%Y-%m-%d}  ({r_.lag:,} d)")
    if len(before):
        print("\n  Known case, upstream not ours: literature EventID 61 carries")
        print("  DateFireStart 13-Sep-2013 with DateOfFlow 1914, sourced to Eaton")
        print("  (1935). A 1913 fire mis-entered as 2013 fits every other field.")
        print("  It does not reach the results: the analysis keys on flow date,")
        print("  and the lag analysis drops it.")

    if len(late):
        print("\n  The long lags are NOT errors, with one exception. Every one is")
        print("  LANDSLIDE-initiated, and landslide susceptibility after fire")
        print("  peaks years later through root decay, where runoff-generated")
        print("  susceptibility peaks in the first wet season. Checked 2026-09-20:")
        print("    litevent327/326  CA San Gabriels, 1960 fire -> 1965 and 1969")
        print("                     (Rice & Foggin 1971; the Jan 1969 storms)")
        print("    litevent325      Boise NF, 1989 fire -> 1997 New Year flood")
        print("                     (Shaub 2001, 'Landslides and wildfire')")
        print("    litevent342      1987 fire -> Jan 1997; MTBS finds a 1996 fire")
        print("                     here and the lag analysis already re-dates it")
        print("    litevent346      1973 fire -> 1995 flow, cited to Gray (1981).")
        print("                     IMPOSSIBLE: the reference predates the flow by")
        print("                     14 years. A genuine upstream error.")

    # ----------------------------------------------- initiation sensitivity
    print("\n" + "-" * 76)
    print("6. PROCESS MIXING — are landslide-initiated flows in the analysis?")
    print("-" * 76)
    # Sources that carry an initiation field in the published data. Everything
    # else is set to "runoff-generated" by our loader from the source's title
    # and stated scope — defensible, but an assumption, not an observation.
    OBSERVES_INITIATION = {"literature": "InitiationMechanism",
                           "oregon2024": "Primary_IT"}
    print("  Which sources actually observe initiation mechanism?\n")
    print(f"    {'source':<16} {'field':<20} {'classes present'}")
    for src, grp in rec.groupby("source_key"):
        field = OBSERVES_INITIATION.get(src, "-- none (assumed by loader) --")
        counts = grp.initiation_class.value_counts().to_dict()
        print(f"    {src:<16} {field:<20} {counts}")
    assumed = sorted(set(rec.source_key) - set(OBSERVES_INITIATION))
    n_assumed = int(rec.source_key.isin(assumed).sum())
    print(f"\n  {len(OBSERVES_INITIATION)} of {rec.source_key.nunique()} sources report the field. "
          f"The other {len(assumed)} contribute\n  {n_assumed} records whose class is an ASSUMPTION. "
          "Say so in the paper rather than\n  implying every source was screened.")
    # If a source outside the declared set ever shows a non-runoff class, the
    # loader changed and this constant is stale.
    stray = rec[~rec.source_key.isin(OBSERVES_INITIATION)
                & (rec.initiation_class != "runoff-generated")]
    if len(stray):
        print(f"\n  !! {len(stray)} records from {sorted(set(stray.source_key))} carry a "
              "non-runoff class\n     but are not in OBSERVES_INITIATION — update this constant.")

    ls = ev.initiation_class.astype(str).str.contains("landslide|mixed", na=False)
    print(f"\n  events involving landslide initiation: {int(ls.sum())} of {len(ev)}")
    print(f"  their months: {sorted(ev[ls].event_date.dt.month.tolist())}")
    print("  They skew hard to the wet season, which is what a saturation-driven")
    print("  process should do.\n")

    # What actually reaches the analysis, computed rather than remembered.
    from pfdf_seasonality.events import load_events
    stages = [("compiled*", dict(window_years=None, runoff_policy=None)),
              ("+ 2-yr window", dict(runoff_policy=None)),
              ("+ runoff filter", dict())]
    print(f"    {'stage':<18} {'events':>7} {'landslide-involved':>20}")
    for label, kw in stages:
        sub = load_events(verbose=False, **kw)
        n_ls = int(sub.initiation_class.astype(str)
                   .str.contains("landslide|mixed", na=False).sum())
        print(f"    {label:<18} {len(sub):>7} {n_ls:>20}")
    print("    * 344, not 345: one event is dated before its own fire and is")
    print("      dropped as impossible before either filter runs.")
    print("\n  The window alone is a POOR process filter — it keeps more than half")
    print("  the landslide events (western Cascades shallow landslides arrive in")
    print("  months, not years) while discarding five runoff-generated events for")
    print("  every landslide one. The initiation filter is what does this job.")
    print("  Events still counted as landslide-involved at the last stage are the")
    print("  mixed ones kept under policy='any': each also has a runoff-generated")
    print("  record, so a runoff-generated flow did occur on that storm date.")

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
