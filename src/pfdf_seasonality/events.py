"""Load compiled events scoped to the postfire, runoff-generated population.

Every analysis takes its events through here, so the two scope decisions —
how long after the fire, and which initiation process — live in one place
instead of being re-implemented per script.

No post-fire window
-------------------
`POSTFIRE_WINDOW_YEARS = None`. Every event is analyzed regardless of how long
after the fire it occurred. This reverses a 2-year window applied earlier on
2026-09-20, for two reasons that both point the same way.

**It could not be defended quantitatively.** The attempt is preserved in
`scripts/checks/postfire_lag_cutoff.py`. Two of its three arguments do not
survive scrutiny: the exposure correction used a per-*source* horizon, which
credits every fire in the literature database with observation through 2021 and
so understates the tail; and the season-mix argument compared the retained
sample against the full sample, which is a NESTED comparison whose difference
goes to zero as the cutoff grows by construction. On first flows per fire the
hazard is already at its plateau by 1.5 yr, so 1.5 was as defensible as 2.0 —
which is another way of saying neither was.

**It does not change the answer.** Across every cutoff from 0.5 yr to none, the
decisive subset holds at 71-75% intense against 5-9% wettest. A filter that
changes nothing except the sample size is a liability in review, not a
safeguard.

Removing it readmits 30 events, the longest 4.2 yr after its fire. None is
landslide-initiated: 27 are runoff-generated and 3 unknown. The landslide tail
that the window was incidentally catching is now handled where it belongs, by
the initiation filter below.

`load_events(window_years=2.0)` reproduces the windowed sample for a
sensitivity table.

The initiation filter
---------------------
`RUNOFF_POLICY = "any"`, set 2026-09-20. The paper is about runoff-generated
postfire debris flows, so events that the source attributes to landsliding do
not belong in the sample.

Only two of the eight point sources observe initiation mechanism:
`literature` (`InitiationMechanism`) and `oregon2024` (`Primary_IT`). The other
six — cavagnaro2025, czu2021, dolan2020, graber2023, graber2024, volumes227 —
are set to "runoff-generated" by the loader on the strength of their titles and
scope, not because the source carries a field. That is an assumption, and it is
recorded here so nobody later mistakes those 3,222 records for observations of
process. See `scripts/checks/date_basis_audit.py`, section 6.

Two consequences follow, and both are deliberate:

* **"unknown" is kept.** 16 events carry only `unknown` records, all from the
  two sources that actually report the field. Dropping them would discard the
  honest labels while keeping 3,222 records assumed runoff-generated with no
  field at all — it would penalize exactly the sources that did the work.
* **Mixed events are kept under `policy="any"`.** Four Oregon events (Archie
  Creek and Riverside, 2021-2022) aggregate both landslide and runoff-generated
  records on one storm date. A runoff-generated debris flow demonstrably
  occurred on that date, so the date belongs in the sample; `policy="all"`
  drops them instead, for a sensitivity test. The four are winter events in the
  Cascades, the region with no Atlas 14 coverage, so the choice is not neutral
  and is reported in the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import PROCESSED

__all__ = ["POSTFIRE_WINDOW_YEARS", "RUNOFF_POLICY", "LANDSLIDE_CLASSES",
           "load_events", "add_postfire_interval", "filter_runoff_generated"]

#: None = no post-fire window; see the module docstring. Pass 2.0 explicitly
#: for the sensitivity table.
POSTFIRE_WINDOW_YEARS: float | None = None

#: `initiation_class` values that mean the source attributed the flow, wholly
#: or partly, to landslide initiation. "mixed" is the harmonized class for
#: literature records described as "runoff-generated and landslide": the
#: single record is both, so it cannot be split the way a mixed event can.
LANDSLIDE_CLASSES = ("landslide", "mixed")

#: "any" — keep an event if any of its records is runoff-generated.
#: "all" — require every record to be runoff-generated (mixed events dropped).
RUNOFF_POLICY = "any"


def add_postfire_interval(ev: pd.DataFrame) -> pd.DataFrame:
    """Attach `ignition_date`, `ignition_source` and `years_since_fire`.

    Ignition comes from MTBS where the event matched, otherwise from the
    inventory's own fire start date.
    """
    mt_path = PROCESSED / "inventory" / "pfdf_events_mtbs.csv"
    out = ev.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out["fire_start_date"] = pd.to_datetime(out.get("fire_start_date"), errors="coerce")
    ign = pd.Series(pd.NaT, index=out.index)
    if mt_path.exists():
        mt = pd.read_csv(mt_path, low_memory=False, parse_dates=["mtbs_ig_date"])
        # `event_id` is positional -- EV00000 upward in row order -- so it is
        # not stable across recompilations. A stale MTBS file therefore does
        # not fail to join; it joins to the WRONG events and silently attaches
        # the wrong ignition dates. Check identity, not just presence.
        missing = set(out.event_id) - set(mt.event_id)
        if missing:
            raise SystemExit(
                f"{mt_path.name} is stale: {len(missing)} of {len(out)} compiled "
                f"events are absent from it (e.g. {sorted(missing)[:3]}). "
                "`event_id` is positional, so a stale file misattributes "
                "ignition dates rather than failing. Rerun 01_match_mtbs.py.")
        shared = out[["event_id", "fire_key", "event_date"]].merge(
            mt[["event_id", "fire_key", "event_date"]].assign(
                event_date=lambda d: pd.to_datetime(d.event_date, errors="coerce")),
            on="event_id", suffixes=("", "_mt"))
        drifted = (shared.fire_key.ne(shared.fire_key_mt)
                   | shared.event_date.ne(shared.event_date_mt)).sum()
        if drifted:
            raise SystemExit(
                f"{mt_path.name} is stale: {drifted} event ids point at a "
                "different fire or date than the compilation does. Rerun "
                "01_match_mtbs.py.")
        m = out[["event_id"]].merge(mt[["event_id", "mtbs_ig_date"]], on="event_id",
                                    how="left")
        ign = pd.to_datetime(m.mtbs_ig_date).to_numpy()
        ign = pd.Series(ign, index=out.index)
    out["ignition_source"] = np.where(ign.notna(), "mtbs",
                                      np.where(out.fire_start_date.notna(),
                                               "inventory", "unknown"))
    out["ignition_date"] = ign.fillna(out.fire_start_date)
    out["years_since_fire"] = ((out.event_date - out.ignition_date).dt.days / 365.25)
    return out


def _class_set(s: pd.Series) -> pd.Series:
    """The pipe-joined `initiation_class` of an event, as a set per row."""
    return s.fillna("unknown").astype(str).str.split("|").map(
        lambda parts: {p.strip() for p in parts if p.strip()})


def filter_runoff_generated(ev: pd.DataFrame, policy: str = RUNOFF_POLICY,
                            verbose: bool = True) -> pd.DataFrame:
    """Drop events the source attributes to landslide initiation.

    `policy="any"` keeps an event that has at least one runoff-generated
    record; `policy="all"` keeps only events with no landslide record at all.
    `policy=None` is a no-op, for sensitivity tests. Events whose records are
    all `unknown` are kept under every policy — see the module docstring.
    """
    if policy is None:
        if verbose:
            print(f"{len(ev)} events; no initiation filter applied")
        return ev.reset_index(drop=True)
    if policy not in ("any", "all"):
        raise ValueError(f"policy must be 'any', 'all' or None, got {policy!r}")

    classes = _class_set(ev.get("initiation_class", pd.Series("unknown", index=ev.index)))
    has_landslide = classes.map(lambda c: bool(c & set(LANDSLIDE_CLASSES)))
    has_runoff = classes.map(lambda c: "runoff-generated" in c)

    drop = has_landslide if policy == "all" else (has_landslide & ~has_runoff)
    kept = ev[~drop]
    if verbose:
        n_mixed = int((has_landslide & has_runoff).sum())
        print(f"{len(ev)} events -> {len(kept)} runoff-generated "
              f"(policy={policy!r}: dropped {int(drop.sum())} landslide-initiated; "
              f"{n_mixed} mixed landslide+runoff "
              f"{'kept' if policy == 'any' else 'dropped'}; "
              f"{int(classes.map(lambda c: c == {'unknown'}).sum())} unknown kept)")
    return kept.reset_index(drop=True)


def load_events(window_years: float | None = POSTFIRE_WINDOW_YEARS,
                runoff_policy: str | None = RUNOFF_POLICY,
                verbose: bool = True) -> pd.DataFrame:
    """Compiled events within the postfire window, runoff-generated only.

    `window_years=None` and `runoff_policy=None` each disable their filter, for
    sensitivity tests. Events with no ignition date are KEPT (they cannot be
    judged, and there is only one); events dated before their own fire are
    dropped as impossible.
    """
    ev = pd.read_csv(PROCESSED / "inventory" / "pfdf_events_compiled.csv",
                     low_memory=False, parse_dates=["event_date"])
    ev = add_postfire_interval(ev)
    n0 = len(ev)

    impossible = ev.years_since_fire < 0
    ev = ev[~impossible]

    if window_years is None:
        if verbose:
            print(f"{n0} compiled events; no post-fire window applied "
                  f"({int(impossible.sum())} dropped as dated before their fire)")
    else:
        too_late = ev.years_since_fire > window_years
        kept = ev[~too_late]
        if verbose:
            print(f"{n0} compiled events -> {len(kept)} within {window_years:g} yr of fire "
                  f"({int(too_late.sum())} beyond the window, "
                  f"{int(impossible.sum())} dated before their fire, "
                  f"{int(kept.years_since_fire.isna().sum())} with no ignition date, kept)")
        ev = kept

    return filter_runoff_generated(ev, policy=runoff_policy, verbose=verbose)
