"""Pinned headline numbers — a durable regression baseline for the results.

Why this exists
---------------
Generated reports are not in git (they live in the Box tree under
``results/reports/``), so there is no committed artifact to diff a refactor
against. This file pins the numbers that are actually quoted in the AGU
abstract and the manuscript, parsed out of the reports. If a refactor moves one
of them, a test fails with the old and new values side by side.

This is the cheap durable version of the byte-for-byte diff used during the
2026-09-20 migration, which caught a real quarter-day shift in
``circular_mean_doy``. It guards the conclusions rather than every byte.

Updating a pin
--------------
A failure here is **not** a licence to edit the expected value. Either the
change is a bug, or it is a deliberate result change — new inventory data, a
corrected method — in which case:

1. Confirm the new number is right, from the report and the code that made it.
2. Update the pin here **and** every place the old number is quoted: the
   manuscript draft, ``DECISIONS.md``, and the frozen AGU abstract's
   "Numbers that could shift" section.
3. Say in the commit message which number moved and why.

Skipped, not failed, when the Box tree is unavailable — a fresh clone on
another machine has the code but not the reports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pytest

from pfdf_seasonality.paths import REPORTS


@dataclass(frozen=True)
class Pin:
    report: str
    label: str            # what the number means, shown on failure
    pattern: str          # regex with one capturing group
    expected: float | str
    tol: float = 0.0      # absolute tolerance; 0 means exact
    after: str | None = None   # only search past this marker (disambiguates repeated lines)
    quoted_in: str = ""   # where the number appears in the writing
    group: int = 1


PINS: list[Pin] = [
    # ---------------------------------------------------------- the database
    Pin("compilation_report.txt", "unique fire-by-date events",
        r"unique events\s+:\s+(\d+)", 320, quoted_in="abstract: 'over 300'"),
    Pin("compilation_report.txt", "debris-flow records after de-duplication",
        r"after de-duplication\s+:\s+(\d+)", 3526, quoted_in="abstract: 'over 3,500'"),

    # ------------------------------------------------ the analysis-stage scope
    # Guards the two filters in `pfdf_seasonality.events`. If either is silently
    # disabled, these move before any result does.
    Pin("analysis_report.txt", "events inside the 2-year post-fire window",
        r"-> (\d+) within 2 yr of fire", 282,
        quoted_in="paper methods: the post-fire window"),
    Pin("analysis_report.txt", "events after the runoff-generated filter",
        r"-> (\d+) runoff-generated", 278,
        quoted_in="paper methods: landslide-initiated events excluded"),

    # -------------------------------------------- the core result (60-min gauge)
    Pin("analysis_report.txt", "60-min M2 minus M1 log-likelihood",
        r"M2 - M1 = \+([\d.]+)", 119.3, tol=0.05,
        after="60m short-duration model",
        quoted_in="abstract: 'intensity model beats both'"),
    Pin("analysis_report.txt", "decisive subset size (60-min)",
        r"60m: (\d+)/\d+ events where wettest", 67,
        quoted_in="abstract: '28% of events'"),
    Pin("analysis_report.txt", "events scored at 60-min",
        r"60m: \d+/(\d+) events where wettest", 250,
        quoted_in="abstract: '28% of events' denominator"),
    Pin("analysis_report.txt", "decisive flows in the most-intense season",
        r"debris flow in the most-intense-60m season\s*:\s*([\d.]+)%", 74.6, tol=0.05,
        quoted_in="abstract: '74%' of the 9:1"),
    Pin("analysis_report.txt", "decisive flows in the wettest season",
        r"debris flow in the wettest season\s*:\s*([\d.]+)%", 6.0, tol=0.05,
        after="60m: 67/250", quoted_in="abstract: '8%' of the 9:1"),

    # ------------------------------------------------------------- the regimes
    Pin("analysis_report.txt", "winter regime, n events",
        r"DJF-dominant rainfall\s+\(n = (\d+)", 126),
    # anchored on the regime header, not on each other's values, so a change in
    # R does not also break the date pin with a misleading "wording changed"
    Pin("analysis_report.txt", "winter regime mean date",
        r"mean debris-flow date (\d+ \w+)", "24 Dec",
        after="DJF-dominant rainfall", quoted_in="abstract: '27 Dec'"),
    Pin("analysis_report.txt", "winter regime concentration R",
        r"concentration R = ([\d.]+)", 0.46, tol=0.005,
        after="DJF-dominant rainfall"),
    Pin("analysis_report.txt", "summer regime, n events",
        r"JJA-dominant rainfall\s+\(n = (\d+)", 123),
    Pin("analysis_report.txt", "summer regime mean date",
        r"mean debris-flow date (\d+ \w+)", "02 Aug",
        after="JJA-dominant rainfall", quoted_in="abstract: '2 Aug'"),
    Pin("analysis_report.txt", "summer regime concentration R",
        r"concentration R = ([\d.]+)", 0.88, tol=0.005,
        after="JJA-dominant rainfall"),

    # -------- the anti-susceptibility-decay control: wettest-first must be HIGHER
    Pin("analysis_report.txt", "intense-season-first: flows in intense season",
        r"intense season arrives first\s+n=\s*\d+\s+flows in intense season\s+([\d.]+)%",
        79.3, tol=0.05, after="60m: decisive subset with a fire date"),
    Pin("analysis_report.txt", "wettest-season-first: flows in intense season",
        r"wettest season arrives first\s+n=\s*\d+\s+flows in intense season\s+([\d.]+)%",
        85.7, tol=0.05, after="60m: decisive subset with a fire date",
        quoted_in="abstract: 'strengthens when the wet season arrives first'"),

    # -------------------------------------------------- fire-to-first-flow lags
    Pin("ignition_lag_report.txt", "winter: first flow in the first intense season",
        r"DJF: caught the first intense season (\d+)%", 88,
        quoted_in="abstract: '84% (winter)'"),
    Pin("ignition_lag_report.txt", "summer: first flow in the first intense season",
        r"JJA: caught the first intense season (\d+)%", 83,
        quoted_in="abstract: '76% (summer)'"),
    Pin("ignition_lag_report.txt", "winter: median |observed - clock|",
        r"DJF: median \|observed - clock prediction\| = (\d+) d", 41),
    Pin("ignition_lag_report.txt", "summer: median |observed - clock|",
        r"JJA: median \|observed - clock prediction\| = (\d+) d", 20),

    # ------------------------------------------------- the a priori forecast
    Pin("first_flow_prediction_report.txt", "pooled median error, clock forecast",
        r"median \|error\|, clock\s+\d+d\s+\d+d\s+(\d+)d", 39,
        quoted_in="abstract: '41-day median error'"),
    Pin("first_flow_prediction_report.txt", "pooled median error, fitted constant lag",
        r"median \|error\|, const-lag fit\s+\d+d\s+\d+d\s+(\d+)d", 50,
        quoted_in="abstract: 'beating a constant-lag model (67 days)'"),
    Pin("first_flow_prediction_report.txt", "summer nearest-branch median error",
        r"JJA: P\(first season\) = \d+%;\s+nearest-branch median \|error\| (\d+) d", 23,
        quoted_in="paper: monsoon is date-predictable"),
    Pin("first_flow_prediction_report.txt", "winter window coverage",
        r"DJF: window width median \d+ mo \(\d+% of the year\); first-window coverage (\d+)%",
        69, quoted_in="paper: winter is window-predictable"),

    # --------------------------------------------------- the product choice
    Pin("seasonality_product_evaluation_report.txt",
        "15-min reanalysis vs Atlas 14 15-min gauges",
        r"15-min reanalysis\s+([\d.]+)", 84.3, tol=0.05,
        quoted_in="paper Section 6"),
    Pin("seasonality_product_evaluation_report.txt",
        "CONUS404 rate vs Atlas 14 15-min gauges",
        r"CONUS404 rate\s+([\d.]+)", 84.4, tol=0.05,
        quoted_in="paper Section 6: finer resolution buys nothing"),
    Pin("seasonality_product_evaluation_report.txt",
        "CONUS404 1-hour vs Atlas 14 15-min gauges",
        r"CONUS404 1-hour\s+([\d.]+)", 79.8, tol=0.05),
    Pin("seasonality_product_evaluation_report.txt",
        "best gauge product on the decisive subset (fixed amount reference)",
        r"A14-AMS\s+15-min\s+\d+\s+[\d.]+ \|\s+\d+\s+([\d.]+)", 75.5, tol=0.05,
        after="FIXED AMOUNT REFERENCE", quoted_in="paper: gauges 71.4% vs models 62-65%"),
    Pin("seasonality_product_evaluation_report.txt",
        "15-min vs CONUS404 1-hour agreement",
        r"15-min vs CONUS404 1-hour :\s+([\d.]+)%", 80.4, tol=0.05),
    Pin("seasonality_product_evaluation_report.txt",
        "15-min vs CONUS404 rate agreement",
        r"15-min vs CONUS404 rate   :\s+([\d.]+)%", 86.6, tol=0.05),
    Pin("seasonality_product_evaluation_report.txt",
        "California winter-dominant share, 15-min product",
        r"15-min reanalysis, exceedance :\s+([\d.]+)%", 40.1, tol=0.05,
        quoted_in="DECISIONS: outside the CONUS404 span, so not a duration variant"),

    # ------------------------------ CONUS404 as the second line of evidence
    Pin("conus404_hourly_analysis_report.txt", "decisive subset size (1-hour)",
        r"(\d+)/270 events where wettest season", 84),
    Pin("conus404_hourly_analysis_report.txt", "decisive flows in the intense season",
        r"debris flow in the most-intense season :\s+([\d.]+)%", 57.1, tol=0.05),
    Pin("conus404_hourly_analysis_report.txt", "decisive flows in the wettest season",
        r"debris flow in the wettest season\s+:\s+([\d.]+)%", 11.9, tol=0.05),
]


def _read(name: str) -> str:
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{path} not present - the Box data tree is not available here")
    return path.read_text()


def _extract(pin: Pin) -> str:
    text = _read(pin.report)
    if pin.after:
        _, sep, rest = text.partition(pin.after)
        assert sep, f"marker {pin.after!r} not found in {pin.report}"
        text = rest
    m = re.search(pin.pattern, text)
    assert m, (f"could not find {pin.label!r} in {pin.report} "
               f"using {pin.pattern!r} - did the report's wording change?")
    return m.group(pin.group)


@pytest.mark.parametrize("pin", PINS, ids=lambda p: f"{p.report.split('_')[0]}:{p.label}")
def test_headline_number_has_not_moved(pin: Pin):
    found = _extract(pin)
    where = f"  (quoted in {pin.quoted_in})" if pin.quoted_in else ""
    if isinstance(pin.expected, str):
        assert found == pin.expected, (
            f"{pin.label}: report says {found!r}, pinned {pin.expected!r}{where}")
    else:
        got = float(found)
        assert got == pytest.approx(pin.expected, abs=pin.tol), (
            f"{pin.label}: report says {got}, pinned {pin.expected}{where}")


def test_the_9_to_1_ratio_still_holds():
    """The abstract's headline claim, as a relationship rather than two numbers."""
    intense = float(_extract(next(p for p in PINS
                                  if p.label.startswith("decisive flows in the most-intense season")
                                  and p.report == "analysis_report.txt")))
    wettest = float(_extract(next(p for p in PINS
                                  if p.label.startswith("decisive flows in the wettest")
                                  and p.report == "analysis_report.txt")))
    assert intense / wettest > 8.0, (
        f"intensity-to-amount ratio fell to {intense / wettest:.1f}:1; "
        "the abstract claims roughly 9:1")


def test_intensity_preference_strengthens_when_the_wet_season_comes_first():
    """Rules out decaying susceptibility. If this inverts, the paper's control fails."""
    intense_first = float(_extract(next(
        p for p in PINS if p.label.startswith("intense-season-first"))))
    wettest_first = float(_extract(next(
        p for p in PINS if p.label.startswith("wettest-season-first"))))
    assert wettest_first > intense_first, (
        f"wettest-first {wettest_first}% is no longer above intense-first "
        f"{intense_first}%; the susceptibility-decay control no longer holds")


def test_gauges_still_beat_every_reanalysis_product():
    """The reason the abstract is gauge-only."""
    text = _read("seasonality_product_evaluation_report.txt")
    _, sep, rest = text.partition("FIXED AMOUNT REFERENCE")
    assert sep, "the fixed-amount-reference table is missing from the report"
    rows = re.findall(r"^\s{4}(\S.*?)\s{2,}(\d+)\s+([\d.]+) \|\s+\d+\s+([\d.]+)",
                      rest, re.M)
    gauges = [float(d) for name, _, _, d in rows if name.startswith("A14")]
    models = [float(d) for name, _, _, d in rows if name.startswith(("C404", "R15"))]
    assert gauges and models, f"could not parse the comparison table (got {rows[:2]})"
    assert max(gauges) > max(models), (
        f"best gauge product {max(gauges)}% no longer beats the best reanalysis "
        f"product {max(models)}%; the gauge-only framing would need revisiting")


def test_no_landslide_initiated_event_reaches_the_analysis():
    """The paper is about runoff-generated debris flows.

    `load_events` is the only door into the analysis, so an event the source
    attributed to landsliding must not come through it. Under the default
    `policy="any"` an event may still carry a landslide record provided it also
    carries a runoff-generated one — the storm date is then a real
    runoff-generated occurrence — so the assertion is 'no landslide WITHOUT
    runoff', not 'no landslide at all'. `policy="all"` is the stricter variant
    and is checked too.
    """
    import pandas as pd
    from pfdf_seasonality.events import (LANDSLIDE_CLASSES, _class_set,
                                         load_events)
    from pfdf_seasonality.paths import PROCESSED
    if not (PROCESSED / "inventory" / "pfdf_events_compiled.csv").exists():
        pytest.skip("compiled events not present")

    def offenders(ev: pd.DataFrame, allow_mixed: bool) -> pd.DataFrame:
        cls = _class_set(ev.initiation_class)
        bad = cls.map(lambda c: bool(c & set(LANDSLIDE_CLASSES)))
        if allow_mixed:
            bad &= cls.map(lambda c: "runoff-generated" not in c)
        return ev[bad]

    for policy, allow_mixed in [("any", True), ("all", False)]:
        ev = load_events(runoff_policy=policy, verbose=False)
        bad = offenders(ev, allow_mixed)
        assert bad.empty, (
            f"policy={policy!r} let {len(bad)} landslide-initiated events through: "
            f"{bad.event_id.tolist()[:5]}")

    # and the filter must actually be doing something, or the guarantee above
    # would be vacuous the day the compilation stops recording initiation
    unfiltered = load_events(runoff_policy=None, verbose=False)
    assert len(offenders(unfiltered, allow_mixed=True)) > 0, (
        "no landslide-initiated events in the compilation at all — either the "
        "inventory changed or initiation_class stopped being populated")


def test_no_event_is_silently_survey_dated():
    """Dolan was dated by field visits and image acquisitions for weeks before
    anyone noticed. Any event built on an observation-dated record must carry
    the `survey_dated` flag so timing work can exclude it."""
    import pandas as pd
    from pfdf_seasonality.paths import PROCESSED
    f = PROCESSED / "inventory" / "pfdf_events_compiled.csv"
    if not f.exists():
        pytest.skip("compiled events not present")
    ev = pd.read_csv(f, low_memory=False)
    assert "date_basis" in ev.columns and "survey_dated" in ev.columns, (
        "the compilation must record what each event's date means")
    flagged = ev.date_basis.fillna("").str.contains("observation")
    flagged_actual = ev.survey_dated.astype("boolean").fillna(False).astype(bool)
    assert (flagged_actual == flagged).all(), (
        "survey_dated disagrees with date_basis")
    # Dolan specifically: one storm, not four survey dates
    dolan = ev[ev.fire_key.astype(str).str.contains("dolan", case=False, na=False)]
    assert len(dolan) <= 2, (
        f"Dolan has {len(dolan)} events; the segment inventory should sit on the "
        "2021-01-27 storm, not on its survey dates")
