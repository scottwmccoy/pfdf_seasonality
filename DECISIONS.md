# Decisions

Choices that are not recoverable from the code, with the evidence behind them.
Newest last.

---

## 2026-08 — Short-duration intensity, not rainfall amount, sets seasonality

Three candidate timing models were scored by total log-likelihood of the
observed event months: uniform (M0), amount (M1, proportional to monthly
normals), and short-duration intensity (M2, proportional to the fraction of
years whose annual maximum fell in that month). M2 wins.

The diagnostic is the **decisive subset**, the events where the locally wettest
and locally most intense seasons differ, because only there do M1 and M2
predict opposite things. On that subset flows follow intensity roughly 9:1
(74% vs 8%).

**Ruled out: susceptibility decay.** If flows merely followed whichever rainy
season arrived first while susceptibility was highest, the intensity preference
should weaken when the wet season arrives first. It strengthens. See
`results/reports/analysis_report.txt`.

## 2026-09 — Gauge-only evidence in the AGU abstract

CONUS404 was dropped from the abstract and held as presentation material. The
gauges outperform every model product (below), the abstract had a hard
2,000-character limit, and a second line of evidence that is weaker than the
first costs space without strengthening the claim.

## 2026-09-18 — Do not acquire the native 15-minute CONUS404 (auxhist24)

Originally argued by **bracketing**: the instantaneous rate (duration -> 0) and
the 1-hour accumulation flank 15 minutes, so wherever they agree the 15-minute
answer is pinned, which was 80.5% of western land.

**That reasoning was later shown to be unsafe**, though its conclusion held.
When the real 15-minute rasters turned out to be on disk already, a direct
per-pixel test showed the 15-minute product is *not* a duration variant of
CONUS404: in California it calls 40.1% of land winter-dominant, outside the
53.7-70.9% span of all four CONUS404 variants. Duration and dataset are
entangled (1980-2021 vs 1980-2024), so the bracket assumption does not hold
across datasets. The metric is not the explanation — recomputing our season on
the exceedance-count metric moves agreement by only about 3 points.

Lesson kept in `CLAUDE.md`: reading a categorical map by eye is not a
substitute for the per-pixel test.

## 2026-09-19 — Gauges are the primary product; reanalysis is for coverage

Eight products scored on the 232 events all of them cover, with the NCEI
normals as a single fixed amount reference so only the intensity side varies:

| | Decisive-subset flows in the predicted intense season |
|---|---|
| Atlas 14 station AMS 15-min, Atlas 14 regional | **71.4%** |
| Every reanalysis product, 15-min raster included | 62-65% |

Gauges also discriminate about twice as sharply (intense-to-wettest near 8:1
against 4:1). Against true 15-minute gauge observations, the 15-minute
reanalysis (84.3%) is indistinguishable from CONUS404 instantaneous rate
(84.4%) and better than 1-hour (79.8%) — finer model resolution buys nothing
we do not already have.

**The reanalysis earns its place on coverage alone.** Atlas 14 never published
Oregon or Washington, which blanks 28 of 349 events; and Volume 1 (semiarid
Southwest) publishes 60-minute annual maxima only, so a 15-minute gauge metric
additionally loses AZ 45, NM 28, UT 4. Gridded products cover 100%.
See `results/reports/seasonality_product_evaluation_report.txt`.

## 2026-09-20 — Repository split from the data tree

Code moved to this git repository; bulk data, generated outputs and
manuscripts stayed in Box, connected by `$PFDF_SEASONALITY_DATA` and
`paths.py`. This mirrors `hma-gof`, keeps 1.6 GB of `.npz` out of git, and
avoids running git inside a cloud-sync folder.

Reports were initially committed to the repository, on the argument that their
diffs show when a number moved. **Reversed the same week**: the repository is
now code only, and `results/figures/` and `results/reports/` sit side by side
in Box. Splitting generated outputs across two trees meant the manuscripts —
which live in Box and cite both — had to reach into the repository for half
of what they referenced, and one directory of outputs is easier to reason
about than two. Provenance is served by the reports being regenerable from
committed code against immutable inputs, not by their being versioned.

**Manuscripts are deliberately not in git** (decided 2026-09-20). The drafts
live in Box and are edited there, so they stay synced across devices; their
version history is whatever Box provides natively, and there are no commit
messages or line-level diffs for prose. This was weighed against moving
`manuscripts/` into this repository — 84 KB of markdown, which would have
given proper diffs — and the workflow cost of editing the paper outside Box
decided it. Do not "fix" this by adding the manuscripts here without asking.

A consequence worth knowing: manuscript edits produce nothing to commit. If a
session changes only the draft or the abstract, `git status` in this repo will
be clean, and that is correct rather than a mistake.

**Verification standard set here.** The migration replaced per-script path
boilerplate in 18 files and consolidated constants duplicated in up to 12, then
re-ran every affected script and diffed all 29 outputs byte-for-byte against
pre-migration copies. That caught one real defect: the shared
`circular_mean_doy` had been written with 365.25 days where the original used
365.0, shifting `clock_doy`, `pred_days` and `residual_d` by a quarter day
across all 308 rows. The original behavior was restored and the inconsistency
documented rather than silently "fixed", because the published seasonal-clock
numbers were produced with 365.0.

## 2026-09-20 — The analysis set: a 2-year window and runoff-generated only

Two scope decisions, made together, that define what the seasonality analysis
treats as a postfire debris flow. Both live in
`src/pfdf_seasonality/events.py` and are applied by `load_events`, which is the
only door into the analysis. The compilation itself keeps every record — it is
a database, not a sample.

**345 compiled events -> 302 within 2 years of fire -> 298 runoff-generated.**

**The window (2 years).** Postfire susceptibility is concentrated in the first
wet season or two; an event five or twenty years later belongs to a different
population. This is a **scope** decision and must be described as one. It is a
poor process filter, which is why the second filter exists: of the 15
landslide-involved events, 8 fall inside 2 years (five within six months —
shallow landsliding in the western Cascades is near-immediate), while the
window discards 35 runoff-generated events, five for every landslide one.

**The initiation filter.** The paper is about runoff-generated flows, so events
a source attributes to landsliding are excluded. `policy="any"` keeps an event
that also has a runoff-generated record; `policy="all"` is the strict variant.

The filter can only act on what the sources report, and **only two of eight
do**: `literature` (`InitiationMechanism`) and `oregon2024` (`Primary_IT`). The
other six — cavagnaro2025, czu2021, dolan2020, graber2023, graber2024,
volumes227 — are set to runoff-generated by our loader on the strength of their
titles and stated scope. That is an assumption, defensible but not an
observation, and the paper says so rather than implying eight sources were
screened.

Two judgment calls inside the filter:

- **`unknown` is kept** (12 events). Dropping it would discard the honest
  labels from the two sources that report the field while keeping 3,222 records
  assumed runoff-generated with no field at all — it would penalize exactly the
  sources that did the work.
- **Mixed landslide+runoff events are kept** (4: Archie Creek and Riverside,
  Oregon). A runoff-generated flow demonstrably occurred on that storm date.
  This is also the conservative choice: all four are in Oregon, which has no
  Atlas 14 coverage, so the **gauge results are bit-identical** under either
  policy, while in CONUS404 — which does cover Oregon — dropping them would
  *improve* the decisive subset from 61.8%/12.7% (n=102) to 64.3%/10.2%
  (n=98). Reporting the weaker number is the right call.

**What this cost the headline numbers.** Everything tightened or held. The
decisive subset did not move at all (76 events, 73.7% vs 7.9%), because none of
the removed events were decisive. The winter regime absorbed the change: n 158
-> 137 (window) -> 134 (filter), mean date 25 -> 29 -> 27 Dec, R 0.46 -> 0.50 ->
0.49. The forecast margin narrowed honestly, clock 41 -> 38 -> 39 d against a
fitted baseline that improved more, 67 -> 54 -> 52 d.

Pinned in `tests/test_headline_numbers.py`, including a structural test that no
landslide-initiated event can reach the analysis under either policy.

## 2026-09-20 — Station Fire date erratum in the Cavagnaro release

The Cavagnaro figshare release
(`data/raw/inventories/cavagnaro_figshare/PublicCodes/DFObsHydroclimatePubNew_attributes.xlsx`)
carries the Station Fire (2009) block with `StormDate` shifted forward by
exactly four years: 11/13/2013 for a storm that happened 2009-11-13. All 489
Station rows are affected, 108 of them `Response == 1`. `scripts/00_compile_inventory.py`
corrects it in `_fix_station_fire_dates`, guarded on the symptom rather than on
the fire name, so a corrected upstream file passes through untouched.

**The evidence.**

1. The shift is upstream, not ours. The loader passes `StormDate` through
   without arithmetic; the spreadsheet itself holds the 2013-2014 dates beside
   `Year = 2009`.
2. Station is the only fire affected. Of 43 fires in the file, 41 have their
   first storm 0 or 1 years after the fire year. Station is at +4. (Boot 2018
   sits at +3 from a personal communication — a separate, 5-row question.)
3. `volumes227`, an independent USGS release, records the same Station Fire
   storms at 2009-11-12, 2009-12-11, 2010-01-18 and 2010-02-06 — Cavagnaro's
   dates minus exactly four years, matching to within a day for four of five.
4. Staley et al. (2016), named in the file as the source for these records,
   tabulates `Station stn California 2009 600 108`: 108 debris-flow basins,
   exactly the 108 `Response == 1` rows. The count survived the transfer; the
   dates did not.

**Why it mattered more than five dates.** The `literature` source (McGuire et
al. 2024) also carries Staley-2016-derived records, correctly dated. With the
four-year offset the compilation held the same observations twice and could not
recognize them, because de-duplication matches on date. Correcting the dates let
them merge:

    records : 3,634 -> 3,526   (104 literature + 4 volumes227 were duplicates)
    events  : 345   -> 339
    Station events now sit in the first winter after the fire, 0.2-0.5 yr,
    where 5 of them had been excluded by the 2-year window at 4.2-4.5 yr

**What it did not change.** The decisive subset is still 76 events at 73.7% vs
7.9%; the robustness control still 74.3 / 86.7; both regime dates, both R
values, 87% / 83%, and 39 vs 52 days are all unmoved. Only denominators and
counts shifted. The database was wrong; the result was not sensitive to it.

**One consequence for our own analysis.** The 4.0-4.5 yr bump in the
exposure-corrected rate panel of `scripts/checks/postfire_lag_cutoff.py` was
this artifact, not a feature of postfire susceptibility. That figure and report
were regenerated.

Found by an independent subagent review, 2026-09-20 (`docs/reviews/compilation.md`
in the Box tree). Reported to the author for correction upstream.
