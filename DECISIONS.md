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

## 2026-09-20 — Literature fire identity: EventID is a storm, not a fire

`load_literature` keyed every record `litevent<EventID>` because the source
carries no fire name. But the source defines EventID as "debris flows grouped
together as being part of a single debris-flow event" — a STORM. Using it as
the fire key produced three distinct defects at once:

1. **One fire split across many keys.** Grizzly Creek (2020-08-10) held seven
   EventIDs; Station six. Every fire-level statistic was wrong: events per
   fire, first flow per fire, and any clustering correction.
2. **Many fires under one key.** 21 of 397 EventIDs span more than one
   `DateFireStart`. EventID 312 covers Wildwood (22-Sep-1997) and Wohlford
   (02-Aug-1997), 94 km apart, joined only by a shared storm on 01-Dec-1997.
3. **No merge with named inventories.** A literature record could never join
   the same fire under its real name, so a flow reported by both the literature
   database and a named inventory became two events.

`resolve_literature_fire_keys` now derives fire identity from what the source
does carry — `DateFireStart` plus location — by union-find over records within
5 km whose start dates agree within 7 days, then adopting a named fire's key
where one lies within the same tolerances. Named records are never modified.

Thresholds were measured, not guessed: every literature record matching a named
fire on start date sits within 4.7 km of it (median 1.5), with nothing else
inside a 40 km search radius.

    973 literature records -> 118 fires (36 matched a named fire)
    54 fires joined that had been split across EventIDs
    8 EventIDs separated that had covered more than one fire
    literature fire keys: 239 -> 118
    events: 339 -> 328

**Effect on the result — it strengthens.**

    decisive subset : 76 at 73.7% / 7.9%  ->  67 at 74.6% / 6.0%
    ratio           : 9.3:1               ->  12.4:1
    winter          : 133, 27 Dec, R 0.49 ->  134, 27 Dec, R 0.48
    summer          : 135, 01 Aug, R 0.87 ->  123, 02 Aug, R 0.88
    60-min M2 - M1  : 126.7               ->  119.0

**One consequence worth carrying into the writing.** The anti-decay control
narrows from 74.3 / 86.7 to 79.3 / 85.7 — a 6.4-point gap where it was 12.4.
The claim that the intensity preference *strengthens* when the wet season
arrives first was already not significant (Fisher p = 0.35 on the old split);
it is now weaker still. Use "does not weaken". See `docs/reviews/TRIAGE.md`, B1.

Found by independent subagent review (`docs/reviews/compilation.md`, corroborated
by `independence.md` and `decisive_subset.md`).

## 2026-09-20 — The fire alias table was written in a space it could not reach

`norm_fire` strips "fire", "complex", "lightning" and similar words BEFORE
looking a name up in `FIRE_ALIASES`, but the table's keys were written against
raw strings containing exactly those words, in a word order the sources do not
use. Two aliases were therefore dead, and two fires stayed split:

    "Fish (San Gabriel Complex)"   -> fishsangabriel   (graber2023, literature)
    "Fish"                         -> fish             (oakley2025)
    alias key "sangabrielcomplexfish" -> unreachable

    "CZU"                          -> czulightningcomplex  (cavagnaro2025)
    "CZU August Lightning Complex" -> czuaugust            (oakley2025)

`czulightningcomplex` is worth noting on its own: nothing but the alias could
ever produce it, so it was a canonical form outside the normalizer's range.

The table is now written as the names actually appear, and both sides pass
through `_basic_norm`, so an alias cannot be expressed in a form the normalizer
cannot produce. `_build_aliases` drops no-ops and raises on two mistakes the old
table could have hidden silently: two raw keys normalizing to the same key with
different values, and an alias whose value is itself a key (which would need
transitive resolution that `norm_fire` does not do). Canonical forms are the
full official names, which the data reaches without an alias at all.

    events: 328 -> 322
    `fish` now draws on graber2023, literature and oakley2025
    `czuaugust` now draws on cavagnaro2025 and oakley2025

**Checked and deliberately NOT merged.** Three other name pairs look like
aliases and are not: cedar/cedarcreek are 1,761 km apart, river/riverside 957
km, and slink/slinkard are 16 km apart but are a 2020 California fire and a
2017 Nevada fire. Name similarity is not evidence; fire year, state and
location were checked in each case.

Effect on the result, with A1 already in place:

    decisive subset : 67 at 74.6% / 6.0%  (unchanged by A2)
    winter          : 128, 26 Dec, R 0.47
    60-min M2 - M1  : 118.9

Found by independent subagent review (`docs/reviews/compilation.md`).
TRIAGE.md item A2.

## 2026-09-20 — One storm under two dates

Events were grouped on an exact `(fire_key, event_date)`, so a storm two
sources dated differently became two events. Sources genuinely disagree about
what date a flow carries — `DATE_BASIS` records storm start for some and the
observation for others, and a multi-day atmospheric river can be dated anywhere
inside it.

Eleven same-fire event pairs sit within three days of each other with no source
in common. **Date proximity does not settle them**, so `canonicalize_storm_dates`
requires the two sets of points to BE the same flows: median nearest-neighbour
distance within 250 m, the same radius record-level de-duplication uses. The
distance does the discriminating; the day window only widens the candidate set.

    Butte  2016-03-04 vs 03-06   median     0 m  -> merged
    Dolan  2021-01-26 vs 01-27   median    22 m  -> merged
    Station (x3), Sayre, Monument, Tadpole  339-3,514 m  -> left alone
    Whitewater-Baldy 09-14/15/16 vs 09-17   7-25 km      -> left alone,
        a real monsoon sequence: graber2024 dates three storms and the
        literature record is 25 km from the nearest of them

The surviving date comes from the highest-priority source present, not from
whichever is earlier, so Dolan keeps 2021-01-27 — the storm date the Cavagnaro
GRL paper assigns — rather than the 26th that `czu2021` carries for the start
of the same atmospheric river. Dolan is now one event of 2,124 flows drawing on
all four sources.

    events: 322 -> 320

**Rejected: widening the record-level de-duplication's date window.** It looks
like the natural fix and is not. At +/-2 days it merges 224 extra records, 208
of them Dolan segments absorbed into higher-priority basin points, because that
inventory maps stream segments about 10 m apart and one point sits within 250 m
of many. That is the record-comparability problem, not a date-convention one,
and conflating them would have quietly deleted 10% of the database.

**A footgun worth naming.** `_to_local_xy` takes its reference latitude from
whatever array it is handed, so projecting two sets separately puts them on
different planes. It bit this session twice — first in the literature fire
matching, then here, where it inflated the Dolan distance enough to suppress
the merge while Butte still passed because its two sets are at identical
coordinates. Any comparison across two frames must project them together.

Found by independent subagent review (`docs/reviews/compilation.md`).
TRIAGE.md item A3.

## 2026-09-20 — Dolan: one stream segment is one debris flow

`load_dolan2020` kept every row of the release as a debris-flow record. But a
row is one OBSERVATION, and `FireSegmentID` is the stream segment it sits on
(release README, field list). A segment is routinely observed at several points
a few tens of metres apart on the same date — the mapped extent of one
response, not several debris flows. 2,144 observations carried only 1,803
segments.

The record-level de-duplication cannot catch these: pass B is cross-source
only, deliberately, because two nearby records inside one inventory are
normally two real adjacent flows. That reasoning is right in general and wrong
for this specific column, which names the unit explicitly.

The loader now collapses to one record per segment, preferring field-verified
(`Response == 3`) over remotely mapped (`Response == 2`).

    records: 3,526 -> 3,204   (341 extra points on already-counted segments)
    events : 320 -> 320       (unchanged)

**Every analysis result is byte-identical.** Only one pinned number moved, the
record count, which confirms what was already believed: `n_debris_flows` is
carried for description and is used by no analysis.

**It does break a published claim.** The AGU abstract says "over 3,500 debris
flows". The compilation holds 3,204. "Over 300 events" still holds. The paper
must not repeat the flow count, and the abstract now carries this in its
superseded table.

Worth stating plainly in the paper: a record count summed across these sources
was never a comparable quantity. Dolan maps channel segments about 10 m apart
and contributed 57% of all records; the other inventories map basin outlets.
Events are the unit the analysis uses and the unit the paper should lead with.

## 2026-09-20 — Butte year conflict: resolved, no action

Reported by the compilation review alongside the Station date shift. Both
sources now agree the Butte fire is 2015, and its two March 2016 dates merged
under `canonicalize_storm_dates`. Nothing further to fix; recorded so it is not
re-investigated.

## 2026-09-20 — The post-fire window is removed

`POSTFIRE_WINDOW_YEARS = None`. Every event is analyzed regardless of how long
after the fire it occurred. This reverses the 2-year window applied earlier the
same day. Scott's call, and the right one.

**It could not be defended quantitatively.** Of the three criteria used to pick
2 years, two do not survive scrutiny:

* the exposure correction used a per-SOURCE horizon, crediting every fire in
  the literature database with observation through 2021, when that database is
  a compilation of studies each of which stopped looking when its own fieldwork
  ended. It gave 70% of fires five or more years of apparent observation; the
  denominator fell 29% over a range where the numerator fell 99%.
* the season-mix criterion compared the retained sample against the full
  sample. That is NESTED: the difference must go to zero as the cutoff grows,
  by construction. It measured its own arithmetic, not the data, and the gap it
  rested on was about one point.

On first flows per fire the rate is already at its plateau by 1.5 yr, so 1.5
was as defensible as 2.0 — which is another way of saying neither was.

**It does not change the answer.** Across every cutoff from 0.5 yr to none the
decisive subset holds. A filter that changes nothing but the sample size is a
liability in review, not a safeguard. The insensitivity table is better
evidence than any particular window.

**Process is filtered on process.** Removing the window readmits 30 events, the
longest 4.2 yr after its fire, of which NONE is landslide-initiated (27
runoff-generated, 3 unknown). The runoff filter now excludes 11 landslide
events rather than 4, because the long-lag tail is back in scope and screened
on what actually distinguishes it.

    analysis set    : 278 -> 308 events, 300 with a trustworthy date
    decisive subset : 67 at 74.6% / 6.0%  ->  77 at 75.3% / 6.5%
    60-min M2 - M1  : 119.3 -> 146.8
    winter          : 138, 18 Dec, R 0.43
    summer          : 135, 02 Aug, R 0.89

`scripts/checks/postfire_lag_cutoff.py` is kept and repurposed. Its exposure
correction is now per PAPER (`original_source` for literature records, the
source itself otherwise), which roughly doubles the tail rates it reports. Its
verdict section now records why there is no cutoff. Sections 1 and 2 stand on
their own as a description of how long after a fire runoff-generated debris
flows occur — a result worth reporting, and no longer measured downstream of
the cutoff it was meant to justify.

TRIAGE.md items D1 and D2, resolved together by removing the thing they
criticized.

## 2026-09-20 — Reproducibility (TRIAGE.md stage C)

**C1. The station climatology can now be regenerated, and the committed code
reproduces it exactly.** `03_` writes `atlas14_station_seasonality.csv`; `04_`
read it and wrote back to the same path, so a second run of `04_` merged the
normals onto columns that already held them and died on `KeyError`, and
rerunning `03_` afterwards silently stripped the normals from five downstream
consumers. `04_` now drops the columns it owns before merging, and fails with
a clear message if `03_` has not run.

The file on disk was dated 2026-08-05, before the reorganization, and nothing
in the tree showed the committed code had ever produced it. It has now:
regenerating from raw and diffing against a copy of the 5 August file gives
**3,301 rows and 101 columns identical, zero differing values**. The
reproducibility gap was real; a correctness problem was not. The sequence
`03 -> 04 -> 04 -> 03 -> 04` is now idempotent and was verified to be.

**C2. A stale MTBS file can no longer misjoin silently.** `event_id` is
positional (`EV00000` upward in row order), so it is not stable across
recompilations — a stale file does not fail to join, it joins to the WRONG
events. `add_postfire_interval` now checks identity, not presence: every
compiled event must appear in the MTBS file, and each id must point at the
same fire and date on both sides. Verified by truncating the file and
confirming the guard fires.

**C3. A missing states shapefile now fails loudly.** `assign_states` returned
the unmodified column when the zip was absent. `load_literature` sets no state
at all, so the caller then dropped 958 records under the message "outside any
US state" — a silent 30% loss reported as a bounding-box clip.

**C4. Small.** `pyproject.toml` now declares `openpyxl`, which `00_` needs for
`read_excel` and which only `environment.yml` had. `paths.ensure_dirs` was
defined and never called, so six scripts did their work and then died on
`savefig` in a clean tree; `style.apply` and `report.tee` now call it, which
between them covers every figure and report writer. `06_fetch_event_durations`
ran its fetch at MODULE level with no guard, so importing it fired 690 requests
at NOAA — note that appending a `__main__` guard below the module-level code
does NOT fix this, because the requests go out while the module is still
executing; the work moved into `main()`. `RESOURCE_MAP.md` said nine reports
and eight figures where there are fourteen and eleven, and described
`data/logs/` as a live output directory when nothing writes it.

**C5. The `data/raw/` invariant was a documentation error, not a code one.**
`paths.py` and `RESOURCE_MAP.md` already said "never written by *analysis*
code", which is accurate; `CLAUDE.md` dropped the qualifier. Acquisition code
does write there — the NOAA endpoint caches and the extracted CONUS404 `.npz`
— and `CLAUDE.md` now says so, including that the declared environment cannot
regenerate the 1.6 GB of CONUS404 inputs without `xarray`/`fsspec`/`s3fs`/`zarr`.

Found by independent subagent review (`docs/reviews/reproducibility.md`).

## 2026-09-20 — How the result is stated (TRIAGE.md stage B)

The review's own figures could not be quoted: they were measured before the
compilation fixes and the removal of the post-fire window, which moved every
denominator. `scripts/checks/review_followups.py` recomputes each disclosed
number on the current sample and writes `review_followups_report.txt`, so the
paper cites a generated report rather than a stale review.

**B1. "Strengthens" is dropped; "does not weaken" replaces it.** The split is
84.8% (n=33) against 77.4% (n=31): +7.4 points, Fisher exact **p = 0.53**. It
was p = 0.35 on the larger pre-correction sample, so this was never supported.
The control still works, because decaying susceptibility predicts the intensity
preference should WEAKEN when the wet season arrives first and it does not —
that is the falsifiable part, and it is what the paper now claims. The pinned
test was renamed from `..._strengthens_...` to `..._does_not_weaken_...` and
now asserts the direction with a margin instead of pinning a claim the data do
not support.

**B2. The decisive test speaks to the monsoon regime only.** 76 of 77 decisive
events sit at summer-intense sites; one is SON; none is winter. Mechanical, not
a sampling accident: in coastal and southern California the wettest and most
intense seasons are both winter, so a winter event can never be decisive. The
winter regime rests on the model comparison and the regime statistics instead.
Stated in the Introduction and in Results item 4.

**B3. Two figures printed literals from old runs.** The story figure's
chi-square label read 59 and 355 where the report said 45.0 and 282.4; `25_`
said "the 349 events" over a 300-event table. Both are computed now. A figure
that looks calculated and is not is worse than no number.

**B4. Fire-weighted numbers are reported alongside event-weighted.** 300 events
from 143 fires, and 34% fall on a date when another fire also produced flows —
one storm, several burn scars. Event-weighted 75.3% / 6.5%; fire-weighted
**64.3% / 7.2%**. Directions hold; precision is what dependence costs.

**B5. Lead with the binning-free score.** For each decisive event, which
climatology assigned more probability to the month the flow actually happened?
Intensity, in **69 of 77 (90%), binomial p = 3e-13**. No seasons, no bin edges.
The ratio moves with where season boundaries are drawn — a one-month shift
takes it from 11.6:1 to about 3:1 — and this does not.

**B6. The concentration concern largely dissolved with A1.** The review found
one apparent storm chain supplying 42% of the decisive subset. After the
fire-keying fix the largest single fire is **13%**. The earlier figure was
substantially an artifact of the defect the same review found. Still reported.

**B7. The suspect-date filter is load-bearing, and now disclosed.** Eight
events carry 1 January placeholders, six in Idaho. Including them moves the
decisive subset from 75.3%/6.5% to **69.9%/12.0%**, ratio 11.6:1 to 5.8:1. The
filter is necessary, not merely defensible — Idaho is summer-intense, so a
1 January placeholder forces the event onto the wettest side and fabricates
winter debris flows in a paper about seasonality — but the reader is owed the
sensitivity.

**B8. The amount model barely commits.** On the decisive subset it places a
median 0.124 probability on its own favoured month, margin 0.013 over the
runner-up. A 12:1 ratio against a near-indifferent alternative is weaker than
it sounds; second reason to lead with B5.

Stale counts through the draft were refreshed at the same time (320 events,
3,204 records, 4,062 raw, 169 fires, 284 MTBS matches, 26 of 300 events with no
Atlas 14 coverage). The published abstract body is untouched; its superseded
table carries the current values.
