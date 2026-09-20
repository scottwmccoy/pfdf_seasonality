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

**Verification standard set here.** The migration replaced per-script path
boilerplate in 18 files and consolidated constants duplicated in up to 12, then
re-ran every affected script and diffed all 29 outputs byte-for-byte against
pre-migration copies. That caught one real defect: the shared
`circular_mean_doy` had been written with 365.25 days where the original used
365.0, shifting `clock_doy`, `pred_days` and `residual_d` by a quarter day
across all 308 rows. The original behavior was restored and the inconsistency
documented rather than silently "fixed", because the published seasonal-clock
numbers were produced with 365.0.
