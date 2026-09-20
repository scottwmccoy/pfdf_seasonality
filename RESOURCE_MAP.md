# Resource map

What every file in both trees is. Updated 2026-09-20, at the reorganization.

---

## This repository — `~/git/code/pfdf_seasonality`

### `src/pfdf_seasonality/`

| File | What it is |
|---|---|
| `paths.py` | Every path in the project. Resolves `$PFDF_SEASONALITY_DATA`; run `python -m pfdf_seasonality.paths` to print and check them all. |
| `seasons.py` | `SEASONS`, `MONTH_TO_SEASON`, `EPS`, `LAT0`, and the shared numerics: `normalize`, `season_of`, `circular`, `circular_mean_doy`, `to_xy`, `doy_to_date`, `circ_diff_days`. Previously copy-pasted across up to 12 scripts. |
| `style.py` | Okabe-Ito palette, season colors, matplotlib defaults (`style.apply()`), western-US map framing. |
| `atlas14.py` | NOAA Atlas 14 client: the undocumented PFDS seasonality endpoint, plus parsers for the station annual-maximum series (four different file layouts across volumes). |

### `scripts/` — ordered pipeline

| Script | Produces |
|---|---|
| `00_compile_inventory.py` | The 349-event / 3,698-record debris-flow database from 12 published inventories |
| `01_match_mtbs.py` | MTBS ignition dates joined to events (313 of 331 MTBS-era matched) |
| `02_fetch_atlas14_grid.py` | 0.5-degree probe of the Atlas 14 seasonality endpoint (87 distinct sub-region curves) |
| `03_build_station_seasonality.py` | Per-station seasonality from the annual-maximum series, 15/30/60-min |
| `04_add_precip_normals.py` | NCEI 1991-2020 monthly normals attached to stations |
| `05_join_rainfall_seasonality.py` | Rainfall climatology joined to each event (IDW, k=5, <=150 km) |
| `06_fetch_event_durations.py` | Endpoint seasonality at event locations |
| `10_analyze_seasonality.py` | **Core result**: M0/M1/M2 comparison, regimes, decisive subset, robustness |
| `11_analyze_ignition_lags.py` | Fire-to-first-flow intervals, the seasonal clock, skipped-season counts |
| `12_predict_first_flow.py` | A priori forecast skill against a fitted constant-lag baseline |
| `20_extract_conus404.py` | CONUS404 instantaneous rate (RAINNCVMAX). ~hours, large cloud reads |
| `21_extract_conus404_hourly.py` | CONUS404 1-hour accumulation (PREC_ACC_NC). ~770 GB of reads |
| `22_analyze_conus404.py` | The seasonality analysis repeated on CONUS404 |
| `23_validate_duration_sensitivity.py` | One-tile rate-vs-1-hour test |
| `24_compare_conus404_durations.py` | Full-domain duration bracket (method superseded by `25_`) |
| `25_evaluate_products.py` | **Eight-product comparison**: gauges vs reanalysis vs native 15-min |
| `plots/*.py` | Five figure scripts, all writing to the Box `results/figures/` |
| `rerun_when_hourly_done.sh` | One-off watcher from the original extraction run; kept for reference |

### `results/reports/` — committed

Eight generated text reports. These are the provenance record for every number
in the manuscript; their diffs show when a result moved.

`analysis_report.txt` (core seasonality) · `ignition_lag_report.txt` ·
`first_flow_prediction_report.txt` · `seasonality_product_evaluation_report.txt` ·
`conus404_analysis_report.txt` · `conus404_hourly_analysis_report.txt` ·
`conus404_duration_comparison_report.txt` · `duration_sensitivity_report.txt` ·
`compilation_report.txt` (written by `00_`, lands here on the next run)

---

## The Box tree — `$PFDF_SEASONALITY_DATA`

`~/Library/CloudStorage/Box-Box/SWMresearch/PostFireDebrisFlows/Seasonality`

### `data/raw/` — immutable inputs, never written by analysis code

| Directory | What it is |
|---|---|
| `inventories/` | 12 published source inventories: `sciencebase_*` (USGS), `oakley_ijwf/` (Oakley et al. 2025 supplement), `cavagnaro_figshare/` (Cavagnaro et al. 2025 figshare) |
| `mtbs/` | MTBS national burned-area perimeters, 390 MB zip |
| `aux/` | `cb_2023_us_state_20m.zip`, Census state boundaries |
| `atlas14/` | Station annual-maximum series (`atlas14_ams/`), the 0.5-degree grid probe, endpoint response caches |
| `ghcn_normals/` | NCEI 1991-2020 monthly normals, tenths of mm |
| `conus404/` | **1.6 GB.** Per-water-year `amax_wy*.npz` (rate) and `hourly/tile_*.npz` (1-hour), plus `grid.npz` and `monthly_climatology.npz`. Do not sweep |
| `reanalysis_15min/` | Native 15-minute rasters from D. Cavagnaro: `season_of_most_exc1yr.tif`, `season_of_most_exc24.tif`, `wettest_season.tif`, and the two reference PNGs they came with. 1980-2021, EPSG:4326 at 0.005 deg |

### `data/processed/` — analysis-ready, regenerable

- `inventory/` — the compiled event and record tables, and the MTBS match
- `seasonality/` — station seasonality, per-event rainfall climatology, the
  CONUS404 event tables, ignition lags, model comparison

### `data/logs/`

Extraction logs from the CONUS404 and Atlas 14 runs.

### `results/figures/`

Eight generated PNGs. `pfdf_seasonality_gauge_story.png` is the core result;
`ignition_lag.png` and `first_flow_prediction.png` are the timing results;
`seasonality_product_evaluation.png` is the product comparison;
`atlas14_station_seasonality_map.png` and `atlas14_seasonality_grid_map.png`
are the climatology; `gauge_vs_conus404.png` and
`conus404_duration_comparison.png` are the reanalysis comparisons.

### `manuscripts/`

- `agu_2026_abstract/abstract.md` — submitted and published; frozen record
- `seasonality_paper/draft.md` — the working paper draft
- `seasonality_paper/figures/`, `refs/` — manuscript-ready exports, references

### `docs/`

- `source_papers/` — the two source-inventory papers and their text
  extractions. **Both are in Zotero** and should be cited from there:
  Cavagnaro et al. 2025 = `D8EX7F9P` (McCoyLab), `KR5CUFHG` (My Library);
  Oakley et al. 2025 = `R3FJIS7K` (McCoyLab). The copies here are
  working convenience only and are not in git
- `notes/` — `FINDINGS_gauge_only.md`, the two pre-reorganization READMEs
  (historical; superseded by this file and the repo README), and
  `REORG_PROPOSAL.md`, which records the pre-2026-09-20 layout
