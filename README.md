# pfdf_seasonality

What sets the time of year that postfire debris flows occur across the western
United States, and how far ahead it can be predicted.

The result: debris-flow seasonality follows the climatology of **intense
short-duration rainfall**, not rainfall amount. Where those two diverge, the
intensity climatology assigns more probability to the month the flow actually
occurred in 90% of cases (69 of 77, binomial p = 3×10⁻¹³). That makes the
expected time of year predictable from prefire climatology and the ignition
date alone, to a 41-day median error — better than a constant-lag model fit to
the observations (54 days).

Built from 320 fire-by-date events across 3,204 published debris-flow records.

Supports the AGU Fall Meeting 2026 abstract and a manuscript in preparation.
**Numbers here are current as of the last run, not final** — the abstract was
submitted before several corrections to the compilation, and `DECISIONS.md`
records what changed and why. Anything quoted from this repository should be
checked against the generated reports rather than against the abstract.

## Two trees

This repository is code only. Everything it reads and everything it writes —
bulk data, generated figures, generated reports, and the manuscripts — lives in
Box:

```
~/git/code/pfdf_seasonality/          this repository
    src/pfdf_seasonality/             importable package
    scripts/                          ordered pipeline
    tests/

$PFDF_SEASONALITY_DATA/               the Box tree
    data/raw/                         immutable inputs
    data/processed/                   analysis-ready tables
    results/figures/                  generated .png
    results/reports/                  generated .txt
    manuscripts/                      abstract and paper draft
```

`src/pfdf_seasonality/paths.py` resolves every path. No script computes one
from its own location, so either tree can move without breaking the other.

## Setup

```bash
conda env create -f environment.yml
conda activate pfdf_seasonality_env
```

The environment is named for the repository with an `_env` suffix, so it is
obvious which is the checkout and which is the interpreter. It is the minimum
needed to run the analysis; `xarray`, `zarr`, `fsspec` and `s3fs` are left out
because they are only required to re-extract CONUS404 from the cloud (see
`environment.yml`).

`environment.yml` installs the package in editable mode. If the Box tree is not
in the default location:

```bash
export PFDF_SEASONALITY_DATA="/path/to/Seasonality"
python -m pfdf_seasonality.paths      # prints every resolved root
```

## Running the pipeline

Scripts are numbered in dependency order and each runs alone. Most read the
processed tables already in Box, so a normal session starts at `10_`.

| Stage | Scripts | Notes |
|---|---|---|
| Build the inventory | `00`, `01` | Compiles published inventories; matches MTBS ignition dates |
| Build the climatology | `02`–`06` | Atlas 14 station AMS, NCEI normals, joins. `02`/`06` hit the NOAA API |
| Core analysis | `10`–`12` | Model comparison, ignition lags, first-flow forecast |
| Reanalysis comparison | `20`–`25` | `20`/`21` re-extract CONUS404: ~770 GB of cloud reads, hours. Do not run casually |
| Figures | `scripts/plots/` | Write to `$PFDF_SEASONALITY_DATA/results/figures/` |
| Checks | `scripts/checks/` | Standalone audits: record comparability, date basis, post-fire lags, independent-review follow-ups |

```bash
python scripts/10_analyze_seasonality.py
python scripts/11_analyze_ignition_lags.py
python scripts/12_predict_first_flow.py
python scripts/plots/plot_seasonality_story.py
```

## Data sources

NOAA Atlas 14 station annual-maximum series (15/30/60-min) and the PFDS
seasonality endpoint; NOAA NCEI 1991-2020 monthly normals; MTBS burn
perimeters and ignition dates; CONUS404 via the HyTEST OSN pod; published
debris-flow inventories (see `NOTICE.md`). Attribution and terms are in
`NOTICE.md`; `RESOURCE_MAP.md` says what every file is.

## License

MIT, see `LICENSE`.
