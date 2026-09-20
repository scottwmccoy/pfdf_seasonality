# pfdf_seasonality — development notes

Start sessions here, not in the Box tree. The Box tree holds data, figures and
manuscripts; this repository holds the code and the generated reports.

## Environment — IMPORTANT

Run the project's Python by absolute path; do not rely on an activated shell.

    /opt/anaconda3/envs/pfdf_seasonality_env/bin/python scripts/10_analyze_seasonality.py

The environment is `pfdf_seasonality_env` — repo name plus `_env`, so the
checkout and the interpreter are never confused. Build it once with
`conda env create -f environment.yml` (conda-forge, strict priority). Never
install into `base`.

It is the **minimum** to run the analysis. `xarray`, `zarr`, `fsspec` and
`s3fs` are deliberately absent because only `20_`, `21_` and `23_` need them,
and those re-extract CONUS404 from the cloud. Add them only if you are actually
re-extracting:

    /opt/anaconda3/bin/conda install -n pfdf_seasonality_env -c conda-forge xarray zarr fsspec s3fs

Note that `conda` here is a shell function that is not available to
non-interactive tooling; call `/opt/anaconda3/bin/conda` by absolute path.

## Data layout

Every path comes from `pfdf_seasonality.paths`. Never write
`Path(__file__).parent` in a script — that idiom is what the 2026-09-20
reorganization removed from 18 files, and re-introducing it re-breaks the
ability to move either tree.

    from pfdf_seasonality.paths import CONUS404, PROCESSED, REPORTS, FIGURES

`python -m pfdf_seasonality.paths` prints every resolved root and flags any
that is missing — the first thing to run when a path looks wrong.

**This repository is code only.** Nothing generated is committed; every output
goes to the Box tree:

| Kind | Goes to |
|---|---|
| Text reports | `REPORTS` — Box `results/reports/` |
| Figures | `FIGURES` — Box `results/figures/` |
| Analysis-ready tables | `PROCESSED` — Box `data/processed/` |
| Raw inputs | `RAW` — Box `data/raw/`, never written by code |

Figures and reports are siblings under `results/` on purpose, and there is
exactly one of each directory. A second copy of a figure or report beside a
manuscript goes stale the first time a script is re-run.

## Gotchas

**1. Box lies about file size.** Files in the Box tree report 0 blocks to `du`
while holding real content; `stat -f%z` gives the true size. Every read is a
network operation. Never sweep `data/raw/conus404` (1.6 GB) or run an unbounded
`find` over the tree.

**2. The two extract scripts are not re-runnable casually.** `20_` and `21_`
read roughly 770 GB from the HyTEST cloud store and take hours. The analysis
runs from the extracted `.npz` files. If you change them, verify against the
existing outputs rather than regenerating.

**3. `circular_mean_doy` uses 365.0, `circular` uses 365.25.** This
inconsistency predates the package and is preserved on purpose: every
seasonal-clock number (clock_doy, pred_days, residual_d, the 41-day forecast
error) was produced with 365.0. See the note in `seasons.py`.

**4. Two scripts need a differently shaped `SEASONS`.** `03_` wants lowercase
keys for its column names, `11_` wants a plain list. Both derive it from the
shared dict rather than redefining it. Keep it that way.

## Conventions

- **Never change a number silently.** `tests/test_headline_numbers.py` pins
  every value quoted in the abstract and the manuscript, parsed from the
  reports, so run the test suite after any refactor that touches analysis code:

      /opt/anaconda3/envs/pfdf_seasonality_env/bin/python -m pytest tests/ -q

  For a larger refactor, also re-run the affected scripts and diff their
  outputs against copies taken beforehand; that is what caught a real
  quarter-day shift during the reorganization, and the only legitimate
  differences are the absolute paths in "wrote ..." lines.

  A pin failing is **not** a licence to edit the expected value. Either it is a
  bug, or it is a deliberate result change — in which case update the pin *and*
  every place the number is quoted, and say in the commit which number moved
  and why.
- **American English** everywhere, including figure labels and comments.
- **Figures**: Okabe-Ito via `style.apply()`; season colors from
  `style.SEASON_COLOR`. Do not paste a palette into a script.
- **Numbers in the manuscript** must trace to a file in `results/reports/`.
  If a number is not in a report, it is not ready to quote.
- Source papers are referenced, not stored in git. Both live in Zotero:
  Cavagnaro et al. 2025 (`D8EX7F9P` in McCoyLab) and Oakley et al. 2025
  (`R3FJIS7K`). Search the local libraries with `~/.claude/bin/zot` before the
  web.
