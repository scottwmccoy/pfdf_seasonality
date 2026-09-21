# Attributions and data sources

`pfdf_seasonality` is released under the [MIT License](LICENSE). This file
records the public data the project uses and the terms it comes under. No
third-party code is vendored here.

## Debris-flow inventories

The compiled database is derived from published inventories. It is a
compilation, not original observation, and the source teams should be credited
in any product that uses it.

| Source | Reference | Licence | Redistributable? |
|---|---|---|---|
| USGS ScienceBase postfire debris-flow inventories (`sciencebase_*`) | Various USGS data releases | Public domain, US Government works | Yes |
| Oakley et al. (2025), *Int. J. Wildland Fire* | doi:10.1071/WF25136 — supplemental material A, 25-year California database | **CC BY-NC-ND 4.0** | **No — see below** |
| Cavagnaro et al. (2025), *JGR Earth Surface* | doi:10.1029/2024JF007781 — figshare code and attribute tables | Not verified (figshare landing page returned 403 on 2026-09-20) | Unknown |

**The compiled database cannot be redistributed as it stands.** Checked
2026-09-20. Oakley et al. is CC BY-**ND**: NoDerivatives, which prohibits
distributing adaptations, and a standardized, de-duplicated, re-keyed merge of
that database is an adaptation. It touches 97 of the 320 compiled events, 36 of
which exist only through it. The Cavagnaro figshare licence could not be
confirmed and contributes a further 389 records and 66 events.

Only the licensors can waive ND. Releasing the full compilation therefore needs
written permission from the Oakley team, and confirmation of the figshare
terms. Until then the compilation stays where it is and this repository remains
code only. A public-domain subset — 2,815 records and 187 events from the USGS
releases alone — could be distributed without permission if a partial product
is ever wanted.

This has to be settled before the manuscript's data-availability statement is
written, not after.

## Rainfall data

- **NOAA Atlas 14** precipitation-frequency estimates: station annual-maximum
  series (15/30/60-min) and the PFDS seasonality endpoint at
  `hdsc.nws.noaa.gov`. US Government work, public domain. The seasonality
  endpoint is undocumented; it is read at a polite rate with an on-disk cache.
  Atlas 14 does not cover Oregon or Washington.
- **NOAA NCEI 1991-2020 monthly normals** (`mly-prcp-normal-metric-30yr`).
  Public domain.
- **CONUS404** (Rasmussen et al.), read from the HyTEST Open Storage Network
  pod at `usgs.osn.mghpcc.org`, anonymous access, no egress charge. Cite the
  CONUS404 dataset and the HyTEST project when using it.
- **Native 15-minute reanalysis rasters** were provided by David B. Cavagnaro
  and are not redistributed here.

## Fire data

- **MTBS** (Monitoring Trends in Burn Severity), USGS/USFS national burned-area
  perimeters with ignition dates. Public domain.

## Boundaries

- **US Census Bureau** cartographic boundary files (`cb_2023_us_state_20m`).
  Public domain.

## Note on redistribution

This repository contains code and generated text reports only. The input data
listed above lives outside the repository and is obtained from its original
providers; nothing here redistributes it.
