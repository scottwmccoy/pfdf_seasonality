# Attributions and data sources

`pfdf_seasonality` is released under the [MIT License](LICENSE). This file
records the public data the project uses and the terms it comes under. No
third-party code is vendored here.

## Debris-flow inventories

The compiled database is derived from published inventories. It is a
compilation, not original observation, and the source teams should be credited
in any product that uses it.

| Source | Reference |
|---|---|
| USGS ScienceBase postfire debris-flow inventories (`sciencebase_*`) | Various USGS data releases; public domain as US Government works |
| Oakley et al. (2025), *Int. J. Wildland Fire* | doi:10.1071/WF25136 — supplemental material A, 25-year California database |
| Cavagnaro et al. (2025), *JGR Earth Surface* | doi:10.1029/2024JF007781 — figshare code and attribute tables |

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
