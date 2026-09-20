"""Every path in the project, resolved in one place.

No script computes a path from its own location. Before this module, 18 scripts
each did ``BASE = Path(__file__).resolve().parent`` and then reached sideways
for data, so moving any file broke its siblings. Import from here instead.

Two trees
---------
**This repository** holds code and the generated text reports. Reports are
small and are the provenance record for every number in the manuscript, so
they are committed and their diffs show when a result moved.

**The Box tree** holds the bulk data (1.6 GB of CONUS404 alone), the generated
figures, and the manuscripts. It is not in git. Point ``PFDF_SEASONALITY_DATA``
at it if it is not in the default location::

    export PFDF_SEASONALITY_DATA="$HOME/Library/CloudStorage/Box-Box/SWMresearch/PostFireDebrisFlows/Seasonality"

A note on Box: files there report 0 blocks to ``du`` while holding real
content, so size accounting lies and every read is a network operation. Check
``stat -f%z`` for a true size, and never sweep ``data/raw/conus404``.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO", "REPORTS", "DATA", "RAW", "PROCESSED", "LOGS", "FIGURES",
    "MANUSCRIPTS", "INVENTORIES", "MTBS", "AUX", "ATLAS14", "GHCN_NORMALS",
    "CONUS404", "CONUS404_HOURLY", "REANALYSIS_15MIN", "EVENTS", "EVENTS_MTBS",
    "STATES_ZIP", "ensure_dirs", "describe",
]

# --------------------------------------------------------------- repository
# paths.py -> pfdf_seasonality -> src -> repo root
REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "results" / "reports"

# ------------------------------------------------------------- the data tree
_DEFAULT_DATA = (Path.home() / "Library/CloudStorage/Box-Box/SWMresearch"
                 / "PostFireDebrisFlows" / "Seasonality")
DATA = Path(os.environ.get("PFDF_SEASONALITY_DATA", _DEFAULT_DATA)).expanduser()

RAW = DATA / "data" / "raw"
PROCESSED = DATA / "data" / "processed"
LOGS = DATA / "data" / "logs"
FIGURES = DATA / "results" / "figures"
MANUSCRIPTS = DATA / "manuscripts"

# raw inputs, never written by analysis code
INVENTORIES = RAW / "inventories"          # published debris-flow inventories
MTBS = RAW / "mtbs"                        # burn-severity perimeters + ignition dates
AUX = RAW / "aux"                          # census boundaries and other helpers
ATLAS14 = RAW / "atlas14"                  # station AMS, grid probe, endpoint cache
GHCN_NORMALS = RAW / "ghcn_normals"        # NCEI 1991-2020 monthly normals
CONUS404 = RAW / "conus404"                # 1.6 GB - do not sweep
CONUS404_HOURLY = CONUS404 / "hourly"
REANALYSIS_15MIN = RAW / "reanalysis_15min"   # native 15-min rasters (from D. Cavagnaro)

# the two analysis-ready tables most scripts start from
EVENTS = PROCESSED / "inventory" / "pfdf_events_compiled.csv"
EVENTS_MTBS = PROCESSED / "inventory" / "pfdf_events_mtbs.csv"
STATES_ZIP = AUX / "cb_2023_us_state_20m.zip"


def ensure_dirs() -> None:
    """Create the output directories a script is about to write into."""
    for d in (REPORTS, FIGURES, PROCESSED / "inventory", PROCESSED / "seasonality",
              LOGS):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable resolution of every root, for debugging a bad path."""
    rows = [("REPO", REPO), ("REPORTS", REPORTS), ("DATA", DATA), ("RAW", RAW),
            ("PROCESSED", PROCESSED), ("FIGURES", FIGURES),
            ("MANUSCRIPTS", MANUSCRIPTS)]
    width = max(len(n) for n, _ in rows)
    return "\n".join(
        f"{n:<{width}}  {p}  {'' if p.exists() else '  <-- MISSING'}"
        for n, p in rows)


if __name__ == "__main__":
    print(describe())
