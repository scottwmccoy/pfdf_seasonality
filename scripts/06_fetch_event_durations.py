"""Fetch Atlas 14 seasonality at the debris-flow event locations for the longer
durations too, so each event has a 60-min / 24-hr / 10-day ladder.

RETAINED AS A NEGATIVE RESULT — NOT PART OF THE PIPELINE.
--------------------------------------------------------
The only recorded run (`data/logs/event_durations_log.txt`, 2026-08-05) reports
`covered 0/349` for both durations: the NOAA endpoint serves no 24-hour or
10-day seasonality at these points. Its four output files do not exist, and no
script reads them.

Until 2026-09-20 the fetch ran at module level with no `__main__` guard, so
merely importing this module fired 690 requests at the NOAA endpoint. A guard
appended below the module-level code would not have helped — the requests go
out while the module is still being executed, before the guard is reached. The
work is therefore now inside `main()`, which is the part that actually fixes it.
"""
from __future__ import annotations

import pandas as pd

from pfdf_seasonality import atlas14 as a
from pfdf_seasonality.paths import ATLAS14, EVENTS

DURATIONS = ["24h", "10d"]


def main() -> None:
    ev = pd.read_csv(EVENTS)
    for dur in DURATIONS:
        print(f"=== {dur} ===", flush=True)
        df = a.seasonality_table(ev, dur=dur, workers=4,
                                 cache_file=ATLAS14 / f"atlas14_cache_events_{dur}.json")
        df.to_csv(ATLAS14 / f"atlas14_events_{dur}.csv", index=False)
        print(f"  covered {int(df.covered.sum())}/{len(df)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(
        "06_fetch_event_durations is retained as the record of a negative "
        "result: the NOAA endpoint returned no 24h or 10d coverage at any "
        "event (0/349, 2026-08-05), and nothing reads its outputs. Running it "
        "makes 690 network requests. To do so deliberately, call main()."
    )
