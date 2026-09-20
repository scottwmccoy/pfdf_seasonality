"""Fetch Atlas 14 seasonality at the debris-flow event locations for the longer
durations too, so each event has a 60-min / 24-hr / 10-day ladder."""
import sys
from pathlib import Path
import pandas as pd
from pfdf_seasonality import atlas14 as a
from pfdf_seasonality.paths import ATLAS14, EVENTS

ev = pd.read_csv(EVENTS)
for dur in ["24h", "10d"]:
    print(f"=== {dur} ===", flush=True)
    df = a.seasonality_table(ev, dur=dur, workers=4,
                             cache_file=ATLAS14 / f"atlas14_cache_events_{dur}.json")
    df.to_csv(ATLAS14 / f"atlas14_events_{dur}.csv", index=False)
    print(f"  covered {int(df.covered.sum())}/{len(df)}", flush=True)
