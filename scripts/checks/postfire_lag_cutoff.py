"""Where should the postfire analysis window end? Choose it from the data.

The 2-year window was set on physical reasoning alone (susceptibility is
concentrated in the first wet seasons). This picks the cutoff from the observed
lag distribution instead, and reports what the choice costs.

Why the existing lag figure cannot answer this
----------------------------------------------
`plot_ignition_lags.py` reads `pfdf_events_ignition_lags.csv`, which is written
downstream of `load_events` and is therefore truncated at exactly the cutoff
under test. Everything here starts from the UNWINDOWED compilation.

The censoring problem
---------------------
A naive lag histogram falls off at long lags for two quite different reasons,
and only one of them is physics:

1. postfire susceptibility decays — the signal we want; and
2. no inventory looked that long — an artifact of how the data were built.

Reason 2 dominates the tail here. `dolan2020` and `czu2021` are SINGLE-STORM
inventories: one survey date each, so a fire in them cannot show a 2-year lag
no matter what happened. So each event gets an observation horizon, the latest
date its source(s) could have recorded anything, and the hazard panel divides
events in a lag bin by the fires actually observable at that lag.

One bias this does NOT fix: a fire enters the database only by producing at
least one flow, so the earliest bins are inflated by selection. The correction
handles right-censoring, which is what distorts the tail and hence the cutoff.

  /opt/anaconda3/envs/pfdf_seasonality_env/bin/python scripts/checks/postfire_lag_cutoff.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfdf_seasonality import style  # noqa: E402
from pfdf_seasonality.events import load_events  # noqa: E402
from pfdf_seasonality.paths import FIGURES, PROCESSED  # noqa: E402
from pfdf_seasonality.report import tee  # noqa: E402
from pfdf_seasonality.seasons import (MONTH_TO_SEASON, circular,  # noqa: E402
                                      doy_to_date, normalize, season_of)

# Reuse the analysis script's own IDW rather than reimplementing it; a second
# copy would be free to drift from the one that makes the published numbers.
_an = importlib.import_module("10_analyze_seasonality")
idw = _an.idw
STATIONS = _an.STATIONS

CUTOFFS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, None]
C = {"DJF": "#56B4E9", "JJA": "#E69F00"}


# ----------------------------------------------------------------- exposure
def observation_horizons() -> dict[str, pd.Timestamp]:
    """Latest date each source could have recorded a debris flow."""
    rec = pd.read_csv(PROCESSED / "inventory" / "pfdf_occurrence_records_compiled.csv",
                      low_memory=False, parse_dates=["event_date"])
    h = rec.groupby("source_key").event_date.max().to_dict()
    oak = PROCESSED / "inventory" / "pfdf_oakley2025_events_standardized.csv"
    if oak.exists():
        o = pd.read_csv(oak, low_memory=False, parse_dates=["event_date"])
        h["oakley2025"] = o.event_date.max()
    return h


def add_exposure(ev: pd.DataFrame) -> pd.DataFrame:
    """`observable_years`: how long after ignition this event's sources looked."""
    h = observation_horizons()
    out = ev.copy()
    horizon = out.sources.fillna("").map(
        lambda s: max((h[k] for k in str(s).split("|") if k in h),
                      default=pd.NaT))
    out["horizon"] = pd.to_datetime(pd.Series(horizon, index=out.index))
    out["observable_years"] = ((out.horizon - out.ignition_date).dt.days / 365.25)
    return out


# ------------------------------------------------- the analysis at a cutoff
def gauge_models(ev: pd.DataFrame) -> pd.DataFrame:
    """Attach the 60-min intensity and the amount climatology, exactly as
    `10_analyze_seasonality` does, once, for the whole unwindowed set."""
    ev = ev[~ev.date_precision.astype(str).str.startswith("suspect")].copy()
    ev["df_month"] = ev.event_date.dt.month
    ev["df_doy"] = ev.event_date.dt.dayofyear
    ev["df_season"] = ev.df_month.map(MONTH_TO_SEASON)

    st = pd.read_csv(STATIONS)
    st = st[st.western_us & ~st.short_record]

    s = st[st.duration == "60m"]
    cols = [f"month_frac_{m:02d}" for m in range(1, 13)]
    P, d0, ok = idw(ev, s, cols)
    ev["int_ok"] = ok
    seas, _ = season_of(normalize(P))
    ev["int_season"] = np.where(ok, seas, None)

    sw = st[(st.duration == "60m") & st.has_normals]
    P, d0, ok = idw(ev, sw, [f"prcp_norm_{m:02d}" for m in range(1, 13)])
    ev["wet_ok"] = ok
    seas, _ = season_of(normalize(P))
    ev["wet_season"] = np.where(ok, seas, None)
    return ev


def headline(ev: pd.DataFrame) -> dict:
    """The numbers the paper quotes, for one cutoff's worth of events."""
    use = ev[ev.int_ok & ev.wet_ok]
    dis = use[use.int_season != use.wet_season]
    row = dict(n_events=len(ev), n_scored=len(use), n_decisive=len(dis))
    # The OBSERVED season mix of the retained sample. A cutoff that slices an
    # annual cycle keeps one season and drops the next, which would manufacture
    # seasonality in a paper about seasonality. Watch these two, not just n.
    row["obs_DJF"] = (ev.df_season == "DJF").mean() * 100
    row["obs_JJA"] = (ev.df_season == "JJA").mean() * 100
    row["intense_pct"] = (dis.df_season == dis.int_season).mean() * 100 if len(dis) else np.nan
    row["wettest_pct"] = (dis.df_season == dis.wet_season).mean() * 100 if len(dis) else np.nan
    row["ratio"] = row["intense_pct"] / row["wettest_pct"] if row["wettest_pct"] else np.nan
    for reg in ("DJF", "JJA"):
        g = use[use.int_season == reg]
        row[f"n_{reg}"] = len(g)
        if len(g) >= 5:
            mean_doy, R, _ = circular(g.df_doy.to_numpy())
            row[f"date_{reg}"], row[f"R_{reg}"] = doy_to_date(mean_doy), R
        else:
            row[f"date_{reg}"], row[f"R_{reg}"] = "-", np.nan
    return row


def main() -> None:
    ev = load_events(window_years=None, verbose=False)   # runoff filter ON
    ev = add_exposure(ev)
    ev = gauge_models(ev)
    lag = ev.years_since_fire

    print("=" * 76)
    print("WHERE SHOULD THE POSTFIRE WINDOW END?")
    print("=" * 76)
    print(f"  {len(ev)} runoff-generated events with a trustworthy date; "
          f"{int(lag.notna().sum())} have an ignition date")

    # ------------------------------------------------------ 1. raw distribution
    print("\n" + "-" * 76)
    print("1. THE RAW LAG DISTRIBUTION")
    print("-" * 76)
    q = lag.quantile([.5, .75, .9, .95, .99]).round(2).to_dict()
    print(f"  median {q[0.5]:.2f} yr   p75 {q[0.75]:.2f}   p90 {q[0.9]:.2f}   "
          f"p95 {q[0.95]:.2f}   p99 {q[0.99]:.2f}   max {lag.max():.1f}")
    print(f"\n  {'bin (yr)':<12} {'events':>7} {'cum %':>7}")
    edges = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 10, 100]
    n_tot = int(lag.notna().sum())
    cum = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((lag >= lo) & (lag < hi)).sum())
        cum += n
        bar = "#" * int(round(n / max(1, n_tot) * 120))
        print(f"  {lo:>5.2f}-{hi:<6.2f} {n:>7} {100*cum/n_tot:>6.1f}%  {bar}")

    print("\n  CAPTURE AT CANDIDATE CUTOFFS")
    for c in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        print(f"    <= {c:>4.1f} yr : {100*(lag <= c).mean():5.1f}% of events "
              f"({int((lag <= c).sum())} of {n_tot})")

    # ------------------------------------------- 2. censoring-corrected hazard
    print("\n" + "-" * 76)
    print("2. EXPOSURE-CORRECTED RATE — how much of the tail is real?")
    print("-" * 76)
    print("  'at risk' = FIRES whose source(s) looked at least this far past")
    print("  ignition. Single-storm inventories leave the denominator early.\n")
    # One row per fire: how far past ignition was this fire ever observed?
    fires = ev.groupby("fire_key").observable_years.max()
    print(f"  {'bin (yr)':<12} {'events':>7} {'fires':>7} {'per 100 fires':>15}")
    haz_x, haz_y = [], []
    for lo, hi in zip(np.arange(0, 5, 0.5), np.arange(0.5, 5.5, 0.5)):
        n = int(((lag >= lo) & (lag < hi)).sum())
        at_risk = int((fires >= hi).sum())
        rate = 100 * n / at_risk if at_risk else np.nan
        haz_x.append((lo + hi) / 2)
        haz_y.append(rate)
        print(f"  {lo:>5.2f}-{hi:<6.2f} {n:>7} {at_risk:>7} {rate:>14.1f}")
    print(f"\n  ({len(fires)} fires total; the denominator is fires still under")
    print("  observation, the numerator events, so this is events per fire.)")
    print("\n  Read the LAST column, not the second: the raw counts fall off")
    print("  partly because the inventories stop looking.")

    # ---------------------------------------------- 3. sensitivity of results
    print("\n" + "-" * 76)
    print("3. DOES THE CUTOFF CHANGE THE ANSWER?")
    print("-" * 76)
    rows = []
    for c in CUTOFFS:
        sub = ev if c is None else ev[lag.isna() | (lag <= c)]
        r = headline(sub)
        r["cutoff"] = "none" if c is None else f"{c:g} yr"
        rows.append(r)
    tab = pd.DataFrame(rows)
    print(f"  {'cutoff':>8} {'n':>5} {'scored':>7} {'decis':>6} {'intense%':>9} "
          f"{'wettest%':>9} {'ratio':>6} {'DJF n':>6} {'DJF date':>9} {'R':>5} "
          f"{'JJA n':>6} {'JJA date':>9} {'R':>5}")
    for _, r in tab.iterrows():
        print(f"  {r.cutoff:>8} {r.n_events:>5} {r.n_scored:>7} {r.n_decisive:>6} "
              f"{r.intense_pct:>8.1f}% {r.wettest_pct:>8.1f}% {r.ratio:>6.1f} "
              f"{r.n_DJF:>6} {r.date_DJF:>9} {r.R_DJF:>5.2f} "
              f"{r.n_JJA:>6} {r.date_JJA:>9} {r.R_JJA:>5.2f}")

    print("\n  SEASON MIX OF THE RETAINED SAMPLE — does the cutoff itself")
    print("  create seasonality? A cutoff landing mid-cycle keeps one season")
    print("  and drops the next.\n")
    print(f"  {'cutoff':>8} {'observed DJF':>13} {'observed JJA':>13}")
    for _, r in tab.iterrows():
        print(f"  {r.cutoff:>8} {r.obs_DJF:>12.1f}% {r.obs_JJA:>12.1f}%")

    # ------------------------------------------------------------- 4. verdict
    print("\n" + "-" * 76)
    print("4. VERDICT — keep 2 years, on three criteria that agree")
    print("-" * 76)
    asym = tab[tab.cutoff == "none"].iloc[0]
    for _, r in tab.iterrows():
        if r.cutoff in ("1 yr", "1.5 yr", "2 yr", "3 yr"):
            print(f"    {r.cutoff:>7}: keeps {r.n_events:>3} events, "
                  f"season mix off the full database by "
                  f"{abs(r.obs_DJF - asym.obs_DJF):.1f} pts (DJF)")
    print("""
  (a) RATE. The exposure-corrected rate falls from 78 events per 100 fires in
      the first half-year to 8 by 1.5-2 yr, then sits on a 1-6 plateau all the
      way to 5 yr. The postfire signal is spent by about 2 years.

  (b) SEASON-MIX NEUTRALITY — the criterion that matters most for THIS paper.
      Flows arrive in discrete seasons, so the lag distribution is modulated by
      the annual clock (panel a). A cutoff landing mid-cycle keeps one season
      and drops the next, which would manufacture seasonality in a paper whose
      subject is seasonality. Cutting at 1 yr leaves the retained sample 5.9
      points too winter-heavy; 1.5 yr, 1.7 points; 2 yr, 0.5 points. Two years
      is the shortest cutoff whose sample looks like the full database.

      This is the argument against following the rate break alone to 1.5 yr.

  (c) CAPTURE. 2 yr keeps 89% of runoff-generated events; going to 3 yr buys
      6 more points and 5 yr buys 11, at the cost of (b).

  And the choice does not carry the result. Across every cutoff from 0.5 yr to
  none, the decisive subset stays at 71-75% intense vs 5-9% wettest, a ratio of
  8:1 to 13:1. Report the 2-yr numbers, and report this table, so the window
  reads as a scope decision the result does not depend on.

  Worth stating separately: after the runoff filter the longest lag in the
  database is 5.0 yr. Every event beyond 5 years was landslide-initiated.""")

    # ------------------------------------------------------------- 5. figure
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    a = ax[0, 0]
    bins = np.arange(0, 5.05, 0.125)
    for reg in ("DJF", "JJA"):
        g = ev[ev.int_season == reg]
        a.hist(g.years_since_fire.clip(upper=5), bins=bins, alpha=.75,
               color=C[reg], label=f"{reg}-intense regime (n={len(g)})")
    a.axvline(2, color="k", ls="--", lw=1.2)
    a.text(2.05, a.get_ylim()[1] * .92, "2-yr window", fontsize=8)
    a.set_xlabel("years since ignition"); a.set_ylabel("events")
    a.set_title("a  lag distribution, by regime", loc="left", fontweight="bold")
    a.legend(fontsize=8, frameon=False)

    b = ax[0, 1]
    s = np.sort(lag.dropna())
    b.step(s, np.arange(1, len(s) + 1) / len(s) * 100, color="0.2", lw=1.6)
    for c, col in [(1, "0.6"), (2, "k"), (3, "0.6"), (5, "0.6")]:
        b.axvline(c, color=col, ls="--", lw=1.2 if c == 2 else .9)
        b.text(c, 8, f" {100*(lag <= c).mean():.0f}%", fontsize=8, color=col)
    b.set_xscale("log"); b.set_xlim(0.03, 30)
    b.set_xlabel("years since ignition (log)"); b.set_ylabel("cumulative % of events")
    b.set_title("b  cumulative capture", loc="left", fontweight="bold")

    c_ax = ax[1, 0]
    c_ax.bar(haz_x, haz_y, width=0.44, color="#009E73")
    c_ax.axvline(2, color="k", ls="--", lw=1.2)
    c_ax.set_xlabel("years since ignition")
    c_ax.set_ylabel("events per 100 fires still observed")
    c_ax.set_title("c  exposure-corrected rate", loc="left", fontweight="bold")

    d = ax[1, 1]
    num = tab[tab.cutoff != "none"].copy()
    num["x"] = [float(s.split()[0]) for s in num.cutoff]
    d.plot(num.x, num.intense_pct, "o-", color="#009E73", label="in most-intense season")
    d.plot(num.x, num.wettest_pct, "s-", color="#CC79A7", label="in wettest season")
    d.axvline(2, color="k", ls="--", lw=1.2)
    d.set_xscale("log"); d.set_ylim(0, 100)
    d.set_xlabel("cutoff (yr, log)"); d.set_ylabel("% of decisive subset")
    d.set_title("d  the result is insensitive to the cutoff", loc="left",
                fontweight="bold")
    d.legend(fontsize=8, frameon=False, loc="center right")

    fig.tight_layout()
    out = FIGURES / "postfire_lag_cutoff.png"
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    style.apply()
    with tee("postfire_lag_cutoff_report.txt"):
        main()
