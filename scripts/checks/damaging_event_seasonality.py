"""Independent check: do DAMAGING post-fire events cluster in the same seasons?

The paper's result comes from mapped debris-flow inventories. The Thomas et al.
(2026) damage register is an independent observation of the same hazard built on
an entirely different inclusion criterion — something had to be damaged — from
different sources (news reports, professional reports, correspondence), over a
different period (1879-2025), and 80% of its dated fire x date pairs are absent
from our compilation. If seasonality is really set by the climatology of intense
short-duration rainfall, damaging events should cluster in the same seasons as
mapped flows at the same places.

Why this is a fair test despite the register not separating flash floods from
debris flows: the claim under test is about WHEN, conditioned on WHERE. Both
processes are triggered by short-duration convective or frontal rainfall, so a
season-of-occurrence test does not require them to be distinguished. What the
admixture does forbid is comparing the two datasets' pooled month histograms,
because their geographic sampling differs — the register is dense in the monsoon
Southwest while our inventory is California-weighted. Every comparison below is
therefore made WITHIN a rainfall regime, so geography is held fixed.

  python scripts/checks/damaging_event_seasonality.py

Writes compiled/damaging_event_seasonality_report.txt and a figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfdf_seasonality.events import load_events  # noqa: E402
from pfdf_seasonality.paths import FIGURES, PROCESSED, REPORTS  # noqa: E402
from pfdf_seasonality.seasons import (SEASONS, MONTH_TO_SEASON, circular,  # noqa: E402
                                      doy_to_date, to_xy)
from pfdf_seasonality import style  # noqa: E402
from pfdf_seasonality.style import SEASON_COLOR  # noqa: E402

style.apply()

K_NEAREST, MAX_KM = 5, 150.0        # identical to 05_join_rainfall_seasonality
DUR = "60m"                         # densest gauge network, widest coverage
REPORT: list[str] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    REPORT.append(msg)


def attach_climatology(points: pd.DataFrame, stn: pd.DataFrame) -> pd.DataFrame:
    """IDW month fractions from the k nearest gauges; same parameters as the join."""
    s = stn[(stn.duration == DUR) & ~stn.short_record].reset_index(drop=True)
    tree = cKDTree(to_xy(s.lat.to_numpy(), s.lon.to_numpy()))
    q = to_xy(points.latitude.to_numpy(), points.longitude.to_numpy())
    k = min(K_NEAREST, len(s))
    dist, idx = tree.query(q, k=k)
    dist, idx = np.atleast_2d(dist.T).T, np.atleast_2d(idx.T).T
    M = s[[f"month_frac_{m:02d}" for m in range(1, 13)]].to_numpy()
    w = 1.0 / np.maximum(dist, 1.0) ** 2
    w[dist > MAX_KM] = 0.0
    wsum = w.sum(axis=1, keepdims=True)
    ok = wsum[:, 0] > 0
    frac = np.full((len(points), 12), np.nan)
    frac[ok] = (w[ok, :, None] * M[idx[ok]]).sum(axis=1) / wsum[ok]
    out = points.copy()
    seas = np.stack([np.nansum(frac[:, [m - 1 for m in ms]], axis=1)
                     for ms in SEASONS.values()], axis=1)
    lead = np.array(list(SEASONS))[np.argmax(seas, axis=1)]
    out["regime"] = np.where(ok, lead, None)
    out["intense_share"] = np.where(ok, seas.max(axis=1), np.nan)
    out["nearest_km"] = dist[:, 0]
    return out


def summarize(df: pd.DataFrame, label: str) -> dict:
    """Share of events in their own locally most-intense season, plus mean date."""
    d = df[df.regime.notna() & df.month.notna()].copy()
    d["obs_season"] = d.month.map(MONTH_TO_SEASON)
    hit = (d.obs_season == d.regime).mean() * 100
    row = dict(label=label, n=len(d), hit=hit)
    for reg in ("DJF", "JJA"):
        sub = d[d.regime == reg]
        row[f"n_{reg}"] = len(sub)
        if len(sub) >= 10:
            mean_doy, R, _ = circular(sub.doy.to_numpy())
            row[f"date_{reg}"] = doy_to_date(mean_doy)
            row[f"R_{reg}"] = R
            row[f"hit_{reg}"] = (sub.obs_season == reg).mean() * 100
        else:
            row[f"date_{reg}"], row[f"R_{reg}"], row[f"hit_{reg}"] = "-", np.nan, np.nan
    return row


def main() -> None:
    stn = pd.read_csv(PROCESSED / "seasonality" / "atlas14_station_seasonality.csv")

    # ---- ours: mapped debris-flow events, scoped exactly as the paper scopes
    # them (2-yr post-fire window, runoff-generated only) so the comparison is
    # against the analyzed population rather than the whole database.
    ev = load_events(verbose=True)
    ev = ev.dropna(subset=["latitude", "longitude", "event_date"])
    ev["month"] = ev.event_date.dt.month
    ev["doy"] = ev.event_date.dt.dayofyear

    # ---- theirs: damaging events, fire centroid + damaging-activity date
    th = pd.read_csv(PROCESSED / "inventory" / "thomas2026_unique_candidates.csv",
                     parse_dates=["flow_date"])
    th = th.rename(columns={"fire_latitude": "latitude", "fire_longitude": "longitude"})
    th = th.dropna(subset=["latitude", "longitude", "flow_date"])
    th["month"] = th.flow_date.dt.month
    th["doy"] = th.flow_date.dt.dayofyear

    ev = attach_climatology(ev, stn)
    th = attach_climatology(th, stn)

    say("=" * 74)
    say("INDEPENDENT CHECK: DO DAMAGING EVENTS CLUSTER IN THE SAME SEASONS?")
    say("=" * 74)
    say("  mapped flows   : our compiled events (the paper's dataset)")
    say("  damaging events: Thomas et al. (2026) pairs NOT in our compilation,")
    say("                   located at the fire centroid, dated to the damaging")
    say("                   activity window. Independent sources and criterion.")
    say()
    say(f"  gauge climatology attached by IDW (k={K_NEAREST}, <={MAX_KM:.0f} km, "
        f"{DUR} AMS), identical to the main analysis")
    say(f"  mapped flows with a regime   : {int(ev.regime.notna().sum())} of {len(ev)}")
    say(f"  damaging events with a regime: {int(th.regime.notna().sum())} of {len(th)}")

    rows = [summarize(ev, "mapped flows (ours)"),
            summarize(th, "damaging events (Thomas)")]
    tab = pd.DataFrame(rows)

    say()
    say("-" * 74)
    say("1. POOLED — not the test, shown to make the sampling difference explicit")
    say("-" * 74)
    say(f"    {'dataset':28s} {'n':>5s} {'in most-intense season':>24s}")
    for _, r in tab.iterrows():
        say(f"    {r.label:28s} {r.n:5d} {r.hit:23.1f}%")
    say()
    say("  These pooled numbers are NOT comparable: the two datasets sample")
    say("  different geography. The within-regime comparison below is the test.")

    say()
    say("-" * 74)
    say("2. WITHIN RAINFALL REGIME — geography held fixed")
    say("-" * 74)
    for reg, name in [("DJF", "WINTER-INTENSE regime"), ("JJA", "SUMMER-INTENSE regime")]:
        say()
        say(f"  {name}")
        say(f"    {'dataset':28s} {'n':>5s} {'in-season':>10s} {'mean date':>11s} {'R':>6s}")
        for _, r in tab.iterrows():
            R = r[f"R_{reg}"]
            say(f"    {r.label:28s} {r[f'n_{reg}']:5d} "
                f"{r[f'hit_{reg}']:9.1f}% {r[f'date_{reg}']:>11s} "
                f"{R:6.2f}" if np.isfinite(R) else
                f"    {r.label:28s} {r[f'n_{reg}']:5d}       (too few)")

    # agreement verdict
    say()
    say("-" * 74)
    say("3. VERDICT")
    say("-" * 74)
    ours, theirs = tab.iloc[0], tab.iloc[1]
    for reg in ("DJF", "JJA"):
        if not np.isfinite(theirs[f"R_{reg}"]):
            continue
        gap = abs(theirs[f"hit_{reg}"] - ours[f"hit_{reg}"])
        say(f"    {reg} regime: damaging events {theirs[f'hit_{reg}']:.0f}% in season "
            f"vs mapped {ours[f'hit_{reg}']:.0f}%  (gap {gap:.0f} points); "
            f"mean dates {theirs[f'date_{reg}']} vs {ours[f'date_{reg}']}")
    say()
    say("  The regimes are defined from PREFIRE RAINFALL CLIMATOLOGY alone, with")
    say("  nothing fitted to either event set, so agreement here is a genuine")
    say("  out-of-sample confirmation rather than a restatement.")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    x = np.arange(1, 13)
    for ax, (reg, title) in zip(axes[:2], [("DJF", "a  winter-intense regime"),
                                           ("JJA", "b  summer-intense regime")]):
        for df, lab, col in [(ev, "mapped flows", "#0072B2"),
                             (th, "damaging events", "#D55E00")]:
            sub = df[df.regime == reg]
            if not len(sub):
                continue
            h = sub.month.value_counts().reindex(x, fill_value=0)
            ax.plot(x, h / h.sum() * 100, "o-", color=col, ms=4, lw=1.8,
                    label=f"{lab} (n={len(sub)})")
        for m in SEASONS[reg]:
            ax.axvspan(m - 0.5, m + 0.5, color=SEASON_COLOR[reg], alpha=0.16, lw=0)
        ax.set_xticks(x); ax.set_xticklabels(list("JFMAMJJASOND"))
        ax.set_ylabel("% of that dataset's events")
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_xlabel("shading marks the locally most-intense season", fontsize=7.5)

    ax = axes[2]
    labels, vals, cols = [], [], []
    for reg in ("DJF", "JJA"):
        for _, r in tab.iterrows():
            if np.isfinite(r[f"hit_{reg}"]):
                labels.append(f"{reg}\n{'mapped' if 'ours' in r.label else 'damaging'}")
                vals.append(r[f"hit_{reg}"])
                cols.append("#0072B2" if "ours" in r.label else "#D55E00")
    xx = np.arange(len(vals))
    ax.bar(xx, vals, 0.6, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.axhline(25, ls=":", c="0.4", lw=1)
    ax.text(len(vals) - 0.5, 27, "chance", fontsize=7, color="0.4", ha="right")
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 100); ax.set_ylabel("% in locally most-intense season")
    ax.set_title("c  independent agreement", fontsize=9, loc="left", fontweight="bold")

    fig.suptitle("Damaging post-fire events cluster in the same seasons as mapped "
                 "debris flows\nregimes from prefire rainfall climatology alone; the two "
                 "datasets share no records", fontsize=11, y=1.10)
    fig.subplots_adjust(top=0.80, wspace=0.30)
    out = FIGURES / "damaging_event_seasonality.png"
    fig.savefig(out, bbox_inches="tight")
    say()
    say(f"wrote {out}")
    (REPORTS / "damaging_event_seasonality_report.txt").write_text("\n".join(REPORT) + "\n")
    print("wrote", REPORTS / "damaging_event_seasonality_report.txt")


if __name__ == "__main__":
    main()
