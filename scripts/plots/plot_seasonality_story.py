"""
Main figure for the gauge-only debris-flow seasonality result.

  /opt/anaconda3/envs/PointMan/bin/python plot_pfdf_seasonality_story.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pfdf_seasonality.paths import PROCESSED, FIGURES
from pfdf_seasonality.seasons import SEASONS, normalize
from pfdf_seasonality import style

style.apply(savefig_dpi=220)

EV = PROCESSED / "seasonality" / "pfdf_events_gauge_seasonality.csv"
MON = [pd.Timestamp(2001, m, 1).strftime("%b")[0] for m in range(1, 13)]

C_WINTER = "#0072B2"     # DJF-dominant rainfall regime
C_MONSOON = "#D55E00"    # JJA-dominant rainfall regime
C_INTENSE = "#009E73"    # short-duration model
C_WET = "#CC79A7"        # wet-season model


def main():
    ev = pd.read_csv(EV, parse_dates=["event_date"])
    ev["df_month"] = ev.event_date.dt.month
    use = ev[ev.int60m_ok & ev.wet_ok].copy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))

    # ---- (a) debris-flow timing, split by gauge-defined rainfall regime
    ax = axes[0, 0]
    x = np.arange(1, 13)
    w = 0.42
    for off, reg, col in [(-w / 2, "DJF", C_WINTER), (w / 2, "JJA", C_MONSOON)]:
        g = use[use.int60m_season == reg]
        cnt = g.df_month.value_counts().reindex(x, fill_value=0)
        ax.bar(x + off, cnt, w, color=col,
               label=f"{reg}-dominant 60-min rainfall  (n={len(g)})")
    ax.set_xticks(x); ax.set_xticklabels(MON)
    ax.set_ylabel("debris-flow events")
    ax.set_title("a  Debris flows split cleanly by local rainfall regime", loc="left",
                 fontweight="bold", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    # ---- (b) observed vs the two competing gauge models
    ax = axes[0, 1]
    obs = use.df_month.value_counts().reindex(x, fill_value=0).to_numpy()
    e_int = normalize(use[[f"int60m_p{m:02d}" for m in x]].to_numpy(float)).sum(axis=0)
    e_wet = normalize(use[[f"wet_p{m:02d}" for m in x]].to_numpy(float)).sum(axis=0)
    ax.bar(x, obs, 0.68, color="0.72", edgecolor="0.35", linewidth=0.4,
           label=f"observed debris flows (n={len(use)})")
    ax.plot(x, e_int, "o-", color=C_INTENSE, ms=4, lw=1.8,
            label="expected if timed by intense 60-min rainfall")
    ax.plot(x, e_wet, "s--", color=C_WET, ms=4, lw=1.8,
            label="expected if timed by monthly precipitation")
    ax.set_xticks(x); ax.set_xticklabels(MON)
    ax.set_ylabel("events per month")
    ax.set_title("b  Timing follows rainfall intensity, not rainfall amount", loc="left",
                 fontweight="bold", fontsize=9.5)
    ax.set_ylim(0, max(obs.max(), e_int.max(), e_wet.max()) * 1.42)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    # Computed, not typed: these read 59 and 355 for months after the run that
    # produced them, while the report said 45.0 and 282.4 -- a figure that
    # looks calculated and is not is worse than no number at all.
    chi_int = float(((obs - e_int) ** 2 / e_int).sum())
    chi_wet = float(((obs - e_wet) ** 2 / e_wet).sum())
    ax.text(0.97, 0.70,
            f"$\\chi^2$ = {chi_int:.0f} (intensity)\nvs {chi_wet:.0f} (amount)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.3")

    # ---- (c) the decisive subset: where the two models disagree
    ax = axes[1, 0]
    labels, hi, hw, ns = [], [], [], []
    for dur in ["15m", "30m", "60m"]:
        ok = ev[f"int{dur}_ok"] & ev.wet_ok
        d = ev[ok & (ev[f"int{dur}_season"] != ev.wet_season)]
        if not len(d):
            continue
        labels.append(dur.replace("m", "-min"))
        hi.append((d.df_season == d[f"int{dur}_season"]).mean() * 100)
        hw.append((d.df_season == d.wet_season).mean() * 100)
        ns.append(len(d))
    xx = np.arange(len(labels))
    ax.bar(xx - 0.2, hi, 0.38, color=C_INTENSE, label="in the most-intense-rainfall season")
    ax.bar(xx + 0.2, hw, 0.38, color=C_WET, label="in the wettest season")
    for i, (a, b, n) in enumerate(zip(hi, hw, ns)):
        ax.text(i - 0.2, a + 2, f"{a:.0f}%", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 2, f"{b:.0f}%", ha="center", fontsize=8)
        ax.text(i, 5, f"n={n}", ha="center", fontsize=7.5, color="0.25")
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of debris flows")
    ax.set_title("c  Where the two disagree, intensity wins ~9:1", loc="left",
                 fontweight="bold", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    # ---- (d) the control: same fire season, opposite flow season
    ax = axes[1, 1]
    f = ev.dropna(subset=["fire_start_date"]).copy()
    f["fire_start_date"] = pd.to_datetime(f.fire_start_date, errors="coerce")
    f = f.dropna(subset=["fire_start_date"])
    f["lag"] = (f.event_date - f.fire_start_date).dt.days
    f = f[(f.lag >= 0) & (f.lag <= 5 * 365) & f.int60m_ok]
    for reg, col, ls in [("DJF", C_WINTER, "-"), ("JJA", C_MONSOON, "-")]:
        g = f[f.int60m_season == reg]
        fm = g.fire_start_date.dt.month.value_counts().reindex(x, fill_value=0)
        dm = g.df_month.value_counts().reindex(x, fill_value=0)
        ax.plot(x, fm / fm.sum() * 100, ls, color=col, lw=1.2, alpha=0.45,
                marker="^", ms=3.5, label=f"{reg} regime: fire ignitions")
        ax.plot(x, dm / dm.sum() * 100, ls, color=col, lw=2.2,
                marker="o", ms=4, label=f"{reg} regime: debris flows")
    ax.set_xticks(x); ax.set_xticklabels(MON)
    ax.set_ylabel("% of that regime's events")
    ax.set_title("d  Both regimes burn in summer; only the flows differ", loc="left",
                 fontweight="bold", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    fig.suptitle("Post-fire debris-flow seasonality in the western US, from rain gauges alone\n"
                 "NOAA Atlas 14 station annual maxima + NCEI 1991–2020 precipitation normals",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGURES / "pfdf_seasonality_gauge_story.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
