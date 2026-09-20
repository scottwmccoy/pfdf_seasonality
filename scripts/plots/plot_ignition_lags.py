"""
Figure: the ignition-to-first-flow interval is set by the intense-rainfall clock.

Four panels from compiled/pfdf_events_ignition_lags.csv (MTBS ignition dates):
  a  first-flow-per-fire lag histograms, winter vs summer regime — the summer
     regime is bimodal with an empty 180-300 d trough; winter is tight
  b  ignition day-of-year vs lag, with the first-opportunity clock curves
  c  intense seasons skipped before the first flow, by regime
  d  summer regime: share of burns catching their first monsoon, by ignition month

  /opt/anaconda3/envs/PointMan/bin/python plot_ignition_lags.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pfdf_seasonality.paths import PROCESSED, FIGURES
from pfdf_seasonality import style

style.apply()

OUT = PROCESSED / "seasonality"
C = {"DJF": "#56B4E9", "JJA": "#E69F00"}
LABEL = {"DJF": "winter-intense regime (DJF)", "JJA": "summer-intense regime (JJA)"}


def main():
    ev = pd.read_csv(OUT / "pfdf_events_ignition_lags.csv",
                     parse_dates=["event_date", "ign_date"])
    first = ev.sort_values("event_date").groupby("fire_id", as_index=False).first()

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.subplots_adjust(hspace=0.34, wspace=0.26)

    # --- (a) lag histograms, first flow per fire
    ax = axs[0, 0]
    bins = np.arange(0, 1151, 50)
    for s in ["DJF", "JJA"]:
        g = first[first.regime == s]
        ax.hist(g.lag_days.clip(upper=1140), bins=bins, alpha=0.55, color=C[s],
                edgecolor="0.3", linewidth=0.4,
                label=f"{LABEL[s]}  (n={len(g)}, median {g.lag_days.median():.0f} d)")
    ax.axvspan(180, 300, color="0.55", alpha=0.18, zorder=0)
    ax.annotate("the monsoon trough:\nonly 3% of summer-regime fires\nfall at 180-300 d",
                xy=(240, 3.2), xytext=(430, 8.6), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax.set_xlabel("ignition → first debris flow (days)")
    ax.set_ylabel("fires")
    ax.set_xlim(0, 1150)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("a  Same season or a year later: lag to the first flow",
                 loc="left", fontsize=9.5, fontweight="bold")

    # --- (b) ignition DOY vs lag with the clock curves
    ax = axs[0, 1]
    doy = np.arange(1, 366)
    for s in ["DJF", "JJA"]:
        g = first[first.regime == s]
        mu = g.clock_doy.median()
        pred = (mu - doy) % 365
        pred[np.abs(np.diff(pred, prepend=pred[0])) > 180] = np.nan  # break wrap
        ax.plot(doy, pred, color=C[s], lw=1.4, alpha=0.9)
        ax.plot(doy, pred + 365, color=C[s], lw=1.4, ls="--", alpha=0.7)
        ax.scatter(g.ign_doy, g.lag_days, s=16, color=C[s], edgecolors="0.25",
                   linewidths=0.3, zorder=4)
    ax.plot([], [], color="0.4", lw=1.4, label="first intense season")
    ax.plot([], [], color="0.4", lw=1.4, ls="--", label="one season skipped")
    mon = np.array([1, 60, 121, 182, 244, 305])
    ax.set_xticks(mon)
    ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.set_xlabel("ignition day of year")
    ax.set_ylabel("ignition → first debris flow (days)")
    ax.set_ylim(-20, 1150)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("b  Lags follow the local intense-season clock",
                 loc="left", fontsize=9.5, fontweight="bold")

    # --- (c) seasons skipped, by regime
    ax = axs[1, 0]
    xx = np.arange(2)
    caught = [100 * (first[first.regime == s].k_skipped <= 0).mean()
              for s in ["DJF", "JJA"]]
    skip1 = [100 * (first[first.regime == s].k_skipped == 1).mean()
             for s in ["DJF", "JJA"]]
    skip2 = [100 * (first[first.regime == s].k_skipped >= 2).mean()
             for s in ["DJF", "JJA"]]
    ax.bar(xx, caught, 0.55, color=[C["DJF"], C["JJA"]], edgecolor="0.3",
           linewidth=0.5)
    ax.bar(xx, skip1, 0.55, bottom=caught, color="0.75", edgecolor="0.3",
           linewidth=0.5, label="skipped one season")
    ax.bar(xx, skip2, 0.55, bottom=np.array(caught) + np.array(skip1),
           color="0.45", edgecolor="0.3", linewidth=0.5,
           label="skipped two or more")
    for i, s in enumerate(["DJF", "JJA"]):
        n = (first.regime == s).sum()
        ax.text(i, caught[i] / 2, f"{caught[i]:.0f}%\nin first\nintense season",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(i, 103, f"n={n}", ha="center", fontsize=8, color="0.25")
    ax.set_xticks(xx)
    ax.set_xticklabels(["winter-intense\n(DJF)", "summer-intense\n(JJA)"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of fires")
    ax.legend(fontsize=8, frameon=False, loc="lower left",
              bbox_to_anchor=(0.02, 0.02))
    ax.set_title("c  Most burns are hit in their first intense season",
                 loc="left", fontsize=9.5, fontweight="bold")

    # --- (d) JJA regime: catching the first monsoon vs ignition month
    ax = axs[1, 1]
    j = first[first.regime == "JJA"].copy()
    j["ign_month"] = j.ign_date.dt.month
    months = [4, 5, 6, 7, 8, 9]
    frac, ns = [], []
    for m in months:
        g = j[j.ign_month == m]
        ns.append(len(g))
        frac.append(100 * (g.k_skipped <= 0).mean() if len(g) else np.nan)
    xb = np.arange(len(months))
    ax.bar(xb, frac, 0.6, color=C["JJA"], edgecolor="0.3", linewidth=0.5)
    for i, (f, n) in enumerate(zip(frac, ns)):
        if n:
            ax.text(i, (f if np.isfinite(f) else 0) + 3, f"n={n}", ha="center",
                    fontsize=8, color="0.25")
    ax.set_xticks(xb)
    ax.set_xticklabels(["Apr", "May", "Jun", "Jul", "Aug", "Sep"])
    ax.set_xlabel("ignition month")
    ax.set_ylabel("% hit in their first monsoon")
    ax.set_ylim(0, 112)
    ax.set_title("d  Summer regime: catching the first monsoon depends on\n"
                 "    ignition month", loc="left", fontsize=9.5, fontweight="bold")

    fig.suptitle("The interval from ignition to the first debris flow is set by the "
                 "intense-rainfall clock\nMTBS ignition dates matched to "
                 f"{len(ev)} compiled events ({first.fire_id.nunique()} fires); "
                 "regimes from Atlas 14 gauge climatology",
                 fontsize=11, y=0.995)
    out = FIGURES / "ignition_lag.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
