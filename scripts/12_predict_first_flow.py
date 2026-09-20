"""
Flip the lag analysis into a forecast: given ONLY the pre-fire rainfall
climatology and an ignition date, how well can we predict when the first
debris flow will arrive?

Nothing here is fit to the debris-flow data. The "clock" (circular mean date of
the local 60-min annual-maximum month distribution) and the "seasonal window"
(local months with above-uniform probability of producing the annual maximum)
are both computed from Atlas 14 gauge climatology alone, and both are knowable
BEFORE any fire ignites. The debris-flow lags are used only for scoring.

Predictions evaluated (first flow per fire):
  1. Point forecast: first-opportunity clock date, pred = (clock - ign_doy) mod
     365. Scored against a per-regime constant-lag predictor that IS fit to the
     observed lags (an in-sample baseline the a priori forecast must beat).
  2. Two-branch forecast: the flow arrives in the first intense season with
     probability p_catch (per regime), else one season later. Skill = error to
     the nearest branch.
  3. Window forecast: the first run of above-uniform intensity months after
     ignition. Skill = coverage (% of first flows inside it) vs sharpness
     (window width as a fraction of the year).
  4. Operational lookup: regime x ignition month -> expected wait.

All forecasts are CONDITIONAL on a debris flow occurring: this predicts when,
not whether.

  /opt/anaconda3/envs/PointMan/bin/python predict_first_flow.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pfdf_seasonality.paths import PROCESSED, REPORTS, FIGURES
from pfdf_seasonality import style
from pfdf_seasonality.report import tee

style.apply()

OUT = PROCESSED / "seasonality"
C = {"DJF": "#56B4E9", "JJA": "#E69F00"}
LABEL = {"DJF": "winter-intense (DJF)", "JJA": "summer-intense (JJA)"}
TOL = 45          # +/- days for "hit" scoring of the point forecast


def month_windows(p: np.ndarray, ign_month: int):
    """First and second runs of above-uniform months at/after ignition month.

    Returns (first_run, second_run) as lists of month-offsets from ignition
    (offset 0 = the ignition month itself)."""
    above = p > 1.0 / 12.0
    flags = [above[(ign_month - 1 + t) % 12] for t in range(36)]
    runs, t = [], 0
    while t < 36 and len(runs) < 2:
        if flags[t]:
            t0 = t
            while t < 36 and flags[t]:
                t += 1
            runs.append(list(range(t0, t)))
        else:
            t += 1
    while len(runs) < 2:
        runs.append([])
    return runs[0], runs[1]


def main():
    ev = pd.read_csv(OUT / "pfdf_events_ignition_lags.csv",
                     parse_dates=["event_date", "ign_date"])
    first = (ev.sort_values("event_date")
             .groupby("fire_id", as_index=False).first())
    first = first[first.regime.isin(["DJF", "JJA"])].copy()

    P = first[[f"int60m_p{m:02d}" for m in range(1, 13)]].to_numpy(float)
    P = P / P.sum(axis=1, keepdims=True)

    # ---- 1. point forecast (a priori) vs fitted constant-lag baseline
    first["err"] = first.lag_days - first.pred_days
    med_lag = first.groupby("regime").lag_days.transform("median")
    first["err_const"] = first.lag_days - med_lag

    # ---- 2. nearest-branch error (= residual to the k the flow chose)
    first["err_branch"] = first.residual_d

    # ---- 3. a priori seasonal-window forecast
    hit1, hit2, width1 = [], [], []
    for i, row in enumerate(first.itertuples()):
        w1, w2 = month_windows(P[i], row.ign_date.month)
        off = ((row.event_date.year - row.ign_date.year) * 12
               + row.event_date.month - row.ign_date.month)
        hit1.append(off in w1)
        hit2.append(off in w2)
        width1.append(len(w1))
    first["win_hit1"] = hit1
    first["win_hit2"] = hit2
    first["win_width_mo"] = width1

    # ------------------------------------------------------------- report
    print("=" * 74)
    print("PREDICTING THE FIRST DEBRIS FLOW FROM PRE-FIRE CLIMATOLOGY ALONE")
    print("=" * 74)
    print(f"first flow per fire, n = {len(first)} "
          f"(DJF {sum(first.regime=='DJF')}, JJA {sum(first.regime=='JJA')}); "
          f"forecasts use only Atlas 14 climatology + ignition date\n")

    print("--- point forecast: first-opportunity clock date ---")
    print(f"{'':>28} {'DJF':>12} {'JJA':>12} {'pooled':>12}")
    rows = [
        ("median |error|, clock", lambda d: d.err.abs().median()),
        ("median |error|, const-lag fit", lambda d: d.err_const.abs().median()),
        (f"% within +/-{TOL} d, clock", lambda d: 100 * (d.err.abs() <= TOL).mean()),
        (f"% within +/-{TOL} d, const fit", lambda d: 100 * (d.err_const.abs() <= TOL).mean()),
    ]
    for name, f in rows:
        v = [f(first[first.regime == s]) for s in ["DJF", "JJA"]] + [f(first)]
        unit = "d" if "median" in name else "%"
        print(f"{name:>28} " + " ".join(f"{x:>11.0f}{unit}" for x in v))

    print("\n--- two-branch forecast (nearest of first season / one later) ---")
    for s in ["DJF", "JJA"]:
        d = first[first.regime == s]
        pc = 100 * (d.k_skipped <= 0).mean()
        print(f"  {s}: P(first season) = {pc:.0f}%;  nearest-branch median |error| "
              f"{d.err_branch.abs().median():.0f} d;  within +/-{TOL} d of a "
              f"branch: {100 * (d.err_branch.abs() <= TOL).mean():.0f}%")

    print("\n--- a priori seasonal-window forecast ---")
    for s in ["DJF", "JJA"]:
        d = first[first.regime == s]
        print(f"  {s}: window width median {d.win_width_mo.median():.0f} mo "
              f"({d.win_width_mo.median() / 12 * 100:.0f}% of the year); "
              f"first-window coverage {100 * d.win_hit1.mean():.0f}%; "
              f"first-or-second {100 * (d.win_hit1 | d.win_hit2).mean():.0f}%")

    print("\n--- operational lookup: ignition month -> median wait (days) ---")
    print(f"{'regime':>7} {'ign mo':>7} {'n':>4} {'predicted':>10} "
          f"{'observed':>9} {'P(1st season)':>14}")
    look = []
    for s in ["DJF", "JJA"]:
        d = first[first.regime == s].copy()
        d["im"] = d.ign_date.dt.month
        for m, g in d.groupby("im"):
            if len(g) < 3:
                continue
            look.append((s, m, len(g), g.pred_days.median(), g.lag_days.median(),
                         100 * (g.k_skipped <= 0).mean()))
            print(f"{s:>7} {m:>7} {len(g):>4} {look[-1][3]:>9.0f}d "
                  f"{look[-1][4]:>8.0f}d {look[-1][5]:>13.0f}%")

    # ------------------------------------------------------------- figure
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.4))
    fig.subplots_adjust(hspace=0.36, wspace=0.27)

    # (a) observed vs predicted date
    ax = axs[0, 0]
    lim = 1150
    xx = np.array([0, 365])
    for b, ls in [(0, "-"), (365, "--")]:
        ax.plot(xx, xx + b, color="0.35", lw=1.1, ls=ls, zorder=2)
        ax.fill_between(xx, xx + b - TOL, xx + b + TOL, color="0.65",
                        alpha=0.22, zorder=1, linewidth=0)
    for s in ["DJF", "JJA"]:
        g = first[first.regime == s]
        ax.scatter(g.pred_days, g.lag_days, s=18, color=C[s], edgecolors="0.25",
                   linewidths=0.3, zorder=4, label=LABEL[s])
    hit_b1 = 100 * (first.err.abs() <= TOL).mean()
    hit_any = 100 * (first.err_branch.abs() <= TOL).mean()
    ax.text(0.03, 0.97, f"within ±{TOL} d of the forecast: {hit_b1:.0f}%\n"
            f"within ±{TOL} d of a branch: {hit_any:.0f}%",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(fc="white", ec="0.7", alpha=0.85))
    ax.set_xlabel("a priori forecast: days from ignition to first intense season")
    ax.set_ylabel("observed days to first debris flow")
    ax.set_xlim(0, 370)
    ax.set_ylim(-15, lim)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("a  Forecast vs outcome (nothing fit to flow data)",
                 loc="left", fontsize=9.5, fontweight="bold")

    # (b) |error| CDFs, clock vs fitted constant
    ax = axs[0, 1]
    for s in ["DJF", "JJA"]:
        g = first[first.regime == s]
        for col, ls, lab in [("err", "-", "climatology clock"),
                             ("err_const", ":", "best constant lag (fit)")]:
            e = np.sort(g[col].abs())
            ax.step(np.r_[0, e], np.r_[0, np.arange(1, len(e) + 1)] / len(e) * 100,
                    color=C[s], ls=ls, lw=1.6,
                    label=f"{s}: {lab}" if s else None)
    ax.axvline(TOL, color="0.6", lw=0.8, ls="--")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 100)
    ax.set_xlabel("|forecast error| (days)")
    ax.set_ylabel("% of fires")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("b  A priori clock beats the fitted baseline",
                 loc="left", fontsize=9.5, fontweight="bold")

    # (c) the forecast product: ignite here -> expect the first flow then
    ax = axs[1, 0]
    doy = np.arange(1, 366)
    for s in ["DJF", "JJA"]:
        g = first[first.regime == s].copy()
        mu = g.clock_doy.median()
        pred = (mu - doy) % 365
        pred[np.abs(np.diff(pred, prepend=pred[0])) > 180] = np.nan
        ax.plot(doy, pred, color=C[s], lw=1.6, label=f"{LABEL[s]} forecast")
        g["im"] = g.ign_date.dt.month
        for m, gg in g.groupby("im"):
            if len(gg) < 3:
                continue
            x = (m - 0.5) / 12 * 365
            q1, q2, q3 = gg.lag_days.quantile([0.25, 0.5, 0.75])
            ax.errorbar(x, q2, yerr=[[q2 - q1], [q3 - q2]], fmt="o", ms=5,
                        color=C[s], mec="0.2", mew=0.4, capsize=3, lw=1.2)
    mon = np.array([1, 60, 121, 182, 244, 305])
    ax.set_xticks(mon)
    ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.set_xlabel("ignition day of year")
    ax.set_ylabel("expected days to first debris flow")
    ax.set_ylim(-15, 820)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("c  The forecast product (dots: observed median + IQR)",
                 loc="left", fontsize=9.5, fontweight="bold")

    # (d) window forecast: sharpness vs coverage
    ax = axs[1, 1]
    xx = np.arange(2)
    cov1 = [100 * first[first.regime == s].win_hit1.mean() for s in ["DJF", "JJA"]]
    cov2 = [100 * (first[first.regime == s].win_hit1
                   | first[first.regime == s].win_hit2).mean()
            for s in ["DJF", "JJA"]]
    share = [first[first.regime == s].win_width_mo.median() / 12 * 100
             for s in ["DJF", "JJA"]]
    ax.bar(xx - 0.22, share, 0.36, color="0.75", edgecolor="0.3", linewidth=0.5,
           label="window width (% of year)")
    ax.bar(xx + 0.22, cov1, 0.36, color=[C["DJF"], C["JJA"]], edgecolor="0.3",
           linewidth=0.5, label="first-window coverage")
    for i in range(2):
        ax.text(i - 0.22, share[i] + 2, f"{share[i]:.0f}%", ha="center", fontsize=8.5)
        ax.text(i + 0.22, cov1[i] + 2, f"{cov1[i]:.0f}%", ha="center", fontsize=8.5,
                fontweight="bold")
        ax.text(i + 0.22, cov2[i] + 9, f"1st or 2nd: {cov2[i]:.0f}%", ha="center",
                fontsize=7.5, color="0.3")
    ax.set_xticks(xx)
    ax.set_xticklabels(["winter-intense\n(DJF)", "summer-intense\n(JJA)"])
    ax.set_ylim(0, 118)
    ax.set_ylabel("%")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("d  Narrow a priori windows capture most first flows",
                 loc="left", fontsize=9.5, fontweight="bold")

    fig.suptitle("Predicting first-debris-flow timing from pre-fire rainfall "
                 "climatology and the ignition date alone\n"
                 f"first flow per fire (n={len(first)}); forecast = Atlas 14 "
                 "intense-season clock; conditional on a flow occurring",
                 fontsize=11, y=0.995)
    out = FIGURES / "first_flow_prediction.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    with tee("first_flow_prediction_report.txt"):
        main()
