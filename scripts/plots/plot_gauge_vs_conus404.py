"""
Gauges vs CONUS404, side by side — do the two independent lines tell the same story?

  /opt/anaconda3/envs/PointMan/bin/python plot_gauge_vs_conus404.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from pfdf_seasonality.paths import CONUS404, PROCESSED, FIGURES, STATES_ZIP
from pfdf_seasonality.seasons import SEASONS, normalize
from pfdf_seasonality import style
from pfdf_seasonality.style import SEASON_COLOR, WEST_EXTENT as EXTENT

style.apply()

C404 = CONUS404
OUT = PROCESSED / "seasonality"

C_INTENSE, C_WET = "#009E73", "#CC79A7"
MON = [pd.Timestamp(2001, m, 1).strftime("%b")[0] for m in range(1, 13)]


def frame(ax, states, title):
    states.boundary.plot(ax=ax, linewidth=0.4, color="0.25", zorder=5)
    ax.set_xlim(*EXTENT[:2]); ax.set_ylim(*EXTENT[2:])
    ax.set_aspect(1 / 0.75)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)


def main():
    states = gpd.read_file(STATES_ZIP).to_crs(4326)
    g = np.load(C404 / "grid.npz")
    lat, lon = g["lat"], g["lon"]
    y0, y1, x0, x1 = [int(v) for v in g["box"]]
    ny, nx = y1 - y0, x1 - x0
    tiles = sorted((C404 / "hourly").glob("tile_y*_x*.npz"))
    if tiles:
        # duration-matched 1-hour accumulation (preferred)
        frac = np.full((12, ny, nx), np.nan, dtype="float32")
        for f in tiles:
            d = np.load(f)
            ty, tx, mon = int(d["ty"]), int(d["tx"]), d["month"]
            T = mon.shape[1]
            dy0, dx0 = max(ty - y0, 0), max(tx - x0, 0)
            dy1, dx1 = min(ty - y0 + T, ny), min(tx - x0 + T, nx)
            if dy1 <= dy0 or dx1 <= dx0:
                continue
            sy0, sx0 = dy0 - (ty - y0), dx0 - (tx - x0)
            sub = mon[:, sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
            for mi, m in enumerate(range(1, 13)):
                frac[mi, dy0:dy1, dx0:dx1] = (sub == m).mean(axis=0)
        metric = "1-hour"
    else:
        files = sorted(C404.glob("amax_wy*.npz"))
        month = np.stack([np.load(f)["month"] for f in files])
        frac = np.stack([(month == m).mean(axis=0) for m in range(1, 13)])
        metric = "instantaneous rate"
    clim = np.load(C404 / "monthly_climatology.npz")["clim"]

    def dom(P):
        seas = np.stack([P[[m - 1 for m in ms]].sum(axis=0) for ms in SEASONS.values()])
        return seas.argmax(axis=0), seas.max(axis=0) / P.sum(axis=0)

    d_int, sh_int = dom(frac)
    d_wet, _ = dom(clim)

    stn = pd.read_csv(OUT / "atlas14_station_seasonality.csv")
    stn = stn[stn.western_us & ~stn.short_record]
    evf = OUT / "pfdf_events_conus404_seasonality_hourly.csv"
    if not evf.exists():
        evf = OUT / "pfdf_events_conus404_seasonality.csv"
    ev = pd.read_csv(evf, parse_dates=["event_date"])
    ev["df_month"] = ev.event_date.dt.month

    order = list(SEASONS)
    cmap = ListedColormap([SEASON_COLOR[s] for s in order])
    norm = BoundaryNorm(np.arange(-0.5, 4.5), 4)

    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.24, wspace=0.20)

    # --- (a) CONUS404 intensity season, gridded
    ax = fig.add_subplot(gs[0, 0])
    ax.pcolormesh(lon, lat, d_int, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
    frame(ax, states, f"a  CONUS404: season of annual max ({metric})")

    # --- (b) the same thing from gauges, on top of the CONUS404 field
    ax = fig.add_subplot(gs[0, 1])
    ax.pcolormesh(lon, lat, d_int, cmap=cmap, norm=norm, shading="nearest",
                  alpha=0.30, rasterized=True)
    s60 = stn[stn.duration == "60m"]
    ax.scatter(s60.lon, s60.lat, s=11, marker="o",
               c=[SEASON_COLOR.get(x, "0.5") for x in s60.dominant_season],
               edgecolors="0.15", linewidths=0.3, zorder=4)
    frame(ax, states, "b  Atlas 14 gauges on the CONUS404 field")

    # --- (c) where the two models disagree, in CONUS404
    ax = fig.add_subplot(gs[0, 2])
    same = d_int == d_wet
    ax.pcolormesh(lon, lat, np.where(same, 0, 1), shading="nearest", rasterized=True,
                  cmap=ListedColormap(["#DDD8CE", "#D55E00"]), vmin=0, vmax=1)
    ax.scatter(ev.longitude, ev.latitude, s=8, facecolors="none", edgecolors="k",
               linewidths=0.45, zorder=6)
    frame(ax, states, "c  Wettest season $\\neq$ most intense (orange)")
    ax.legend(handles=[
        Line2D([], [], marker="s", linestyle="none", color="#D55E00", markersize=7,
               label="different"),
        Line2D([], [], marker="s", linestyle="none", color="#DDD8CE", markersize=7,
               label="same"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="k", markersize=6, label="debris flow")],
        loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)

    # --- (d) pooled month distribution, CONUS404 models vs observed
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(1, 13)
    obs = ev.df_month.value_counts().reindex(x, fill_value=0).to_numpy()
    Pi = normalize(ev[[f"c404_int_p{m:02d}" for m in x]].to_numpy(float))
    Pw = normalize(ev[[f"c404_wet_p{m:02d}" for m in x]].to_numpy(float))
    ax.bar(x, obs, 0.68, color="0.72", edgecolor="0.35", linewidth=0.4,
           label=f"observed (n={len(ev)})")
    ax.plot(x, Pi.sum(axis=0), "o-", color=C_INTENSE, ms=4, lw=1.8, label="CONUS404 intensity")
    ax.plot(x, Pw.sum(axis=0), "s--", color=C_WET, ms=4, lw=1.8, label="CONUS404 amount")
    ax.set_xticks(x); ax.set_xticklabels(MON)
    ax.set_ylabel("events per month")
    ax.set_ylim(0, max(obs.max(), Pi.sum(axis=0).max(), Pw.sum(axis=0).max()) * 1.40)
    ax.set_title("d  CONUS404: same answer as the gauges", fontsize=9, loc="left",
                 fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # --- (e) headline numbers, both lines of evidence
    ax = fig.add_subplot(gs[1, 1])
    parts = [pd.read_csv(OUT / f) for f in
             ["gauge_vs_conus404_summary.csv", "gauge_vs_conus404_summary_hourly.csv"]
             if (OUT / f).exists()]
    gauge = (pd.concat(parts, ignore_index=True)
             .drop_duplicates("label", keep="last").reset_index(drop=True))
    gauge["is_gauge"] = ~gauge.label.str.startswith("CONUS404")
    gauge = gauge.sort_values(["is_gauge", "label"], ascending=[False, True]).reset_index(drop=True)
    xx = np.arange(len(gauge))
    ax.bar(xx - 0.2, gauge.pct_intense, 0.38, color=C_INTENSE,
           label="in most-intense season")
    ax.bar(xx + 0.2, gauge.pct_wet, 0.38, color=C_WET, label="in wettest season")
    for i, row in gauge.iterrows():
        ax.text(i - 0.2, row.pct_intense + 2, f"{row.pct_intense:.0f}%", ha="center", fontsize=8)
        ax.text(i + 0.2, row.pct_wet + 2, f"{row.pct_wet:.0f}%", ha="center", fontsize=8)
        ax.text(i, 4, f"n={int(row.n)}", ha="center", fontsize=7.5, color="0.25")
    ax.set_xticks(xx); ax.set_xticklabels(gauge.label, fontsize=7.5)
    ax.set_ylim(0, 100); ax.set_ylabel("% of debris flows")
    ax.set_title("e  Decisive subset, both lines of evidence", fontsize=9, loc="left",
                 fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    # --- (f) gauge vs CONUS404 agreement at the same points
    ax = fig.add_subplot(gs[1, 2])
    af = OUT / "gauge_conus404_agreement_hourly.csv"
    if not af.exists():
        af = OUT / "gauge_conus404_agreement.csv"
    agr = pd.read_csv(af)
    ax.bar(agr.duration, agr.pct_same_season, 0.55, color="#0072B2")
    for i, row in agr.iterrows():
        ax.text(i, row.pct_same_season + 1.5, f"{row.pct_same_season:.0f}%",
                ha="center", fontsize=8)
        ax.text(i, 4, f"n={int(row.n)}", ha="center", fontsize=7.5, color="w")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% same dominant season")
    ax.set_title("f  Do gauge and model agree point-by-point?", fontsize=9, loc="left",
                 fontweight="bold")

    handles = [Line2D([], [], marker="s", linestyle="none", color=SEASON_COLOR[s],
                      markersize=8, label=s) for s in order]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.925), fontsize=9)
    fig.suptitle("Two independent lines of evidence for post-fire debris-flow seasonality\n"
                 "rain gauges (NOAA Atlas 14 + NCEI normals)  vs  CONUS404 4-km reanalysis, "
                 "water years 1980–2024", fontsize=11, y=1.005)
    out = FIGURES / "gauge_vs_conus404.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
