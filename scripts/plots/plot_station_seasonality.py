"""
Map the seasonality of intense short-duration rainfall from Atlas 14 station
annual maxima, for visual comparison with the CONUS404-derived maps.

  /opt/anaconda3/envs/PointMan/bin/python plot_station_seasonality.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from pfdf_seasonality.paths import PROCESSED, FIGURES, STATES_ZIP
from pfdf_seasonality import style
from pfdf_seasonality.style import WEST_EXTENT as EXTENT

style.apply()


# Season colours chosen to echo the CONUS404 figures (DJF blue, MAM green,
# JJA amber, SON brown) while staying inside the Okabe-Ito family. Each season
# also gets its own marker so the map survives grayscale.
SEASON_STYLE = {
    "DJF": dict(color="#56B4E9", marker="o", label="DJF"),
    "MAM": dict(color="#009E73", marker="^", label="MAM"),
    "JJA": dict(color="#E69F00", marker="s", label="JJA"),
    "SON": dict(color="#8B4513", marker="D", label="SON"),
    "bimodal": dict(color="#CC79A7", marker="X", label="bimodal"),
}


def load():
    df = pd.read_csv(PROCESSED / "seasonality" / "atlas14_station_seasonality.csv")
    return df[df.western_us & ~df.short_record]


def panel(ax, d, col, title, states):
    states.boundary.plot(ax=ax, linewidth=0.4, color="0.35")
    for key, sty in SEASON_STYLE.items():
        s = d[d[col] == key]
        if not len(s):
            continue
        ax.scatter(s.lon, s.lat, s=13, c=sty["color"], marker=sty["marker"],
                   linewidths=0.25, edgecolors="0.2", zorder=3)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect(1 / 0.75)
    ax.set_title(f"{title}\n(n = {len(d)} stations)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    df = load()
    states = gpd.read_file(STATES_ZIP).to_crs(4326)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))

    panel(axes[0, 0], df[df.duration == "15m"], "seasonality_class",
          "Season of annual maximum 15-min rainfall", states)
    panel(axes[0, 1], df[df.duration == "30m"], "seasonality_class",
          "Season of annual maximum 30-min rainfall", states)
    panel(axes[1, 0], df[df.duration == "60m"], "seasonality_class",
          "Season of annual maximum 60-min rainfall", states)
    panel(axes[1, 1], df[df.duration == "60m"], "exc2_season",
          "Season of most exceedances of the 2-yr 60-min value", states)

    handles = [Line2D([], [], linestyle="none", marker=s["marker"], color=s["color"],
                      markeredgecolor="0.2", markeredgewidth=0.3, markersize=6,
                      label=s["label"]) for s in SEASON_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Seasonality of intense short-duration rainfall, "
                 "NOAA Atlas 14 station annual maxima", fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))

    out = FIGURES / "atlas14_station_seasonality_map.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
