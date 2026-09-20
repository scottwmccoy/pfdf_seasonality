"""
Map the Atlas 14 60-min seasonality product over the western US, from the
0.5-degree probe of the PFDS endpoint.

The point of the left panel is to show the product's *actual* spatial
resolution: it is piecewise-constant over regional-frequency sub-regions, so
"classify a pixel" really means "classify a sub-region".

  /opt/anaconda3/envs/PointMan/bin/python plot_atlas14_grid.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from pfdf_seasonality.paths import ATLAS14, PROCESSED, FIGURES, STATES_ZIP
from pfdf_seasonality.seasons import SEASONS
from pfdf_seasonality import style
from pfdf_seasonality.style import SEASON_COLOR, WEST_EXTENT as EXTENT

style.apply()

EVENTS = PROCESSED / "seasonality" / "pfdf_events_with_rainfall_seasonality.csv"
GRID = ATLAS14 / "atlas14_grid_60m_0.5.csv"


def frame(ax, states, title):
    states.boundary.plot(ax=ax, linewidth=0.4, color="0.25", zorder=5)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect(1 / 0.75)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    g = pd.read_csv(GRID)
    states = gpd.read_file(STATES_ZIP).to_crs(4326)
    cov = g[g.covered].copy()

    mcols = [f"aep1_2_m{m:02d}" for m in range(1, 13)]
    A = cov[mcols].to_numpy(dtype=float)
    cov["dominant_month"] = A.argmax(axis=1) + 1
    seas = np.column_stack([A[:, [m - 1 for m in ms]].sum(axis=1) for ms in SEASONS.values()])
    cov["dominant_season"] = np.array(list(SEASONS))[seas.argmax(axis=1)]
    tot = A.sum(axis=1)
    cov["season_share"] = np.where(tot > 0, seas.max(axis=1) / tot, np.nan)

    ev = pd.read_csv(EVENTS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))

    # --- panel 1: how coarse is it, really
    ax = axes[0]
    rng = np.random.default_rng(0)
    ids = np.sort(cov.region_id.unique())
    shuffled = rng.permutation(len(ids))
    lut = dict(zip(ids, shuffled))
    ax.scatter(cov.lon, cov.lat, s=8, marker="s",
               c=[lut[i] for i in cov.region_id], cmap="tab20", linewidths=0)
    ax.scatter(g.loc[~g.covered, "lon"], g.loc[~g.covered, "lat"], s=8, marker="s",
               c="0.88", linewidths=0)
    frame(ax, states, f"Atlas 14 seasonality sub-regions (60-min)\n"
                      f"{len(ids)} distinct curves; grey = no coverage")

    # --- panel 2: how strongly seasonal (share of exceedances in the top season).
    # A cyclic "which month" map was tried here first, but no cyclic colormap
    # separates Jul from Sep legibly at this marker size; strength is both more
    # useful and unambiguous, and panel 3 already carries "which season".
    ax = axes[1]
    sc = ax.scatter(cov.lon, cov.lat, s=8, marker="s", c=100 * cov.season_share,
                    cmap="viridis", vmin=25, vmax=100, linewidths=0)
    ax.scatter(g.loc[~g.covered, "lon"], g.loc[~g.covered, "lat"], s=8, marker="s",
               c="0.88", linewidths=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("% of 60-min exceedances in the leading season", fontsize=8)
    frame(ax, states, "Strength of seasonality\n(25% = aseasonal, 100% = one season only)")

    # --- panel 3: season, with debris flows on top
    ax = axes[2]
    order = list(SEASONS)
    cmap = ListedColormap([SEASON_COLOR[s] for s in order])
    idx = cov.dominant_season.map({s: i for i, s in enumerate(order)}).to_numpy()
    ax.scatter(cov.lon, cov.lat, s=8, marker="s", c=idx, cmap=cmap,
               norm=BoundaryNorm(np.arange(-0.5, 4.5), 4), linewidths=0)
    ax.scatter(g.loc[~g.covered, "lon"], g.loc[~g.covered, "lat"], s=8, marker="s",
               c="0.88", linewidths=0)
    ax.scatter(ev.longitude, ev.latitude, s=9, facecolors="none",
               edgecolors="k", linewidths=0.5, zorder=6)
    frame(ax, states, "Season with most 60-min exceedances,\n"
                      "with compiled debris-flow events (circles)")
    handles = [Line2D([], [], linestyle="none", marker="s", color=SEASON_COLOR[s],
                      markersize=7, label=s) for s in order]
    handles.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                          markeredgecolor="k", markersize=6, label="debris-flow event"))
    handles.append(Line2D([], [], linestyle="none", marker="s", color="0.88",
                          markersize=7, label="no Atlas 14 coverage"))
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=True,
              framealpha=0.9, borderpad=0.4)

    fig.suptitle("NOAA Atlas 14 60-min rainfall seasonality, western US "
                 "(0.5° probe of the PFDS seasonality endpoint)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGURES / "atlas14_seasonality_grid_map.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)

    print(f"\ncovered {len(cov)}/{len(g)} grid points ({100*len(cov)/len(g):.0f}%)")
    print(f"distinct sub-regions: {len(ids)}")
    print("\nsub-regions per Atlas 14 volume:")
    print(cov.groupby('region').region_id.nunique().to_string())
    print("\nmedian sub-region size (0.5-deg cells):",
          int(cov.groupby('region_id').size().median()))
    print("\ndominant season of covered cells:")
    print(cov.dominant_season.value_counts().to_string())


if __name__ == "__main__":
    main()
