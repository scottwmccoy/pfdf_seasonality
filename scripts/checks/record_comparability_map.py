"""Map every debris-flow record, to test whether records are comparable.

A "record" is nominally one mapped debris flow, but the eight point sources do
not mean the same thing by it, and the compiled `location_type` field hides the
difference: seven of the eight are lumped under `observation_point`. The Dolan
inventory alone contributes 2,144 records from a single fire — stream-network
segments — against a literature-derived source averaging 2.8 records per fire.

This figure exists to make that visible and to measure it:

  a  where the records are, over terrain, colored by source
  b-d  the same three fires at matched scale, showing that a "record" is a
       channel segment in one source and a basin-scale point in another
  e  nearest-neighbour spacing by source — the quantitative version of the
     same observation, and the one that does not depend on picking good
     example fires

Terrain is ETOPO 2022 60 arc-second (NOAA NCEI), cached under data/raw/aux/.
StormScape was considered for this map and is the wrong tool: its point overlay
is hardcoded to one style, it carries no state boundaries, and its DEM path
(py3dep, 60 m at coarsest) cannot cover a western-US extent.

  python scripts/checks/record_comparability_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfdf_seasonality.paths import AUX, FIGURES, PROCESSED, REPORTS, STATES_ZIP  # noqa: E402
from pfdf_seasonality.seasons import to_xy  # noqa: E402
from pfdf_seasonality import style  # noqa: E402

style.apply()

DEM = AUX / "etopo2022_60s_west.tif"
ALBERS = "EPSG:5070"
WEST_STATES = ["WA", "OR", "CA", "NV", "ID", "MT", "WY", "UT", "CO", "AZ", "NM"]
REPORT: list[str] = []

# One color per source, Okabe-Ito plus two neutrals; ordered by record count.
SOURCE_COLOR = {
    "dolan2020": "#D55E00", "cavagnaro2025": "#0072B2", "literature": "#009E73",
    "graber2023": "#E69F00", "graber2024": "#CC79A7", "volumes227": "#56B4E9",
    "czu2021": "#000000", "oregon2024": "#8B4513",
}


def say(msg: str = "") -> None:
    print(msg, flush=True)
    REPORT.append(msg)


def hillshade_albers(bounds_ll, out_shape=(1500, 1500)):
    """Read the cached DEM, reproject to Albers, and shade it."""
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    with rasterio.open(DEM) as src:
        tr, w, h = calculate_default_transform(
            src.crs, ALBERS, src.width, src.height, *src.bounds)
        dem = np.empty((h, w), dtype="float32")
        reproject(rasterio.band(src, 1), dem, src_transform=src.transform,
                  src_crs=src.crs, dst_transform=tr, dst_crs=ALBERS,
                  resampling=Resampling.bilinear)
    dem = np.where(dem < -1000, np.nan, dem)          # drop deep ocean
    ls = LightSource(azdeg=315, altdeg=45)
    filled = np.nan_to_num(dem, nan=0.0)
    hs = ls.hillshade(filled, vert_exag=12, dx=tr.a, dy=abs(tr.e))
    hs = np.where(np.isnan(dem), np.nan, hs)
    extent = (tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f)
    return hs, extent


def main() -> None:
    rec = pd.read_csv(PROCESSED / "inventory" / "pfdf_occurrence_records_compiled.csv",
                      low_memory=False)
    rec = rec.dropna(subset=["latitude", "longitude"])
    g = gpd.GeoDataFrame(rec, geometry=gpd.points_from_xy(rec.longitude, rec.latitude),
                         crs="EPSG:4326").to_crs(ALBERS)
    states = gpd.read_file(STATES_ZIP).to_crs(ALBERS)
    west = states[states.STUSPS.isin(WEST_STATES)]

    order = list(rec.source_key.value_counts().index)

    say("=" * 74)
    say("ARE DEBRIS-FLOW RECORDS COMPARABLE ACROSS SOURCES?")
    say("=" * 74)
    say(f"  {len(g):,} located records from {rec.source_key.nunique()} point sources")

    # ------------------------------------------------ nearest-neighbour spacing
    say()
    say("-" * 74)
    say("  NEAREST-NEIGHBOUR SPACING WITHIN A FIRE, BY SOURCE")
    say("-" * 74)
    say("  If records meant the same thing everywhere, spacing would be similar.")
    say(f"    {'source':16s} {'records':>8s} {'fires':>6s} {'median NN':>11s} {'p10':>8s} {'p90':>8s}")
    nn_by_source: dict[str, np.ndarray] = {}
    for src_key in order:
        sub = rec[rec.source_key == src_key]
        d_all = []
        for _, grp in sub.groupby("group_key"):
            if len(grp) < 2:
                continue
            xy = to_xy(grp.latitude.to_numpy(), grp.longitude.to_numpy())
            dist, _ = cKDTree(xy).query(xy, k=2)
            d_all.append(dist[:, 1] * 1000.0)          # km -> m
        if not d_all:
            say(f"    {src_key:16s} {len(sub):8d} {sub.group_key.nunique():6d}"
                f" {'(single-record fires)':>28s}")
            continue
        d = np.concatenate(d_all)
        nn_by_source[src_key] = d
        say(f"    {src_key:16s} {len(sub):8d} {sub.group_key.nunique():6d} "
            f"{np.median(d):10.0f}m {np.percentile(d, 10):7.0f}m "
            f"{np.percentile(d, 90):7.0f}m")

    if "dolan2020" in nn_by_source and "volumes227" in nn_by_source:
        ratio = np.median(nn_by_source["volumes227"]) / np.median(nn_by_source["dolan2020"])
        say()
        say(f"  Dolan's records sit ~{ratio:.0f}x closer together than the deposit")
        say("  inventory's, which is the signature of a channel-segment sampling")
        say("  scheme rather than one-point-per-debris-flow.")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(13.0, 11.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.75, 1.0], hspace=0.30, wspace=0.24)

    # --- (a) overview
    ax = fig.add_subplot(gs[0, :2])
    hs, extent = hillshade_albers(None)
    ax.imshow(hs, cmap="Greys_r", extent=extent, origin="upper",
              vmin=0.05, vmax=1.05, interpolation="bilinear", zorder=0)
    west.boundary.plot(ax=ax, linewidth=0.5, color="0.25", zorder=2)
    for src_key in order:
        sub = g[g.source_key == src_key]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=5, linewidths=0,
                   color=SOURCE_COLOR.get(src_key, "0.4"), alpha=0.85, zorder=3,
                   label=f"{src_key} ({len(sub):,})")
    b = west.total_bounds
    ax.set_xlim(b[0] - 6e4, b[2] + 6e4); ax.set_ylim(b[1] - 6e4, b[3] + 6e4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title("a  every located debris-flow record, by source",
                 fontsize=10, loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower left", frameon=True, framealpha=0.92,
              markerscale=2.2, title="source (records)", title_fontsize=7.5)

    # --- (b-d) matched-scale detail on three sources
    picks = []
    for src_key in ["dolan2020", "graber2024", "volumes227"]:
        sub = rec[rec.source_key == src_key]
        if not len(sub):
            continue
        fire = sub.group_key.value_counts().index[0]
        picks.append((src_key, fire, sub[sub.group_key == fire]))
    HALF_KM = 9.0
    for i, (src_key, fire, sub) in enumerate(picks[:3]):
        ax = fig.add_subplot(gs[1, i])
        cx, cy = sub.longitude.mean(), sub.latitude.mean()
        p = gpd.GeoDataFrame(sub, geometry=gpd.points_from_xy(sub.longitude, sub.latitude),
                             crs="EPSG:4326").to_crs(ALBERS)
        c = gpd.GeoSeries(gpd.points_from_xy([cx], [cy]), crs="EPSG:4326").to_crs(ALBERS)
        x0, y0 = c.geometry.x.iloc[0], c.geometry.y.iloc[0]
        ax.scatter(p.geometry.x, p.geometry.y, s=9, linewidths=0,
                   color=SOURCE_COLOR.get(src_key, "0.4"))
        ax.set_xlim(x0 - HALF_KM * 1000, x0 + HALF_KM * 1000)
        ax.set_ylim(y0 - HALF_KM * 1000, y0 + HALF_KM * 1000)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
        # 5 km scale bar, identical in all three panels
        ax.plot([x0 - 8000, x0 - 3000], [y0 - 7800] * 2, "k-", lw=2.5)
        ax.text(x0 - 5500, y0 - 7200, "5 km", fontsize=7.5, ha="center")
        med = np.median(nn_by_source[src_key]) if src_key in nn_by_source else np.nan
        ax.set_title(f"{'bcd'[i]}  {src_key} — {fire[:20]}\n"
                     f"{len(sub):,} records, median spacing {med:.0f} m",
                     fontsize=8.5, loc="left", fontweight="bold")

    # --- (e) spacing distributions
    ax = fig.add_subplot(gs[0, 2])
    bins = np.logspace(0.5, 4.6, 45)
    for src_key in order:
        if src_key not in nn_by_source:
            continue
        ax.hist(nn_by_source[src_key], bins=bins, histtype="step", linewidth=1.8,
                density=True, color=SOURCE_COLOR.get(src_key, "0.4"), label=src_key)
    ax.set_xscale("log")
    ax.set_xlabel("nearest-neighbour distance\nwithin a fire (m)", fontsize=8.5)
    ax.set_ylabel("density")
    ax.set_title("e  a 'record' is not one thing", fontsize=10, loc="left",
                 fontweight="bold")
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Debris-flow records are not equivalent between sources\n"
                 "spacing within a fire spans two orders of magnitude, from "
                 "channel segments to one point per basin", fontsize=11.5, y=0.96)
    out = FIGURES / "record_comparability_map.png"
    fig.savefig(out, bbox_inches="tight")
    say()
    say(f"wrote {out}")
    (REPORTS / "record_comparability_report.txt").write_text("\n".join(REPORT) + "\n")
    print("wrote", REPORTS / "record_comparability_report.txt")


if __name__ == "__main__":
    main()
