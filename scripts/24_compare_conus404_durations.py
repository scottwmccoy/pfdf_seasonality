"""
Is the native 15-minute CONUS404 output worth acquiring? A bracketing test.

The two products already extracted BRACKET 15 minutes in duration:

  rate    RAINNCVMAX, instantaneous precipitation rate at model timestep
          (duration -> 0), annual max + month, raw/conus404/amax_wy*.npz
  1-hour  PREC_ACC_NC, true 60-minute accumulation, raw/conus404/hourly/

The 15-minute accumulation (auxhist24; monthly tar bundles in the 815 TB NCAR
GDEX archive, no cloud copy) sits strictly between the two. Wherever the two
brackets give the same seasonal answer, the 15-minute answer is pinned; where
they disagree, 15-minute data could genuinely add information. This script
maps that disagreement over the full western box (600 x 585 cells, 45 water
years each), checks whether it touches the places debris flows actually occur,
and brings in the one source of TRUE 15-minute data we already have — the
Atlas 14 station annual-maximum series at 15/30/60 min — to measure real-world
duration sensitivity across exactly the 15-to-60-minute range in question.

It also rebuilds, from each product, the exceedance-count maps of the external
15-minute reference figures (season_exc1yr_total_19802021.png,
season_of_most_exc_wettest_with1yr.png in the project root): threshold = the
46th-largest of the 540 monthly maxima (mean one exceedance per year), count
of exceedances per season, dominant season = argmax.

  /opt/anaconda3/envs/PointMan/bin/python compare_conus404_durations.py

Outputs: compiled/conus404_duration_comparison_report.txt
         compiled/conus404_duration_comparison.png
"""
from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from pfdf_seasonality.paths import CONUS404, PROCESSED, REPORTS, FIGURES, STATES_ZIP, REANALYSIS_15MIN
from pfdf_seasonality.seasons import SEASONS, LAT0, to_xy, circ_diff_days
from pfdf_seasonality import style
from pfdf_seasonality.style import SEASON_COLOR, WEST_EXTENT as EXTENT

style.apply()

C404 = CONUS404
OUT = PROCESSED / "seasonality"
WEST_STATES = ["WA", "OR", "CA", "NV", "ID", "MT", "WY", "UT", "CO", "AZ", "NM"]

SEASON_NAMES = np.array(list(SEASONS))

REPORT = []


def say(msg=""):
    print(msg, flush=True)
    REPORT.append(msg)


def season_fields(frac):
    """frac (12, ny, nx) -> dominant-season index, share margin, circular mean doy."""
    seas = np.stack([frac[[m - 1 for m in ms]].sum(axis=0) for ms in SEASONS.values()])
    tot = frac.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = seas / tot
    dom = np.where(np.isfinite(tot) & (tot > 0), seas.argmax(axis=0), -1).astype("int8")
    srt = np.sort(np.nan_to_num(shares, nan=0.0), axis=0)
    margin = srt[-1] - srt[-2]
    theta = 2 * np.pi * (np.arange(12) + 0.5) / 12
    with np.errstate(invalid="ignore", divide="ignore"):
        p = frac / tot
    C = np.tensordot(np.cos(theta), p, axes=(0, 0))
    S = np.tensordot(np.sin(theta), p, axes=(0, 0))
    doy = (np.arctan2(S, C) % (2 * np.pi)) * 365.25 / (2 * np.pi)
    return dom, margin, doy


# ------------------------------------------------------------------ load rate
def load_rate():
    files = sorted(C404.glob("amax_wy*.npz"))
    months, mstack = [], []
    for f in files:
        d = np.load(f)
        months.append(d["month"])
        mstack.append(d["monthly_max"].astype("float32"))
    month = np.stack(months)                             # (nyr, ny, nx)
    monthly_max = np.stack(mstack)                       # (nyr, 12, ny, nx)
    frac = np.stack([(month == m).mean(axis=0) for m in range(1, 13)]).astype("float32")
    return frac, monthly_max, len(files)


# ---------------------------------------------------------------- load 1-hour
def load_hourly(ny, nx, y0, x0):
    files = sorted((C404 / "hourly").glob("tile_y*_x*.npz"))
    nyr = 45
    frac = np.full((12, ny, nx), np.nan, dtype="float32")
    monthly_max = np.full((nyr, 12, ny, nx), np.nan, dtype="float32")
    for f in files:
        d = np.load(f)
        ty, tx, mon, mmax = int(d["ty"]), int(d["tx"]), d["month"], d["monthly_max"]
        T = mon.shape[1]
        dy0, dx0 = max(ty - y0, 0), max(tx - x0, 0)
        dy1, dx1 = min(ty - y0 + T, ny), min(tx - x0 + T, nx)
        if dy1 <= dy0 or dx1 <= dx0:
            continue
        sy0, sx0 = dy0 - (ty - y0), dx0 - (tx - x0)
        sub = mon[:, sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        for mi, m in enumerate(range(1, 13)):
            frac[mi, dy0:dy1, dx0:dx1] = (sub == m).mean(axis=0)
        monthly_max[:, :, dy0:dy1, dx0:dx1] = \
            mmax[:, :, sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    return frac, monthly_max


# ------------------------------------------- exceedance-count season (1-yr level)
def exceedance_season(monthly_max):
    """Season with most exceedances of the local mean-1-per-year level.

    monthly_max (nyr, 12, ny, nx); threshold = 46th largest of the nyr*12
    monthly maxima (so ~nyr values exceed it strictly); returns dominant-season
    index (-1 where the threshold is zero, i.e. too dry to define a level).
    """
    nyr, _, ny, nx = monthly_max.shape
    flat = monthly_max.reshape(nyr * 12, ny, nx)
    k = nyr + 1                                          # 46th largest
    thr = np.partition(np.nan_to_num(flat, nan=0.0), -k, axis=0)[-k]
    exc = np.nan_to_num(flat, nan=0.0) > thr[None]
    exc = exc.reshape(nyr, 12, ny, nx)
    counts = np.stack([exc[:, [m - 1 for m in ms]].sum(axis=(0, 1))
                       for ms in SEASONS.values()])      # (4, ny, nx)
    dom = counts.argmax(axis=0).astype("int8")
    dom[thr <= 0] = -1
    return dom, (thr <= 0)


def frame(ax, states, title):
    states.boundary.plot(ax=ax, linewidth=0.4, color="0.25", zorder=5)
    ax.set_xlim(*EXTENT[:2]); ax.set_ylim(*EXTENT[2:])
    ax.set_aspect(1 / 0.75)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)


def cat_map(ax, lon, lat, dom, mask, states, title):
    z = np.where(mask & (dom >= 0), dom.astype(float), np.nan)
    cmap = ListedColormap([SEASON_COLOR[s] for s in SEASONS])
    cmap.set_bad("white")
    ax.pcolormesh(lon, lat, z, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
                  shading="nearest", rasterized=True)
    frame(ax, states, title)


def main():
    g = np.load(C404 / "grid.npz")
    lat, lon = g["lat"], g["lon"]
    y0, y1, x0, x1 = [int(v) for v in g["box"]]
    ny, nx = lat.shape

    states = gpd.read_file(STATES_ZIP).to_crs(4326)
    west = states[states.STUSPS.isin(WEST_STATES)]
    try:
        west_geom = west.geometry.union_all()
    except AttributeError:
        west_geom = west.geometry.unary_union
    shapely.prepare(west_geom)
    land = shapely.contains_xy(west_geom, lon.ravel(), lat.ravel()).reshape(lat.shape)

    state_label = np.full(lat.shape, -1, dtype="int8")
    for i, geom in enumerate(west.geometry):
        shapely.prepare(geom)
        inside = shapely.contains_xy(geom, lon.ravel(), lat.ravel()).reshape(lat.shape)
        state_label[inside] = i
    state_names = west.STUSPS.to_numpy()

    say("=" * 74)
    say("CONUS404 DURATION BRACKET: instantaneous rate vs 1-hour accumulation")
    say("=" * 74)

    frac_r, mmax_r, nyr = load_rate()
    say(f"rate    : {nyr} water years, grid {ny} x {nx}")
    frac_h, mmax_h = load_hourly(ny, nx, y0, x0)
    say(f"1-hour  : {np.isfinite(frac_h[0]).mean() * 100:.0f}% of box covered")
    say(f"land mask: {land.sum():,} of {land.size:,} cells inside the 11 western states")

    clim = np.load(C404 / "monthly_climatology.npz")["clim"]

    dom_r, mar_r, doy_r = season_fields(frac_r)
    dom_h, mar_h, doy_h = season_fields(frac_h)
    dom_w, _, _ = season_fields(clim)

    ok = land & (dom_r >= 0) & (dom_h >= 0)
    same = dom_r == dom_h

    say()
    say("-" * 74)
    say("1. SEASON OF THE ANNUAL MAXIMUM (the metric used in the analysis)")
    say("-" * 74)
    say(f"  same dominant season, rate vs 1-hour : {same[ok].mean() * 100:.1f}%  "
        f"(n = {ok.sum():,} land cells)")
    ct = pd.crosstab(pd.Series(SEASON_NAMES[dom_h[ok]], name="hourly"),
                     pd.Series(SEASON_NAMES[dom_r[ok]], name="rate"))
    say(ct.reindex(index=list(SEASONS), columns=list(SEASONS), fill_value=0).to_string())
    say()
    dd = circ_diff_days(doy_r, doy_h)
    say(f"  |circular mean date difference|      : median {np.nanmedian(dd[ok]):.0f} d, "
        f"90th pct {np.nanpercentile(dd[ok], 90):.0f} d")
    say(f"  where they AGREE    median margin (top - 2nd season share, 1-hour): "
        f"{np.nanmedian(mar_h[ok & same]):.2f}")
    say(f"  where they DISAGREE median margin                                : "
        f"{np.nanmedian(mar_h[ok & ~same]):.2f}")
    say(f"  -> disagreement lives in weakly seasonal cells" if
        np.nanmedian(mar_h[ok & ~same]) < 0.5 * np.nanmedian(mar_h[ok & same])
        else "  -> disagreement is NOT confined to weakly seasonal cells")

    say()
    say("  disagreeing land cells by state:")
    dis = ok & ~same
    rows = []
    for i, nm in enumerate(state_names):
        in_st = state_label == i
        n_st = (ok & in_st).sum()
        if n_st:
            rows.append((nm, (dis & in_st).sum() / n_st * 100, n_st))
    for nm, pct, n_st in sorted(rows, key=lambda r: -r[1]):
        say(f"    {nm}  {pct:5.1f}% of {n_st:6,} cells")

    # ------------------------------------------------ exceedance-count basis
    say()
    say("-" * 74)
    say("2. SEASON WITH MOST EXCEEDANCES OF THE 1-YR LEVEL "
        "(the reference maps' metric)")
    say("-" * 74)
    exc_r, dry_r = exceedance_season(mmax_r)
    del mmax_r
    exc_h, dry_h = exceedance_season(mmax_h)
    del mmax_h
    ok_e = land & (exc_r >= 0) & (exc_h >= 0)
    same_e = exc_r == exc_h
    say(f"  same dominant season, rate vs 1-hour : {same_e[ok_e].mean() * 100:.1f}%  "
        f"(n = {ok_e.sum():,}; {int((land & (dry_r | dry_h)).sum()):,} too-dry cells excluded)")
    ct = pd.crosstab(pd.Series(SEASON_NAMES[exc_h[ok_e]], name="hourly"),
                     pd.Series(SEASON_NAMES[exc_r[ok_e]], name="rate"))
    say(ct.reindex(index=list(SEASONS), columns=list(SEASONS), fill_value=0).to_string())

    # ------------------------------------------------ decisive status
    say()
    say("-" * 74)
    say("3. THE DECISIVE MAP (wettest season != most intense season)")
    say("-" * 74)
    dec_r = (dom_r != dom_w)
    dec_h = (dom_h != dom_w)
    say(f"  decisive share of land, rate   : {dec_r[ok].mean() * 100:.1f}%")
    say(f"  decisive share of land, 1-hour : {dec_h[ok].mean() * 100:.1f}%")
    agree_dec = (dec_r == dec_h)
    say(f"  same decisive STATUS           : {agree_dec[ok].mean() * 100:.1f}%")
    say(f"  both decisive AND same intense season: "
        f"{(dec_r & dec_h & same)[ok].mean() * 100:.1f}% of land")

    # ------------------------------------------------ events
    say()
    say("-" * 74)
    say("4. DOES ANY OF THIS TOUCH THE DEBRIS FLOWS?")
    say("-" * 74)
    ev_r = pd.read_csv(OUT / "pfdf_events_conus404_seasonality.csv")
    ev_h = pd.read_csv(OUT / "pfdf_events_conus404_seasonality_hourly.csv")
    ev = ev_r[["event_id", "latitude", "longitude", "df_season",
               "c404_int_season", "c404_wet_season"]].merge(
        ev_h[["event_id", "c404_int_season", "c404_wet_season"]],
        on="event_id", suffixes=("_rate", "_hr"))
    n = len(ev)
    same_ev = ev.c404_int_season_rate == ev.c404_int_season_hr
    say(f"  events with both products            : {n}")
    say(f"  same intense season at the event     : {same_ev.mean() * 100:.1f}%  "
        f"({(~same_ev).sum()} differ)")
    dec_ev_r = ev.c404_int_season_rate != ev.c404_wet_season_rate
    dec_ev_h = ev.c404_int_season_hr != ev.c404_wet_season_hr
    say(f"  decisive under rate / 1-hour / both  : {dec_ev_r.sum()} / "
        f"{dec_ev_h.sum()} / {(dec_ev_r & dec_ev_h).sum()}")
    hit_r = (ev.df_season == ev.c404_int_season_rate)
    hit_h = (ev.df_season == ev.c404_int_season_hr)
    say(f"  flow in intense season (decisive set): rate {hit_r[dec_ev_r].mean() * 100:.1f}%  "
        f"1-hour {hit_h[dec_ev_h].mean() * 100:.1f}%")
    tree = cKDTree(to_xy(lat.ravel(), lon.ravel()))
    _, idx = tree.query(to_xy(ev.latitude.to_numpy(), ev.longitude.to_numpy()))
    ev_dis = ~same.ravel()[idx]
    say(f"  events sitting in bracket-DISAGREEMENT cells: {ev_dis.sum()} of {n} "
        f"({ev_dis.mean() * 100:.0f}%)")

    say()
    say("  headline numbers, quoted from the two analysis reports:")
    say("                                       rate        1-hour")
    say("    M2 - M1 log-likelihood            +140.8       +84.7")
    say("    decisive subset                 123 events   103 events")
    say("      flow in intense season           60.2%        57.3%")
    say("      flow in wettest season           12.2%        11.7%")
    say("    regime mean dates              29 Dec / 31 Jul   24 Dec / 31 Jul")
    say("    agreement with 15-min gauges      83.7%        80.3%")
    say("    agreement with 30-min gauges      86.7%        82.8%")
    say("    agreement with 60-min gauges      83.4%        83.2%")

    # ------------------------------------------------ gauge truth
    say()
    say("-" * 74)
    say("5. GROUND TRUTH: real 15- vs 30- vs 60-min duration sensitivity (gauges)")
    say("-" * 74)
    st = pd.read_csv(OUT / "atlas14_station_seasonality.csv")
    st = st[st.western_us & ~st.short_record]
    mf = [f"month_frac_{m:02d}" for m in range(1, 13)]
    pieces = {d: st[st.duration == d].set_index("station_id") for d in ["15m", "30m", "60m"]}
    gauge_pairs = []
    for a, b in [("15m", "30m"), ("30m", "60m"), ("15m", "60m")]:
        j = pieces[a].join(pieces[b], lsuffix="_a", rsuffix="_b", how="inner")
        same_g = (j.dominant_season_a == j.dominant_season_b)
        tvd = 0.5 * np.abs(j[[c + "_a" for c in mf]].to_numpy()
                           - j[[c + "_b" for c in mf]].to_numpy()).sum(axis=1)
        dgap = circ_diff_days(j.circular_mean_doy_a.to_numpy(),
                              j.circular_mean_doy_b.to_numpy())
        say(f"  {a} vs {b}: same dominant season {same_g.mean() * 100:.1f}%  "
            f"(n={len(j)});  median TVD {np.median(tvd):.2f};  "
            f"median |mean date diff| {np.median(dgap):.0f} d")
        gauge_pairs.append((f"gauges\n{a} vs {b}", same_g.mean() * 100, len(j)))

    say()
    say("  For contrast, the model bracket (rate vs 1-hour) spans a far wider")
    say("  duration range than 15-to-60 min, so gauge 15m-vs-60m agreement is an")
    say("  UPPER bound on how much a 15-min model product could differ from the")
    say("  1-hour product we already have.")

    # ------------------------------------------------ figure
    ev_lat, ev_lon = ev.latitude.to_numpy(), ev.longitude.to_numpy()
    fig = plt.figure(figsize=(13.6, 13.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.95], hspace=0.22, wspace=0.16)

    ax = fig.add_subplot(gs[0, 0])
    cat_map(ax, lon, lat, dom_r, land, states,
            "a  season of annual max — instantaneous rate")
    ax = fig.add_subplot(gs[0, 1])
    cat_map(ax, lon, lat, dom_h, land, states,
            "b  season of annual max — 1-hour accumulation")

    ax = fig.add_subplot(gs[0, 2])
    z = np.where(ok, (~same).astype(float), np.nan)
    cmap = ListedColormap(["#DDD8CE", "#D55E00"]); cmap.set_bad("white")
    ax.pcolormesh(lon, lat, z, cmap=cmap, vmin=0, vmax=1, shading="nearest",
                  rasterized=True)
    ax.scatter(ev_lon, ev_lat, s=8, facecolors="none", edgecolors="k",
               linewidths=0.45, zorder=6)
    frame(ax, states, f"c  bracket disagrees ({(~same)[ok].mean() * 100:.0f}% of land) "
                      f"— 15-min could differ here")
    ax.legend(handles=[
        Line2D([], [], marker="s", linestyle="none", color="#D55E00", markersize=7,
               label="rate and 1-hour differ"),
        Line2D([], [], marker="s", linestyle="none", color="#DDD8CE", markersize=7,
               label="agree (15-min pinned)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="k", markersize=6, label="debris flow")],
        loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)

    ax = fig.add_subplot(gs[1, 0])
    cat_map(ax, lon, lat, exc_r, land, states,
            "d  most exceedances of 1-yr level — rate")
    ax = fig.add_subplot(gs[1, 1])
    cat_map(ax, lon, lat, exc_h, land, states,
            "e  most exceedances of 1-yr level — 1-hour")

    ax = fig.add_subplot(gs[1, 2])
    ref = plt.imread(REANALYSIS_15MIN / "season_of_most_exc_wettest_with1yr.png")
    h, w = ref.shape[:2]
    ax.imshow(ref[int(0.01 * h):int(0.50 * h), int(0.50 * w):])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("f  reference: 15-min product (most exc of 1-yr I$_{15}$)",
                 fontsize=9, loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[2, 0])
    z = np.full(lat.shape, np.nan)
    z[ok & ~dec_r & ~dec_h] = 0
    z[ok & dec_r & dec_h] = 1
    z[ok & (dec_r != dec_h)] = 2
    cmap = ListedColormap(["#DDD8CE", "#D55E00", "#0072B2"]); cmap.set_bad("white")
    ax.pcolormesh(lon, lat, z, cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest",
                  rasterized=True)
    ax.scatter(ev_lon, ev_lat, s=8, facecolors="none", edgecolors="k",
               linewidths=0.45, zorder=6)
    frame(ax, states, "g  wettest $\\neq$ intense: do the products agree?")
    ax.legend(handles=[
        Line2D([], [], marker="s", linestyle="none", color="#D55E00", markersize=7,
               label="both say decisive"),
        Line2D([], [], marker="s", linestyle="none", color="#DDD8CE", markersize=7,
               label="both say not"),
        Line2D([], [], marker="s", linestyle="none", color="#0072B2", markersize=7,
               label="status differs")],
        loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)

    ax = fig.add_subplot(gs[2, 1])
    bins = np.linspace(0, 1, 41)
    ax.hist(mar_h[ok & same], bins=bins, density=True, alpha=0.75,
            color="#009E73", label=f"agree (n={(ok & same).sum():,})")
    ax.hist(mar_h[ok & ~same], bins=bins, density=True, alpha=0.75,
            color="#D55E00", label=f"disagree (n={(ok & ~same).sum():,})")
    ax.set_xlabel("seasonal margin: top share $-$ 2nd share (1-hour)")
    ax.set_ylabel("density")
    ax.set_title("h  disagreement lives where seasonality is weak",
                 fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False)

    ax = fig.add_subplot(gs[2, 2])
    bars = gauge_pairs + [("model\nrate vs 1-hr", same[ok].mean() * 100, int(ok.sum()))]
    xx = np.arange(len(bars))
    cols = ["#0072B2"] * len(gauge_pairs) + ["#D55E00"]
    ax.bar(xx, [b[1] for b in bars], 0.6, color=cols)
    for i, (lab, v, nn) in enumerate(bars):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
        ax.text(i, 6, f"n={nn:,}", ha="center", fontsize=7, color="w", rotation=90)
    ax.set_xticks(xx); ax.set_xticklabels([b[0] for b in bars], fontsize=7.5)
    ax.set_ylim(0, 100); ax.set_ylabel("% same dominant season")
    ax.set_title("i  real duration sensitivity (gauges) vs the bracket",
                 fontsize=9, loc="left", fontweight="bold")

    handles = [Line2D([], [], marker="s", linestyle="none", color=SEASON_COLOR[s],
                      markersize=8, label=s) for s in SEASONS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.925), fontsize=9)
    fig.suptitle("Does 15-minute CONUS404 data have room to change the answer?\n"
                 "instantaneous rate and 1-hour accumulation bracket the 15-min "
                 "duration; water years 1980–2024", fontsize=11, y=1.0)
    out = FIGURES / "conus404_duration_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    say()
    say(f"wrote {out}")

    (REPORTS / "conus404_duration_comparison_report.txt").write_text("\n".join(REPORT) + "\n")
    print("wrote", REPORTS / "conus404_duration_comparison_report.txt")


if __name__ == "__main__":
    main()
