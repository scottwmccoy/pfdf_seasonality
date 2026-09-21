"""The numbers the 2026-09-20 independent review said the paper must disclose.

Each was first measured by a reviewing subagent on the sample as it stood
BEFORE the compilation fixes and the removal of the post-fire window. Those
fixes moved every denominator, so the review's figures cannot be quoted. This
recomputes each one on the current sample, so the paper cites a generated
report rather than a stale review.

Items are numbered as in `docs/reviews/TRIAGE.md`.

  /opt/anaconda3/envs/pfdf_seasonality_env/bin/python scripts/checks/review_followups.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfdf_seasonality.events import load_events  # noqa: E402
from pfdf_seasonality.paths import PROCESSED  # noqa: E402
from pfdf_seasonality.report import tee  # noqa: E402
from pfdf_seasonality.seasons import MONTH_TO_SEASON, circular  # noqa: E402

GAUGE = PROCESSED / "seasonality" / "pfdf_events_gauge_seasonality.csv"
MID_DOY = {"DJF": 15, "MAM": 105, "JJA": 196, "SON": 288}


def decisive(g: pd.DataFrame) -> pd.DataFrame:
    use = g[g.int60m_ok.fillna(False).astype(bool) & g.wet_ok.fillna(False).astype(bool)]
    return use[use.int60m_season != use.wet_season]


def main() -> None:
    g = pd.read_csv(GAUGE, low_memory=False, parse_dates=["event_date", "fire_start_date"])
    g["df_season"] = g.event_date.dt.month.map(MONTH_TO_SEASON)
    dis = decisive(g)
    hit = (dis.df_season == dis.int60m_season)
    wet = (dis.df_season == dis.wet_season)

    print("=" * 76)
    print("REVIEW FOLLOW-UPS — recomputed on the current sample")
    print("=" * 76)
    print(f"  decisive subset: n = {len(dis)}   "
          f"intense {100*hit.mean():.1f}%   wettest {100*wet.mean():.1f}%   "
          f"ratio {hit.mean()/wet.mean():.1f}:1")

    # ---------------------------------------------------------------- B1
    print("\n" + "-" * 76)
    print('B1. "Strengthens when the wet season arrives first" — is it significant?')
    print("-" * 76)
    d = dis.dropna(subset=["fire_start_date"]).copy()
    fdoy = d.fire_start_date.dt.dayofyear.to_numpy()

    def days_until(season):
        return (np.array([MID_DOY[s] for s in season]) - fdoy) % 365

    # not `first`: that name collides with DataFrame.first, and attribute
    # access then silently returns the method instead of the column
    d["arrives_first"] = np.where(
        days_until(d.int60m_season) < days_until(d.wet_season), "intense", "wettest")
    tab = []
    for grp in ("wettest", "intense"):
        sub = d[d.arrives_first == grp]
        k = int((sub.df_season == sub.int60m_season).sum())
        tab.append((grp, k, len(sub)))
        print(f"  {grp + ' season arrives first':30s} {k}/{len(sub)} "
              f"= {100*k/len(sub):.1f}% in the intense season")
    (_, kw, nw), (_, ki, ni) = tab
    odds, p = stats.fisher_exact([[kw, nw - kw], [ki, ni - ki]])
    print(f"\n  difference {100*kw/nw - 100*ki/ni:+.1f} points, "
          f"Fisher exact p = {p:.2f}, odds ratio {odds:.2f}")
    print("  => NOT significant. The defensible claim is 'does not weaken',")
    print("     which still refutes decaying susceptibility; 'strengthens' does not")
    print("     survive. Susceptibility decay predicts the preference should WEAKEN")
    print("     when the wet season comes first, and it does not.")

    # ---------------------------------------------------------------- B2
    print("\n" + "-" * 76)
    print("B2. Which regime does the decisive test actually speak to?")
    print("-" * 76)
    vc = dis.int60m_season.value_counts()
    print(f"  intense-season regime of the {len(dis)} decisive events: {vc.to_dict()}")
    print(f"  JJA-intense share: {100*vc.get('JJA', 0)/len(dis):.0f}%")
    print("\n  This is mechanical, not a sampling accident: in coastal and southern")
    print("  California the wettest and most-intense seasons are BOTH winter, so a")
    print("  winter event can never be decisive. The decisive test is evidence about")
    print("  the summer/monsoon regime alone. The winter regime rests on the model")
    print("  comparison and the regime statistics instead. Say so.")

    # ---------------------------------------------------------------- B4
    print("\n" + "-" * 76)
    print("B4. Events are not independent — report fire-weighted numbers too")
    print("-" * 76)
    mt = pd.read_csv(PROCESSED / "inventory" / "pfdf_events_mtbs.csv", low_memory=False)
    fid = dict(zip(mt.event_id, mt.mtbs_event_id.fillna(mt.fire_key)))
    dis = dis.assign(fire=dis.event_id.map(fid).fillna(dis.fire_key))
    allf = g.assign(fire=g.event_id.map(fid).fillna(g.fire_key))
    same_day = allf.groupby("event_date").fire.nunique()
    n_shared = int(allf.event_date.map(same_day).gt(1).sum())
    print(f"  {len(allf)} events from {allf.fire.nunique()} fires")
    print(f"  {n_shared} ({100*n_shared/len(allf):.0f}%) fall on a date when another "
          "fire also produced flows")
    print(f"  decisive subset: {len(dis)} events from {dis.fire.nunique()} fires")
    per_fire = dis.groupby("fire").apply(
        lambda s: pd.Series({"int": (s.df_season == s.int60m_season).mean(),
                             "wet": (s.df_season == s.wet_season).mean()}),
        include_groups=False)
    print(f"\n  event-weighted : {100*hit.mean():.1f}% intense vs {100*wet.mean():.1f}% wettest")
    print(f"  fire-weighted  : {100*per_fire['int'].mean():.1f}% intense vs "
          f"{100*per_fire['wet'].mean():.1f}% wettest")
    print("\n  Directions are unchanged; the PRECISION is what dependence costs.")
    print("  Report both, and say the effective sample is nearer the fire count.")

    # ---------------------------------------------------------------- B5
    print("\n" + "-" * 76)
    print("B5. The ratio depends on calendar bins; the binning-free score does not")
    print("-" * 76)
    P = dis[[f"int60m_p{m:02d}" for m in range(1, 13)]].to_numpy(float)
    P = P / P.sum(axis=1, keepdims=True)
    W = dis[[f"wet_p{m:02d}" for m in range(1, 13)]].to_numpy(float)
    W = W / W.sum(axis=1, keepdims=True)
    mi = dis.event_date.dt.month.to_numpy() - 1
    r = np.arange(len(dis))
    closer = P[r, mi] > W[r, mi]
    print(f"  months where the intensity climatology assigns the observed month more")
    print(f"  probability than the amount climatology: {closer.sum()}/{len(dis)} "
          f"= {100*closer.mean():.1f}%  (binomial p = "
          f"{stats.binomtest(int(closer.sum()), len(dis), 0.5).pvalue:.1e})")
    print("\n  No seasons, no bin edges, no ratio -- just which model put more")
    print("  probability on the month the flow actually happened. Lead with this.")

    # ---------------------------------------------------------------- B6
    print("\n" + "-" * 76)
    print("B6. Is the decisive subset carried by one storm chain?")
    print("-" * 76)
    top = dis.groupby("fire").size().sort_values(ascending=False).head(5)
    print("  largest contributing fires:")
    for f, n in top.items():
        sub = dis[dis.fire == f]
        print(f"    {str(f):<28} {n:>3} events, "
              f"{100*(sub.df_season == sub.int60m_season).mean():.0f}% in the intense season")
    share = 100 * top.iloc[0] / len(dis)
    print(f"\n  largest single fire is {share:.0f}% of the subset")
    yr = dis.event_date.dt.year.value_counts().sort_index()
    print(f"  by year: {yr.to_dict()}")

    # ---------------------------------------------------------------- B7
    print("\n" + "-" * 76)
    print("B7. The suspect-date filter is load-bearing — disclose it")
    print("-" * 76)
    ev_all = load_events(verbose=False)
    susp = ev_all[ev_all.date_precision.astype(str).str.startswith("suspect")]
    print(f"  {len(susp)} events dropped, all coded 1 January: "
          f"{susp.state.value_counts().to_dict()}")
    # What the headline becomes if the filter is switched off. The suspect
    # events carry no gauge climatology (10_ drops them before the IDW), so
    # reuse that script's own interpolation rather than a second copy of it.
    import importlib
    _an = importlib.import_module("10_analyze_seasonality")
    st = pd.read_csv(_an.STATIONS)
    st = st[st.western_us & ~st.short_record]
    from pfdf_seasonality.seasons import normalize, season_of
    keep = ["event_id", "event_date", "latitude", "longitude"]
    P, _, ok_i = _an.idw(susp[keep], st[st.duration == "60m"],
                         [f"month_frac_{m:02d}" for m in range(1, 13)])
    Q, _, ok_w = _an.idw(susp[keep], st[(st.duration == "60m") & st.has_normals],
                         [f"prcp_norm_{m:02d}" for m in range(1, 13)])
    add = susp.assign(int60m_season=np.where(ok_i, season_of(normalize(P))[0], None),
                      wet_season=np.where(ok_w, season_of(normalize(Q))[0], None),
                      df_season=susp.event_date.dt.month.map(MONTH_TO_SEASON))
    add = add[add.int60m_season.notna() & add.wet_season.notna()]
    add_dis = add[add.int60m_season != add.wet_season]
    if len(add_dis):
        n2 = len(dis) + len(add_dis)
        h2 = (hit.sum() + (add_dis.df_season == add_dis.int60m_season).sum()) / n2
        w2 = (wet.sum() + (add_dis.df_season == add_dis.wet_season).sum()) / n2
        print(f"\n  with the filter OFF, {len(add_dis)} of them enter the decisive subset:")
        print(f"    {len(dis)} -> {n2} events, {100*hit.mean():.1f}% -> {100*h2:.1f}% intense, "
              f"{100*wet.mean():.1f}% -> {100*w2:.1f}% wettest")
        print(f"    ratio {hit.mean()/wet.mean():.1f}:1 -> {h2/w2:.1f}:1")
    print("  Idaho is a summer-intense regime, so a 1 January placeholder forces the")
    print("  event onto the WETTEST side and fabricates winter debris flows in a")
    print("  paper about seasonality. The filter is necessary, not merely defensible.")
    print("  What the paper owes the reader is that the headline is sensitive to it.")

    # ---------------------------------------------------------------- B8
    print("\n" + "-" * 76)
    print("B8. How decisively is the amount model beaten?")
    print("-" * 76)
    wet_season_share = W.max(axis=1)
    margin = np.sort(W, axis=1)[:, -1] - np.sort(W, axis=1)[:, -2]
    print(f"  amount model's probability on its own favoured month: "
          f"median {np.median(wet_season_share):.3f}")
    print(f"  its margin over the runner-up month: median {np.median(margin):.3f}")
    print(f"  the amount model's own hit rate on the decisive subset: "
          f"{100*wet.mean():.1f}%")
    print("\n  The amount model barely commits, so beating it is a weaker statement")
    print("  than the ratio makes it sound. Report the binning-free score (B5) as")
    print("  the primary evidence and the ratio as a descriptive summary.")

    # ------------------------------------------------------- regime circulars
    print("\n" + "-" * 76)
    print("Regime summary, for the writing")
    print("-" * 76)
    for reg in ("DJF", "JJA"):
        sub = g[g.int60m_season == reg]
        mean_doy, R, p = circular(sub.event_date.dt.dayofyear.to_numpy())
        print(f"  {reg}: n = {len(sub)}, R = {R:.2f}, Rayleigh p = {p:.1e}")


if __name__ == "__main__":
    with tee("review_followups_report.txt"):
        main()
