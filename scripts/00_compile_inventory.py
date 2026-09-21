"""
Compile a de-duplicated post-fire debris-flow (PFDF) occurrence dataset for the western US.

Required output fields: debris-flow location (lat/lon) + date of occurrence.
Everything else is carried through as provenance / optional attributes.

Sources
-------
Core (user specified):
  cavagnaro2025  Cavagnaro et al. (2025) JGR-ES 10.1029/2024JF007781
                 data: figshare 10.6084/m9.figshare.25569603
  oakley2025     Oakley et al. (2025) IJWF 10.1071/WF25136, Supplementary Material S1

Added from literature/data-repository search:
  graber2023     USGS 10.5066/P98Q4CDH  (17 fires, AZ CA CO NM WA)
  graber2024     USGS 10.5066/P13ZFKJS  (18 fires, AZ CA CO NM UT WA)
  volumes227     USGS 10.5066/P13EZSWW  (227 DF volumes, 34 fires, 6 states; pub. 2025-11)
  czu2021        USGS 10.5066/P91O03Y7  (CZU/River/Camel/Dolan, Jan-2021 AR sequence)
  dixie2023      USGS 10.5066/P9YPX1BM  (2021 Dixie Fire)
  dolan2020      USGS 10.5066/P13ZGR6F  (2020 Dolan Fire, segment-scale)
  oregon2024     USGS 10.5066/P13TPP8J  (W. Cascade Range OR; burned + unburned,
                 we keep only the 28 field visits recording a debris-flow response)
  literature     USGS 10.5066/P13STASQ  (McGuire et al. 2024, literature-derived)

Full citations, verified against each DOI's ScienceBase landing page, are in
SOURCE_META below. Do not take them from a citing paper's reference list.

Run:
  /opt/anaconda3/envs/PointMan/bin/python compile_pfdf_occurrence.py
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pfdf_seasonality.paths import INVENTORIES, PROCESSED, REPORTS, STATES_ZIP

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RAW = INVENTORIES
OUT = PROCESSED / "inventory"
OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

# Western US bounding box used to keep the compilation regional.
WEST_LON = (-126.0, -101.0)
WEST_LAT = (30.0, 50.0)

# Source priority when collapsing duplicate records: lower number wins.
SOURCE_PRIORITY = {
    "cavagnaro2025": 1,
    "graber2024": 2,
    "graber2023": 3,
    "volumes227": 4,
    "czu2021": 5,
    "dixie2023": 6,
    "dolan2020": 7,
    "oregon2024": 8,
    "literature": 9,
    "oakley2025": 10,  # event-level only; never a point record
}

SOURCE_META = {
    "cavagnaro2025": dict(
        citation="Cavagnaro, D.B., McCoy, S.W., Lindsay, D.N., McGuire, L.A., Kean, J.W., "
        "Trugman, D.T., 2025, JGR Earth Surface 130, e2024JF007781",
        doi="10.1029/2024JF007781",
        data_doi="10.6084/m9.figshare.25569603",
    ),
    "oakley2025": dict(
        citation="Oakley, N.S., Cheung, D.J., Lindsay, D.N., Nash, D., 2025, "
        "Int. J. Wildland Fire 34, WF25136",
        doi="10.1071/WF25136",
        data_doi="10.1071/WF25136 (Supplementary Material S1)",
    ),
    "graber2023": dict(
        citation="Graber, A.P., 2023, Compilation of runoff-generated debris-flow inventories "
        "for 17 fires across AZ, CA, CO, NM, and WA, USA: USGS data release",
        doi="10.5066/P98Q4CDH",
        data_doi="10.5066/P98Q4CDH",
    ),
    "graber2024": dict(
        citation="Graber, A.P., 2024, Compilation of runoff-generated debris-flow inventories "
        "for 18 fires across AZ, CA, CO, NM, UT, and WA, USA: USGS data release",
        doi="10.5066/P13ZFKJS",
        data_doi="10.5066/P13ZFKJS",
    ),
    # Citations verified 2026-09-20 against the authoritative ScienceBase landing
    # page for each DOI (doi.org -> sciencebase.gov/catalog/item/...). An earlier
    # pass took them from citing papers' reference lists and from a README shipped
    # with the data; both proved unreliable in detail. Reference lists abbreviate
    # (the CZU release lost three of its eleven authors that way, including
    # McCoy), and the Oregon README's title differs from the published one.
    # Use the landing page, not a secondary citation.
    "volumes227": dict(
        citation="Gorr, A.N., Rengers, F.K., Barnhart, K.R., Thomas, M.A., Kean, J.W., "
        "and Crowder, C.A., 2025, Inventory of 227 postfire debris-flow volumes for 34 fires "
        "in the western United States: USGS data release",
        doi="10.5066/P13EZSWW",
        data_doi="10.5066/P13EZSWW",
    ),
    "czu2021": dict(
        # 11 authors per ScienceBase; a citing paper's reference list showed only 8,
        # dropping Kean, McCoy and Cavagnaro.
        citation="Thomas, M.A., Kean, J.W., McCoy, S.W., Lindsay, D.N., Kostelnik, J., "
        "Cavagnaro, D.B., Rengers, F.K., East, A.E., Schwartz, J.Y., Smith, D., and "
        "Collins, B.D., 2023, Field-verified inventory of postfire hydrologic response "
        "for the 2020 CZU Lightning Complex, River, Camel, and Dolan Fires following a "
        "26-29 January 2021 atmospheric river storm sequence: USGS data release",
        doi="10.5066/P91O03Y7",
        data_doi="10.5066/P91O03Y7",
    ),
    "dixie2023": dict(
        citation="Thomas, M.A., Lindsay, D.N., Cavagnaro, D.B., Kean, J.W., McCoy, S.W., "
        "and Graber, A.P., 2023, Field-verified inventory of postfire debris flows for the "
        "2021 Dixie Fire following a 23-25 October 2021 atmospheric river storm and "
        "12 June 2022 thunderstorm: USGS data release",
        doi="10.5066/P9YPX1BM",
        data_doi="10.5066/P9YPX1BM",
    ),
    "dolan2020": dict(
        citation="Cavagnaro, D.B., McCoy, S.W., Thomas, M.A., Kostelnik, J., and "
        "Lindsay, D.N., 2025, Inventory of fluvial erosion and debris-flow activity "
        "following the 2020 Dolan Fire, California: USGS data release",
        doi="10.5066/P13ZGR6F",
        data_doi="10.5066/P13ZGR6F",
    ),
    "oregon2024": dict(
        # "Cascade Range", not "Cascades" (the shipped README says the latter), and
        # the authors carry middle initials the README omits.
        citation="Selander, B.D., Calhoun, N.C., Burns, W.J., Rengers, F.K., Kean, J.W., "
        "Moffett, K.B., Patton, A.I., Quinn, D.S., and Roering, J.J., 2024, Inventory of "
        "debris flows in burned (2020-2022) and unburned (1995-2020) areas in the western "
        "Cascade Range of Oregon: USGS data release",
        doi="10.5066/P13TPP8J",
        data_doi="10.5066/P13TPP8J",
    ),
    "literature": dict(
        citation="McGuire, L.A., Ebel, B.A., Rengers, F.K., Vieira, D.C.S., and Nyman, P., 2024, "
        "Postfire Debris-Flow Database (Literature Derived): USGS data release",
        doi="10.5066/P13STASQ",
        data_doi="10.5066/P13STASQ",
    ),
}

STD_COLS = [
    "record_id",
    "source_key",
    "fire_name",
    "fire_name_norm",
    "group_key",
    "fire_year",
    "fire_start_date",
    "state",
    "latitude",
    "longitude",
    "location_type",
    "location_quality",
    "event_date",
    "date_precision",
    "date_basis",
    "observation_date",
    "date_reference",
    "n_flows",
    "initiation_type",
    "df_evidence",
    "peak_i15_mmh",
    "peak_i30_mmh",
    "peak_i60_mmh",
    "volume_m3",
    "source_record_id",
    "original_source",
    "notes",
]

# location_quality:
#   point       - a mapped debris-flow location (initiation point, outlet, or deposit)
#   general     - somewhere in the burned area / study area, km-scale uncertainty
#   fire_scale  - fire-perimeter centroid; represents the fire, not the debris flow

# date_basis - WHAT `event_date` ACTUALLY MEANS. Added 2026-09-20 after the Dolan
# inventory was found to be dated by survey rather than by storm, which split one
# atmospheric river into four spurious events. `event_date` is always the best
# available estimate of WHEN THE DEBRIS FLOW OCCURRED; `observation_date` keeps the
# raw date the source reported when the two differ.
#   storm                - the source gives the triggering storm's date
#   debris_flow          - the source gives the flow's own date
#   observation          - the source gives only a survey / imagery date, used as a
#                          proxy. NOT a date of occurrence; suspect for timing work
#   assigned_literature  - we set the date from a cited publication; see
#                          `date_reference`
DATE_BASIS = {
    "cavagnaro2025": "storm",           # StormDate
    "graber2023": "storm",              # StormStart, falls back to ObservationDate
    "graber2024": "storm",              # StormStart, falls back to ObservationDate
    "volumes227": "debris_flow",        # DebrisFlowDate
    "czu2021": "storm",                 # StormStart
    "dixie2023": "storm",               # StormStart
    "dolan2020": "assigned_literature",  # see DOLAN_STORM_DATE below
    "oregon2024": "observation",        # Observation_Date; no storm date published
    "literature": "debris_flow",        # DateOfFlow, else ApproxDateOfFlow
    "oakley2025": "storm",              # EventDate
}

# The Dolan release records ObservationDate, defined in its README as the Google
# Earth image-acquisition date for remotely mapped segments and the field-visit
# date otherwise. Treating those as dates of occurrence split one storm into four
# events (2021-02-23, 2021-02-26, 2021-07-01, 2022-01-15). The triggering storm is
# a single, well-documented atmospheric river.
DOLAN_STORM_DATE = "2021-01-27"
DOLAN_DATE_REFERENCE = (
    "Cavagnaro, D.B., McCoy, S.W., Thomas, M.A., Kostelnik, J., and Lindsay, D.N., "
    "2025, Improved prediction of postfire debris flows through rainfall anomaly "
    "maps: Geophysical Research Letters 52, e2025GL114791, doi:10.1029/2025GL114791 "
    '("On 27 January 2021, the area received intense rainfall ... triggering '
    'numerous debris flows and floods across the burn area")'
)

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

_FIRE_DROP = re.compile(
    r"\b(fire|fires|complex|lightning|wildfire|incident|the)\b", flags=re.I
)

# Fires that are named differently between sources. Maps normalized -> canonical.
def _basic_norm(name) -> str:
    """Drop-and-strip only, with no alias lookup, so it can build the table."""
    if pd.isna(name):
        return ""
    return re.sub(r"[^a-z0-9]+", "", _FIRE_DROP.sub(" ", str(name)).lower())


#: Fires named differently between sources, written as the names ACTUALLY
#: appear in the data. Both sides go through `_basic_norm`, so an alias can
#: never be written in a form the normalizer is unable to produce.
#:
#: That was the defect here until 2026-09-20: the table was written against
#: raw strings, but `_FIRE_DROP` strips "complex" and "lightning" BEFORE the
#: lookup, and the keys used a word order the sources do not use. So
#: `sangabrielcomplexfish` was unreachable while the data actually produced
#: `fishsangabriel` from "Fish (San Gabriel Complex)" and `fish` from Oakley —
#: one fire under two keys. Likewise "CZU" reached `czulightningcomplex`, a
#: string nothing else can produce, while Oakley's "CZU August Lightning
#: Complex" reached `czuaugust`.
#:
#: Canonical forms are therefore the full official names, which are reachable
#: from the data without an alias at all.
_FIRE_ALIAS_NAMES = {
    # 2003 Grand Prix/Old burned as one area and are reported jointly
    "Grand Prix-Old": "Grand Prix",
    "Old": "Grand Prix",
    # point sources abbreviate; Oakley gives the full incident name
    "CZU": "CZU August Lightning Complex",
    # point sources qualify Fish with its complex; Oakley does not
    "Fish (San Gabriel Complex)": "Fish",
    "San Gabriel Complex": "Fish",
    "Whitewater-Baldy Complex": "Whitewater-Baldy",
    "Twentyfive Mile": "25 Mile",
    "Cub Creek 2": "Cub Creek",
    "Beachie Creek-Lionshead": "Beachie Creek",
    "Inyo Complex": "Inyo",
}


def _build_aliases(names: dict[str, str]) -> dict[str, str]:
    """Normalize both sides, dropping no-ops and refusing to hide a mistake."""
    out: dict[str, str] = {}
    for raw_k, raw_v in names.items():
        k, v = _basic_norm(raw_k), _basic_norm(raw_v)
        if k == v:
            continue  # the drop rules already collapse this one
        if out.get(k, v) != v:
            raise ValueError(f"fire alias {raw_k!r} normalizes to {k!r}, which is "
                             f"already mapped to {out[k]!r}, not {v!r}")
        out[k] = v
    chained = {k for k, v in out.items() if v in out}
    if chained:
        raise ValueError(f"fire aliases chain, which is not resolved: {sorted(chained)}")
    return out


FIRE_ALIASES = _build_aliases(_FIRE_ALIAS_NAMES)


def norm_fire(name) -> str:
    s = _basic_norm(name)
    return FIRE_ALIASES.get(s, s)


def parse_date(series) -> pd.Series:
    """Robustly parse dates that may be Timestamps, 'M/D/YYYY' strings, or yyyymmdd ints."""

    def one(v):
        if pd.isna(v):
            return pd.NaT
        if isinstance(v, (pd.Timestamp,)):
            return pd.Timestamp(v).normalize()
        s = str(v).strip()
        if s in ("", "-9999", "nan", "NaT", "None"):
            return pd.NaT
        if re.fullmatch(r"\d{8}", s):
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        if re.fullmatch(r"\d{8}\.0", s):
            return pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce")
        out = pd.to_datetime(s, errors="coerce")
        return pd.NaT if pd.isna(out) else pd.Timestamp(out).normalize()

    return series.map(one)


def clean_num(series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s <= -9990)


def blank(n) -> pd.Series:
    return pd.Series([np.nan] * n)


def assign_states(df: pd.DataFrame) -> pd.Series:
    """Fill missing `state` by point-in-polygon against US Census state boundaries."""
    import geopandas as gpd

    shp = STATES_ZIP
    if not shp.exists():
        # Not optional. `load_literature` sets no state at all, so returning
        # here leaves 958 records with a null state, which the caller then
        # drops under the message "outside any US state" -- a silent 30% loss
        # of the database reported as a bounding-box clip.
        raise SystemExit(
            f"{shp} is missing. It is required: the literature source carries "
            "no state field, and without point-in-polygon every one of its "
            "records is dropped as 'outside any US state'. Download "
            "cb_2023_us_state_20m.zip from the US Census TIGER/Line site.")
    states = gpd.read_file(shp)[["STUSPS", "geometry"]].to_crs(4326)
    need = df["state"].isna() | df["state"].astype(str).str.strip().isin(["", "nan", "None"])
    if not need.any():
        return df["state"]
    pts = gpd.GeoDataFrame(
        df.loc[need, ["latitude", "longitude"]],
        geometry=gpd.points_from_xy(df.loc[need, "longitude"], df.loc[need, "latitude"]),
        crs=4326,
    )
    joined = gpd.sjoin(pts, states, how="left", predicate="within")
    out = df["state"].copy()
    out.loc[need] = joined["STUSPS"].reindex(pts.index).values
    return out


def finish(df: pd.DataFrame, source_key: str) -> pd.DataFrame:
    """Fill missing standard columns, normalize types, tag ids, clip to western US."""
    df = df.copy()
    df["source_key"] = source_key
    for c in STD_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["fire_name_norm"] = df["fire_name"].map(norm_fire)
    if df["group_key"].isna().all():
        df["group_key"] = df["fire_name_norm"]
    df["latitude"] = clean_num(df["latitude"])
    df["longitude"] = clean_num(df["longitude"])
    df["event_date"] = parse_date(df["event_date"])
    df["fire_start_date"] = parse_date(df["fire_start_date"])
    df["observation_date"] = parse_date(df["observation_date"])

    # What does `event_date` mean for this source? See DATE_BASIS.
    if df["date_basis"].isna().all():
        df["date_basis"] = DATE_BASIS.get(source_key, "unknown")
    # Where the source gives only a survey date, event_date IS the observation
    # date; record that explicitly rather than leaving the field empty.
    obs_only = df["date_basis"].eq("observation") & df["observation_date"].isna()
    df.loc[obs_only, "observation_date"] = df.loc[obs_only, "event_date"]
    for c in ["peak_i15_mmh", "peak_i30_mmh", "peak_i60_mmh", "volume_m3", "fire_year"]:
        df[c] = clean_num(df[c])

    n0 = len(df)
    df = df[df.latitude.notna() & df.longitude.notna() & df.event_date.notna()]
    n1 = len(df)
    df = df[
        df.longitude.between(*WEST_LON) & df.latitude.between(*WEST_LAT)
    ]
    n2 = len(df)
    df = df.reset_index(drop=True)
    df["record_id"] = [f"{source_key}_{i:05d}" for i in range(len(df))]
    print(
        f"  {source_key:14s} kept {n2:5d}  (dropped {n0 - n1} lacking lat/lon/date, "
        f"{n1 - n2} outside western US bbox)"
    )
    return df[STD_COLS]


# ----------------------------------------------------------------------------
# loaders (one per source) -> standardized point records of DEBRIS-FLOW OCCURRENCE
# ----------------------------------------------------------------------------


#: Upstream erratum, found 2026-09-20. In the Cavagnaro figshare release the
#: Station Fire (2009) block carries `StormDate` shifted forward by exactly four
#: years — 11/13/2013 for a storm that happened 2009-11-13. All 489 Station rows
#: are affected (108 of them Response == 1); no other fire in the file is.
#: See DECISIONS.md, "Station Fire date erratum", for the evidence.
#: Reported to the author; drop this correction once the release is fixed.
STATION_SHIFT_YEARS = 4


def _fix_station_fire_dates(c: pd.DataFrame) -> pd.DataFrame:
    """Undo the +4-year shift on the Station Fire block.

    Guarded on the symptom rather than on the fire name alone, so a corrected
    upstream file passes through untouched instead of being shifted twice.
    """
    c = c.copy()
    sd = pd.to_datetime(c["StormDate"], errors="coerce")
    bad = (c["Fire Name"].astype(str).str.strip().eq("Station")
           & c["Year"].eq(2009)
           & ((sd.dt.year - c["Year"]) >= 3))
    if bad.any():
        c.loc[bad, "StormDate"] = sd[bad] - pd.DateOffset(years=STATION_SHIFT_YEARS)
        print(f"  cavagnaro2025: corrected {int(bad.sum())} Station Fire dates by "
              f"-{STATION_SHIFT_YEARS} yr (upstream erratum; see DECISIONS.md)")
    return c


def load_cavagnaro() -> pd.DataFrame:
    p = RAW / "cavagnaro_figshare/PublicCodes/DFObsHydroclimatePubNew_attributes.xlsx"
    c = pd.read_excel(p)
    c = _fix_station_fire_dates(c)
    c = c[c["Response"] == 1].copy()
    out = pd.DataFrame(
        {
            "fire_name": c["Fire Name"],
            "fire_year": c["Year"],
            "state": c["State"],
            "latitude": c["Lat"],
            "longitude": c["Lon"],
            "location_type": "observation_point",
            "location_quality": "point",
            "event_date": c["StormDate"],
            "date_precision": "day",
            "n_flows": 1,
            "initiation_type": "runoff-generated",
            "df_evidence": "reported",
            "peak_i15_mmh": c["Peak_I15_mm/h"],
            "source_record_id": c["Fire_SegID"].astype(str),
            "original_source": c["Citation"],
            "notes": "Cavagnaro et al. (2025) subset " + c["Database"].astype(str),
        }
    )
    return finish(out, "cavagnaro2025")


def _load_graber_style(path: Path, source_key: str) -> pd.DataFrame:
    g = pd.read_csv(path, low_memory=False)
    g = g[g["Response"] == 1].copy()
    out = pd.DataFrame(
        {
            "fire_name": g["FireName"],
            "fire_year": g["FireYear"],
            "fire_start_date": g.get("FireStartDate"),
            "state": g["FireState"],
            "latitude": g["ObservationLatitude"],
            "longitude": g["ObservationLongitude"],
            "location_type": "observation_point",
            "location_quality": "point",
            # StormStart is the triggering-storm onset; ObservationDate is when it was
            # seen. StormStart is the better "date of occurrence" for seasonality.
            "event_date": g["StormStart"].where(
                g["StormStart"].astype(str).str.match(r"^\d{8}"), g["ObservationDate"]
            ),
            "date_precision": "day",
            "n_flows": 1,
            "initiation_type": "runoff-generated",
            "df_evidence": "field_verified",
            "peak_i15_mmh": g.get("PeakI15"),
            "peak_i30_mmh": g.get("PeakI30"),
            "peak_i60_mmh": g.get("PeakI60"),
            "source_record_id": g["FireSegmentID"].astype(str),
            "original_source": g["ObservationSource"],
        }
    )
    return finish(out, source_key)


def load_graber2023() -> pd.DataFrame:
    return _load_graber_style(RAW / "sciencebase_17fires/Combined_Inventory.csv", "graber2023")


def load_graber2024() -> pd.DataFrame:
    return _load_graber_style(RAW / "sciencebase_18fires/Combined_Inventory.csv", "graber2024")


def load_volumes227() -> pd.DataFrame:
    v = pd.read_csv(RAW / "sciencebase_volumes227/DebrisFlowVolume_Inventory.csv")
    out = pd.DataFrame(
        {
            "fire_name": v["FireName"],
            "fire_year": parse_date(v["FireStartDate"]).dt.year,
            "fire_start_date": v["FireStartDate"],
            "state": v["State"],
            "latitude": v["DepositLatitude"],
            "longitude": v["DepositLongitude"],
            "location_type": "deposit",
            "location_quality": "point",
            "event_date": v["DebrisFlowDate"],
            "date_precision": "day",
            "n_flows": 1,
            "initiation_type": "runoff-generated",
            "df_evidence": "field_verified",
            "peak_i15_mmh": v["i15_mm/h"],
            "peak_i30_mmh": v["i30_mm/h"],
            "peak_i60_mmh": v["i60_mm/h"],
            "volume_m3": v["Volume_m3"],
            "source_record_id": v["WatershedID"].astype(str),
            "original_source": v["Source"],
        }
    )
    return finish(out, "volumes227")


def load_czu2021() -> pd.DataFrame:
    z = pd.read_csv(RAW / "sciencebase_czu2021/Inventory.csv", low_memory=False)
    z = z[z["MajorResponse"] == 1].copy()
    out = pd.DataFrame(
        {
            "fire_name": z["FireName"],
            "fire_year": z["FireYear"],
            "state": z["FireState"],
            "latitude": z["ObservationLatitude"],
            "longitude": z["ObservationLongitude"],
            "location_type": "observation_point",
            "location_quality": "point",
            "event_date": z["StormStart"],
            "date_precision": "day",
            "n_flows": 1,
            "initiation_type": "runoff-generated",
            "df_evidence": "field_verified_major_response",
            "peak_i15_mmh": z["PeakI15"],
            "peak_i30_mmh": z["PeakI30"],
            "peak_i60_mmh": z["PeakI60"],
            "source_record_id": z["FireSegmentID"].astype(str),
            "original_source": z["ObservationSource"],
            "notes": "MajorResponse=1 (major field-verified postfire hydrologic response; "
            "debris flow inferred, not explicitly coded in source)",
        }
    )
    return finish(out, "czu2021")


def load_dixie2023() -> pd.DataFrame:
    d = pd.read_csv(RAW / "sciencebase_dixie/Inventory.csv", low_memory=False)
    d = d[d["Response"] == 1].copy()
    out = pd.DataFrame(
        {
            "fire_name": d["FireName"],
            "fire_year": d["FireYear"],
            "state": d["FireState"],
            "latitude": d["ObservationLatitude"],
            "longitude": d["ObservationLongitude"],
            "location_type": "observation_point",
            "location_quality": "point",
            "event_date": d["StormStart"],
            "date_precision": "day",
            "n_flows": 1,
            "initiation_type": "runoff-generated",
            "df_evidence": "field_verified",
            "peak_i15_mmh": d["PeakI15"],
            "peak_i30_mmh": d["PeakI30"],
            "peak_i60_mmh": d["PeakI60"],
            "source_record_id": d["FireSegmentID"].astype(str),
            "original_source": d["ObservationSource"],
        }
    )
    return finish(out, "dixie2023")


def load_dolan2020() -> pd.DataFrame:
    # Response: 0 = no erosion, 1 = flood, 2 = remotely mapped DF, 3 = field-verified DF
    d = pd.read_csv(RAW / "sciencebase_dolan2020/Inventory.csv", low_memory=False)
    d = d[d["Response"].isin([2, 3])].copy()

    # One row is one OBSERVATION; `FireSegmentID` is the stream segment it sits
    # on (release README, field list). A segment is routinely observed at
    # several points a few tens of metres apart on the same date -- the mapped
    # extent of one response, not several debris flows -- so 2,080 rows carry
    # only 1,759 segments. We already treat the segment as the unit
    # (`location_type` below), and the record-level de-duplication cannot catch
    # these because they are within a single source, where two nearby records
    # are normally two real adjacent flows.
    #
    # Field-verified (3) wins over remotely mapped (2) where a segment has both.
    before = len(d)
    d = (d.sort_values(["FireSegmentID", "Response"], ascending=[True, False])
           .drop_duplicates(subset="FireSegmentID", keep="first"))
    print(f"  dolan2020: {before} observations -> {len(d)} stream segments "
          f"({before - len(d)} extra points on an already-counted segment)")
    out = pd.DataFrame(
        {
            "fire_name": d["FireName"],
            "fire_year": d["FireYear"],
            "state": d["FireState"],
            "latitude": d["ObservationLatitude"],
            "longitude": d["ObservationLongitude"],
            "location_type": "channel_segment",
            "location_quality": "point",
            # ObservationDate is the Google Earth image-acquisition date or the
            # field-visit date (see the release README), NOT a date of occurrence.
            # All of these segments record the response to one atmospheric river.
            "event_date": DOLAN_STORM_DATE,
            "observation_date": d["ObservationDate"],
            "date_basis": "assigned_literature",
            "date_reference": DOLAN_DATE_REFERENCE,
            "date_precision": "day",
            "n_flows": 1,
            "initiation_type": "runoff-generated",
            "df_evidence": np.where(
                d["Response"] == 3, "field_verified", "remotely_mapped"
            ),
            "source_record_id": d["FireSegmentID"].astype(str),
            "original_source": d["ObservationSource"],
            "notes": "stream-segment-scale mapping of one fire; date assigned from "
                     "the triggering storm, not the survey",
        }
    )
    return finish(out, "dolan2020")


def load_oregon2024() -> pd.DataFrame:
    f = pd.read_csv(RAW / "sciencebase_oregon/OR_field_observations.csv", low_memory=False)
    f = f[f["Response"] == 1].copy()
    out = pd.DataFrame(
        {
            "fire_name": f["Fire_Name"],
            "fire_year": f["Year"],
            "fire_start_date": f["Date_Of_Fire_Initiation"],
            "state": f["State"],
            "latitude": f["Observation_Lat"],
            "longitude": f["Observation_Long"],
            "location_type": "observation_point",
            "location_quality": "point",
            "event_date": f["Observation_Date"],
            "date_precision": "day",
            "n_flows": 1,
            "initiation_type": f["Primary_IT"].astype(str).str.strip(),
            "df_evidence": "field_verified",
            "peak_i15_mmh": f["Peak_I15"],
            "peak_i30_mmh": f["Peak_I30"],
            "peak_i60_mmh": f["Peak_I60"],
            "source_record_id": f["Fire_SegID"].astype(str),
            "original_source": f["SourceOfOb"],
            "notes": "western Cascades OR; includes landslide-initiated flows - filter on "
            "initiation_type for runoff-generated only",
        }
    )
    return finish(out, "oregon2024")


#: How close two records must be, in space and in fire start date, to be taken
#: as the same fire. Measured 2026-09-20: every literature record that matches a
#: named fire on start date sits within 4.7 km of it (median 1.5), and nothing
#: else lies inside a 40 km search radius, so the separation is clean.
FIRE_MATCH_KM = 5.0
FIRE_MATCH_DAYS = 7


def resolve_literature_fire_keys(points: pd.DataFrame) -> pd.DataFrame:
    """Give literature records a FIRE key instead of an EVENT key.

    The literature source carries no fire name, so `load_literature` keys each
    record `litevent<EventID>`. But EventID is the source's own identifier for
    "debris flows grouped together as being part of a single debris-flow event"
    — a STORM, not a fire. Using it as `fire_key` has two separate consequences:

    1. A literature record can never merge with the same fire under its real
       name, so a flow reported by both the literature database and a named
       inventory becomes two events. This is the one that inflates the event
       count.
    2. One fire is split into as many "fires" as it had storms — up to seven
       for Grizzly Creek (2020-08-10). Event counts are unaffected, because
       those storms have different dates, but every fire-level statistic is:
       events per fire, first flow per fire, and any clustering correction.

    Both are fixed by resolving fire identity from what the source does carry,
    `DateFireStart` plus location. Named records are never modified; only
    literature keys are reassigned, so the blast radius stays small.
    """
    p = points.copy()
    is_lit = p.source_key.eq("literature")
    geo = p.latitude.notna() & p.longitude.notna() & p.fire_start_date.notna()
    usable = is_lit & geo
    if not usable.any():
        return p

    lit = p[usable]
    named = p[~is_lit & geo & p.group_key.notna()]

    # ONE projection over both frames: `_to_local_xy` takes its reference
    # latitude from the data it is handed, so projecting them separately would
    # put the two sets on different planes.
    both = pd.concat([lit, named])
    XY = _to_local_xy(both.latitude.to_numpy(), both.longitude.to_numpy())
    xy_lit, xy_named = XY[:len(lit)], XY[len(lit):]
    radius_m = FIRE_MATCH_KM * 1000.0

    # --- 1. cluster literature records into FIRES: same start date (within
    # tolerance) and close together. This both joins one fire split across
    # several EventIDs and separates several fires filed under one.
    parent = list(range(len(lit)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    start = lit.fire_start_date.to_numpy()
    for a, b in cKDTree(xy_lit).query_pairs(r=radius_m):
        if abs((start[a] - start[b]) / np.timedelta64(1, "D")) <= FIRE_MATCH_DAYS:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    cluster = pd.Series([find(k) for k in range(len(lit))], index=lit.index)

    # --- 2. per cluster, adopt a named fire's key, else synthesize a stable one
    tree = cKDTree(xy_named) if len(named) else None
    pos = {ix: k for k, ix in enumerate(lit.index)}
    adopted_fires, synthesized, split, joined = set(), 0, 0, 0

    for _, idx in cluster.groupby(cluster).groups.items():
        idx = list(idx)
        originals = set(p.loc[idx, "group_key"])
        key = name = None
        if tree is not None:
            for ix in idx:                      # nearest named fire, any member
                cand = tree.query_ball_point(xy_lit[pos[ix]], r=radius_m)
                if not cand:
                    continue
                c = named.iloc[cand]
                dt = (c.fire_start_date - p.at[ix, "fire_start_date"]).dt.days.abs()
                near = (dt <= FIRE_MATCH_DAYS).to_numpy()
                if not near.any():
                    continue
                d = np.linalg.norm(xy_named[cand] - xy_lit[pos[ix]], axis=1)
                j = int(np.argmin(np.where(near, d, np.inf)))
                key, name = c.iloc[j].group_key, c.iloc[j].fire_name
                break
        if key is None:
            # Deterministic, and carries the fire date so an EventID covering
            # two fires yields two obviously different keys.
            #
            # Note this key is coarser than the 5 km clustering above: two
            # clusters sharing an EventID AND a start date collapse back to one
            # key. That is deliberate — it reconnects a single burn scar whose
            # mapped points are more than 5 km apart, which happens for three
            # keys here, spread up to 11.8 km from their centroid. Geography
            # therefore only ever SPLITS an EventID that covers distinct fires;
            # it never splits one fire on distance alone.
            stamp = pd.Timestamp(p.at[idx[0], "fire_start_date"]).strftime("%Y%m%d")
            stem = min(originals).replace("litevent", "")
            key = f"litfire{stem}_{stamp}"
            synthesized += 1
        else:
            adopted_fires.add(key)
        for ix in idx:
            p.at[ix, "group_key"] = key
            if pd.isna(p.at[ix, "fire_name"]) and name is not None:
                p.at[ix, "fire_name"] = name
                p.at[ix, "fire_name_norm"] = norm_fire(name)
        if len(originals) > 1:
            joined += 1

    # an original key landing in more than one cluster was several fires
    split = int((p.loc[usable].groupby(points.loc[usable, "group_key"])
                 .group_key.nunique() > 1).sum())

    before = points.loc[usable, "group_key"].nunique()
    after = p.loc[usable, "group_key"].nunique()
    print("\nResolving literature fire identity (EventID is a storm, not a fire) ...")
    print(f"  {len(lit)} records -> {after} fires "
          f"({len(adopted_fires)} matched a named fire, {synthesized} literature-only)")
    print(f"  {joined} fires were split across several EventIDs and are now joined")
    print(f"  {split} EventIDs covered more than one fire and are now separated")
    print(f"  literature fire keys: {before} -> {after}")
    return p


def load_literature() -> pd.DataFrame:
    """McGuire et al. (2024) literature-derived database (global; clipped to western US).

    Has no fire-name field, so `group_key` is built from the source's own EventID,
    which the authors define as "debris flows grouped together as being part of a
    single debris-flow event".
    """
    lit = pd.read_csv(
        RAW / "sciencebase_literature/PFDF_database_sortedbyReference.txt", low_memory=False
    )
    d = parse_date(lit["DateOfFlow"])
    approx = parse_date(lit["ApproxDateOfFlow"])
    precision = np.where(d.notna(), "day", np.where(approx.notna(), "approximate", "unknown"))
    loc = lit["LocationInformation"].astype(str)
    quality = np.where(loc.str.startswith("General"), "general", "point")
    quality = np.where(loc.eq("-9999"), "unknown", quality)
    out = pd.DataFrame(
        {
            "fire_name": np.nan,
            "group_key": "litevent" + lit["EventID"].astype(str),
            "fire_start_date": lit["DateFireStart"],
            "latitude": lit["Latitude"],
            "longitude": lit["Longitude"],
            "location_type": loc.replace("-9999", np.nan),
            "location_quality": quality,
            "event_date": d.fillna(approx),
            "date_precision": precision,
            "n_flows": clean_num(lit["NumberOfDebrisFlows"]),
            "initiation_type": lit["InitiationMechanism"]
            .astype(str)
            .replace({"-9999": np.nan, "Runoff": "runoff-generated", "Landslide": "landslide",
                      "Runoff and Landslide": "runoff-generated and landslide"}),
            "df_evidence": "reported",
            "source_record_id": lit["SiteID"].astype(str),
            "original_source": lit["Reference"],
            "notes": "literature-derived global PFDF database, clipped to western US",
        }
    )
    out["fire_year"] = parse_date(lit["DateFireStart"]).dt.year
    return finish(out, "literature")


def load_oakley_events() -> pd.DataFrame:
    """Oakley is EVENT-level with fire-perimeter-centroid locations, not DF points."""
    o = pd.read_excel(
        RAW / "oakley_ijwf/SupplementalMaterialA_PFDF_Events_rev1.xlsx", "EventsDatabase"
    )
    out = pd.DataFrame(
        {
            "fire_name": o["FireName"],
            "fire_year": o["FireYear"],
            "fire_start_date": o["FireDate"],
            "state": "CA",
            "latitude": o["FireCentroidLat"],
            "longitude": o["FireCentroidLong"],
            "location_type": "fire_centroid",
            "location_quality": "fire_scale",
            "event_date": o["EventDate"],
            "date_precision": "day",
            "n_flows": pd.to_numeric(o["FlowNum"], errors="coerce"),
            "initiation_type": "runoff-generated",
            "df_evidence": o["Confidence"].astype(str).str.strip() + "_confidence",
            "source_record_id": o["EventName"].astype(str),
            "original_source": o["Source1_Name"],
            "notes": (
                "flow-count bin: "
                + o["FlowNumBin"].astype(str)
                + "; storm class: "
                + o["StormClass"].astype(str)
            ),
        }
    )
    return finish(out, "oakley2025")


# ----------------------------------------------------------------------------
# de-duplication
# ----------------------------------------------------------------------------


#: One storm, two dates: how far apart the dates may be, and how close the two
#: sets of points must be to count as the same flows. The distance does the
#: discriminating; the day window only widens the candidate set.
STORM_DAY_TOL = 2
STORM_SAME_FLOW_M = 250.0


def canonicalize_storm_dates(p: pd.DataFrame,
                             day_tol: int = STORM_DAY_TOL,
                             radius_m: float = STORM_SAME_FLOW_M) -> pd.DataFrame:
    """Give one storm one date, where the evidence says it is one storm.

    Sources disagree about what date a flow carries — `DATE_BASIS` records
    storm start for some and the observation for others, and a multi-day
    atmospheric river can be dated anywhere inside it. Grouping events on an
    exact date therefore splits a single storm in two when two sources date it
    differently.

    Measured 2026-09-20, eleven same-fire event pairs sit within three days of
    each other with no source in common. Date proximity alone does not settle
    them, so this requires the two sets of points to BE the same flows: the
    median nearest-neighbour distance between them must be within `radius_m`,
    the same radius the record-level de-duplication uses.

    That test is what separates the cases:

        Butte  2016-03-04 vs 03-06   median   0 m  -> one storm
        Dolan  2021-01-26 vs 01-27   median  22 m  -> one storm
        Station, Sayre, Monument, Tadpole  339-3,514 m  -> left alone
        Whitewater-Baldy 09-14/15/16 vs 09-17  7-25 km -> a real monsoon
                                                          sequence, left alone

    The surviving date comes from the highest-priority source present, not from
    whichever is earlier, so Dolan keeps 2021-01-27 — the storm date the
    Cavagnaro GRL paper assigns, not the 26th that `czu2021` carries for the
    start of the same atmospheric river.

    Deliberately NOT done at record level: widening the de-duplication's own
    date window merges 208 Dolan segment records into higher-priority basin
    points, because that inventory maps stream segments about 10 m apart and a
    single point sits within 250 m of many of them. That is a scale mismatch,
    not a date-convention problem, and it belongs to record comparability
    rather than here.
    """
    p = p.copy()
    prio = p.source_key.map(SOURCE_PRIORITY).fillna(99)
    merges = []
    for fk, g in p.groupby("fire_key", sort=False):
        dates = sorted(d for d in g.event_date.dropna().unique())
        if len(dates) < 2:
            continue
        for i in range(len(dates) - 1):
            for j in range(i + 1, len(dates)):
                gap = (dates[j] - dates[i]) / np.timedelta64(1, "D")
                if gap > day_tol:
                    break
                A = g[g.event_date == dates[i]]
                B = g[g.event_date == dates[j]]
                if set(A.source_key) & set(B.source_key):
                    continue          # one source dates them apart on purpose
                if not len(A) or not len(B):
                    continue
                # ONE projection over both sets: `_to_local_xy` derives its
                # reference latitude from the data it is handed, so projecting
                # A and B separately puts them on different planes and inflates
                # the distance between them.
                XY = _to_local_xy(
                    np.r_[A.latitude.to_numpy(), B.latitude.to_numpy()],
                    np.r_[A.longitude.to_numpy(), B.longitude.to_numpy()])
                xa, xb = XY[:len(A)], XY[len(A):]
                d, _ = cKDTree(xb).query(xa)
                if np.median(d) > radius_m:
                    continue          # different flows, so a different storm
                keep = dates[i] if prio[A.index].min() <= prio[B.index].min() else dates[j]
                drop = dates[j] if keep == dates[i] else dates[i]
                merges.append((fk, drop, keep, float(np.median(d))))

    if merges:
        print("\nOne storm under two dates (same fire, no shared source, "
              "same flows within "
              f"{radius_m:.0f} m) ...")
        for fk, drop, keep, med in merges:
            m = p.fire_key.eq(fk) & p.event_date.eq(drop)
            print(f"  {fk:<16} {str(pd.Timestamp(drop).date())} -> "
                  f"{str(pd.Timestamp(keep).date())}  "
                  f"({int(m.sum())} records, median {med:.0f} m apart)")
            p.loc[m, "event_date"] = keep
    return p


def _to_local_xy(lat, lon):
    """Equirectangular projection to metres; fine over a few-hundred-metre search."""
    r = 6371000.0
    lat0 = np.deg2rad(np.nanmean(lat))
    x = r * np.deg2rad(lon) * np.cos(lat0)
    y = r * np.deg2rad(lat)
    return np.column_stack([x, y])


def dedup_points(df: pd.DataFrame, radius_m: float = 250.0) -> pd.DataFrame:
    """Collapse records that describe the same debris flow.

    Two passes:
      A. identical rounded coordinates (5 dp, ~1 m) on the same date; applied
         within and across sources
      B. within `radius_m` of each other on the same date, ACROSS DIFFERENT
         SOURCES ONLY

    Pass B is restricted to cross-source pairs because each source has already
    been de-duplicated internally by its own authors: two records 200 m apart in
    the same inventory are two real, adjacent debris flows (this matters most for
    the segment-scale Dolan Fire inventory), whereas the same debris flow mapped
    by two different groups routinely lands 50-200 m apart.

    Pass B deliberately does NOT require the fire names to match: fire naming is
    inconsistent between sources (e.g. "Grand Prix-Old" vs separate "Grand Prix"
    and "Old" entries) and several sources carry no fire name at all.

    Clusters are grown by leader assignment in source-priority order rather than
    by transitive linkage, so a line of closely spaced points cannot chain into
    one giant cluster.

    The surviving record is the one from the highest-priority source; collapsed
    partners are written to the audit file and summarized in
    `corroborating_sources` / `n_independent_sources`.
    """
    df = df.copy()
    df["_prio"] = df["source_key"].map(SOURCE_PRIORITY).fillna(99)
    df = df.sort_values(["_prio", "record_id"]).reset_index(drop=True)

    df["_latr"] = df.latitude.round(5)
    df["_lonr"] = df.longitude.round(5)

    # ---- pass A: exact coordinate + date
    df["_keyA"] = list(zip(df._latr, df._lonr, df.event_date))
    df["_clusterA"] = df.groupby("_keyA", sort=False)["record_id"].transform("first")

    # ---- pass B: cross-source proximity on the same date (KD-tree per date)
    reps = df[df.record_id == df["_clusterA"]]  # one row per pass-A cluster
    remap = {}
    for _date, grp in reps.groupby("event_date", sort=False):
        if grp.source_key.nunique() < 2:
            continue  # nothing to merge across sources on this date
        grp = grp.sort_values("_prio")  # leaders come from the best source first
        xy = _to_local_xy(grp.latitude.to_numpy(), grp.longitude.to_numpy())
        rid = grp.record_id.to_numpy()
        src = grp.source_key.to_numpy()
        tree = cKDTree(xy)
        neighbours = tree.query_ball_point(xy, radius_m)
        taken = np.zeros(len(grp), dtype=bool)
        for i in range(len(grp)):
            if taken[i]:
                continue
            taken[i] = True  # i is a cluster leader
            for j in neighbours[i]:
                if taken[j] or src[j] == src[i]:
                    continue
                taken[j] = True
                remap[rid[j]] = rid[i]
    df["cluster_id"] = df["_clusterA"].map(lambda c: remap.get(c, c))

    agg = (
        df.groupby("cluster_id", sort=False)
        .agg(
            n_source_records=("record_id", "size"),
            corroborating_sources=("source_key", lambda s: "|".join(sorted(set(s)))),
            n_independent_sources=("source_key", lambda s: len(set(s))),
        )
        .reset_index()
    )

    keep = df[df.record_id == df.cluster_id].merge(agg, on="cluster_id", how="left")
    dropped = df[df.record_id != df.cluster_id].copy()
    dropped = dropped.rename(columns={"cluster_id": "duplicate_of"})

    keep = keep.drop(columns=["_prio", "_latr", "_lonr", "_keyA", "_clusterA"])
    dropped = dropped.drop(columns=["_prio", "_latr", "_lonr", "_keyA", "_clusterA"])
    return keep, dropped


def build_events(points: pd.DataFrame, oakley: pd.DataFrame, day_tol: int = 1):
    """Collapse point records to fire x date events, then fold Oakley events in.

    Oakley is matched on normalized fire name with a +/- `day_tol` day window,
    because Oakley records the calendar date the flow was observed (PST) while the
    point inventories usually record the triggering storm's start date.
    """
    p = points.copy()
    p["fire_key"] = np.where(
        p.group_key.eq("") | p.group_key.isna(),
        "unnamed_" + p.record_id,
        p.group_key,
    )
    p = canonicalize_storm_dates(p)
    ev = (
        p.groupby(["fire_key", "event_date"], sort=False)
        .agg(
            fire_name=("fire_name", "first"),
            fire_year=("fire_year", "first"),
            fire_start_date=("fire_start_date", "first"),
            state=("state", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            n_debris_flows=("record_id", "size"),
            peak_i15_mmh=("peak_i15_mmh", "max"),
            peak_i30_mmh=("peak_i30_mmh", "max"),
            peak_i60_mmh=("peak_i60_mmh", "max"),
            sources=("source_key", lambda s: "|".join(sorted(set(s)))),
            # An event can draw on records with different date bases. Keep the
            # full set, and flag the event as survey-dated if ANY record is,
            # because that is the weakest link for timing work.
            date_basis=("date_basis", lambda s: "|".join(sorted(set(s.dropna())))),
            survey_dated=("date_basis", lambda s: bool((s == "observation").any())),
            initiation_class=(
                "initiation_class",
                lambda s: "|".join(sorted(set(s.dropna()))) or "unknown",
            ),
            date_precision=(
                "date_precision",
                lambda s: "|".join(sorted(set(s.dropna()))) or "unknown",
            ),
            location_quality=(
                "location_quality",
                lambda s: "|".join(sorted(set(s.dropna()))) or "unknown",
            ),
        )
        .reset_index()
    )
    ev["location_type"] = "mean of debris-flow observation points"
    ev["in_oakley2025"] = False
    ev["oakley_event_name"] = np.nan
    ev["oakley_match_type"] = np.nan

    o = oakley.copy().reset_index(drop=True)
    matched_rows = []
    new_rows = []
    for _, r in o.iterrows():
        in_window = ev.event_date.sub(r.event_date).abs() <= pd.Timedelta(days=day_tol)
        cand = ev[(ev.fire_key == r.fire_name_norm) & in_window]
        match_type = "fire name + date"
        if not len(cand):
            # Fallback for the literature-derived database, whose records carry no
            # fire name: accept an *unnamed* event on the same date whose debris
            # flows sit within 25 km of the Oakley fire centroid. Restricting this
            # to unnamed events (and to events not already claimed) keeps adjacent
            # but genuinely distinct fires - e.g. Apple vs El Dorado 2020, Dog Rock
            # vs El Portal 2014 - from being merged into one event.
            near = ev[
                in_window
                & ev.fire_key.str.startswith(("litevent", "unnamed"))
                & ~ev.in_oakley2025
            ].copy()
            if len(near):
                d_km = np.hypot(
                    (near.latitude - r.latitude) * 111.0,
                    (near.longitude - r.longitude)
                    * 111.0
                    * np.cos(np.deg2rad(r.latitude)),
                )
                near = near[d_km <= 25.0]
                if len(near):
                    cand = near
                    match_type = "date + within 25 km (unnamed literature event)"
        if len(cand):
            idx = (cand.event_date - r.event_date).abs().idxmin()
            ev.at[idx, "in_oakley2025"] = True
            ev.at[idx, "oakley_event_name"] = r.source_record_id
            ev.at[idx, "oakley_match_type"] = match_type
            ev.at[idx, "sources"] = "|".join(
                sorted(set(ev.at[idx, "sources"].split("|")) | {"oakley2025"})
            )
            matched_rows.append(
                dict(
                    oakley_event=r.source_record_id,
                    oakley_fire=r.fire_name,
                    oakley_date=r.event_date,
                    matched_event_key=ev.at[idx, "fire_key"],
                    matched_event_date=ev.at[idx, "event_date"],
                    matched_sources=ev.at[idx, "sources"],
                    match_type=match_type,
                )
            )
        else:
            new_rows.append(
                dict(
                    fire_key=r.fire_name_norm or f"oakley_{r.source_record_id}",
                    event_date=r.event_date,
                    fire_name=r.fire_name,
                    fire_year=r.fire_year,
                    fire_start_date=r.fire_start_date,
                    state=r.state,
                    latitude=r.latitude,
                    longitude=r.longitude,
                    n_debris_flows=r.n_flows,
                    initiation_class="runoff-generated",
                    date_precision="day",
                    location_quality="fire_scale",
                    peak_i15_mmh=np.nan,
                    peak_i30_mmh=np.nan,
                    peak_i60_mmh=np.nan,
                    sources="oakley2025",
                    location_type="fire perimeter centroid",
                    in_oakley2025=True,
                    oakley_event_name=r.source_record_id,
                )
            )
    if new_rows:
        ev = pd.concat([ev, pd.DataFrame(new_rows)], ignore_index=True)

    ev["n_sources"] = ev.sources.str.count(r"\|") + 1
    ev = ev.sort_values(["event_date", "fire_name"]).reset_index(drop=True)
    ev.insert(0, "event_id", [f"EV{i:05d}" for i in range(len(ev))])
    ev["month"] = ev.event_date.dt.month
    ev["day_of_year"] = ev.event_date.dt.dayofyear
    ev["water_year"] = np.where(ev.month >= 10, ev.event_date.dt.year + 1, ev.event_date.dt.year)
    return ev, matched_rows


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main():
    print("Loading and standardizing sources ...")
    loaders = [
        load_cavagnaro,
        load_graber2024,
        load_graber2023,
        load_volumes227,
        load_czu2021,
        load_dixie2023,
        load_dolan2020,
        load_oregon2024,
        load_literature,
    ]
    frames = [fn() for fn in loaders]
    points_raw = pd.concat(frames, ignore_index=True)
    points_raw = resolve_literature_fire_keys(points_raw)
    oakley = load_oakley_events()

    print("\nAssigning missing state codes by point-in-polygon ...")
    points_raw["state"] = assign_states(points_raw)
    outside_us = points_raw.state.isna()
    if outside_us.any():
        # The literature-derived database is global; the bounding box lets a few
        # southern-British-Columbia records through. Drop anything that does not
        # land inside a US state polygon.
        print(f"  dropping {int(outside_us.sum())} records outside any US state "
              "(southern British Columbia)")
        points_raw = points_raw[~outside_us].reset_index(drop=True)

    # Harmonize the free-text initiation descriptors used by different sources.
    INIT_MAP = {
        "runoff-generated": "runoff-generated",
        "distributed runoff and erosion": "runoff-generated",
        "in channel failure": "runoff-generated",
        "runoff-generated and landslide": "mixed",
        "landslide": "landslide",
        "shallow landslide": "landslide",
        "deep landslide": "landslide",
        "unknown": "unknown",
        "nan": "unknown",
        "": "unknown",
    }
    points_raw["initiation_class"] = (
        points_raw.initiation_type.astype(str).str.strip().str.lower().map(INIT_MAP).fillna("unknown")
    )

    # Flag literature dates that look like year-only placeholders coded as 1 Jan.
    jan1 = (
        (points_raw.source_key == "literature")
        & (points_raw.event_date.dt.month == 1)
        & (points_raw.event_date.dt.day == 1)
    )
    points_raw.loc[jan1, "date_precision"] = "suspect_year_only (coded 1 January)"
    print(f"  flagged {int(jan1.sum())} literature dates as suspect year-only (1 January)")

    print(f"\nTotal point records before de-duplication: {len(points_raw)}")
    keep, dropped = dedup_points(points_raw)
    print(f"Unique debris-flow records after de-duplication: {len(keep)}")
    print(f"Records collapsed as duplicates: {len(dropped)}")

    # add convenience seasonality fields
    for df in (keep, oakley):
        df["month"] = df.event_date.dt.month
        df["day_of_year"] = df.event_date.dt.dayofyear
        df["water_year"] = np.where(
            df.month >= 10, df.event_date.dt.year + 1, df.event_date.dt.year
        )

    events, matched = build_events(keep, oakley)
    print(f"\nUnique fire x date debris-flow events: {len(events)}")
    print(f"  Oakley events matched to an existing point-derived event: {len(matched)}")
    print(f"  Oakley events contributing a new event: {events.in_oakley2025.sum() - len(matched)}")

    # ---------------- write outputs ----------------
    keep_cols = [c for c in STD_COLS if c != "record_id"]
    keep_out = keep[
        ["record_id"] + keep_cols + [
            "initiation_class", "month", "day_of_year", "water_year",
            "n_source_records", "n_independent_sources", "corroborating_sources",
        ]
    ]
    keep_out.to_csv(OUT / "pfdf_occurrence_records_compiled.csv", index=False)
    dropped.to_csv(OUT / "pfdf_occurrence_records_duplicates_removed.csv", index=False)
    oakley.to_csv(OUT / "pfdf_oakley2025_events_standardized.csv", index=False)
    events.to_csv(OUT / "pfdf_events_compiled.csv", index=False)

    # Oakley <-> western-US-inventory cross-check table
    xcheck = pd.DataFrame(matched)
    unmatched = oakley[~oakley.source_record_id.isin(xcheck.get("oakley_event", pd.Series([])))]
    unmatched_out = unmatched[
        ["source_record_id", "fire_name", "fire_year", "event_date", "n_flows", "df_evidence"]
    ].rename(columns={"source_record_id": "oakley_event"})
    unmatched_out = unmatched_out.assign(match_type="no match - unique to Oakley et al. 2025")
    pd.concat([xcheck, unmatched_out], ignore_index=True).to_csv(
        OUT / "oakley_vs_westernUS_crosscheck.csv", index=False
    )

    # ---------------- report ----------------
    lines = []
    lines.append("POST-FIRE DEBRIS-FLOW OCCURRENCE COMPILATION")
    lines.append("=" * 78)
    lines.append("")
    lines.append("SOURCES")
    for k, m in SOURCE_META.items():
        n = int((points_raw.source_key == k).sum()) if k != "oakley2025" else len(oakley)
        lines.append(f"  [{k}] n={n}")
        lines.append(f"      {m['citation']}")
        lines.append(f"      article DOI: {m['doi']}   data DOI: {m['data_doi']}")
    lines.append("")
    lines.append("POINT-LEVEL DEBRIS-FLOW RECORDS (true debris-flow locations)")
    lines.append(f"  raw records                : {len(points_raw)}")
    lines.append(f"  after de-duplication       : {len(keep)}")
    lines.append(f"  removed as duplicates      : {len(dropped)}")
    lines.append(
        f"  date range                 : {keep.event_date.min():%Y-%m-%d} to "
        f"{keep.event_date.max():%Y-%m-%d}"
    )
    lines.append("")
    lines.append("  per-source contribution (raw -> unique kept):")
    for k in SOURCE_PRIORITY:
        if k == "oakley2025":
            continue
        raw_n = int((points_raw.source_key == k).sum())
        kept_n = int((keep.source_key == k).sum())
        if raw_n:
            lines.append(f"    {k:14s} {raw_n:6d} -> {kept_n:6d}   ({raw_n - kept_n} dropped as dup)")
    lines.append("")
    lines.append("  records corroborated by >1 independent source: "
                 f"{int((keep.n_independent_sources > 1).sum())}")
    lines.append("")
    lines.append("  records by location quality:")
    for s, n in keep.location_quality.value_counts(dropna=False).items():
        lines.append(f"    {str(s):10s} {n}")
    lines.append("  records by date precision:")
    for s, n in keep.date_precision.value_counts(dropna=False).items():
        lines.append(f"    {str(s):10s} {n}")
    lines.append("  records by initiation class:")
    for s, n in keep.initiation_class.value_counts(dropna=False).items():
        lines.append(f"    {str(s)[:34]:36s} {n}")
    lines.append("")
    lines.append("  records by state:")
    for s, n in keep.state.value_counts(dropna=False).items():
        lines.append(f"    {str(s):6s} {n}")
    lines.append("")
    lines.append("EVENT-LEVEL (fire x date), includes Oakley et al. 2025")
    lines.append(f"  unique events              : {len(events)}")
    lines.append(f"  events involving Oakley    : {int(events.in_oakley2025.sum())}")
    lines.append(f"    of which cross-confirmed by a point inventory : {len(matched)}")
    lines.append(f"    of which unique to Oakley                     : "
                 f"{int(events.in_oakley2025.sum()) - len(matched)}")
    lines.append(
        f"  date range                 : {events.event_date.min():%Y-%m-%d} to "
        f"{events.event_date.max():%Y-%m-%d}"
    )
    lines.append("")
    lines.append("  events per month (1=Jan):")
    mc = events.month.value_counts().sort_index()
    for m, n in mc.items():
        lines.append(f"    {int(m):2d}  {'#' * int(n):<40s} {n}")
    lines.append("")
    lines.append("  events per state:")
    for s, n in events.state.value_counts(dropna=False).items():
        lines.append(f"    {str(s):6s} {n}")
    lines.append("")
    lines.append("OUTPUT FILES")
    lines.append("  pfdf_occurrence_records_compiled.csv            de-duplicated point records")
    lines.append("  pfdf_occurrence_records_duplicates_removed.csv  audit trail of collapsed rows")
    lines.append("  pfdf_events_compiled.csv                        unique fire x date events")
    lines.append("  pfdf_oakley2025_events_standardized.csv         Oakley events, standardized")
    lines.append("  oakley_vs_westernUS_crosscheck.csv              Oakley <-> western US match table")

    report = "\n".join(lines)
    (REPORTS / "compilation_report.txt").write_text(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
