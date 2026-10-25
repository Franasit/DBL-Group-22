#!/usr/bin/env python3
"""13h | Hazard-first Top-10 ADM3 profiles with THRESHOLDED factors.

Run from anywhere after saving under scripts/multivariate_flood_modeling/:
    python scripts/multivariate_flood_modeling/13h_map_top10_hazard_first_thresholded_factors.py

Read-only inputs:
  13d full screening table + methodology JSON (four-factor scenario by default);
  05a_monthly_panel/adm3_monthly_panel.csv;
  raw_data/administrative_boundaries/ssd_admin3.geojson;
  a saved 13e river_data/01_rivers_display.gpkg, when available.

This script does NOT calculate new hazard/asset thresholds or validate population.
Monthly means use a common set of complete table years across all 12 months.
Auto mode recognizes EXPLORATORY_EVENT_DATE_PROXY: source zeros mean no event
records, NOT confirmed flood-free days. Unknown other statuses remain excluded.
--monthly-mode strict disables the proxy branch. Missing data remain missing.
Per-month availability is also exported, independently of peak eligibility.
The underlying observation completeness still requires independent source review.
Display days are rounded HALF UP to integers; computations and ranks are not rounded.
No automatic downloads. No lake/wetland/terrain dependency. No previous .py import.
Display-only revision: retain the layout, rankings and monthly calculation; show
saved ADM3 quantities beside qualifying factor badges and define n once below
the table. Population N/A is not replaced by a raw unapproved estimate.
Only display values are rounded. The source values and flags are unchanged.
Units are printed beside every known factor quantity: facilities, km², people.
Missing population remains neutral text (4 N/A), never a zero quantity.

Dependencies: numpy pandas matplotlib geopandas shapely pyproj pyogrio
Primary API documentation:
  https://docs.python.org/3/library/decimal.html
  https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.days_in_month.html
  https://matplotlib.org/stable/api/_as_gen/matplotlib.artist.Artist.set_clip_path.html
"""
from __future__ import annotations
import argparse
import calendar
import hashlib
import io
import json
import math
import re
import string
import sys
import textwrap
import time
import warnings
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm, is_color_like
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, PathPatch
    from matplotlib.path import Path as MplPath
    import geopandas as gpd
    import shapely
    from shapely.ops import unary_union
    from shapely.geometry.polygon import orient
    from shapely.validation import make_valid
    from pyproj import Transformer, Geod
except ImportError as exc:
    raise SystemExit(f'Missing dependency: {exc}\nInstall in the active environment:\n'
                     'python -m pip install numpy pandas matplotlib geopandas shapely pyproj pyogrio') from exc

VERSION = '13h-4.0-thresholded-qc-visible-na'
# Source semantics checked against uploaded 05a_build_monthly_panel_UPDATED.py.
# This status counts unique recorded event dates, not verified inundation days.
PROXY_STATUS = 'exploratory_event_date_proxy'
REVIEWED_05A_SHA256 = 'b4c1a0740565b23de6c1e9cbc5d36421a11474676a82d3dd53badc483ab5c89a'
TABLE = '01_adm3_screening_full.csv'
META = '07_quality_and_methodology.json'
MONTH_TABLE = 'adm3_monthly_panel.csv'
H = 'historical_flood_days_mean_2019_2025'
D = 'historical_flood_days_change_after_2019'
B = 'mean_annual_flood_days_2000_2018'
P = 'population_estimate_2025'
PF = 'population_for_screening_2025'
BASE_FACTORS = (
    (1, 'health', 'health_facilities_total', 'Health facilities', 'facilities'),
    (2, 'cropland', 'cropland_area_km2', 'Cropland', 'km²'),
    (3, 'rangeland', 'rangeland_area_km2', 'Rangeland', 'km²'),
)
POPULATION_FACTOR = (4, 'population', PF, 'Population 2025', 'people')
FACTOR_COLORS = {1: '#C43C39', 2: '#E69F00', 3: '#00845A', 4: '#7B4AB5'}
FACTOR_TEXT = {1: 'white', 2: '#1E293B', 3: 'white', 4: 'white'}
MAP_CRS = '+proj=laea +lat_0=8 +lon_0=30 +datum=WGS84 +units=m +no_defs'
INK, MUTED = '#1E293B', '#54616F'
RIVER_MINOR = '#365B66'
LABEL_GUIDE = '#8A2BE2'
LABEL_GUIDE_HALO = 'white'

# Display-only QC overrides for known source-data conflicts.
# These do NOT alter the stored upstream value or any hazard ranking.
# Key = (ADM3 name casefolded, parent ADM2 name casefolded, factor code).
FACTOR_DISPLAY_NA_OVERRIDES = {
    ('pagil', 'ayod', 2): (
        'ASAP v04 cropland mask reports zero, but independent local evidence indicates cultivation; '
        'treat the cropland quantity as unavailable for display rather than as confirmed absence.'
    ),
}

def factor_display_override(data: dict, code: int):
    region = str(data.get('region_name', '')).strip().casefold()
    parent = str(data.get('parent_adm2_name', '')).strip().casefold()
    reason = FACTOR_DISPLAY_NA_OVERRIDES.get((region, parent, code))
    return (reason is not None), reason
VIEWS = {
    'recent': {'flag': 'candidate_via_recent_activity', 'metric': H,
               'title': 'Highest recent flood activity', 'stem': '01_top10_high_recent_activity',
               'metric_label': 'Activity\ndays/year', 'short': 'high recent activity'},
    'change': {'flag': 'increasing_flood_and_exposure', 'metric': D,
               'title': 'Largest increase in flood activity', 'stem': '02_top10_flood_change',
               'metric_label': 'Change\ndays/year', 'short': 'high positive change'},
}
# Exact source-status labels, NOT substring guesses. Their daily/spatial meaning
# must still be reviewed in 05a. Unknown labels are excluded and inventoried.
DEFAULT_ALLOWED_STATUS = {
    'observed', 'valid', 'complete', 'verified', 'observed_flood', 'observed_positive',
    'observed_zero', 'confirmed_zero', 'verified_zero', 'observed_no_flood',
    'confirmed_no_flood', 'observed_non_flood',
}
DEFAULT_ZERO_STATUS = {
    'observed_zero', 'confirmed_zero', 'verified_zero', 'observed_no_flood',
    'confirmed_no_flood', 'observed_non_flood',
}
INVALID_STATUS = {'missing', 'no_data', 'nodata', 'unobserved', 'invalid',
                  'cloud', 'cloudy', 'not_observed', 'missing_target',
                  'structural_missing', 'outside_coverage', 'unknown'}


def log(message: str) -> None:
    print(message, flush=True)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    def convert(x):
        if isinstance(x, dict): return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)): return [convert(v) for v in x]
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, (float, np.floating)): return float(x) if math.isfinite(x) else None
        if isinstance(x, Path): return str(x)
        if x is pd.NA: return None
        return x
    path.write_text(json.dumps(convert(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def display_days(value, signed: bool = False) -> str:
    """Round only at display time: 14.5 -> 15, not bankers' rounding."""
    if pd.isna(value) or not np.isfinite(float(value)):
        return 'N/A'
    rounded = Decimal(str(float(value))).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    number = int(rounded)
    return f'{number:+d}' if signed else str(number)


def project_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for parent in (start, *start.parents):
            if (parent / 'processed_data' / 'multivariate_flood_modeling').is_dir():
                return parent
    here = Path(__file__).resolve()
    if here.parent.name == 'multivariate_flood_modeling' and here.parent.parent.name == 'scripts':
        return here.parents[2]
    return Path.cwd()


def resolve_path(path: Path, root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def active_factors(df=None, cutoffs=None):
    enabled = PF in cutoffs if cutoffs is not None else bool(df.population_factor_enabled.iloc[0])
    return BASE_FACTORS + (POPULATION_FACTOR,) if enabled else BASE_FACTORS


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype('string').str.strip().str.upper()


def boolean_values(series: pd.Series, label: str, nullable: bool = False) -> pd.Series:
    """Do not turn the nonempty string 'False' into True."""
    values = series.astype("string").str.strip().str.casefold()
    missing = values.isna() | values.isin(["", "nan", "none", "<na>"])
    valid = values.isin(["true", "false", "1", "0", "1.0", "0.0"]) | missing
    if not valid.all():
        raise ValueError(f"Invalid boolean values in {label}: {values[~valid].unique().tolist()[:5]}")
    result = values.isin(["true", "1", "1.0"]).astype("boolean").mask(missing, pd.NA)
    if not nullable and result.isna().any():
        raise ValueError(f"Unknown boolean values in required column {label}.")
    return result if nullable else result.astype(bool)

def pick_13d(base: Path) -> Path:
    for stem in ("13d_adm3_humanitarian_priority_population", "13d_adm3_humanitarian_priority_screening"):
        for path in (base / stem, base / (stem + ".zip")):
            if path.is_dir() and (path / TABLE).is_file():
                return path
            if path.is_file() and zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    if any(PurePosixPath(n).name == TABLE and "__MACOSX" not in n for n in archive.namelist()):
                        return path
    raise FileNotFoundError(f"No completed 13d output found under {base}. Run 13d first or pass --input PATH.")

def read_scenarios(path: Path, choice: str) -> list[dict]:
    """Read scenario files directly; do not extract ZIP contents or execute code."""
    if path.suffix.lower() == ".csv":
        if path.name != TABLE:
            raise ValueError(f"Use the full 13d table {TABLE}, not the candidate-only table.")
        path = path.parent
    scenarios = []
    if path.is_dir():
        root = path
        if not (root / TABLE).is_file():
            matches = [p.parent for p in root.rglob(TABLE)
                       if "__MACOSX" not in p.parts and "population_sensitivity" not in p.parts]
            if len(matches) != 1:
                raise ValueError(f"Expected one 13d root table under {root}; found {len(matches)}. Use --input exact_folder.")
            root = matches[0]
        candidates = [("main", root), ("population_sensitivity", root / "population_sensitivity")]
        for label, folder in candidates:
            if not (folder / TABLE).is_file():
                continue
            if not (folder / META).is_file():
                raise FileNotFoundError(f"13d methodology/threshold file missing: {folder / META}. Rerun 13d; thresholds will not be guessed.")
            scenarios.append({"label": label, "csv_bytes": (folder / TABLE).read_bytes(),
                              "meta_bytes": (folder / META).read_bytes(), "location": str(folder),
                              "input_csv_path": str(folder / TABLE)})
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if "__MACOSX" not in n and not PurePosixPath(n).name.startswith("._")]
            roots = [n for n in names if PurePosixPath(n).name == TABLE and "population_sensitivity" not in PurePosixPath(n).parts]
            if len(roots) != 1:
                raise ValueError("ZIP must contain exactly one main 13d output. Unzip and use --input exact_folder if ambiguous.")
            prefix = str(PurePosixPath(roots[0]).parent)
            prefix = "" if prefix == "." else prefix + "/"
            for label, sub in (("main", ""), ("population_sensitivity", "population_sensitivity/")):
                table_name, meta_name = prefix + sub + TABLE, prefix + sub + META
                if table_name not in names:
                    continue
                if meta_name not in names:
                    raise ValueError(f"Missing metadata in ZIP: {meta_name}")
                scenarios.append({"label": label, "csv_bytes": archive.read(table_name),
                                  "meta_bytes": archive.read(meta_name), "location": f"{path}::{prefix + sub}",
                                  "input_csv_path": None})
    else:
        raise FileNotFoundError(f"13d input is not a folder, ZIP, or full CSV: {path}")
    if choice == "main":
        scenarios = scenarios[:1]
    elif choice == "population-sensitivity":
        scenarios = [s for s in scenarios if s["label"] == "population_sensitivity"
                     or json.loads(s["meta_bytes"]).get("scenario") == "population_sensitivity"]
        if not scenarios:
            raise ValueError("The input has no population_sensitivity result. Run 13d v4 first; 13e will not create a new scenario.")
    return scenarios

def validate_saved_result(item: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Validate existing flags. Derive only display aliases missing in old 13d files."""
    meta = json.loads(item["meta_bytes"])
    df = pd.read_csv(io.BytesIO(item["csv_bytes"]), dtype={"adm_id": "string", "factor_codes": "string"})
    required = ["adm_id", "region_name", "parent_adm2_name", H, D,
                "high_recent_flood", "high_flood_increase", "priority_review_candidate", "increasing_flood_and_exposure"]
    absent = [c for c in required if c not in df]
    if absent:
        raise ValueError(f"Not a supported full 13d result; missing: {absent}. Run 13d v3/v4 first.")
    df["adm_id"] = normalize_ids(df.adm_id)
    if df.empty or df.adm_id.isna().any() or df.adm_id.eq("").any() or df.adm_id.duplicated().any():
        raise ValueError("13d ADM3 IDs must be nonempty and unique.")
    cutoffs = meta.get("numeric_thresholds")
    if not isinstance(cutoffs, dict):
        raise ValueError("13d numeric_thresholds missing; 13f does not recalculate cutoffs.")
    cutoffs = {k: float(v) for k, v in cutoffs.items() if v is not None}
    enabled = PF in cutoffs
    if "population_factor_enabled" in df:
        flag = boolean_values(df.population_factor_enabled, "population_factor_enabled")
        if flag.nunique() != 1 or bool(flag.iloc[0]) != enabled:
            raise ValueError("Population flag disagrees with the saved 13d thresholds.")
    df["population_factor_enabled"] = enabled
    known_scenario = meta.get("scenario")
    if "analysis_scenario" in df:
        present = df.analysis_scenario.dropna().unique().tolist()
        if len(present) != 1 or (known_scenario and str(present[0]) != known_scenario):
            raise ValueError("Scenario labels in 13d table and metadata disagree.")
        known_scenario = str(present[0])
    if not known_scenario:
        known_scenario = "population_sensitivity" if enabled else "baseline"
    if enabled and known_scenario == "baseline":
        raise ValueError("Population is enabled but the scenario is labelled baseline; review 13d first.")
    df["analysis_scenario"] = known_scenario
    for c in ("high_recent_flood", "high_flood_increase", "priority_review_candidate", "increasing_flood_and_exposure"):
        df[c] = boolean_values(df[c], c)
    for _, short, column, *_ in active_factors(df):
        c = f"high_{short}"
        if c not in df or column not in df or column not in cutoffs:
            raise ValueError(f"Saved factor {short} lacks a value, flag or cutoff. Do not infer factors from a picture.")
        df[c] = boolean_values(df[c], c, nullable=(short == "population"))
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if short != "population" and not np.isfinite(df[column]).all():
            raise ValueError(f"Non-finite input in {column}.")
    for c in (H, D):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if not np.isfinite(df[c]).all() or c not in cutoffs:
            raise ValueError(f"Invalid or undocumented flood metric: {c}")
    if df[H].lt(0).any() or df[H].gt(366).any():
        raise ValueError("Mean annual flood-activity days must be between 0 and 366.")
    if B in df:
        if not np.allclose(df[D], df[H] - pd.to_numeric(df[B]), atol=1e-6, rtol=0):
            raise ValueError("The change column is not the stated difference between period means.")
    flags = [f"high_{f[1]}" for f in active_factors(df)]
    num = df[flags].fillna(False).sum(axis=1).astype(int)
    codes = df.apply(lambda r: "+".join(str(code) for code, short, *_ in active_factors(df)
                                      if pd.notna(r[f"high_{short}"]) and bool(r[f"high_{short}"])), axis=1)
    derived = {"high_exposure_count": num,
               "candidate_via_recent_activity": df.high_recent_flood & num.gt(0),
               "candidate_via_change_only": ~df.high_recent_flood & df.high_flood_increase & num.gt(0)}
    added = []
    for c, val in derived.items():
        if c in df:
            old = pd.to_numeric(df[c]) if c == "high_exposure_count" else boolean_values(df[c], c)
            if not np.array_equal(old.to_numpy(), val.to_numpy()):
                raise ValueError(f"13d column {c} is inconsistent with its saved flags. No automatic reclassification performed.")
        else:
            added.append(c)
        df[c] = val
    if "factor_codes" in df:
        if not df.factor_codes.fillna("").eq(codes).all():
            raise ValueError("Saved factor_codes do not match the saved factor flags.")
    else:
        added.append("factor_codes")
    df["factor_codes"] = codes
    if not df.priority_review_candidate.eq(df.candidate_via_recent_activity | df.candidate_via_change_only).all():
        raise ValueError("Saved 13d candidate flags are inconsistent; rerun 13d rather than changing selection inside 13f.")
    if not df.increasing_flood_and_exposure.eq(df.high_flood_increase & num.gt(0)).all():
        raise ValueError("Saved worsening flags do not cover all high-change+factor cases (an early 13d issue). Run the corrected 13d v3/v4 first; 13e will not silently replace these flags.")
    # Older versions had reference-only population columns but no v4 display fields.
    for c in (P, PF):
        if c not in df:
            df[c] = pd.to_numeric(df[P], errors="coerce") if c == PF and P in df else np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("population_factor_unknown", "selection_undetermined_population"):
        if c in df:
            df[c] = boolean_values(df[c], c)
        elif not enabled:
            df[c] = False
            added.append(c)
        else:
            raise ValueError(f"Population result lacks {c}; use the full v4 13d result.")
    if not enabled:
        df["high_population"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    quantile = float(meta.get("quantile", float("nan")))
    if not 0 < quantile < 1:
        raise ValueError("The 13d quantile is not documented correctly.")
    candidates = sorted(df.loc[df.priority_review_candidate, "adm_id"].tolist())
    worsening = sorted(df.loc[df.increasing_flood_and_exposure, "adm_id"].tolist())
    recent_candidates = sorted(df.loc[df.candidate_via_recent_activity, "adm_id"].tolist())
    for key, actual in (("review_candidates", len(candidates)), ("worsening_candidates", len(worsening))):
        if key in meta and int(meta[key]) != actual:
            raise ValueError(f"13d metadata {key} disagrees with its CSV; stop to avoid mixed runs.")
    audit = {"source": item["location"], "source_csv_sha256": hashlib.sha256(item["csv_bytes"]).hexdigest(),
             "source_metadata_sha256": hashlib.sha256(item["meta_bytes"]).hexdigest(), "rows": len(df),
             "scenario": known_scenario, "population_factor_enabled": enabled,
             "review_candidates": len(candidates), "worsening_candidates": len(worsening),
             "candidate_ids": candidates, "worsening_candidate_ids": worsening,
             "high_recent_candidates": len(recent_candidates), "high_recent_candidate_ids": recent_candidates,
             "added_legacy_display_aliases": added, "screening_thresholds_recalculated": False,
             "original_candidate_flags_changed": False, "original_13d_files_modified": False}
    return df, meta, audit

def polygonal(geom):
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [polygonal(g) for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon", "GeometryCollection")]
        parts = [g for g in parts if g is not None and not g.is_empty]
        return unary_union(parts) if parts else None
    return None

def load_geography(path: Path, results: pd.DataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"ADM3 boundaries not found: {path}\nPass --adm3-geojson PATH.")
    geo = gpd.read_file(path)
    required = ["adm3_pcode", "adm2_pcode", "geometry"]
    if any(c not in geo for c in required):
        raise ValueError(f"ADM3 boundaries require {required}; actual columns: {geo.columns.tolist()}")
    geo["adm_id"] = normalize_ids(geo.adm3_pcode)
    if geo.adm_id.isna().any() or geo.adm_id.eq("").any() or geo.adm_id.duplicated().any():
        raise ValueError("Boundary ADM3 IDs are missing or duplicated.")
    if geo.crs is None:
        raise ValueError("Boundary CRS is missing. The script will not guess it.")
    if geo.geometry.isna().any() or geo.geometry.is_empty.any():
        raise ValueError("Boundary file contains empty or missing geometries.")
    invalid = ~geo.geometry.is_valid
    repaired = geo.loc[invalid, "adm_id"].tolist()
    if invalid.any():
        geo.loc[invalid, "geometry"] = geo.loc[invalid, "geometry"].apply(lambda g: polygonal(make_valid(g)))
        warnings.warn(f"Repaired {len(repaired)} invalid ADM3 geometries for display; see audit JSON.")
    if (geo.geometry.isna().any() or geo.geometry.is_empty.any() or not geo.geometry.is_valid.all()
            or not geo.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Non-polygon or invalid geometry remains after repair.")
    left, right = set(results.adm_id), set(geo.adm_id)
    if left != right:
        raise ValueError(f"PCODE mismatch: {len(left-right)} analysis IDs without polygons {sorted(left-right)[:8]}; "
                         f"{len(right-left)} polygons without analysis {sorted(right-left)[:8]}. No rows were dropped.")
    source_crs = str(geo.crs)
    name_mismatches = 0
    if "adm3_name" in geo:
        names = geo.set_index("adm_id").adm3_name.reindex(results.adm_id).astype("string").str.strip().str.casefold()
        actual = results.set_index("adm_id").region_name.astype("string").str.strip().str.casefold()
        name_mismatches = int(names.ne(actual).fillna(True).sum())
    keep = [c for c in ("adm_id", "adm3_pcode", "adm2_pcode", "adm1_pcode", "adm1_name", "geometry") if c in geo]
    overlapping = [c for c in keep if c != "adm_id" and c in results]
    merged = geo[keep].merge(results.drop(columns=overlapping), on="adm_id", how="left", validate="one_to_one")
    projected = merged.to_crs(MAP_CRS)
    if not np.isfinite(projected.total_bounds).all():
        raise ValueError("Non-finite projected map bounds.")
    return projected, {
        "boundary_source": str(path), "boundary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "boundary_crs": source_crs, "map_crs": MAP_CRS,
        "boundary_rows": len(geo), "matched_rows": len(merged),
        "name_mismatches_with_matching_pcode": name_mismatches,
        "repaired_geometry_ids": repaired,
    }


def choose_scenario(base: Path, supplied: Path | None, choice: str):
    source = supplied if supplied is not None else pick_13d(base)
    items = read_scenarios(source, 'all')
    ready = [(item, *validate_saved_result(item)) for item in items]
    if choice == 'main':
        return ready[0]
    if choice == 'population-sensitivity':
        candidates = [r for r in ready if r[3]['scenario'] == 'population_sensitivity']
    else:
        candidates = [r for r in ready if r[3]['population_factor_enabled']]
        candidates.sort(key=lambda r: r[3]['scenario'] == 'population_sensitivity')
    if not candidates:
        raise ValueError('No saved four-factor result found. Run 13d v4 first, or deliberately '
                         'use --scenario main for a three-factor baseline. 13f does not invent population flags.')
    return candidates[0]


def rank_view(df: pd.DataFrame, view: str, top_n: int) -> tuple[pd.DataFrame, dict]:
    """Rank hazard first across ALL ADM3; exposure factors do not affect selection."""
    spec = VIEWS[view]
    candidate = df.copy()
    if view == 'change':
        candidate = candidate.loc[candidate[D].gt(0)].copy()
    candidate = candidate.sort_values([spec['metric'], 'adm_id'], ascending=[False, True], kind='stable')
    selected = candidate.head(top_n).copy().reset_index(drop=True)
    selected.insert(0, 'rank', np.arange(1, len(selected) + 1))
    selected.insert(0, 'map_letter', list(string.ascii_uppercase[:len(selected)]))
    selected['ranking_metric'] = spec['metric']
    selected['ranking_value'] = selected[spec['metric']]
    selected['ranking_display'] = selected.ranking_value.map(lambda v: display_days(v, view == 'change'))
    selected['view'] = view
    cutoff_ties = 0
    if len(selected):
        cutoff_ties = int(candidate[spec['metric']].eq(selected.iloc[-1][spec['metric']]).sum())
    return selected, {
        'view': view, 'eligible_candidates': len(candidate), 'selected_count': len(selected),
        'ranking_metric': spec['metric'], 'descending': True,
        'selection_basis': 'ALL ADM3 hazard metric; exposure factors do not filter or weight ranking',
        'tie_break': 'ADM3 PCODE ascending; ranking uses unrounded values',
        'number_tied_at_last_selected_metric': cutoff_ties,
        'selected_ids_in_rank_order': selected.adm_id.tolist(),
        'not_a_validated_humanitarian_need_ranking': True,
    }


def read_monthly_bytes(path: Path) -> tuple[bytes, str]:
    if path.is_dir():
        matches = [p for p in path.rglob(MONTH_TABLE) if '__MACOSX' not in p.parts]
        if len(matches) != 1:
            raise ValueError(f'Expected one {MONTH_TABLE} in {path}, found {len(matches)}.')
        return matches[0].read_bytes(), str(matches[0])
    if path.is_file() and path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if PurePosixPath(n).name == MONTH_TABLE
                     and '__MACOSX' not in PurePosixPath(n).parts]
            if len(names) != 1:
                raise ValueError(f'Expected one {MONTH_TABLE} in {path}, found {len(names)}.')
            return z.read(names[0]), f'{path}::{names[0]}'
    if path.is_file() and path.suffix.lower() == '.csv':
        return path.read_bytes(), str(path)
    raise FileNotFoundError(f'Monthly input must be a CSV, directory or ZIP: {path}')


def locate_monthly(base: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f'Explicit monthly input does not exist: {explicit}')
        return explicit
    for p in (base / '05a_monthly_panel' / MONTH_TABLE,
              base / '05a_monthly_panel.zip', base / MONTH_TABLE):
        if p.exists():
            return p
    # Restrict fallback discovery to monthly-panel filenames, not forecast tables.
    matches = [p for p in base.rglob(MONTH_TABLE) if '__MACOSX' not in p.parts
               and not any(part.startswith('13f_') for part in p.parts)]
    if len(matches) > 1:
        raise ValueError('Several monthly panels found. Use --monthly PATH to select the original 05a panel:\n'
                         + '\n'.join(map(str, matches)))
    return matches[0] if matches else None


def column_alias(df: pd.DataFrame, names: tuple[str, ...], required: bool = True):
    columns = {str(c).casefold(): c for c in df.columns}
    for n in names:
        if n.casefold() in columns:
            return columns[n.casefold()]
    if required:
        raise ValueError(f'Missing monthly column; expected one of {names}. Found {list(df.columns)}')
    return None


def status_key(value) -> str:
    if pd.isna(value):
        return ''
    return re.sub(r'[\s-]+', '_', str(value).strip().casefold())


def load_monthly_review(path: Path | None, checksum: str) -> dict:
    review = {} if path is None else json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(review, dict):
        raise ValueError('Monthly review must be a JSON object.')
    expected = review.get('source_csv_sha256')
    if expected and expected != checksum:
        raise ValueError('Monthly review belongs to a different CSV (SHA256 mismatch).')
    for key in ('additional_allowed_status_values', 'additional_zero_valid_status_values'):
        if key in review and not isinstance(review[key], list):
            raise ValueError(f'{key} must be a list of exact status strings.')
    if any(review.get(k) for k in ('additional_allowed_status_values',
                                  'additional_zero_valid_status_values', 'all_source_zeros_reviewed')):
        if not review.get('status_meanings_reviewed') or not str(review.get('review_notes', '')).strip():
            raise ValueError('Status/zero overrides need status_meanings_reviewed=true and substantive review_notes. '
                             'Review the source code; do not enable these just to fill the table.')
    return review


def monthly_profiles(path: Path | None, adm_ids: list[str], args, audit_dir: Path):
    """Audit 05a counts before computing equal-year same-calendar-month means.

    Common years across all 12 months avoid comparing different years as seasons.
    The reviewed event-date status has an explicit descriptive-record pathway.
    Other unresolved statuses remain blocked; zeros are not dropped from means.
    Missing/invalid rows are NOT filled; invalid counts stop execution.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(args.start_year, args.end_year + 1))
    summary_columns = ['adm_id', 'month', 'month_name', 'mean_days', 'median_days', 'min_days',
                       'max_days', 'std_days', 'n_years', 'included_years', 'mean_days_display',
                       'eligible_for_peak', 'profile_status']
    if path is None:
        if args.require_monthly:
            raise FileNotFoundError(f'{MONTH_TABLE} not found. Pass --monthly PATH.')
        profiles = {i: {'status': 'monthly_input_missing', 'n_years': 0, 'peak_text': 'N/A',
                        'peak_months': [], 'included_years': []} for i in adm_ids}
        report = {'source': None, 'status': 'monthly_input_missing', 'display_is_placeholder_not_zero': True}
        write_json(audit_dir / '04_monthly_quality.json', report)
        log('[MONTHLY] Source not found: the figures will show N/A, not invented monthly days.')
        return pd.DataFrame(columns=summary_columns), profiles, report
    content, location = read_monthly_bytes(path)
    checksum = sha256_bytes(content)
    review = load_monthly_review(args.monthly_review, checksum)
    raw = pd.read_csv(io.BytesIO(content), low_memory=False)
    idcol = column_alias(raw, ('adm_id', 'adm3_id', 'adm3_pcode'))
    yearcol = column_alias(raw, ('year',))
    monthcol = column_alias(raw, ('month',))
    daycol = column_alias(raw, ('flood_days', 'recorded_flood_days', 'recorded_activity_days', 'event_date_proxy'))
    statuscol = column_alias(raw, ('target_status',), required=False)
    raw['__id'] = normalize_ids(raw[idcol])
    if raw['__id'].isna().any() or raw['__id'].eq('').any():
        raise ValueError('Monthly panel contains empty ADM3 IDs.')
    raw['__year'] = pd.to_numeric(raw[yearcol], errors='coerce')
    raw['__month'] = pd.to_numeric(raw[monthcol], errors='coerce')
    if raw[['__year', '__month']].isna().any().any():
        raise ValueError('Monthly panel has missing/non-numeric year or month.')
    if (raw.__year % 1 != 0).any() or (raw.__month % 1 != 0).any() or not raw.__month.between(1, 12).all():
        raise ValueError('Year and month must be integers; month must be 1..12.')
    raw = raw.loc[raw.__year.between(args.start_year, args.end_year) & raw.__id.isin(adm_ids)].copy()
    df = pd.DataFrame({'adm_id': raw.__id, 'year': raw.__year.astype(int), 'month': raw.__month.astype(int),
                       'raw_days': pd.to_numeric(raw[daycol], errors='coerce')})
    supplied_non_numeric = raw[daycol].notna() & df.raw_days.isna()
    if supplied_non_numeric.any():
        raise ValueError('Monthly flood_days contains non-numeric nonmissing text.')
    if df.duplicated(['adm_id', 'year', 'month']).any():
        duplicates = df[df.duplicated(['adm_id', 'year', 'month'], keep=False)].head(12)
        raise ValueError('Duplicate ADM3/year/month rows. Do not sum overlapping flood types.\n' + duplicates.to_string(index=False))
    df['calendar_days'] = [calendar.monthrange(y, m)[1] for y, m in zip(df.year, df.month)]
    finite = df.raw_days.notna()
    bad = finite & (~np.isfinite(df.raw_days) | df.raw_days.lt(0) | df.raw_days.gt(df.calendar_days)
                    | (df.raw_days - df.raw_days.round()).abs().gt(1e-7))
    if bad.any():
        df.loc[bad].to_csv(audit_dir / 'INVALID_MONTHLY_COUNTS.csv', index=False)
        raise ValueError('Invalid monthly day counts: must be integer counts within the actual calendar month. '
                         f'See {audit_dir / "INVALID_MONTHLY_COUNTS.csv"}. No clamping or silent correction applied.')
    df['target_status'] = raw[statuscol].map(status_key) if statuscol else ''
    observed_col = args.observed_days_column or review.get('observed_days_column')
    # Hydrology coverage does not measure observation coverage of flood masks.
    if observed_col and any(token in observed_col.casefold() for token in ('rainfall', 'runoff', 'et0', 'hydrology')):
        raise ValueError('A rainfall/runoff/ET0 coverage column cannot be used as a flood-observation denominator.')
    if observed_col and observed_col not in raw:
        raise ValueError(f'Reviewed flood-observation denominator column absent: {observed_col}')
    df['observed_days'] = pd.to_numeric(raw[observed_col], errors='coerce') if observed_col else np.nan
    if observed_col:
        invalid_obs = (df.observed_days.notna() & (~np.isfinite(df.observed_days) | df.observed_days.lt(0)
                       | df.observed_days.gt(df.calendar_days) | df.observed_days.lt(df.raw_days)
                       | (df.observed_days - df.observed_days.round()).abs().gt(1e-7)))
        if invalid_obs.any():
            raise ValueError('Invalid flood observation-day denominator or activity days > observed days.')
    full_observation = df.observed_days.notna() & df.observed_days.eq(df.calendar_days)
    mode = getattr(args, 'monthly_mode', 'auto')
    if mode not in ('auto', 'strict'):
        raise ValueError('monthly_mode must be auto or strict.')
    allowed = DEFAULT_ALLOWED_STATUS | {status_key(v) for v in review.get('additional_allowed_status_values', [])}
    zero_states = DEFAULT_ZERO_STATUS | {status_key(v) for v in review.get('additional_zero_valid_status_values', [])}
    if PROXY_STATUS in zero_states:
        raise ValueError('The event-date proxy is not confirmed zero flooding. Use --monthly-mode auto instead.')
    allowed |= zero_states
    if allowed & INVALID_STATUS:
        raise ValueError('Known missing/invalid statuses cannot be promoted to usable.')
    proxy_source = df.target_status.eq(PROXY_STATUS)
    proxy_enabled = proxy_source & (mode == 'auto')
    contradictory_zero = df.target_status.isin(zero_states) & df.raw_days.gt(0)
    if contradictory_zero.any():
        raise ValueError('A zero/no-flood status has a positive count; review the source.')
    known_accepted = (df.target_status.isin(allowed) | df.target_status.eq('')) & ~proxy_source
    df['row_reason'] = 'accepted_record_count'
    df.loc[df.raw_days.isna(), 'row_reason'] = 'missing_count'
    df.loc[df.raw_days.notna() & df.target_status.isin(INVALID_STATUS), 'row_reason'] = 'status_missing_or_invalid'
    unknown_status = df.raw_days.notna() & ~known_accepted & ~proxy_source & ~df.target_status.isin(INVALID_STATUS)
    df.loc[unknown_status, 'row_reason'] = 'status_meaning_unreviewed'
    zero_reviewed = df.target_status.isin(zero_states) | full_observation
    if review.get('all_source_zeros_reviewed'):
        zero_reviewed = pd.Series(True, index=df.index)
    unknown_zero = df.raw_days.eq(0) & ~zero_reviewed & ~proxy_source & df.row_reason.eq('accepted_record_count')
    df.loc[unknown_zero, 'row_reason'] = 'zero_meaning_unreviewed'
    if observed_col:
        df.loc[~proxy_source & df.row_reason.eq('accepted_record_count') & ~full_observation,
               'row_reason'] = 'incomplete_observation_month'
    # Existing zero counts are legitimate zeros OF RECORDED EVENTS. They are
    # never promoted to verified absence of water. No NaN or absent row is filled.
    df.loc[proxy_enabled & df.raw_days.notna(), 'row_reason'] = 'accepted_event_record_proxy'
    df.loc[proxy_source & ~proxy_enabled & df.raw_days.notna(), 'row_reason'] = 'proxy_disabled_in_strict_mode'
    df['row_usable'] = df.row_reason.isin(['accepted_record_count', 'accepted_event_record_proxy'])
    df['usable_days'] = df.raw_days.where(df.row_usable)
    df['measure_kind'] = np.where(proxy_source, 'event_date_proxy', 'source_record_count')
    df['zero_interpretation'] = np.where(proxy_source & df.raw_days.eq(0),
        'no_event_records_not_confirmed_dry', np.where(df.raw_days.eq(0),
        'source_zero_requires_own_review', 'not_a_zero_value'))
    df['proxy_status_recognized'] = proxy_source
    df.to_csv(audit_dir / '01_monthly_row_audit.csv', index=False)
    df.groupby(['target_status', 'row_reason'], dropna=False).size().rename('rows').reset_index().to_csv(
        audit_dir / '02_status_inventory.csv', index=False)
    summaries, profiles, profile_rows, per_month_rows = [], {}, [], []
    blocked_reasons = {'status_meaning_unreviewed', 'zero_meaning_unreviewed', 'proxy_disabled_in_strict_mode'}
    for aid in adm_ids:
        part = df.loc[df.adm_id.eq(aid)].copy()
        index = pd.MultiIndex.from_product([years, range(1, 13)], names=['year', 'month'])
        values = part.set_index(['year', 'month']).usable_days.reindex(index)
        grid = values.unstack('month')
        complete_years = grid.index[grid.notna().all(axis=1)].astype(int).tolist()
        unresolved = part.row_reason.isin(blocked_reasons).any()
        kinds = sorted(part.loc[part.row_usable, 'measure_kind'].unique().tolist())
        mixed = len(kinds) > 1
        measure = kinds[0] if len(kinds) == 1 else ('mixed' if mixed else 'unavailable')
        enough = len(complete_years) >= args.min_years
        state = ('mixed_monthly_measure_definitions' if mixed else
                 'source_status_or_zero_unreviewed' if unresolved else
                 'insufficient_common_years' if not enough else
                 'eligible_event_date_proxy' if measure == 'event_date_proxy' else 'eligible_recorded_proxy')
        common = grid.loc[complete_years]
        profile_months = []
        # Single-month values remain available even when the cross-month peak
        # comparison fails. They are NOT substituted into peak ranking.
        for month in range(1, 13):
            local = grid[month].dropna()
            per_month_rows.append({
                'adm_id': aid, 'month': month, 'month_name': calendar.month_abbr[month],
                'available_years': len(local), 'included_years': ';'.join(map(str, local.index)),
                'mean_record_count': float(local.mean()) if len(local) and not mixed else np.nan,
                'median_record_count': float(local.median()) if len(local) and not mixed else np.nan,
                'table_year_fraction': len(local) / len(years), 'measure_kind': measure,
                'used_directly_for_peak_ranking': False,
                'daily_spatial_flood_observation_coverage': 'not_established',
            })
            vals = common[month].dropna()
            mean = float(vals.mean()) if len(vals) else np.nan
            eligible = not unresolved and not mixed and enough and np.isfinite(mean)
            record = {'adm_id': aid, 'month': month, 'month_name': calendar.month_abbr[month],
                      'mean_days': mean, 'median_days': float(vals.median()) if len(vals) else np.nan,
                      'min_days': float(vals.min()) if len(vals) else np.nan,
                      'max_days': float(vals.max()) if len(vals) else np.nan,
                      'std_days': float(vals.std(ddof=1)) if len(vals) >= 2 else np.nan,
                      'n_years': len(vals), 'included_years': ';'.join(map(str, complete_years)),
                      'mean_days_display': display_days(mean) if eligible else 'N/A',
                      'eligible_for_peak': eligible, 'profile_status': state,
                      'measure_kind': measure,
                      'zero_record_years': int(vals.eq(0).sum()),
                      'available_month_values_before_common_year_filter': int(grid[month].notna().sum())}
            summaries.append(record)
            if eligible and mean > 0:
                profile_months.append(record)
        profile_months.sort(key=lambda r: (-r['mean_days'], r['month']))
        peaks = profile_months[:args.peak_months]
        peak_text = '\n'.join(f"{r['month_name']}: {r['mean_days_display']}" for r in peaks)
        if state.startswith('eligible') and not peaks:
            peak_text = 'No records'
            state = 'eligible_all_zero_records'
        elif not peaks:
            peak_text = 'N/A'
        profiles[aid] = {'status': state, 'n_years': len(complete_years), 'peak_text': peak_text,
                         'peak_months': peaks, 'included_years': complete_years, 'measure_kind': measure}
        profile_rows.append({
            'adm_id': aid, 'profile_status': state, 'n_common_years': len(complete_years),
            'common_years': ';'.join(map(str, complete_years)), 'expected_years': len(years),
            'expected_year_month_rows': len(index), 'present_year_month_rows': len(part),
            'missing_year_month_rows': len(index) - len(part),
            'table_row_fraction': len(part)/len(index),
            'numeric_month_count': int(part.raw_days.notna().sum()),
            'numeric_month_fraction': float(part.raw_days.notna().sum()/len(index)),
            'unresolved_status_rows': int(part.row_reason.eq('status_meaning_unreviewed').sum()),
            'unreviewed_zero_rows': int(part.row_reason.eq('zero_meaning_unreviewed').sum()),
            'proxy_zero_record_rows': int((part.target_status.eq(PROXY_STATUS) & part.raw_days.eq(0)).sum()),
            'daily_spatial_flood_observation_coverage': 'not_established',
            'measure_kind': measure,
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(audit_dir / '03_adm3_calendar_month_statistics.csv', index=False)
    pd.DataFrame(profile_rows).to_csv(audit_dir / '05_profile_coverage.csv', index=False)
    pd.DataFrame(per_month_rows).to_csv(audit_dir / '06_per_month_available_statistics.csv', index=False)
    report = {
        'source': location, 'source_csv_sha256': checksum, 'years': years,
        'rows_in_analysis_period_and_ids': len(df), 'status_column': statuscol,
        'observed_days_column': observed_col, 'monthly_mode': mode,
        'recognized_proxy_status': PROXY_STATUS,
        'proxy_rows': int(proxy_source.sum()),
        'accepted_proxy_rows': int(df.row_reason.eq('accepted_event_record_proxy').sum()),
        'proxy_zero_rows_included': int((df.row_reason.eq('accepted_event_record_proxy') & df.raw_days.eq(0)).sum()),
        'min_common_years': args.min_years, 'complete_calendar_months_required_per_year': 12,
        'common_years_across_all_months': True, 'all_missing_or_invalid_rows_filled_zero': False,
        'mean_type': 'Equal-weight arithmetic mean of same-calendar-month recorded date counts over common table years',
        'display_rounding': 'ROUND_HALF_UP integer; full precision retained in CSV and ranking',
        'unknown_statuses': sorted(df.loc[unknown_status, 'target_status'].unique().tolist()),
        'unreviewed_zero_rows': int(unknown_zero.sum()),
        'eligible_profiles': sum(p['status'].startswith('eligible') for p in profiles.values()),
        'total_profiles': len(adm_ids), 'review_configuration': review,
        'observation_semantics_independently_verified_by_this_script': False,
        'flood_observation_coverage': 'not_established',
        'source_definition_reviewed_file': '05a_build_monthly_panel_UPDATED.py',
        'reviewed_05a_sha256': REVIEWED_05A_SHA256,
        'cautions': [
            'The recognized proxy is a count of distinct dates with event records somewhere inside an ADM3.',
            'A source zero means no event records; it is not evidence of a verified flood-free month.',
            'Table completeness is not daily/spatial satellite observation completeness.',
            'Rainfall/runoff/ET0 valid fractions cannot establish flood-mask coverage.',
            'Missing rows and NaN remain missing; zero counts are not discarded to inflate positive-only means.',
            'Common-year filtering is retained for cross-month comparability; independent monthly statistics are also exported.',
            'These are peaks of the recorded-date proxy, not verified peak inundation area or continuous flood duration.',
            'No original flood Parquet files or observation masks were independently audited in this script.',
        ],
    }
    write_json(audit_dir / '04_monthly_quality.json', report)
    log(f"[MONTHLY] {report['eligible_profiles']}/{len(adm_ids)} profiles have usable recorded-date summaries.")
    if report['accepted_proxy_rows']:
        log(f"[PROXY] Included {report['accepted_proxy_rows']} record counts, including "
            f"{report['proxy_zero_rows_included']} zero-record months. This does NOT validate absence of flooding.")
    if report['unknown_statuses']:
        log('[MONTHLY] Unreviewed statuses still excluded: ' + ', '.join(report['unknown_statuses'][:12]))
    return summary, profiles, report


def attach_profiles(table: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    out = table.copy()
    out['monthly_measure_kind'] = out.adm_id.map(lambda i: profiles[i].get('measure_kind', 'unavailable'))
    out['monthly_profile_status'] = out.adm_id.map(lambda i: profiles[i]['status'])
    out['seasonality_n_years'] = out.adm_id.map(lambda i: profiles[i]['n_years'])
    out['seasonality_years'] = out.adm_id.map(lambda i: ';'.join(map(str, profiles[i]['included_years'])))
    out['peak_months_integer_display'] = out.adm_id.map(lambda i: profiles[i]['peak_text'])
    for j in range(3):
        for target, source in [('month', 'month'), ('mean_days', 'mean_days'), ('mean_days_display', 'mean_days_display')]:
            out[f'peak_{j+1}_{target}'] = out.adm_id.map(
                lambda i, j=j, source=source: profiles[i]['peak_months'][j][source]
                if j < len(profiles[i]['peak_months']) else pd.NA)
    return out


def polygon_path(geometry) -> MplPath:
    """Respect exterior rings and holes for a country-only rendering clip."""
    polygons = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
    vertices, codes = [], []
    for poly in polygons:
        if poly.geom_type != 'Polygon':
            continue
        poly = orient(poly, sign=1.0)
        for ring in (poly.exterior, *poly.interiors):
            points = np.asarray(ring.coords)[:, :2]
            vertices.extend(points)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 2) + [MplPath.CLOSEPOLY])
    if not vertices:
        raise ValueError('No polygon rings available for country clipping.')
    return MplPath(np.asarray(vertices), np.asarray(codes, dtype=np.uint8))


def load_saved_rivers(base: Path, explicit: Path | None, geo, country):
    """Reuse the small, documented 13e display export, not continental raw data."""
    if explicit is not None:
        candidates = [explicit]
        if not explicit.is_file():
            raise FileNotFoundError(f'Saved 13e river layer does not exist: {explicit}')
    else:
        candidates = sorted(base.glob('13e*/river_data/01_rivers_display.gpkg'),
                            key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
    if not candidates:
        log('[RIVERS] No saved 13e river layer found. Maps will disclose that no river overlay is available.')
        return None, {'source': None, 'available': False, 'legend': []}
    chosen = candidates[0]
    log(f'[RIVERS] Reusing saved display layer: {chosen}')
    river = gpd.read_file(chosen)
    if river.crs is None:
        raise ValueError('Saved river layer CRS missing; it will not be guessed.')
    river = river.loc[river.geometry.notna() & ~river.geometry.is_empty].copy().to_crs(geo.crs)
    if not river.geom_type.isin(['LineString', 'MultiLineString']).all():
        raise ValueError('River layer must contain line geometry.')
    if not river.geometry.is_valid.all():
        raise ValueError('Saved river layer has invalid geometry. Repair/re-export it in 13e first.')
    for col in ('is_major', 'named_colour'):
        if col not in river:
            river[col] = False
        else:
            river[col] = boolean_values(river[col], col)
    if 'display_name' not in river:
        river['display_name'] = river['river_name'].fillna('') if 'river_name' in river else ''
    if 'line_color' not in river:
        river['line_color'] = RIVER_MINOR
    river['display_name'] = river.display_name.fillna('').astype(str)
    if not river.line_color.map(is_color_like).all():
        raise ValueError('Invalid saved river colour value; re-export from 13e.')
    # Prepared predicates only; do not repeat expensive full-country intersections.
    shapely.prepare(country)
    keep = shapely.intersects(country, river.geometry.to_numpy())
    river = river.loc[keep].copy()
    named = river.loc[river.named_colour & river.display_name.ne('')]
    legend = []
    for name, part in named.groupby('display_name', sort=True):
        colours = part.line_color.unique()
        if len(colours) != 1:
            raise ValueError(f'Conflicting saved colours for river {name}.')
        legend.append({'name': name, 'color': str(colours[0])})
    return river, {'source': str(chosen), 'source_sha256': file_hash(chosen), 'available': True,
                   'line_records_kept': len(river), 'legend': legend,
                   'only_country_visible': True, 'names_inferred': False,
                   'other_candidate_exports': [str(p) for p in candidates[1:]],
                   'choice_rule': 'explicit input or most recently modified saved 13e display export',
                   'legend_note': 'Uses saved source/reference geometry; not inferred tributary identities.'}


def draw_rivers(ax, river, country_path):
    if river is None or river.empty:
        return
    clip = PathPatch(country_path, transform=ax.transData, facecolor='none', edgecolor='none')
    def layer(part, colour, width, order):
        if part.empty:
            return
        start = len(ax.collections)
        part.plot(ax=ax, color=colour, linewidth=width, zorder=order, alpha=1.0)
        for collection in ax.collections[start:]:
            collection.set_clip_path(clip)
    minor = river.loc[~river.is_major]
    unnamed = river.loc[river.is_major & ~river.named_colour]
    named = river.loc[river.named_colour]
    layer(minor, 'white', 1.05, 3.05)
    layer(minor, RIVER_MINOR, .55, 3.1)
    layer(unnamed, 'white', 1.5, 3.15)
    layer(unnamed, '#78838C', .85, 3.2)
    layer(named, 'white', 3.7, 3.25)
    layer(named, '#39434C', 2.55, 3.3)
    layer(named, named.line_color.tolist(), 2.3, 3.35)


def guides(ax, crs) -> None:
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    dx, dy = xmax - xmin, ymax - ymin
    to_lonlat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    from_lonlat = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    geod = Geod(ellps='WGS84')
    x, y = xmin + .07 * dx, ymin + .065 * dy
    lon, lat = to_lonlat.transform(x, y)
    lon2, lat2, _ = geod.fwd(lon, lat, 90, 200000)
    end, endy = from_lonlat.transform(lon2, lat2)
    ax.plot([x, end], [y, endy], lw=1.4, color=INK, zorder=6)
    for xx, yy in [(x, y), (end, endy)]:
        ax.plot([xx, xx], [yy - .006 * dy, yy + .006 * dy], lw=1.2, color=INK, zorder=6)
    ax.text((x + end) / 2, min(y, endy) - .022 * dy, '200 km', ha='center', va='top', size=9)
    nx, ny = xmin + .94 * dx, ymin + .86 * dy
    lon, lat = to_lonlat.transform(nx, ny)
    nlon, nlat, _ = geod.fwd(lon, lat, 0, 65000)
    ex, ey = from_lonlat.transform(nlon, nlat)
    ax.annotate('', xy=(ex, ey), xytext=(nx, ny),
                arrowprops={'arrowstyle': '-|>', 'color': INK, 'lw': 1.4}, zorder=6)
    ax.text(ex, ey + .014 * dy, 'N', ha='center', size=11, weight='bold', zorder=6)


def rect_overlap(a, b) -> float:
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def letter_labels(fig, ax, selected_geo, diameter: float) -> pd.DataFrame:
    """Deterministic collision-aware A-J labels. Leaders remain tied to ADM3 points."""
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.patches import Circle
    from matplotlib.text import Text
    columns = ['map_letter', 'adm_id', 'anchor_x', 'anchor_y', 'offset_x_points',
               'offset_y_points', 'leader_line', 'estimated_overlap', 'anchor_inside_adm3']
    if selected_geo.empty:
        return pd.DataFrame(columns=columns)
    selected = selected_geo.sort_values('rank').reset_index(drop=True)
    points = selected.geometry.representative_point()
    fig.canvas.draw()
    anchors = ax.transData.transform(np.column_stack([points.x, points.y]))
    px = fig.dpi / 72.
    box = ax.get_window_extent()
    distances = np.sqrt(((anchors[:, None] - anchors[None, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances.min(axis=1), kind='stable')
    candidates = [(0., 0.)]
    for radius in (24, 38, 56, 76, 100, 130, 165, 205):
        for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            candidates.append((radius * math.cos(angle), radius * math.sin(angle)))
    occupied = []
    obstacles = [(box.x0, box.y0, box.x0 + .34 * box.width, box.y0 + .12 * box.height),
                 (box.x0 + .86 * box.width, box.y0 + .81 * box.height, box.x1, box.y1)]
    half = (diameter / 2 + 3) * px
    output = []
    for i in order:
        best = None
        for ox, oy in candidates:
            centre = anchors[i] + np.array([ox, oy]) * px
            rect = (centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half)
            outside = (max(0., box.x0 - rect[0]) + max(0., rect[2] - box.x1)
                       + max(0., box.y0 - rect[1]) + max(0., rect[3] - box.y1))
            overlap = sum(rect_overlap(rect, other) for other in occupied + obstacles)
            covered_others = sum(rect[0] <= a[0] <= rect[2] and rect[1] <= a[1] <= rect[3]
                                 for j, a in enumerate(anchors) if j != i)
            score = outside * 1e8 + overlap * 1e5 + covered_others * 1e4 + ox*ox + oy*oy
            if best is None or score < best[0]:
                best = (score, ox, oy, rect, outside + overlap)
            if outside == 0 and overlap == 0 and covered_others == 0:
                break
        _, ox, oy, rect, collision = best
        occupied.append(rect)
        point, row = points.iloc[i], selected.iloc[i]
        moved = math.hypot(ox, oy) > 1
        if moved:
            anchor_display = anchors[i]
            displacement = np.array([ox, oy]) * px
            norm = np.linalg.norm(displacement)
            endpoint_display = anchor_display + displacement * max(0., (norm - diameter/2*px)/norm)
            endpoint = ax.transData.inverted().transform(endpoint_display)
            # High-contrast guide system: white halo below a purple leader line
            # and purple anchor point so the intended ADM3 remains easy to trace
            # even on dark blue/red polygons and dense river networks.
            ax.plot([point.x, endpoint[0]], [point.y, endpoint[1]], color=LABEL_GUIDE_HALO, lw=2.9, zorder=7)
            ax.plot([point.x, endpoint[0]], [point.y, endpoint[1]], color=LABEL_GUIDE, lw=1.35, zorder=7.1)
            ax.plot(point.x, point.y, marker='o', markersize=5.0, markerfacecolor=LABEL_GUIDE_HALO,
                    markeredgecolor=LABEL_GUIDE_HALO, zorder=7.15)
            ax.plot(point.x, point.y, marker='o', markersize=3.2, markerfacecolor=LABEL_GUIDE,
                    markeredgecolor=LABEL_GUIDE, zorder=7.2)
        drawing = DrawingArea(diameter, diameter, 0, 0)
        drawing.add_artist(Circle((diameter/2, diameter/2), diameter/2 - .7,
                                  facecolor='white', edgecolor=INK, linewidth=1.2))
        drawing.add_artist(Text(diameter/2, diameter/2, row.map_letter, ha='center', va='center',
                                fontsize=diameter*.57, fontweight='bold', color=INK))
        badge = AnnotationBbox(drawing, (point.x, point.y), xybox=(ox, oy),
                               xycoords='data', boxcoords='offset points', frameon=False, pad=0, zorder=8)
        ax.add_artist(badge)
        output.append({'map_letter': row.map_letter, 'adm_id': row.adm_id,
                       'anchor_x': point.x, 'anchor_y': point.y,
                       'offset_x_points': ox, 'offset_y_points': oy, 'leader_line': moved,
                       'estimated_overlap': bool(collision > 0),
                       'anchor_inside_adm3': bool(row.geometry.covers(point))})
    return pd.DataFrame(output, columns=columns).sort_values('map_letter')



def factor_value_display(code: int, value) -> str:
    """Format an ADM3 total, never a count of affected people/assets.

    Facilities must be a count. Population estimates are rounded HALF UP with
    thousands separators. Area totals use one decimal, retaining tiny positives
    as '<0.1' rather than suggesting zero. Original numbers are not modified.
    """
    if code not in (1, 2, 3, 4):
        raise ValueError(f'Unknown factor code: {code}')
    if pd.isna(value):
        return 'N/A'
    number = float(value)
    if not math.isfinite(number):
        return 'N/A'
    if number < 0:
        raise ValueError(f'Factor {code}: a negative ADM3 total cannot be displayed.')
    if code == 1:
        if not math.isclose(number, round(number), abs_tol=1e-8, rel_tol=0):
            raise ValueError('Health facilities must be an integer count; review the saved input.')
        return f'{int(round(number)):,}'
    precision = '1' if code == 4 else '0.1'
    rounded = Decimal(str(number)).quantize(Decimal(precision), rounding=ROUND_HALF_UP)
    if code == 4:
        return f'{int(rounded):,}'
    if 0 < number < 0.1:
        return '<0.1'
    return f'{rounded:,.1f}'


def factor_value_with_unit(code: int, value) -> str:
    """Display an ADM3 quantity together with its unit; never alter raw data.

    Numeric-only formatting remains in factor_value_display for CSV compatibility.
    Missing quantities remain 'N/A', not '0 people' or a qualifying badge.
    """
    number_text = factor_value_display(code, value)
    if number_text == 'N/A':
        return number_text
    if code == 1:
        unit = 'facility' if int(round(float(value))) == 1 else 'facilities'
    elif code in (2, 3):
        unit = 'km²'
    else:
        unit = 'person' if number_text == '1' else 'people'
    return f'{number_text} {unit}'


def row_factor_entries(row) -> list[dict]:
    """Return only factors that should be visible in the thresholded figure.

    Display rules:
    - threshold-qualified factor -> show saved ADM3 value;
    - known QC/data-quality conflict -> show N/A even when the factor does not qualify;
    - population explicitly unavailable -> show N/A;
    - otherwise below-threshold factors remain blank.

    These are display rules only. Saved thresholds, flags, source values and hazard
    rankings are never changed.
    """
    data = row._asdict() if hasattr(row, '_asdict') else dict(row)
    columns = {1: 'health_facilities_total', 2: 'cropland_area_km2',
               3: 'rangeland_area_km2', 4: PF}
    shorts = {1: 'health', 2: 'cropland', 3: 'rangeland', 4: 'population'}

    population_enabled = bool(data.get('population_factor_enabled', False))
    population_na_flag = data.get('population_factor_unknown', False)
    population_na = pd.notna(population_na_flag) and bool(population_na_flag)

    entries = []
    for code in (1, 2, 3, 4):
        if code == 4 and not population_enabled:
            continue

        short = shorts[code]
        flag = data.get(f'high_{short}', pd.NA)
        qualifies = pd.notna(flag) and bool(flag)
        qc_na, qc_reason = factor_display_override(data, code)
        explicit_na = code == 4 and population_na

        # Below-threshold, reliable values are intentionally blank in 13h.
        if not qualifies and not qc_na and not explicit_na:
            continue

        if explicit_na and qualifies:
            raise ValueError('A saved population N/A cannot also have a qualifying factor 4.')

        value = data.get(columns[code], np.nan)
        if qc_na or explicit_na:
            displayed = 'N/A'
            known = False
            shown_value = None
        else:
            displayed = factor_value_with_unit(code, value)
            if displayed == 'N/A':
                raise ValueError(f'Qualifying factor {code} lacks a finite total; review the saved input.')
            known = True
            shown_value = float(value)

        entries.append({
            'code': code,
            'value': shown_value,
            'display': displayed,
            'qualifies': qualifies,
            'known': known,
            'qc_display_na_override': qc_na,
            'qc_reason': qc_reason,
            'explicit_na': explicit_na,
        })
    return entries

def factor_quantity_table(tables: dict[str, pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    """Audit both shown and unshown totals without changing factor flags."""
    records = []
    for view, table in tables.items():
        for row in table.itertuples():
            data = row._asdict()
            for code, short, column, label, unit in active_factors(df):
                flag = data.get(f'high_{short}', pd.NA)
                qualifies = pd.notna(flag) and bool(flag)
                unknown = code == 4 and bool(data.get('population_factor_unknown', False))
                value = data.get(column, np.nan)
                if unknown:
                    value = np.nan
                qc_na, qc_reason = factor_display_override(data, code)
                display = 'N/A' if qc_na else factor_value_display(code, value)
                display_with_unit = 'N/A' if qc_na else factor_value_with_unit(code, value)
                records.append({
                    'view': view, 'rank': row.rank, 'map_letter': row.map_letter,
                    'adm_id': row.adm_id, 'region_name': row.region_name,
                    'parent_adm2_name': row.parent_adm2_name,
                    'factor_number': code, 'factor_name': label, 'source_column': column,
                    'unit': 'count' if code == 1 else unit,
                    'adm3_total_unrounded': value, 'display_value': display,
                    'display_text_with_unit': display_with_unit,
                    'qc_display_na_override': qc_na, 'qc_override_reason': qc_reason,
                    'qualifies_saved_threshold': qualifies,
                    'quantity_na': display == 'N/A',
                    'displayed_in_figure': qualifies or qc_na or unknown,
                    'population_scenario': row.analysis_scenario,
                })
    return pd.DataFrame(records)


def render_factor_values(ax, row, y: float, xs: list[float], row_step: float) -> None:
    """Use the existing Factors column; compact stacks fit inside each row."""
    entries = row_factor_entries(row)
    if not entries:
        return
    badge_x, value_x = .812, .840
    # Normalize spacing to the existing row height: four entries remain within
    # a 44-point row at the default physical figure size.
    spacing = row_step / 4.08
    total = len(entries)
    badge_size = 7.2 if total >= 4 else 8.0
    value_size = 8.3 if total >= 4 else 9.0
    for j, entry in enumerate(entries):
        yy = y + ((total - 1) / 2 - j) * spacing
        if entry['known']:
            artist = factor_circle(ax, badge_x, yy, entry['code'], size=badge_size,
                                   transform=ax.transAxes, pad=.12)
            artist.set_gid(f"factor-badge-{row.map_letter}-{entry['code']}")
        else:
            # No purple filled circle for missing population: N/A is not
            # membership of the high-population category.
            artist = ax.text(badge_x, yy, str(entry['code']), ha='center', va='center',
                             fontsize=badge_size, color=MUTED, transform=ax.transAxes)
            artist.set_gid(f"factor-unknown-{row.map_letter}-{entry['code']}")
        value_artist = ax.text(value_x, yy, entry['display'], ha='left', va='center',
                               fontsize=value_size, color=INK if entry['known'] else MUTED,
                               transform=ax.transAxes)
        value_artist.set_gid(f"factor-value-{row.map_letter}-{entry['code']}")
        # Guard unusually large totals from overflowing the existing column.
        renderer = ax.figure.canvas.get_renderer()
        max_width = (.995 - value_x) * ax.get_window_extent(renderer).width
        measured = value_artist.get_window_extent(renderer).width
        if measured > max_width:
            value_artist.set_fontsize(max(6.5, value_size * max_width / measured))


def table_explanatory_notes(ax, start_year: int, end_year: int) -> None:
    """Approved notes, once under the table; preserve n = 7 in individual rows."""
    n_note = (f'n = number of years used to compute monthly mean flood-activity days '
              f'({start_year}\u2013{end_year}).')
    factor_note = ('Only threshold-qualified factors are shown. N/A marks a known data-quality or availability issue; '
                   'blank means the factor did not meet its saved threshold.')
    # Explicit wrapping is visual only; the two sentences remain unchanged.
    n_artist = ax.text(0, .047, n_note, fontsize=8.15, color=MUTED, va='top',
                       transform=ax.transAxes)
    n_artist.set_gid('monthly-years-explanation')
    wrapped = factor_note.replace(', \u2462 rangeland', ',\n\u2462 rangeland')
    f_artist = ax.text(0, .021, wrapped, fontsize=8.15, color=MUTED, va='top',
                       linespacing=1.18, transform=ax.transAxes)
    f_artist.set_gid('factor-totals-explanation')


def factor_circle(parent, x, y, code, size=10, transform=None, pad=.24):
    return parent.text(x, y, str(code), ha='center', va='center', fontsize=size, weight='bold',
                       color=FACTOR_TEXT[code], transform=transform or parent.transFigure,
                       bbox={'boxstyle': f'circle,pad={pad}', 'fc': FACTOR_COLORS[code], 'ec': 'white', 'lw': 1.2})


def factor_legend(fig, df, cutoffs):
    positions = [(0.057, .137), (.345, .137), (.057, .089), (.345, .089)]
    for (x, y), (code, _, column, name, unit) in zip(positions, active_factors(df)):
        factor_circle(fig, x, y, code, size=11)
        threshold = cutoffs[column]
        # Asset thresholds are not monthly day values: preserve decimal precision where needed.
        number = f'{threshold:,.0f}' if float(threshold).is_integer() else f'{threshold:,.2f}'
        comparison = '>' if threshold == 0 else '≥'
        fig.text(x + .018, y + .007, name, size=11, weight='bold', va='top', color=INK)
        fig.text(x + .018, y - .013, f'{comparison} {number} {unit} per ADM3', size=9, va='top', color=MUTED)


def render_table(ax, table, view, profiles, groups, start_year, end_year):
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, .99, f'TOP {len(table)} ADM3', size=14, weight='bold', va='top', color=INK)
    ax.text(0, .945, 'Regions ranked by the flood metric across all ADM3', size=9, color=MUTED, va='top')
    xs = [.03, .105, .455, .595, .88]
    labels = ['ID', 'ADM3\nParent ADM2', VIEWS[view]['metric_label'], 'Peak months\nMean record days', 'Threshold-qualified\nfactors only']
    for x, label in zip(xs, labels):
        ax.text(x, .884, label, size=10, weight='bold', va='center',
                ha='left' if x in (xs[1], xs[3]) else 'center', color=INK)
    ax.plot([0, 1], [.85, .85], color='#A5B0BC', lw=1)
    step = .067
    for k, row in enumerate(table.itertuples()):
        y = .811 - k * step
        if k % 2 == 0:
            ax.add_patch(Rectangle((0, y - step / 2), 1, step, facecolor='#F3F6F8', edgecolor='none', zorder=-1))
        ax.text(xs[0], y, row.map_letter, ha='center', va='center', fontsize=12, weight='bold', color=INK)
        name = textwrap.fill(str(row.region_name), width=21, break_long_words=False)
        parent = textwrap.fill(str(row.parent_adm2_name), width=23, break_long_words=False)
        text = name + '\n(' + parent + ')'
        lines = text.count('\n') + 1
        fs = 10.5 if lines <= 3 else max(8.2, 10.5 * 3 / lines)
        ax.text(xs[1], y, text, ha='left', va='center', fontsize=fs, color=INK, linespacing=1.13)
        ax.text(xs[2], y + .007, row.ranking_display, ha='center', va='center', fontsize=12, color=INK)
        info = profiles[row.adm_id]
        if info['status'].startswith('eligible'):
            ax.text(xs[3], y, info['peak_text'], ha='left', va='center', fontsize=10.8, color=INK, linespacing=1.2)
            ax.text(xs[2], y - .016, f"n = {info['n_years']}", fontsize=8, color=MUTED, ha='center', va='center')
        else:
            ax.text(xs[3], y + .007, 'N/A', fontsize=11, color=INK, va='center')
            reason = {'monthly_input_missing': 'Month table missing',
                      'source_status_or_zero_unreviewed': 'Status / zero review',
                      'insufficient_common_years': 'Too few common years',
                      'mixed_monthly_measure_definitions': 'Mixed count meanings'}.get(info['status'], 'See monthly audit')
            ax.text(xs[3], y - .013, reason, fontsize=7.8, color=MUTED, va='center')
        render_factor_values(ax, row, y, xs, step)
    bottom = .811 - max(0, len(table) - 1) * step - .05
    ax.plot([0, 1], [bottom, bottom], color='#A5B0BC', lw=.8)
    grouptext = '; '.join(f'{r.parent_adm2_name} ({r.selected_adm3})' for r in groups.itertuples())
    ax.text(0, bottom - .025, f'{len(table)} ADM3 across {len(groups)} ADM2', size=10, weight='bold', va='top', color=INK)
    ax.text(0, bottom - .053, textwrap.fill(grouptext, width=66), size=8.5, va='top', color=MUTED, linespacing=1.25)
    table_explanatory_notes(ax, start_year, end_year)


def river_legend(fig, metadata):
    ax = fig.add_axes([.655, .070, .323, .145])
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1, 'RIVERS', fontsize=11, weight='bold', va='top', color=INK)
    if not metadata['available']:
        ax.text(0, .68, 'River overlay unavailable in this run.', fontsize=9.5, color=MUTED)
        return
    entries = [(r['name'], r['color'], 2.3) for r in metadata['legend']]
    entries += [('Other shown reaches', RIVER_MINOR, .55)]
    # Two columns; source only supplies at most 12 named rivers in 13e.
    nrows = max(1, math.ceil(len(entries)/2))
    step = min(.22, .73 / nrows)
    for i, (name, colour, width) in enumerate(entries):
        col, row = divmod(i, nrows)
        x, y = col*.51, .76 - row*step
        ax.plot([x+.005, x+.105], [y, y], color='white', lw=width+1.1, solid_capstyle='round')
        ax.plot([x+.005, x+.105], [y, y], color='#39434C', lw=width+.3, solid_capstyle='round')
        ax.plot([x+.005, x+.105], [y, y], color=colour, lw=width, solid_capstyle='round')
        ax.text(x+.13, y, name, fontsize=9.5 if nrows <= 5 else 8.5, va='center', color=INK)
    ax.text(0, .015, 'Source: saved river display layer; only in-country lines are visible.', size=8, color=MUTED)


def draw_figure(geo, country, adm2, adm1, rivers, river_meta, table, groups, profiles,
                df, metadata, view, directory: Path, args):
    spec = VIEWS[view]
    metric = spec['metric']
    limit_h = max(1, math.ceil(float(df[H].max()) / 25) * 25)
    limit_d = max(1, math.ceil(float(df[D].abs().max()) / 20) * 20)
    norm = Normalize(0, limit_h) if view == 'recent' else TwoSlopeNorm(vmin=-limit_d, vcenter=0, vmax=limit_d)
    cmap = 'Blues' if view == 'recent' else 'RdBu_r'
    fig = plt.figure(figsize=(25, 14.5), dpi=110, facecolor='white')
    title = f"{spec['title']}: Top {len(table)} ADM3 with threshold-qualified factors"
    fig.text(.035, .94, title, fontsize=24, weight='bold', color=INK)
    subtitle = ('Mean annual recorded flood-activity days, 2019–2025' if view == 'recent'
                else 'Difference between period means: 2019–2025 minus 2000–2018')
    fig.text(.035, .907, subtitle, fontsize=12, color=MUTED)
    scenario = str(df.analysis_scenario.iloc[0])
    scenario_text = ('PROVISIONAL POPULATION SENSITIVITY' if scenario == 'population_sensitivity'
                     else 'POPULATION PROCESSING CHECKED' if bool(df.population_factor_enabled.iloc[0])
                     else 'THREE-FACTOR BASELINE — POPULATION EXCLUDED')
    fig.text(.98, .975, scenario_text, ha='right', size=9.5,
             color='#9A3412' if scenario == 'population_sensitivity' else MUTED, weight='bold')
    ax = fig.add_axes([.025, .245, .615, .63])
    table_ax = fig.add_axes([.655, .245, .323, .63])
    geo.plot(column=metric, cmap=cmap, norm=norm, edgecolor='#F7F9FA', linewidth=.22, ax=ax, zorder=1)
    if adm2 is not None:
        adm2.boundary.plot(ax=ax, color='#7B8792', linewidth=.35, zorder=2)
    if adm1 is not None:
        adm1.boundary.plot(ax=ax, color='#475767', linewidth=.62, zorder=2.2)
    draw_rivers(ax, rivers, polygon_path(country))
    selected_geo = geo.set_index('adm_id').loc[table.adm_id].reset_index()
    for c in ('rank', 'map_letter'):
        selected_geo[c] = table.set_index('adm_id').loc[selected_geo.adm_id, c].to_numpy()
    selected_geo.boundary.plot(ax=ax, color='white', linewidth=2.4, zorder=4)
    selected_geo.boundary.plot(ax=ax, color=INK, linewidth=1.35, zorder=4.2)
    gpd.GeoSeries([country], crs=geo.crs).boundary.plot(ax=ax, color=INK, linewidth=.9, zorder=4.4)
    xmin, ymin, xmax, ymax = geo.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - .07*dx, xmax + .07*dx); ax.set_ylim(ymin - .12*dy, ymax + .075*dy)
    ax.set_aspect('equal'); ax.set_axis_off()
    guides(ax, geo.crs)
    cax = fig.add_axes([.095, .208, .48, .015])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    colorbar.ax.tick_params(labelsize=9)
    caption = ('Mean annual recorded activity (days/year)' if view == 'recent'
               else 'Difference in mean annual recorded activity (days/year)')
    fig.text(.335, .234, caption, ha='center', size=11, color=INK)
    render_table(table_ax, table, view, profiles, groups, args.start_year, args.end_year)
    factor_legend(fig, df, metadata['numeric_thresholds'])
    river_legend(fig, river_meta)
    fig.add_artist(Line2D([.643, .643], [.067, .878], transform=fig.transFigure, color='#CED6DF', lw=.9))
    fig.text(.035, .041, 'A–J identify ADM3 regions. Coloured digits identify exposure factors, not severity or rank.', fontsize=9, color=MUTED)
    proxy_used = any(p.get('measure_kind') == 'event_date_proxy' for p in profiles.values())
    footnote = ('Monthly values show days with recorded flood activity, not continuous flooding duration. Zero means no event records; observation coverage is unverified.'
                if proxy_used else 'Historical screening, not a forecast or actual losses. Monthly data status and observation limits are documented in the figure caption.')
    fig.text(.035, .022, footnote, fontsize=9, color=MUTED)
    log(f"[LABELS] {view}: placing {len(selected_geo)} labels with inside-region anchors / leader lines")
    labels = letter_labels(fig, ax, selected_geo, args.label_diameter)
    labels.to_csv(directory / f'{spec["stem"]}_label_audit.csv', index=False)
    if len(labels) != len(table) or not labels.anchor_inside_adm3.all():
        raise ValueError('Top-10 labels do not match selected regions or anchors lie outside their ADM3.')
    collisions = int(labels.estimated_overlap.sum())
    if collisions:
        warnings.warn(f'{collisions} estimated label collisions in {view}. Inspect local output or lower --label-diameter.')
    outputs = []
    for extension in args.formats:
        name = spec['stem'] + '.' + extension
        fig.savefig(directory / name, dpi=args.dpi, facecolor='white')
        outputs.append(name)
        log(f'[SAVED] {directory / name}')
    plt.close(fig)
    return {'files': outputs, 'labels_drawn': len(labels), 'estimated_label_overlaps': collisions,
            'map_colour_limits': [0, limit_h] if view == 'recent' else [-limit_d, limit_d],
            'figure_inches': [25, 14.5], 'map_axes_width_fraction': .615,
            'selection_ids': table.adm_id.tolist(), 'river_names_on_map': False,
            'population_factor_in_table': bool(df.population_factor_enabled.iloc[0])}


def adm2_summary(table, geo):
    codes = geo.set_index('adm_id').adm2_pcode
    out = table.copy()
    out['parent_adm2_pcode'] = out.adm_id.map(codes)
    rows = []
    for (code, name), part in out.groupby(['parent_adm2_pcode', 'parent_adm2_name'], dropna=False):
        rows.append({'view': str(part.view.iloc[0]), 'parent_adm2_pcode': code, 'parent_adm2_name': name,
                     'selected_adm3': len(part), 'map_letters': ', '.join(part.map_letter),
                     'adm3_names': '; '.join(part.region_name), 'adm3_ids': ';'.join(part.adm_id)})
    return pd.DataFrame(rows).sort_values(['selected_adm3', 'parent_adm2_pcode'], ascending=[False, True])


def write_captions(out, reports, df, monthly_report, rank_reports, args):
    text = ['# Figure captions and interpretation', '']
    for view in ('recent', 'change'):
        r = rank_reports[view]
        text += [f'## {VIEWS[view]["title"]}',
                 f'The map shows the national distribution of {VIEWS[view]["metric"]} and highlights '
                 f'the {r["selected_count"]} highest unrounded values within the {r["eligible_candidates"]} '
                 'ADM3 areas nationwide, ranked by the hazard metric without exposure filtering. '
                 'Equal metric values are ordered by ADM3 PCODE; this tie-break is administrative, not substantive. '
                 'A–J link the mapped polygons to the table. This is not a validated ranking of humanitarian need.', '']
    text += ['## Seasonal information',
             f'Displayed monthly values are equal-year arithmetic means of recorded activity days in the same '
             f'calendar month during {args.start_year}–{args.end_year}, rounded half up to whole days only for display. '
             f'A minimum of {args.min_years} common complete accepted years across all 12 months is required. '
             'This minimum is a project screening rule, not a statistical guarantee of reliability. '
             'Means, medians, variability, included years and original row decisions are exported to CSV. '
             'The highest three positive monthly means are shown. Ties use calendar order. '
             'For EXPLORATORY_EVENT_DATE_PROXY, source zeros mean absence of event records and are included in record-count averages, not relabelled as flood-free. Missing months remain missing. '
             'Other unresolved status/zero semantics continue to withhold peaks. Table-year availability does not establish daily or spatial observation coverage. Per-month summaries remain separate from common-year peak comparisons. '
             'Available years can differ between regions; compare with the audit, particularly for a changing regime. '
             'No record-date statistic proves continuous flooding, full-area inundation, loss, or adequate daily/spatial coverage.', '',
             '## Factors and population',
             '1: health facilities; 2: cropland; 3: rangeland; 4: modelled 2025 population. '
             'Saved 13d factor thresholds are reused exactly. Threshold-qualified factors are displayed; known QC/data-quality conflicts are shown as N/A even when they do not qualify. '
             'The table displays saved ADM3 totals beside qualifying coloured badges; below-threshold reliable factors remain blank. '
             'N/A means the selected source does not provide a reliable display value and must not be interpreted as zero. '
             'Facility counts and modelled population are displayed as integers with thousands separators; '
             'cropland and rangeland areas use one decimal. Units are printed after each displayed quantity: facilities, km² or people. Display rounding does not alter selection. '
             'These are administrative totals, not affected population or losses. Known source-data conflicts may be displayed as N/A while preserving the raw upstream value in the audit CSV. '

             f'Population status: {df.analysis_scenario.iloc[0]}. '
             'Population N/A is not treated as zero or below threshold. A 2025 snapshot does not reconstruct historical exposure.', '',
             '## Rivers and geography',
             'River lines reuse the saved 13e display dataset. Names are not assigned by proximity. '
             'All river strokes are visually clipped to the matched ADM3 national outline. '
             'The display export may itself contain generalized reference lines; this is not a network-flow or causal analysis.', '',
             '## Data availability', json.dumps(monthly_report, ensure_ascii=False, indent=2), '']
    (out / '08_figure_captions.md').write_text('\n'.join(text), encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, help='Project root; normally discovered automatically')
    p.add_argument('--input', type=Path, help='13d output folder, ZIP or full CSV with adjacent methodology JSON')
    p.add_argument('--scenario', choices=['auto', 'main', 'population-sensitivity'], default='auto',
                   help='Default: saved four-factor scenario, retaining provisional status')
    p.add_argument('--monthly', type=Path, help='Original 05a ADM3 monthly CSV, folder or ZIP')
    p.add_argument('--monthly-mode', choices=['auto', 'strict'], default='auto',
                   help='auto: recognize reviewed event-date record counts; strict: retain prior blocking of proxy data')
    p.add_argument('--monthly-review', type=Path, help='Optional reviewed status/zero-meaning configuration JSON')
    p.add_argument('--observed-days-column', help='Optional REVIEWED flood-observation denominator; never use ERA5 available days')
    p.add_argument('--require-monthly', action='store_true', help='Stop if no monthly panel is present instead of drawing N/A')
    p.add_argument('--start-year', type=int, default=2019)
    p.add_argument('--end-year', type=int, default=2025)
    p.add_argument('--min-years', type=int, default=5, help='Minimum common complete accepted years across all 12 months (default 5/7)')
    p.add_argument('--peak-months', type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--top-n', type=int, default=10, choices=range(1, 11))
    p.add_argument('--adm3-geojson', type=Path, help='Default raw_data/administrative_boundaries/ssd_admin3.geojson')
    p.add_argument('--rivers', type=Path, help='Saved 13e river_data/01_rivers_display.gpkg, not raw continental HydroRIVERS')
    p.add_argument('--no-rivers', action='store_true', help='Explicitly omit river overlay')
    p.add_argument('--out', type=Path, help='Default processed_data/multivariate_flood_modeling/13h_hazard_first_thresholded_factors_v4_qc_visible_na')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--formats', nargs='+', choices=['png', 'svg'], default=['png', 'svg'])
    p.add_argument('--label-diameter', type=float, default=22., help='A-J badge diameter in points')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1900 <= args.start_year <= args.end_year <= 2100:
        raise ValueError('Invalid seasonal year range.')
    if not 1 <= args.min_years <= args.end_year - args.start_year + 1:
        raise ValueError('--min-years must fit inside the selected seasonal period.')
    if not 72 <= args.dpi <= 600 or not 14 <= args.label_diameter <= 36:
        raise ValueError('--dpi must be 72..600; --label-diameter must be 14..36 points.')
    args.formats = list(dict.fromkeys(args.formats))
    root = args.root.expanduser().resolve() if args.root else project_root()
    base = root / 'processed_data' / 'multivariate_flood_modeling'
    for attr in ('input', 'monthly', 'monthly_review', 'adm3_geojson', 'rivers'):
        value = getattr(args, attr)
        if value is not None:
            setattr(args, attr, resolve_path(value, root))
    out = resolve_path(args.out, root) if args.out else base / '13h_hazard_first_thresholded_factors_v4_qc_visible_na'
    # Refuse to put derived output in the same directory as an input scenario.
    if out.name.startswith(('13d_', '13e_', '05a_')):
        raise ValueError('Use a separate 13h output folder; do not overwrite prior analysis outputs.')
    start = time.perf_counter()
    log(f'[START] {VERSION}\n[ROOT] {root}')
    item, df, metadata, preservation = choose_scenario(base, args.input, args.scenario)
    log(f"[13D] {preservation['scenario']}; {len(df)} ADM3; factor 4 enabled={preservation['population_factor_enabled']}")
    out.mkdir(parents=True, exist_ok=True)
    maps = out / 'maps'; maps.mkdir(exist_ok=True)
    tables, rank_reports = {}, {}
    for view in VIEWS:
        tables[view], rank_reports[view] = rank_view(df, view, args.top_n)
        if tables[view].empty:
            raise ValueError(f'No saved candidates in view {view}; no invented Top 10 will be drawn.')
        log(f'[RANK] {view}: {len(tables[view])} of {rank_reports[view]["eligible_candidates"]} candidates')
    selected_ids = sorted(set(tables['recent'].adm_id) | set(tables['change'].adm_id))
    monthly_path = locate_monthly(base, args.monthly)
    monthly_stats, profiles, monthly_report = monthly_profiles(monthly_path, selected_ids, args, out / 'monthly_audit')
    for view in tables:
        tables[view] = attach_profiles(tables[view], profiles)
    boundary = args.adm3_geojson or root / 'raw_data' / 'administrative_boundaries' / 'ssd_admin3.geojson'
    log('[GEOMETRY] Validate ADM3 codes and project country geometry')
    geo, geography_report = load_geography(boundary, df)
    log('[GEOMETRY] Build parent/national outlines once')
    country = unary_union(geo.geometry)
    adm2 = geo[['adm2_pcode', 'geometry']].dissolve(by='adm2_pcode').reset_index()
    adm1 = (geo[['adm1_pcode', 'geometry']].dissolve(by='adm1_pcode').reset_index()
            if 'adm1_pcode' in geo and geo.adm1_pcode.notna().all() else None)
    if args.no_rivers:
        rivers, river_report = None, {'available': False, 'source': None, 'legend': [], 'explicitly_omitted': True}
    else:
        rivers, river_report = load_saved_rivers(base, args.rivers, geo, country)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'axes.unicode_minus': True, 'savefig.facecolor': 'white'})
    draw_reports, groupframes = {}, []
    for view in tables:
        table = tables[view]
        groups = adm2_summary(table, geo)
        groupframes.append(groups)
        table.to_csv(out / (VIEWS[view]['stem'] + '.csv'), index=False)
        log(f'[DRAW] {view}: large map / letter labels / Top-{len(table)} table')
        draw_reports[view] = draw_figure(geo, country, adm2, adm1, rivers, river_report, table, groups,
                                         profiles, df, metadata, view, maps, args)
    pd.concat(groupframes, ignore_index=True).to_csv(out / '03_parent_adm2_summary.csv', index=False)
    union = df.loc[df.adm_id.isin(selected_ids), ['adm_id', 'region_name', 'parent_adm2_name', H, D, 'factor_codes']].copy()
    for view in tables:
        rank = tables[view].set_index('adm_id')
        union[view + '_rank'] = union.adm_id.map(rank['rank']).astype('Int64')
        union[view + '_letter'] = union.adm_id.map(rank['map_letter'])
    union.to_csv(out / '04_top10_membership_comparison.csv', index=False)
    factor_rows = []
    for view, table in tables.items():
        for code, short, column, label, unit in active_factors(df):
            factor_rows.append({'view': view, 'factor_number': code, 'factor_name': label,
                                'selected_adm3': len(table), 'qualifying_selected_adm3': int(table[f'high_{short}'].fillna(False).sum()),
                                'na_selected_adm3': int(table[f'high_{short}'].isna().sum()),
                                'saved_cutoff': metadata['numeric_thresholds'][column], 'unit': unit})
    pd.DataFrame(factor_rows).to_csv(out / '05_top10_factor_counts.csv', index=False)
    factor_quantity_table(tables, df).to_csv(out / '09_top10_factor_quantities.csv', index=False)
    write_json(out / '06_river_layer_audit.json', river_report)
    report = {'version': VERSION, 'run_time_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.perf_counter() - start, 'saved_13d': preservation,
              'geography': geography_report, 'ranking': rank_reports, 'monthly': monthly_report,
              'figures': draw_reports, 'rivers': river_report, 'script_sha256': file_hash(Path(__file__)),
              'upstream_files_written': False, 'screening_thresholds_recalculated': False,
              'ranking_rounded_before_selection': False, 'seasonality_used_to_change_ranking': False,
              'factor_quantity_display': {
                  'source': 'saved ADM3 totals; qualifying factors shown, with known QC/data-quality conflicts shown as N/A',
                  'facility_count_decimals': 0, 'population_display_decimals': 0,
                  'area_display_decimals': 1, 'thousands_separator': True,
                  'units_after_every_known_value': True,
                  'inline_units': {'1': 'facilities', '2': 'km²', '3': 'km²', '4': 'people'},
                  'na_population_fallback_to_raw_estimate': False,
                  'rounding': 'display only, ROUND_HALF_UP; ranking and thresholds unchanged',
                  'quantities_are_confirmed_impacts': False,
                  'audit_csv': '09_top10_factor_quantities.csv',
              },
              'scope': 'Hazard-first Top 10 across all ADM3; saved threshold-qualified factors are displayed, with known QC/data-quality conflicts shown as N/A'}
    write_json(out / '07_methodology_and_quality.json', report)
    write_captions(out, draw_reports, df, monthly_report, rank_reports, args)
    log(f'[OK] Complete: {out}')
    log('[NOTE] Monthly values are mean recorded-date counts, displayed as integers; they show recorded flood activity, not continuous flooding duration. See monthly_audit for table years and source limitations.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n[STOPPED] Interrupted by the user; incomplete output is not a successful run.', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, FileNotFoundError, KeyError, OSError, zipfile.BadZipFile) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        raise SystemExit(1)
