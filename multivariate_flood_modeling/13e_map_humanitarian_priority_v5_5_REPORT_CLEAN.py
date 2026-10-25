#!/usr/bin/env python3
"""13e | Render saved 13d screening results with REAL river-network overlays.

Save under scripts/multivariate_flood_modeling/ and Run; no required CLI arguments.
This standalone script does NOT import or execute 13d, change its candidate flags,
recalculate screening thresholds, validate population, or infer river causality.

Input: 13d population output (preferred) or earlier 13d screening output, plus
ssd_admin3.geojson and a LOCAL river dataset, including HydroRIVERS FileGDB.
Default display: the EXISTING four-factor 13d scenario (including population 4)
inside the top-level maps/ folder. A sensitivity scenario REMAINS visibly
PROVISIONAL; its flags and thresholds are not changed or newly validated.
Use --include-baseline to also draw the three-factor comparison separately.
Factor badges use FIXED categorical colours across maps, the lower legend and the
sidebar: 1 health (red), 2 cropland (amber), 3 rangeland (green), 4 population (purple).
The digits remain visible; colours identify factors, NOT their magnitude or rank.
Factor 4 is shown only when the saved 13d scenario enables population. No new
population threshold or verified status is assigned by this visual update.

River emphasis is a CARTOGRAPHIC rule, not a calibrated risk class:
ORD_STRA >= 6 by default, with named main rivers also emphasized when actual names
exist. Small headwaters (ORD_STRA < 3) are omitted by default to reduce clutter.
No river name is inferred from coordinates. Optional reviewed HYRIV_ID/name CSV.

Version 5 completes the paired exposure views without changing saved analysis:
    08: recent activity background (2019-2025), no factors;
    09: difference between period means, no factors;
    10: existing UNION of high-activity and high-change candidates + factors;
    11: high-positive-change candidates + factors;
    12: NEW high-recent-activity candidates ONLY + factors (no change-only cases).
The same saved factor codes 1-4, thresholds and population status are retained.
Map-specific tables, label audits, population subtotals and sidebars use the same
selection mask. The two focused sets may overlap; do not add their totals.
These are historical activity screens, not live disaster areas or loss estimates.

Named major rivers keep the high-contrast colours and separate river-name key.
Other displayed reaches use opaque dark teal-grey lines with NARROW white casings.
Version 5.1 changes display style only: minor-width 0.60 pt, casing 0.50 pt wider
(total casing width 1.10 pt); both are opaque and above flood fills. Main rivers
and factor-circle colours are unchanged. These reaches are NOT automatically
classified as tributaries: colour/width alone does not establish flow topology.
Map filtering is for display, NOT for drainage-density calculations.
No lakes, wetlands, terrain or seasonal-rainfall layer is added here; those are
reserved for 13f with real input data. Original 13d and earlier 13e files are not
changed. The new default output is 13e_humanitarian_maps_report_clean/.
If the local river network has no names, auto mode downloads the small official
Natural Earth 1:10 million named-reference layer (~2 MB), caches it locally and
plots its OWN generalized lines. It does NOT transfer names onto HydroRIVERS by
nearest distance. Use --named-rivers PATH for a local reference; --offline prevents
downloads. No local user data are sent. Missing names are never invented.
Default run requires a four-factor saved scenario and usable named river lines;
otherwise it stops with instructions rather than silently returning old maps.

Report-layout correction 5.2.1: import Rectangle for the river-key frame;
check index uniqueness explicitly without the deprecated set_index keyword.
River labels remain off on the map by default; the right-side key retains
names, line samples and source notes. Factor colours and saved flags do not change.

Report edition 5.5:
- Figure text is self-contained: no script numbers, internal cross-map references,
  or "reading this map" instructions. Periods, units, thresholds, source types,
  and provisional population status remain visible.
- Longer methodological explanations and ready-to-edit captions are exported to
  11_figure_captions.md beside the tables, not printed across the map.
- Country-only river geometry is the default: exact national mask, ZERO outward
  mask buffer. Any read-window padding is only for safe data loading.
- Prepared predicates retain interior reaches and remove exterior reaches; only
  boundary-crossing reaches need expensive intersection. Work is batched/cached.
- New cache schema prevents reuse of the old rectangular/buffered-country cache.
- Input analysis, colour normalization, factor flags and thresholds are unchanged.

Primary documentation:
https://docs.python.org/3/library/exceptions.html#KeyboardInterrupt
https://geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html
https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html
https://www.hydrosheds.org/products/hydrorivers
https://data.hydrosheds.org/file/technical-documentation/HydroRIVERS_TechDoc_v10.pdf
https://geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html
https://pyogrio.readthedocs.io/en/latest/introduction.html
https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-rivers-lake-centerlines/
"""
from __future__ import annotations

import argparse
import colorsys
from types import SimpleNamespace
import hashlib
import io
import json
import math
import os
import re
import sys
import textwrap
import unicodedata
import warnings
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

try:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import pyogrio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.text import Text
    from pyproj import Geod, Transformer
    from shapely.geometry import box, MultiLineString
    import shapely
    from shapely.prepared import prep
    from shapely.validation import make_valid
    from shapely.ops import unary_union, linemerge
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\nInstall in the active environment:\n"
        "python -m pip install numpy pandas matplotlib geopandas shapely pyproj pyogrio"
    ) from exc

VERSION = "13e-5.5-report-clean-country-only"
TABLE = "01_adm3_screening_full.csv"
META = "07_quality_and_methodology.json"
H = "historical_flood_days_mean_2019_2025"
D = "historical_flood_days_change_after_2019"
B = "mean_annual_flood_days_2000_2018"
P = "population_estimate_2025"
PF = "population_for_screening_2025"
BASE_FACTORS = (
    (1, "health", "health_facilities_total", "Health facilities", "facilities"),
    (2, "cropland", "cropland_area_km2", "Cropland", "km²"),
    (3, "rangeland", "rangeland_area_km2", "Rangeland", "km²"),
)
POPULATION_FACTOR = (4, "population", PF, "Population 2025", "people")
MAP_CRS = "+proj=laea +lat_0=8 +lon_0=30 +datum=WGS84 +units=m +no_defs"
INK = "#1E293B"
MUTED = "#54616F"
# Fixed factor identity: do not renumber these when a scenario omits population.
# Change these dictionaries to customize the palette; all badge locations share it.
FACTOR_COLORS = {
    1: "#C43C39",  # Health facilities: red
    2: "#E69F00",  # Cropland: amber/orange
    3: "#00845A",  # Rangeland: green
    4: "#7B4AB5",  # Estimated population 2025: purple
}
FACTOR_TEXT_COLORS = {1: "#FFFFFF", 2: "#1E293B", 3: "#FFFFFF", 4: "#FFFFFF"}
FACTOR_COLOR_NAMES = {1: "red", 2: "amber/orange", 3: "green", 4: "purple"}
RIVER_MAJOR = "#00638C"
RIVER_MINOR = "#365B66"  # Dark teal-grey OTHER reaches; not verified tributaries.
MINOR_ALPHA = 1.0
MINOR_CASING_COLOR = "#FFFFFF"
MINOR_CASING_ZORDER = 5.25
MINOR_CORE_ZORDER = 5.30
NE_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_rivers_lake_centerlines.zip"
KNOWN_MAIN_NAMES = (
    "White Nile", "Nile", "Sobat", "Bahr el Ghazal", "Bahr al Ghazal",
    "Bahr el Jebel", "Bahr al Jabal", "Bahr el Jabal", "Jur", "Jur River",
    "Bahr el Zeraf", "Bahr ez Zeraf", "Pibor", "Baro",
)


# High-contrast categorical identities. These are NOT severity colours.
# Colours are shared by geometry, the named-river key and exported style records.
RIVER_COLORS = {
    "White Nile": "#0066FF",       # vivid blue
    "Sobat": "#F28E00",            # orange (not the old teal)
    "Bahr el Ghazal": "#009E73",    # green (not the old brown/gold)
    "Jur": "#D500A5",              # magenta
    "Bahr el Jebel": "#202020",     # near-black (not the old indigo)
    "Bahr el Zeraf": "#F0D400",     # yellow (not the old dark green)
    "Pibor": "#00BFC4",            # cyan
    "Baro": "#7B2CBF",             # violet
    "Nile": "#8C564B",             # brown; generic Nile is NOT renamed White Nile
}
RIVER_COLOR_NAMES = {
    "White Nile": "vivid blue", "Sobat": "orange", "Bahr el Ghazal": "green",
    "Jur": "magenta", "Bahr el Jebel": "near-black", "Bahr el Zeraf": "yellow",
    "Pibor": "cyan", "Baro": "violet", "Nile": "brown",
}
# A yellow label on a white halo is hard to read; retain the hue but darken the ink.
RIVER_LABEL_COLORS = {"Bahr el Zeraf": "#786600"}
RIVER_UNKNOWN = "#78838C"
RIVER_CASING_EXTRA_PT = 1.8
RIVER_EDGE_EXTRA_PT = .35
RIVER_EDGE_COLOR = "#39434C"


def river_label_colour(name: str, line_colour: str) -> str:
    """Map labels use the line hue, darkened only when needed for readable text."""
    return RIVER_LABEL_COLORS.get(name, line_colour)


RIVER_NAME_ALIASES = {
    "white nile": "White Nile", "white nile river": "White Nile",
    "sobat": "Sobat", "sobat river": "Sobat",
    "bahr el ghazal": "Bahr el Ghazal", "bahr al ghazal": "Bahr el Ghazal",
    "jur": "Jur", "jur river": "Jur",
    "bahr el jebel": "Bahr el Jebel", "bahr el jabal": "Bahr el Jebel",
    "bahr al jabal": "Bahr el Jebel", "bahr al jebel": "Bahr el Jebel",
    "bahr el zeraf": "Bahr el Zeraf", "bahr ez zeraf": "Bahr el Zeraf",
    "pibor": "Pibor", "pibor river": "Pibor", "baro": "Baro", "baro river": "Baro",
    "nile": "Nile", "nile river": "Nile",
}


def active_factors(df=None, cutoffs=None):
    enabled = PF in cutoffs if cutoffs is not None else bool(df["population_factor_enabled"].iloc[0])
    return BASE_FACTORS + (POPULATION_FACTOR,) if enabled else BASE_FACTORS


def factor_style(code: int) -> tuple[str, str]:
    """Return the same fill and digit colours for map, legend and sidebar badges."""
    if code not in FACTOR_COLORS or code not in FACTOR_TEXT_COLORS:
        raise ValueError(f"Unknown factor code {code!r}; supported codes are 1, 2, 3, 4.")
    return FACTOR_COLORS[code], FACTOR_TEXT_COLORS[code]


def factor_badge_text(parent, x: float, y: float, code: int, *, fontsize: float = 12,
                      pad: float = .28, transform=None):
    """Draw a colour-filled numbered circle on a Figure or an Axes.

    A white rim and thin dark outer halo separate it from both blue/red flood
    polygons and blue river lines. Digits remain usable without colour cues.
    """
    fill, text_color = factor_style(code)
    kw = {"transform": transform} if transform is not None else {}
    artist = parent.text(
        x, y, str(code), ha="center", va="center", fontsize=fontsize,
        fontweight="bold", color=text_color,
        bbox={"boxstyle": f"circle,pad={pad}", "fc": fill, "ec": "white", "lw": 1.0},
        **kw,
    )
    artist.get_bbox_patch().set_path_effects([
        pe.Stroke(linewidth=1.8, foreground=INK), pe.Normal()
    ])
    return artist


def factor_colour_key(df: pd.DataFrame) -> dict:
    """Export the visual key separately; never modify screening flags or counts."""
    enabled = {code for code, *_ in active_factors(df)}
    return {
        "meaning": "Categorical factor identity only; not severity, rank, or a causal driver",
        "population_code": 4,
        "digits_retained": True,
        "same_palette_in": ["map_circles", "lower_legend", "sidebar_counts"],
        "factors": [
            {"code": code, "label": label, "fill_hex": factor_style(code)[0],
             "digit_hex": factor_style(code)[1], "colour_name": FACTOR_COLOR_NAMES[code],
             "enabled_in_this_scenario": code in enabled}
            for code, _, _, label, _ in BASE_FACTORS + (POPULATION_FACTOR,)
        ],
    }


def project_root() -> Path:
    """Prefer the script's project, then the current working directory's project."""
    here = Path(__file__).resolve()
    for start in (here.parent, Path.cwd()):
        for candidate in (start, *start.parents):
            if (candidate / "processed_data" / "multivariate_flood_modeling").is_dir():
                return candidate
    if here.parent.name == "multivariate_flood_modeling" and here.parent.parent.name == "scripts":
        return here.parents[2]
    return Path.cwd()


def resolve_path(path: Path, root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def factor_summary(df: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    rows = []
    candidate = df.priority_review_candidate
    for code, short, column, label, unit in active_factors(df):
        flag = df[f"high_{short}"]
        true = flag.fillna(False).astype(bool)
        rows.append({"factor_number": code, "factor": label, "source_column": column,
            "threshold": cutoffs[column], "unit": unit,
            "all_adm3_count": int(true.sum()), "all_adm3_denominator": len(df),
            "known_factor_denominator": int(flag.notna().sum()), "unknown_factor_count": int(flag.isna().sum()),
            "candidate_count": int((candidate & true).sum()), "candidate_denominator": int(candidate.sum()),
            "candidate_unknown_factor_count": int((candidate & flag.isna()).sum()),
            "high_recent_flood_and_factor": int((df.high_recent_flood & true).sum()),
            "high_increase_and_factor": int((df.high_flood_increase & true).sum())})
    return pd.DataFrame(rows)


MAP_VIEWS = {
    "combined_review": {
        "mask_column": "priority_review_candidate", "hazard_column": None,
        "title": "COMBINED REVIEW", "count_column": "candidate_count",
        "map_stem": "10_humanitarian_review_factors_map_rivers",
        "rule": "(high_recent_flood OR high_flood_increase) AND at least one saved high factor",
    },
    "high_change": {
        "mask_column": "increasing_flood_and_exposure", "hazard_column": "high_flood_increase",
        "title": "HIGH-POSITIVE-CHANGE REVIEW", "count_column": "high_increase_and_factor",
        "map_stem": "11_flood_worsening_factors_map_rivers",
        "rule": "high_flood_increase AND at least one saved high factor",
    },
    "high_recent_activity": {
        "mask_column": "candidate_via_recent_activity", "hazard_column": "high_recent_flood",
        "title": "HIGH-RECENT-ACTIVITY REVIEW", "count_column": "high_recent_flood_and_factor",
        "map_stem": "12_high_recent_flood_activity_factors_map_rivers",
        "rule": "high_recent_flood AND at least one saved high factor",
    },
}


def view_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    """Use saved 13d flags, not newly tuned thresholds or spatial proximity."""
    if mode not in MAP_VIEWS:
        raise ValueError(f"Unsupported map view: {mode}")
    column = MAP_VIEWS[mode]["mask_column"]
    if column not in df or df[column].isna().any():
        raise ValueError(f"Missing/unknown selection flags for map view {mode}: {column}")
    return boolean_values(df[column], column)


def build_view_tables(df: pd.DataFrame, counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tables correspond exactly to outlined/numbered regions, not all red/blue pixels."""
    summaries, factors = [], []
    source_counts = counts.set_index("factor_number")
    if not source_counts.index.is_unique:
        raise ValueError("Duplicate factor_number values in factor summary; counts must be unique.")
    recent, change = view_mask(df, "high_recent_activity"), view_mask(df, "high_change")
    union = view_mask(df, "combined_review")
    if not union.eq(recent | change).all():
        raise ValueError("Combined candidates are not the union of the two focused views.")
    for mode, spec in MAP_VIEWS.items():
        selected = view_mask(df, mode)
        hazard = (df.high_recent_flood | df.high_flood_increase) if spec["hazard_column"] is None else df[spec["hazard_column"]]
        values = df.loc[selected, PF]
        summaries.append({
            "map_view": mode, "map_file_stem": spec["map_stem"], "selection_rule": spec["rule"],
            "analysis_scenario": str(df.analysis_scenario.iloc[0]), "all_adm3": len(df),
            "hazard_threshold_adm3": int(hazard.sum()), "selected_adm3": int(selected.sum()),
            "selected_with_two_or_more_factors": int((selected & df.high_exposure_count.ge(2)).sum()),
            "selected_population_factor4": int((selected & df.high_population.fillna(False)).sum()),
            "selected_population_estimate_available": int(values.notna().sum()),
            "selected_population_estimate_missing": int(values.isna().sum()),
            "selected_population_estimate_subtotal_2025": values.sum(min_count=1),
            "population_subtotal_is_affected_population": False,
        })
        for code, short, column, label, unit in active_factors(df):
            raw_flag = df[f"high_{short}"]
            flag = raw_flag.fillna(False).astype(bool)
            selected_count = int((selected & flag).sum())
            if selected_count != int(source_counts.loc[code, spec["count_column"]]):
                raise ValueError(f"Map selection and saved-factor counts disagree: {mode}, code {code}")
            factors.append({
                "map_view": mode, "factor_number": code, "factor": label,
                "all_adm3_count": int(flag.sum()), "all_adm3_denominator": len(df),
                "selected_count": selected_count, "selected_denominator": int(selected.sum()),
                "selected_unknown_count": int((selected & raw_flag.isna()).sum()),
                "source_column": column, "threshold": float(source_counts.loc[code, "threshold"]), "unit": unit,
            })
    membership = df[["adm_id", "region_name", "parent_adm2_name", "factor_codes", "analysis_scenario"]].copy()
    membership["in_high_recent_activity_map"] = recent
    membership["in_high_change_map"] = change
    membership["in_combined_review_map"] = union
    membership["display_membership"] = np.select(
        [recent & change, recent & ~change, ~recent & change],
        ["Both focused views", "High recent activity only", "High positive change only"],
        default="Not selected by either focused view")
    return pd.DataFrame(summaries), pd.DataFrame(factors), membership


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


def setup_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.titlesize": 15, "svg.fonttype": "none",
                         "savefig.facecolor": "white"})


def map_canvas(title: str, subtitle: str):
    # Three independent columns: map / factor counts / named-river key.
    print(f"[DRAW] {title}", flush=True)
    fig = plt.figure(figsize=(20.5, 10.8), dpi=120, facecolor="white")
    fig.text(.035, .967, "SOUTH SUDAN | ADM3", size=10, weight="bold", color=MUTED)
    fig.text(.035, .925, title, size=24, weight="bold")
    fig.text(.035, .893, subtitle, size=11, color=MUTED)
    ax = fig.add_axes([.025, .285, .525, .555])
    side = fig.add_axes([.575, .275, .205, .58]); side.set_axis_off()
    side.set_xlim(0, 1); side.set_ylim(0, 1)
    river_side = fig.add_axes([.805, .285, .175, .555]); river_side.set_axis_off()
    river_side.set_xlim(0, 1); river_side.set_ylim(0, 1)
    fig._named_river_sidebar = river_side
    cax = fig.add_axes([.063, .247, .445, .017])
    for x in (.56, .793):
        fig.add_artist(Line2D([x, x], [.28, .86], transform=fig.transFigure, color="#D8DFE6", lw=.9))
    # subtle frame around the right-side river key so the legend/explanation reads as one report-ready block
    fig.add_artist(Rectangle((.798, .275), .187, .565, transform=fig.transFigure,
                             fill=False, ec="#D8DFE6", lw=.9))
    return fig, ax, side, cax


def map_base(ax, geo, adm2, adm1, state_labels: bool, rivers, river_meta, args) -> None:
    geo.boundary.plot(ax=ax, color="#F9FAFB", linewidth=.22, zorder=2)
    if adm2 is not None:
        adm2.boundary.plot(ax=ax, color="#697580", linewidth=.38, zorder=3)
    if adm1 is not None:
        adm1.boundary.plot(ax=ax, color="#34414D", linewidth=.68, zorder=4)
        if state_labels and "adm1_name" in adm1:
            for _, row in adm1.iterrows():
                point = row.geometry.representative_point()
                label = "\n".join(textwrap.wrap(str(row.adm1_name), width=19))
                ax.text(point.x, point.y, label, size=7.2, ha="center", va="center", color="#36424E",
                        alpha=.72, zorder=4, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    xmin, ymin, xmax, ymax = geo.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - .065 * dx, xmax + .065 * dx)
    ax.set_ylim(ymin - .10 * dy, ymax + .07 * dy)
    ax.set_aspect("equal"); ax.set_axis_off()
    # Matched ADM3 polygons define the national/parent outlines; no mixed boundary versions.
    outline = getattr(args, "_national_outline", None)
    if outline is None:
        outline = gpd.GeoSeries([unary_union(geo.geometry)], crs=geo.crs).boundary
    outline.plot(ax=ax, color="#27323E", linewidth=.85, zorder=4.2)
    draw_rivers(ax, rivers, river_meta, args)
    river_key(ax.figure, river_meta, args)
    cartographic_guides(ax, geo.crs)


def cartographic_guides(ax, crs) -> None:
    """Geodesic local scale bar and true-north direction at the arrow's location."""
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geod = Geod(ellps="WGS84")
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    sx, sy = xmin + .09 * width, ymin + .045 * height
    lon, lat = inverse.transform(sx, sy)
    end_lon, end_lat, _ = geod.fwd(lon, lat, 90, 200_000)
    ex, ey = forward.transform(end_lon, end_lat)
    ax.plot([sx, ex], [sy, ey], color=INK, lw=1.4, zorder=6)
    for x, y in ((sx, sy), (ex, ey)):
        ax.plot([x, x], [y - .005 * height, y + .005 * height], color=INK, lw=1.2, zorder=6)
    ax.text((sx + ex) / 2, min(sy, ey) - .015 * height, "200 km",
            ha="center", va="top", size=8, zorder=6)
    nx, ny = xmin + .945 * width, ymin + .865 * height
    lon, lat = inverse.transform(nx, ny)
    nlon, nlat, _ = geod.fwd(lon, lat, 0, 60_000)
    tx, ty = forward.transform(nlon, nlat)
    ax.annotate("", xy=(tx, ty), xytext=(nx, ny),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2}, zorder=6)
    ax.text(tx, ty + .015 * height, "N", ha="center", size=10, weight="bold", zorder=6)


def side_text(ax, x: float, y: float, text: str, **kw) -> None:
    ax.text(x, y, text, transform=ax.transAxes, va="top", **kw)


def stat_line(ax, y: float, label: str, value, size: float = 11) -> None:
    side_text(ax, 0, y, label, fontsize=size)
    side_text(ax, .98, y, str(value), fontsize=size, ha="right", weight="bold")


def common_footer(fig, df=None, river_meta=None) -> None:
    source = (river_meta or {}).get("source_label", "river vectors")
    fig.text(.040, .026,
             f"Data: historical flood records; ADM3 exposure estimates; administrative boundaries; {source}.",
             size=8, color=MUTED)
    fig.text(.040, .010,
             "Retrospective screening. Regional totals are not observed flood impacts. River centre-lines are cartographic references.",
             size=7.8, color=MUTED)
    if df is not None:
        scenario = str(df.analysis_scenario.iloc[0])
        notice = {
            "baseline": "BASELINE | POPULATION EXCLUDED",
            "population_sensitivity": "PROVISIONAL POPULATION SENSITIVITY | 2025 ESTIMATES",
            "population_processing_checked": "POPULATION PROCESSING CHECKED | NOT CENSUS VALIDATION",
        }.get(scenario, scenario)
        fig.text(.97, .967, notice, ha="right", size=8.5, weight="bold",
                 color="#9A3412" if scenario == "population_sensitivity" else MUTED)


def threshold_text(cutoff: float, unit: str) -> str:
    comparison = ">" if cutoff == 0 else "\u2265"
    number = f"{cutoff:,.0f}" if float(cutoff).is_integer() else f"{cutoff:,.2f}"
    return f"{comparison} {number} {unit}"


def bottom_factor_legend(fig, cutoffs: dict, quantile: float) -> None:
    """Compact categorical legend confined to the space below the left-hand map."""
    factors = active_factors(cutoffs=cutoffs)
    xs = (.040, .173, .306, .439) if len(factors) == 4 else (.040, .215, .390)
    for x, (code, _, column, label, unit) in zip(xs, factors):
        factor_badge_text(fig, x, .144, code, fontsize=11.5, pad=.24)
        fig.text(x + .014, .159, label, size=10.3, weight="bold", va="top")
        fig.text(x + .014, .131, threshold_text(cutoffs[column], unit), size=9, va="top")
    fig.text(.040, .080,
             f"Thresholds per ADM3: Q{100*quantile:g}, including ties; positive values only at zero cutoffs.",
             size=8.6, color=MUTED)
    message = ("Hatching: population data unavailable. Population: 2025 estimates."
               if len(factors) == 4 else "Population excluded from baseline selection.")
    fig.text(.040, .055, message, size=8.5, color=MUTED)


def exposure_sidebar(side, df: pd.DataFrame, counts: pd.DataFrame) -> None:
    factor_sidebar(side, df, counts, mode="combined_review")


def worsening_sidebar(side, df: pd.DataFrame, counts: pd.DataFrame) -> None:
    factor_sidebar(side, df, counts, mode="high_change")


def recent_activity_sidebar(side, df: pd.DataFrame, counts: pd.DataFrame) -> None:
    factor_sidebar(side, df, counts, mode="high_recent_activity")


def factor_sidebar(side, df: pd.DataFrame, counts: pd.DataFrame, mode: str) -> None:
    selected = view_mask(df, mode)
    spec = MAP_VIEWS[mode]
    n = int(selected.sum())
    side_text(side, 0, .99, spec["title"], fontsize=10.4, weight="bold", color=MUTED)
    side_text(side, 0, .935, str(n), fontsize=35, weight="bold")
    side_text(side, .32, .905, f"of {len(df)} ADM3 areas", fontsize=11)
    side_text(side, 0, .80, "FACTOR COUNTS (ADM3)", fontsize=11, weight="bold")
    side_text(side, 0, .745, "Factor", fontsize=10, color=MUTED)
    side_text(side, .75, .745, "All", fontsize=9, color=MUTED, ha="right")
    side_text(side, .99, .745, "Selected", fontsize=9, color=MUTED, ha="right")
    summary = counts.set_index("factor_number")
    if not summary.index.is_unique:
        raise ValueError("Duplicate factor_number values in sidebar summary; counts must be unique.")
    for y, (code, short, _, label, _) in zip((.68, .615, .55, .485), active_factors(df)):
        flag = df[f"high_{short}"].fillna(False).astype(bool)
        all_count, selected_count = int(flag.sum()), int((selected & flag).sum())
        if all_count != int(summary.loc[code, "all_adm3_count"]) or selected_count != int(summary.loc[code, spec["count_column"]]):
            raise ValueError(f"Sidebar and CSV disagree for {mode}, factor {code}.")
        factor_badge_text(side, .035, y-.014, code, fontsize=9.5, pad=.22, transform=side.transAxes)
        side_text(side, .105, y, label, fontsize=9.5)
        side_text(side, .75, y, str(all_count), fontsize=10.5, ha="right")
        side_text(side, .99, y, str(selected_count), fontsize=10.5, ha="right", weight="bold")
    side_text(side, 0, .412, "Factors overlap; do not add the counts.", fontsize=9, color=MUTED)
    if mode == "high_change":
        stat_line(side, .355, "High positive change (all)", int(df.high_flood_increase.sum()), 9.5)
        stat_line(side, .308, "Selected with 2+ factors", int((selected & df.high_exposure_count.ge(2)).sum()), 9.5)
    elif mode == "high_recent_activity":
        stat_line(side, .355, "High recent activity (all)", int(df.high_recent_flood.sum()), 9.5)
        stat_line(side, .308, "Selected with 2+ factors", int((selected & df.high_exposure_count.ge(2)).sum()), 9.5)
    else:
        stat_line(side, .355, "High activity + factor(s)", int(df.candidate_via_recent_activity.sum()), 9.5)
        stat_line(side, .308, "Change-only + factor(s)", int(df.candidate_via_change_only.sum()), 9.5)
    values = df.loc[selected, PF]
    known = int(values.notna().sum())
    side_text(side, 0, .245, "2025 POPULATION IN SELECTED ADM3", fontsize=9.5, weight="bold")
    total = values.sum(min_count=1)
    value = "Unknown" if pd.isna(total) else f"{total:,.0f}"
    stat_line(side, .192, "Available estimates: subtotal", value, 9.5)
    side_text(side, 0, .145, f"Available for {known}/{n} selected ADM3;\nnot affected or inundated population.", fontsize=9, color=MUTED)
    unknown = int(df.population_factor_unknown.sum())
    note = (f"Population factor unknown: {unknown} ADM3."
            if bool(df.population_factor_enabled.iloc[0]) else "Population does not select areas here.")
    side_text(side, 0, .064, note + "\nNo validated humanitarian-risk ranking.", fontsize=9, color=MUTED)


def circle_group(codes: list[int], diameter: float) -> DrawingArea:
    if not codes or len(codes) != len(set(codes)):
        raise ValueError("Factor badges require at least one unique factor code.")
    gap = 1.8
    width = len(codes) * diameter + max(0, len(codes) - 1) * gap
    drawing = DrawingArea(width, diameter, 0, 0)
    for j, code in enumerate(codes):
        x, y = diameter / 2 + j * (diameter + gap), diameter / 2
        fill, text_color = factor_style(code)
        circle = Circle((x, y), diameter / 2 - .7, facecolor=fill, edgecolor="white", lw=1.0)
        circle.set_path_effects([pe.Stroke(linewidth=1.8, foreground=INK), pe.Normal()])
        drawing.add_artist(circle)
        drawing.add_artist(Text(x, y, str(code), ha="center", va="center", fontsize=diameter * .62,
                                fontweight="bold", color=text_color))
    return drawing


def intersection_area(a, b) -> float:
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def add_factor_badges(fig, ax, geo, diameter: float) -> pd.DataFrame:
    """Label ALL candidates; deterministic local collision reduction, with leader lines.

    Digits repeat factor codes. The leader-line anchor is inside the ADM3 polygon,
    not an observed facility/crop location. Unresolved collisions are audited.
    """
    candidates = geo.loc[geo.priority_review_candidate].sort_values("adm_id").copy()
    columns = ["adm_id", "region_name", "factor_codes", "anchor_lon", "anchor_lat",
               "offset_x_points", "offset_y_points", "unresolved_overlap", "leader_line"]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    points = candidates.geometry.representative_point()
    fig.canvas.draw()
    anchors = ax.transData.transform(np.column_stack([points.x, points.y]))
    px_per_point = fig.dpi / 72
    # Harder, locally crowded positions are placed first; this is NOT priority ordering.
    pairwise = np.sqrt(((anchors[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(pairwise, np.inf)
    order = np.argsort(pairwise.min(axis=1), kind="stable")
    offsets = [(0., 0.)]
    for radius in (12, 20, 30, 42, 56, 72, 92):
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            offsets.append((radius * math.cos(angle), radius * math.sin(angle)))
    occupied = []
    box = ax.get_window_extent()
    # Reserve the local scale bar and north arrow regions.
    obstacles = [(box.x0 + .04*box.width, box.y0, box.x0 + .34*box.width, box.y0 + .10*box.height),
                 (box.x0 + .89*box.width, box.y0 + .82*box.height, box.x1, box.y1)]
    renderer = fig.canvas.get_renderer()
    for artist in ax.texts:
        if artist.get_visible() and artist.get_text().strip():
            bb = artist.get_window_extent(renderer).expanded(1.10, 1.15)
            obstacles.append((bb.x0, bb.y0, bb.x1, bb.y1))
    to_wgs84 = Transformer.from_crs(geo.crs, "EPSG:4326", always_xy=True)
    records = []
    for i in order:
        row = candidates.iloc[i]
        point = points.iloc[i]
        codes = [int(v) for v in row.factor_codes.split("+") if v]
        width = (len(codes)*diameter + (len(codes)-1)*1.8 + 4) * px_per_point
        height = (diameter + 4) * px_per_point
        best = None
        for dx, dy in offsets:
            cx, cy = anchors[i] + np.array([dx, dy]) * px_per_point
            rect = (cx-width/2, cy-height/2, cx+width/2, cy+height/2)
            outside = (max(0, box.x0-rect[0]) + max(0, rect[2]-box.x1)
                       + max(0, box.y0-rect[1]) + max(0, rect[3]-box.y1))
            overlap = sum(intersection_area(rect, other) for other in occupied + obstacles)
            score = 1e6 * outside + 1e3 * overlap + dx*dx + dy*dy
            if best is None or score < best[0]:
                best = (score, dx, dy, rect, overlap, outside)
            if outside == 0 and overlap == 0:
                break
        _, dx, dy, rect, overlap, outside = best
        occupied.append(rect)
        moved = math.hypot(dx, dy) > 2
        if moved:
            target = ax.transData.inverted().transform(anchors[i] + np.array([dx, dy]) * px_per_point)
            ax.plot([point.x, target[0]], [point.y, target[1]], color="#414B55", lw=.55, alpha=.85, zorder=7)
            ax.plot(point.x, point.y, marker=".", markersize=2, color=INK, zorder=8)
        artist = AnnotationBbox(circle_group(codes, diameter), (point.x, point.y),
                                xybox=(dx, dy), xycoords="data", boxcoords="offset points",
                                box_alignment=(.5, .5), frameon=False, pad=0, zorder=10)
        ax.add_artist(artist)
        lon, lat = to_wgs84.transform(point.x, point.y)
        records.append({"adm_id": row.adm_id, "region_name": row.region_name,
                        "factor_codes": row.factor_codes, "anchor_lon": lon, "anchor_lat": lat,
                        "offset_x_points": dx, "offset_y_points": dy,
                        "unresolved_overlap": bool(overlap > 0 or outside > 0), "leader_line": moved})
    return pd.DataFrame(records, columns=columns).sort_values("adm_id")


def save_map(fig, directory: Path, stem: str, dpi: int, formats: list[str]) -> list[str]:
    paths = []
    for fmt in formats:
        path = directory / f"{stem}.{fmt}"
        # Write one complete file, then atomically replace; no half-written PNG/SVG.
        temp = path.with_name(path.stem + "." + uuid.uuid4().hex + ".tmp" + path.suffix)
        try:
            with Progress(f"Save {path.name}"):
                fig.savefig(temp, dpi=dpi, facecolor="white", format=fmt)
                os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        paths.append(str(path.name))
    plt.close(fig)
    return paths


def plot_spatial_maps(geo, adm2, adm1, df, cutoffs, counts, directory: Path, args, rivers, river_meta) -> tuple[list[str], dict]:
    directory.mkdir(parents=True, exist_ok=True)
    max_h = max(1., math.ceil(float(df[H].max()) / 25.) * 25.)
    hnorm = Normalize(vmin=0, vmax=max_h)
    qlabel = f"Q{100 * args.quantile:g}"
    figure_paths = []

    # A: recent activity, without turning a continuous variable into factor digits.
    fig, ax, side, cax = map_canvas("Recent Recorded Flood Activity",
        "Mean annual recorded activity, 2019–2025")
    geo.plot(column=H, cmap="Blues", norm=hnorm, ax=ax, edgecolor="none", zorder=1)
    map_base(ax, geo, adm2, adm1, not args.no_state_labels, rivers, river_meta, args)
    cb = fig.colorbar(ScalarMappable(norm=hnorm, cmap="Blues"), cax=cax, orientation="horizontal")
    cb.set_label('Mean annual recorded flood-activity days (2019–2025)', fontsize=10)
    cax.axvline(cutoffs[H], color=INK, lw=1.1)
    cax.text(cutoffs[H], 1.4, qlabel, transform=cax.get_xaxis_transform(), ha="center", fontsize=8)
    side_text(side, 0, .99, "RECENT ACTIVITY", fontsize=11, weight="bold", color=MUTED)
    side_text(side, 0, .91, str(int(df.high_recent_flood.sum())), fontsize=37, weight="bold")
    side_text(side, 0, .79, f"ADM3 at/above the screening cutoff\nout of {len(df)} areas", fontsize=12)
    stat_line(side, .64, f"{qlabel} threshold (days)", f"{cutoffs[H]:.2f}")
    stat_line(side, .55, "Median (days)", f"{df[H].median():.2f}")
    stat_line(side, .46, "Maximum (days)", f"{df[H].max():.2f}")
    stat_line(side, .37, "Zero recorded activity", int(df[H].eq(0).sum()))
    side_text(side, 0, .24, "Historical observations\n2019–2025", fontsize=11, color=MUTED, linespacing=1.6)
    common_footer(fig, df, river_meta)
    figure_paths += save_map(fig, directory, "08_recent_flood_activity_map_rivers", args.dpi, args.formats)

    # B: symmetric normalization; identical absolute differences have equal colour distance from zero.
    limit = max(1., math.ceil(float(df[D].abs().max()) / 20.) * 20.)
    dnorm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, ax, side, cax = map_canvas("Change in Recorded Flood Activity",
        "Difference between period means: 2019–2025 minus 2000–2018")
    geo.plot(column=D, cmap="RdBu_r", norm=dnorm, ax=ax, edgecolor="none", zorder=1)
    map_base(ax, geo, adm2, adm1, not args.no_state_labels, rivers, river_meta, args)
    cb = fig.colorbar(ScalarMappable(norm=dnorm, cmap="RdBu_r"), cax=cax, orientation="horizontal",
                      ticks=[-limit, -limit/2, 0, limit/2, limit])
    cb.set_label('Difference in mean annual flood-activity days (2019–2025 − 2000–2018)', fontsize=10)
    positive, negative = int(df[D].gt(1e-9).sum()), int(df[D].lt(-1e-9).sum())
    side_text(side, 0, .99, "BETWEEN-PERIOD CHANGE", fontsize=11, weight="bold", color=MUTED)
    stat_line(side, .885, "Positive difference", positive)
    stat_line(side, .79, "Near-zero difference", len(df) - positive - negative)
    stat_line(side, .695, "Negative difference", negative)
    side_text(side, 0, .565, "HIGH POSITIVE CHANGE", fontsize=11, weight="bold")
    stat_line(side, .495, f"{qlabel} cutoff (days)", f"{cutoffs[D]:.2f}")
    stat_line(side, .405, "Qualifying ADM3", int(df.high_flood_increase.sum()))
    side_text(side, 0, .29, "High change: positive difference\nat or above the quantile cutoff.", fontsize=11, color=MUTED, linespacing=1.6)
    side_text(side, 0, .15, "Period-mean difference;\nnot an annual trend estimate.", fontsize=10, color=MUTED, linespacing=1.6)
    common_footer(fig, df, river_meta)
    figure_paths += save_map(fig, directory, "09_post2019_flood_change_map_rivers", args.dpi, args.formats)

    # C: keep recent-activity colour and overlay only numbered exposure factors.
    n = int(df.priority_review_candidate.sum())
    fig, ax, side, cax = map_canvas("Flood Activity and Exposure: Combined Screening",
        "High recent activity or high positive change, with one or more exposure factors")
    geo.plot(column=H, cmap="Blues", norm=hnorm, ax=ax, edgecolor="none", zorder=1)
    map_base(ax, geo, adm2, adm1, not args.no_state_labels, rivers, river_meta, args)
    if bool(df.population_factor_enabled.iloc[0]):
        unknown = geo[geo.population_factor_unknown]
        if not unknown.empty:
            unknown.plot(ax=ax, facecolor="none", edgecolor="#5D6670", hatch="///", linewidth=.3, zorder=4.5)
    recent = geo[geo.candidate_via_recent_activity]
    change_only = geo[geo.candidate_via_change_only]
    if not recent.empty:
        recent.boundary.plot(ax=ax, color="#111111", linewidth=1.05, zorder=5)
    if not change_only.empty:
        change_only.boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--", zorder=5)
    cb = fig.colorbar(ScalarMappable(norm=hnorm, cmap="Blues"), cax=cax, orientation="horizontal")
    # Put the caption above this colour bar: leave a separate row for the outline key.
    fig.text(.285, .273, 'Mean annual recorded flood-activity days (2019–2025)',
             ha="center", va="bottom", fontsize=10)
    exposure_sidebar(side, df, counts)
    bottom_factor_legend(fig, cutoffs, args.quantile)
    # Explanatory outline key sits above the bottom factor legend, outside the map.
    fig.legend(handles=[Line2D([0], [0], color="#111111", lw=1.2,
                               label="High recent activity + exposure"),
                        Line2D([0], [0], color="#111111", lw=1.2, linestyle="--",
                               label="High increase + exposure; below activity cutoff")],
               loc="center left", bbox_to_anchor=(.05, .211), frameon=False, ncol=2,
               fontsize=8.6, handlelength=2.8, columnspacing=2)
    badges = add_factor_badges(fig, ax, geo, args.badge_size)
    badges.to_csv(directory / "13_marker_placement_audit.csv", index=False)
    conflicts = int(badges.unresolved_overlap.sum()) if len(badges) else 0
    if conflicts:
        warnings.warn(f"{conflicts} factor groups still have estimated label overlap. Inspect map 10; try --badge-size 10.")
    common_footer(fig, df, river_meta)
    figure_paths += save_map(fig, directory, "10_humanitarian_review_factors_map_rivers", args.dpi, args.formats)

    # D: worsening map with the same factor-number system, focused on high positive change + assets.
    worsening_n = int(df.increasing_flood_and_exposure.sum())
    fig, ax, side, cax = map_canvas("Flood Activity Increase and Exposure Factors",
        "High positive change in mean annual activity: 2019–2025 relative to 2000–2018")
    geo.plot(column=D, cmap="RdBu_r", norm=dnorm, ax=ax, edgecolor="none", zorder=1)
    map_base(ax, geo, adm2, adm1, not args.no_state_labels, rivers, river_meta, args)
    if bool(df.population_factor_enabled.iloc[0]):
        unknown = geo[geo.population_factor_unknown]
        if not unknown.empty:
            unknown.plot(ax=ax, facecolor="none", edgecolor="#5D6670", hatch="///", linewidth=.3, zorder=4.5)
    worsening = geo[geo.increasing_flood_and_exposure]
    if not worsening.empty:
        worsening.boundary.plot(ax=ax, color="#111111", linewidth=1.15, zorder=5)
    cb = fig.colorbar(ScalarMappable(norm=dnorm, cmap="RdBu_r"), cax=cax, orientation="horizontal",
                      ticks=[-limit, -limit/2, 0, limit/2, limit])
    fig.text(.285, .273, 'Difference in mean annual flood-activity days (2019–2025 − 2000–2018)',
             ha="center", va="bottom", fontsize=10)
    worsening_sidebar(side, df, counts)
    bottom_factor_legend(fig, cutoffs, args.quantile)
    fig.legend(handles=[Line2D([0], [0], color="#111111", lw=1.2,
                               label=f"High positive change {threshold_text(cutoffs[D], 'days')} ({qlabel}) + at least one exposure factor")],
               loc="center left", bbox_to_anchor=(.05, .211), frameon=False, ncol=1,
               fontsize=8.6, handlelength=2.8, columnspacing=2)
    worsening_geo = geo.copy()
    worsening_geo["priority_review_candidate"] = worsening_geo["increasing_flood_and_exposure"]
    worsening_badges = add_factor_badges(fig, ax, worsening_geo, args.badge_size)
    worsening_badges.to_csv(directory / "14_worsening_marker_placement_audit.csv", index=False)
    worsening_conflicts = int(worsening_badges.unresolved_overlap.sum()) if len(worsening_badges) else 0
    if worsening_conflicts:
        warnings.warn(f"{worsening_conflicts} worsening-factor groups still have estimated label overlap. Inspect map 11; try --badge-size 10.")
    common_footer(fig, df, river_meta)
    figure_paths += save_map(fig, directory, "11_flood_worsening_factors_map_rivers", args.dpi, args.formats)

    # E: the NEW focused view. Do not include high-change-only candidates here.
    recent_n = int(view_mask(df, "high_recent_activity").sum())
    fig, ax, side, cax = map_canvas("High Recent Flood Activity and Exposure Factors",
        "2019–2025 | High-activity areas with one or more exposure factors")
    geo.plot(column=H, cmap="Blues", norm=hnorm, ax=ax, edgecolor="none", zorder=1)
    map_base(ax, geo, adm2, adm1, not args.no_state_labels, rivers, river_meta, args)
    if bool(df.population_factor_enabled.iloc[0]):
        unknown = geo[geo.population_factor_unknown]
        if not unknown.empty:
            unknown.plot(ax=ax, facecolor="none", edgecolor="#5D6670", hatch="///", linewidth=.3, zorder=4.5)
    recent_selected = geo.loc[view_mask(geo, "high_recent_activity")]
    if not recent_selected.empty:
        recent_selected.boundary.plot(ax=ax, color="#111111", linewidth=1.15, zorder=5)
    fig.colorbar(ScalarMappable(norm=hnorm, cmap="Blues"), cax=cax, orientation="horizontal")
    if 0 <= cutoffs[H] <= max_h:
        cax.axvline(cutoffs[H], color=INK, lw=1.0)
    fig.text(.285, .273, 'Mean annual recorded flood-activity days (2019–2025)',
             ha="center", va="bottom", fontsize=10)
    recent_activity_sidebar(side, df, counts)
    bottom_factor_legend(fig, cutoffs, args.quantile)
    fig.legend(handles=[Line2D([0], [0], color="#111111", lw=1.2,
                               label=f"High recent activity {threshold_text(cutoffs[H], 'days')} ({qlabel}) + at least one exposure factor")],
               loc="center left", bbox_to_anchor=(.05, .211), frameon=False, ncol=1,
               fontsize=8.6, handlelength=2.8)
    recent_geo = geo.copy()
    recent_geo["priority_review_candidate"] = view_mask(geo, "high_recent_activity")
    recent_badges = add_factor_badges(fig, ax, recent_geo, args.badge_size)
    recent_badges.to_csv(directory / "15_high_recent_marker_placement_audit.csv", index=False)
    recent_conflicts = int(recent_badges.unresolved_overlap.sum()) if len(recent_badges) else 0
    if recent_conflicts:
        warnings.warn(f"{recent_conflicts} high-activity factor groups still have estimated overlap. Inspect map 12; try --badge-size 10.")
    common_footer(fig, df, river_meta)
    figure_paths += save_map(fig, directory, "12_high_recent_flood_activity_factors_map_rivers", args.dpi, args.formats)
    if len(recent_badges) != recent_n:
        raise ValueError("Not all high-recent-activity candidates received a factor label.")

    quality = {"candidate_labels_drawn": len(badges), "candidate_labels_expected": n,
               "estimated_unresolved_label_overlaps": conflicts,
               "worsening_labels_drawn": len(worsening_badges), "worsening_labels_expected": worsening_n,
               "worsening_estimated_unresolved_label_overlaps": worsening_conflicts,
               "recent_activity_colour_limits": [0, max_h], "change_colour_limits": [-limit, limit],
               "high_recent_labels_drawn": len(recent_badges), "high_recent_labels_expected": recent_n,
               "high_recent_estimated_unresolved_label_overlaps": recent_conflicts,
               "high_recent_marker_audit": "maps/15_high_recent_marker_placement_audit.csv",
               "high_recent_selection_rule": MAP_VIEWS["high_recent_activity"]["rule"],
               "high_recent_excludes_change_only": True, "map_count": 5,
               "lakes_wetlands_terrain_layers_included": False,
               "report_text": "Self-contained labels; methods in 11_figure_captions.md",
               "marker_anchor": "representative point inside ADM3, not the location of an exposed asset",
               "factor_codes": {str(c): label for c, _, _, label, _ in active_factors(df)},
               "factor_colour_key": factor_colour_key(df),
               "marker_audit": "maps/13_marker_placement_audit.csv",
               "worsening_marker_audit": "maps/14_worsening_marker_placement_audit.csv",
               "worsening_selection_rule": "high_flood_increase AND high_exposure_count > 0",
               "worsening_factor_counts": {str(code): int((df.increasing_flood_and_exposure & df[f"high_{short}"]).sum())
                                           for code, short, *_ in active_factors(df)}}
    if len(badges) != n:
        raise ValueError("Not all review candidates received a factor label.")
    if len(worsening_badges) != worsening_n:
        raise ValueError("Not all worsening candidates received a factor label.")
    return figure_paths, quality


def write_figure_captions(directory: Path, df: pd.DataFrame, cutoffs: dict,
                          quantile: float, river_meta: dict) -> Path:
    """Write standalone report captions and method notes without expanding figure text."""
    factors = active_factors(df)
    lines = ["# Figure captions and method notes", "",
             "Captions refer to the files in maps/. Adapt figure numbering to the report.", "",
             "## Recent recorded flood activity",
             "Mean annual number of days with recorded flood activity during 2019–2025, by ADM3. "
             "The measure denotes activity recorded somewhere in the administrative unit, not the "
             "duration of full-area inundation, water depth, or observed loss.", "",
             "## Change in recorded flood activity",
             "Mean annual activity in 2019–2025 minus the corresponding mean in 2000–2018. "
             "This is a difference between period means, not a trend slope, a significance test, "
             "or evidence of increasing activity in every individual year.", "",
             "## Combined flood and exposure screening",
             "Areas with either high recent activity or high positive period change and at least "
             "one qualifying exposure indicator. Solid outlines identify the high-activity route; "
             "dashed outlines identify the high-change route below the activity threshold.", "",
             "## Flood activity increase and exposure factors",
             "Only areas with high positive change and at least one qualifying exposure indicator "
             "are outlined and labelled. They may also qualify as high-activity areas.", "",
             "## High recent flood activity and exposure factors",
             "Only areas meeting the high recent-activity threshold and at least one exposure "
             "threshold are outlined and labelled; change-only areas are not selected.", "",
             "## Shared definitions",
             f"High recent activity: {threshold_text(cutoffs[H], 'days')}.",
             f"High positive change: positive AND {threshold_text(cutoffs[D], 'days')}.",
             f"High values use Q{100*quantile:g} including ties; a zero cutoff requires a positive value.",
             "The activity maps retain a common numerical scale; the change maps retain a symmetric "
             "scale about zero. Removing cross-map references from the pictures does not change these scales.", ""]
    for code, _, column, label, unit in factors:
        lines.append(f"{code}. {label}: {threshold_text(cutoffs[column], unit)} per ADM3.")
    lines += ["", "Factor colours/numbers indicate categories, not severity or rankings. "
              "Factor totals and focused selection sets overlap and must not be summed as distinct areas.",
              "Population is a 2025 snapshot. Its subtotal is for selected administrative areas, "
              "not affected or inundated population. Unknown values remain unknown.",
              f"Population/analysis status inherited from the saved result: {df.analysis_scenario.iloc[0]}.",
              "All exposure indicators are administrative totals, without asset/flood-footprint intersection. "
              "No vulnerability or loss model is implied; these are retrospective screening results, "
              "not an operational forecast.", "", "## River and boundary context",
              f"River source: {river_meta.get('source_label', 'vector data')}.",
              "Named reference lines, when used, retain their own generalized geometry. They are not "
              "names assigned to HydroRIVERS by nearest distance, measured channel widths, or a verified "
              "tributary classification. Different river colours identify names only.",
              "Default river clipping uses the exact union of matched ADM3 polygons with zero outward "
              "buffer. Read-window padding does not permit drawing rivers beyond that union.",
              "ADM1/ADM2 and national outlines derive from the matched ADM3 data. Detailed source "
              "paths, checksums, thresholds, provenance and processing checks remain in the JSON/CSV outputs."]
    path = directory / "11_figure_captions.md"
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return path


def json_ready(value):
    """Convert numerical scalars to strict JSON; non-finite diagnostics become null."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(json_ready(data), indent=2, allow_nan=False), encoding="utf-8")

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
        raise ValueError("13d numeric_thresholds missing; 13e does not recalculate cutoffs.")
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
        raise ValueError("Saved 13d candidate flags are inconsistent; rerun 13d rather than changing selection inside 13e.")
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


def normalized_name(value) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def canonical_river_name(value) -> str:
    """Spelling aliases only. No positional matching or inferred tributary identity."""
    raw = str(value).strip()
    return RIVER_NAME_ALIASES.get(normalized_name(raw), raw)


def select_display_scenarios(ready: list, args) -> list:
    """Choose saved four-factor results for default presentation, NOT new selection."""
    population = [r for r in ready if r[3]["population_factor_enabled"]]
    baseline = [r for r in ready if not r[3]["population_factor_enabled"]]
    if args.scenario == "baseline":
        if not baseline:
            raise ValueError("No three-factor baseline is saved in this input.")
        return [baseline[0]]
    if args.scenario == "main":
        main = [r for r in ready if r[0]["label"] == "main"]
        return main[:1] or ready[:1]
    if args.scenario == "population-sensitivity":
        population = [r for r in population if r[3]["scenario"] == "population_sensitivity"]
    if not population:
        raise ValueError(
            "Population factor 4 is required for this display, but no saved FOUR-FACTOR 13d scenario was found. "
            "13e will not invent its flags or threshold.\n"
            "Run the supplied 13d v4 population script once (default --population-mode auto), then run 13e again.\n"
            "Expected: 13d_adm3_humanitarian_priority_population/population_sensitivity/"
            "01_adm3_screening_full.csv, or a processing-checked four-factor main table.\n"
            "An explicit --scenario baseline still permits a three-factor comparison."
        )
    population.sort(key=lambda r: (r[3]["scenario"] != "population_processing_checked", r[0]["label"] != "main"))
    chosen = [population[0]]
    if args.include_baseline or args.scenario == "all":
        chosen += baseline[:1]
    return chosen


def named_reference_path(root: Path, args) -> Path:
    if args.named_rivers:
        path = resolve_path(args.named_rivers, root)
        if not path.exists():
            raise FileNotFoundError(f"Named river reference not found: {path}")
        return path
    local = river_candidates(root / "raw_data")
    natural = [p for p in local if "ne_10m_rivers" in p.name.casefold()]
    if natural:
        return natural[0]
    if args.offline:
        raise FileNotFoundError(
            "The network has no named major reaches, and no local named-reference layer was found. "
            "Offline mode forbids downloads. Supply --named-rivers PATH or a reviewed --river-names-csv PATH."
        )
    try:
        return download_ne(root)
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain the official Natural Earth named-river reference. "
            "No river names were guessed. Check the internet connection or supply --named-rivers PATH "
            "(.zip/.shp/.gpkg/.geojson). To explicitly accept generic unnamed lines, use --allow-unnamed-rivers. "
            f"Original error: {exc}"
        ) from exc


def assemble_named_rivers(root: Path, geo, args) -> tuple[gpd.GeoDataFrame, dict]:
    """Keep reference geometry separate; never transfer its names to HydroRIVERS."""
    try:
        network, metadata = load_river_source(root, geo, args)
    except FileNotFoundError:
        if args.rivers:
            raise
        path = named_reference_path(root, args)
        local_args = SimpleNamespace(**vars(args))
        local_args.river_names_csv = None
        local_args.river_layer = args.named_river_layer
        network, metadata = read_rivers(path, geo, local_args)
        metadata["background_fallback"] = "Named reference used because no local network was found"
    network = network.copy()
    network["is_named_reference_overlay"] = False
    network["geometry_source"] = metadata["source_label"]
    network["display_name"] = network.river_name.map(canonical_river_name)
    named_major = network.is_major & network.display_name.ne("")
    use_reference = bool(args.named_rivers) or not named_major.any()
    reference_metadata = None
    parts = [network]
    if use_reference:
        try:
            path = named_reference_path(root, args)
            local_args = SimpleNamespace(**vars(args))
            local_args.river_layer = args.named_river_layer
            local_args.river_names_csv = None
            local_args.min_order = 1
            ref, reference_metadata = read_rivers(path, geo, local_args)
            ref["display_name"] = ref.river_name.map(canonical_river_name)
            ref = ref.loc[ref.is_major & ref.display_name.ne("")].copy()
            if ref.empty:
                raise ValueError("The named-reference layer has no named main river lines within the map extent.")
            # Prefer actual source names for a river already named in the network.
            native_names = set(network.loc[named_major, "display_name"])
            ref = ref.loc[~ref.display_name.isin(native_names)].copy()
            if not ref.empty:
                ref["is_named_reference_overlay"] = True
                ref["geometry_source"] = reference_metadata["source_label"]
                parts.append(ref)
        except Exception as exc:
            if not args.allow_unnamed_rivers:
                raise
            warnings.warn(f"Explicit unnamed-river fallback: {exc}. The legend will disclose missing names.")
    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=geo.crs)
    named = combined.loc[combined.is_major & combined.display_name.ne("")]
    lengths = named.assign(_length=named.geometry.length).groupby("display_name")._length.sum()
    # At most 12 distinct coloured rivers on the side key; other lines stay neutral.
    names = sorted(lengths.index, key=lambda n: (n not in RIVER_COLORS, -float(lengths[n]), n))[:args.max_coloured_rivers]
    selected_names = set(names)
    palette = dict(RIVER_COLORS)
    used = set(palette.values())
    for name in sorted(selected_names - set(palette)):
        hue = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) / 0xffffffff
        for _ in range(80):
            rgb = colorsys.hsv_to_rgb(hue, .72, .66)
            color = "#" + "".join(f"{round(c*255):02X}" for c in rgb)
            if color not in used:
                break
            hue = (hue + .137) % 1
        palette[name] = color
        used.add(color)
    combined["named_colour"] = combined.is_major & combined.display_name.isin(selected_names)
    combined["line_color"] = np.where(combined.is_major, RIVER_UNKNOWN, RIVER_MINOR)
    combined.loc[combined.named_colour, "line_color"] = combined.loc[combined.named_colour, "display_name"].map(palette)
    if not selected_names and not args.allow_unnamed_rivers:
        raise ValueError("No named major rivers available; refusing to invent names. Supply a named line layer or --allow-unnamed-rivers.")
    key = []
    for name in sorted(selected_names, key=lambda x: (list(RIVER_COLORS).index(x) if x in RIVER_COLORS else 99, x)):
        group = combined.loc[combined.named_colour & combined.display_name.eq(name)]
        key.append({"river_name": name, "colour_hex": palette[name],
                    "colour_name": RIVER_COLOR_NAMES.get(name, "additional named river"),
                    "label_colour_hex": river_label_colour(name, palette[name]),
                    "line_width_points": float(args.major_width),
                    "line_records": len(group),
                    "name_sources": sorted(group.name_source.dropna().unique().tolist()),
                    "geometry_sources": sorted(group.geometry_source.dropna().unique().tolist()),
                    "separate_reference_geometry": bool(group.is_named_reference_overlay.any())})
    metadata.update({
        "named_reference": reference_metadata,
        "named_reference_overlay_used": bool(combined.is_named_reference_overlay.any()),
        "named_river_colour_key": key,
        "named_rivers_coloured": len(key),
        "river_palette_version": "13e-v4-high-contrast",
        "river_style_meaning": "Categorical river identity only; not magnitude, severity or flow direction",
        "opaque_white_casing_extra_points": RIVER_CASING_EXTRA_PT,
        "fine_dark_edge_extra_points": RIVER_EDGE_EXTRA_PT,
        "river_edge_colour": RIVER_EDGE_COLOR,
        "factor_badge_palette_unchanged": True,
        "unnamed_or_other_major_line_records": int((combined.is_major & ~combined.named_colour).sum()),
        "minor_line_records": int((~combined.is_major).sum()),
        "other_reaches_colour_hex": RIVER_MINOR,
        "other_reaches_line_width_points": float(args.minor_width),
        "other_reaches_alpha": MINOR_ALPHA,
        "other_reaches_casing_colour_hex": MINOR_CASING_COLOR,
        "other_reaches_casing_extra_points": float(args.minor_casing_extra),
        "other_reaches_casing_total_width_points": float(args.minor_width + args.minor_casing_extra),
        "other_reaches_casing_zorder": MINOR_CASING_ZORDER,
        "other_reaches_core_zorder": MINOR_CORE_ZORDER,
        "other_reaches_display_note": "Opaque core above flood polygons; narrow white casing provides separation on dark fills. Not a measured channel width.",
        "other_reaches_are_verified_tributaries": False,
        "display_layer_suitable_for_drainage_density": False,
        "name_transfer_to_network_by_nearest_distance": False,
        "reference_geometry_not_a_local_survey": bool(combined.is_named_reference_overlay.any()) or "Natural Earth" in metadata["source_label"],
        "generic_nile_not_renamed_to_white_nile": True,
        "all_named_major_names_available": sorted(lengths.index.tolist()),
        "reference_does_not_assign_names_to_hydrorivers": True,
        "display_line_records_after_reference_overlay": len(combined),
    })
    if metadata["named_reference_overlay_used"]:
        metadata["source_label"] += " + named reference lines (separate geometry)"
    print("[RIVERS] Coloured named rivers: " + (", ".join(e["river_name"] for e in key) or "NONE (explicit fallback)"), flush=True)
    return combined, metadata


def river_candidates(folder: Path) -> list[Path]:
    found = []
    if not folder.is_dir():
        return found
    for base, dirs, names in os.walk(folder):
        for dirname in list(dirs):
            path = Path(base) / dirname
            if dirname.startswith(".") or dirname in ("__MACOSX", "node_modules"):
                dirs.remove(dirname)
            elif dirname.lower().endswith(".gdb"):
                if re.search(r"river|waterway", str(path), re.I):
                    found.append(path)
                dirs.remove(dirname)
        for name in names:
            path = Path(base) / name
            if path.suffix.lower() in (".shp", ".gpkg", ".geojson", ".json", ".zip") and not name.startswith("._"):
                if re.search(r"river|waterway", str(path), re.I):
                    found.append(path)
    order = {".gdb": 0, ".shp": 1, ".gpkg": 2, ".geojson": 3, ".json": 4, ".zip": 5}
    return sorted(set(found), key=lambda p: (0 if "hydrorivers" in str(p).lower() else 1,
                                            order.get(p.suffix.lower(), 9), str(p).lower()))


def vector_uri(path: Path) -> str:
    if path.suffix.lower() != ".zip":
        return str(path)
    # GDAL can read the FileGDB in a zip via /vsizip/; do not unzip 100k files.
    with zipfile.ZipFile(path) as archive:
        gdbs = set()
        for name in archive.namelist():
            if "__MACOSX" in name:
                continue
            parts = PurePosixPath(name).parts
            for i, part in enumerate(parts):
                if part.lower().endswith(".gdb"):
                    gdbs.add("/".join(parts[:i+1]))
        if len(gdbs) > 1:
            raise ValueError("ZIP contains multiple FileGDBs; unzip and pass one .gdb directory.")
    return "/vsizip/" + str(path.resolve()) + (("/" + next(iter(gdbs))) if gdbs else "")


def choose_river_layer(uri: str, explicit: str | None) -> str:
    layers = pyogrio.list_layers(uri)
    names = [str(row[0]) for row in layers]
    if explicit:
        if explicit not in names:
            raise ValueError(f"Layer {explicit!r} not in dataset. Available: {names}")
        return explicit
    lines = [(str(name), str(kind)) for name, kind in layers
             if kind is not None and ("line" in str(kind).casefold() or "unknown" in str(kind).casefold())]
    preferred = [name for name, kind in lines if "hydrorivers" in name.casefold()]
    if not preferred:
        preferred = [name for name, kind in lines if re.search(r"river|waterway", name, re.I)]
    choices = preferred or [name for name, kind in lines]
    if len(choices) != 1:
        raise ValueError(f"Expected one river line layer; found {choices}. Use --river-layer EXACT_NAME.")
    return choices[0]



class Progress:
    """Visible long-operation status, including while native GIS code is running."""
    def __init__(self, label: str, interval: float = 10.0):
        self.label = label
        self.interval = interval
        self.done = threading.Event()
        self.worker = None

    def __enter__(self):
        self.started = time.perf_counter()
        print(f"[START] {self.label}", flush=True)
        def heartbeat():
            while not self.done.wait(self.interval):
                elapsed = time.perf_counter() - self.started
                print(f"[WORKING] {self.label} | elapsed {elapsed:.0f}s", flush=True)
        self.worker = threading.Thread(target=heartbeat, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.done.set()
        self.worker.join(timeout=1)
        status = "DONE" if exc_type is None else "STOPPED"
        print(f"[{status}] {self.label} | {time.perf_counter() - self.started:.2f}s", flush=True)
        return False


RIVER_CACHE_SCHEMA = "regional-geometry-3-country-zero-buffer"


def dataset_stamp(path: Path) -> dict:
    """Cheap source change signature; do NOT read an Africa-wide GDB into memory."""
    path = path.resolve()
    if path.is_dir():
        components = sorted(p for p in path.rglob("*") if p.is_file()
                            and ".lock" not in p.name.lower())
        relative_to = path
    elif path.suffix.lower() == ".shp":
        extensions = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix"}
        components = sorted(p for p in path.parent.glob(path.stem + ".*")
                            if p.is_file() and p.suffix.lower() in extensions)
        relative_to = path.parent
    else:
        components = [path]
        relative_to = path.parent
    files = []
    for part in components:
        st = part.stat()
        files.append([str(part.relative_to(relative_to)), st.st_size, st.st_mtime_ns])
    if not files:
        raise FileNotFoundError(f"No source dataset components: {path}")
    return {"path": str(path), "components": files}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def geometry_digest(geo) -> str:
    h = hashlib.sha256()
    h.update(str(geo.crs).encode("utf-8"))
    for geom in geo.geometry:
        h.update(geom.wkb)
    return h.hexdigest()


def load_geometry_cache(directory: Path, key: str, signature: dict, target_crs):
    gpkg, sidecar = directory / (key + ".gpkg"), directory / (key + ".json")
    if not gpkg.is_file() or not sidecar.is_file():
        return None
    try:
        stored = json.loads(sidecar.read_text(encoding="utf-8"))
        if stored.get("signature") != signature or stored.get("sha256") != digest_file(gpkg):
            raise ValueError("signature/checksum mismatch")
        cached = gpd.read_file(gpkg, layer="regional_geometry", engine="pyogrio")
        if cached.crs != target_crs or len(cached) != stored["rows"]:
            raise ValueError("cache CRS/row count mismatch")
        cached = cached.rename(columns=stored["field_map"])
        for field, dtype in stored["dtypes"].items():
            cached[field] = cached[field].astype(dtype)
        if (cached.geometry.isna().any() or cached.geometry.is_empty.any()
                or not cached.geom_type.isin(["LineString", "MultiLineString"]).all()
                or not cached.geometry.is_valid.all()):
            raise ValueError("invalid/non-line cached geometry")
        print(f"[CACHE HIT] {len(cached):,} regional river records: {gpkg.name}", flush=True)
        return cached, {**stored["spatial_meta"], "cache_hit": True, "cache_file": str(gpkg)}
    except Exception as exc:
        warnings.warn(f"Ignoring unusable regional cache {gpkg.name}: {exc}; rebuilding from source.")
        return None


def save_geometry_cache(directory: Path, key: str, signature: dict, river, metadata: dict):
    """Atomic per-file writes; JSON checksum prevents reuse of an incomplete pair."""
    directory.mkdir(parents=True, exist_ok=True)
    temp_id = uuid.uuid4().hex
    temp_data = directory / (key + "." + temp_id + ".tmp.gpkg")
    temp_json = directory / (key + "." + temp_id + ".tmp.json")
    try:
        attrs = [c for c in river if c != river.geometry.name]
        if any(not isinstance(c, str) for c in attrs) or len(set(attrs)) != len(attrs):
            raise ValueError("Source attributes must have unique string names for caching.")
        rename = {name: f"field_{i:04d}" for i, name in enumerate(attrs)}
        encoded = river.rename(columns=rename).copy()
        # Preserve LineString/MultiLineString identity; do not auto-promote singles.
        encoded.to_file(temp_data, layer="regional_geometry", driver="GPKG", engine="pyogrio",
                        geometry_type="Unknown", promote_to_multi=False)
        payload = {"signature": signature, "sha256": digest_file(temp_data),
                   "rows": len(river), "field_map": {v: k for k, v in rename.items()},
                   "dtypes": {name: str(river[name].dtype) for name in attrs},
                   "spatial_meta": metadata}
        temp_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp_data, directory / (key + ".gpkg"))
        os.replace(temp_json, directory / (key + ".json"))
        print(f"[CACHE SAVE] {len(river):,} regional records cached", flush=True)
    except Exception as exc:
        # A cache failure must not invalidate a successfully read source.
        warnings.warn(f"Regional cache could not be saved; continuing without cache: {exc}")
    finally:
        temp_data.unlink(missing_ok=True)
        temp_json.unlink(missing_ok=True)


def keep_line_geometry(geometry):
    """Discard only zero-length contacts; preserve all line parts of a collection."""
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in ("LineString", "MultiLineString"):
        return geometry
    parts = line_parts(geometry)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiLineString(parts)


def clip_river_batches(river, mask, batch_size: int = 512, mode: str = "bbox"):
    """Exact intersection with a SIMPLE box by default; no arbitrary line sampling.

    In optional country mode, use prepared predicates to skip full intersections
    for lines that are completely inside/outside the detailed national mask.
    """
    outputs = []
    n = len(river)
    prepared = prep(mask) if mode == "country" else None
    full_intersections = 0
    for begin in range(0, n, batch_size):
        part = river.geometry.iloc[begin:begin + batch_size]
        if prepared is None:
            clipped = part.intersection(mask)
            full_intersections += len(part)
        else:
            wholly_inside = part.apply(prepared.covers).to_numpy(dtype=bool)
            intersects = part.apply(prepared.intersects).to_numpy(dtype=bool)
            crossing = intersects & ~wholly_inside
            clipped = part.copy()
            clipped.iloc[np.flatnonzero(~intersects)] = None
            if crossing.any():
                indexes = np.flatnonzero(crossing)
                clipped.iloc[indexes] = part.iloc[indexes].intersection(mask).values
            full_intersections += int(crossing.sum())
        outputs.extend(clipped.tolist())
        stop = min(begin + batch_size, n)
        print(f"[CLIP {mode}] {stop:,}/{n:,} records", flush=True)
    clipped_frame = river.copy()
    clipped_frame.geometry = gpd.GeoSeries(outputs, index=river.index, crs=river.crs)
    bad = (~clipped_frame.geometry.isna()) & ~clipped_frame.geometry.is_valid
    repair_count = int(bad.sum())
    if repair_count:
        clipped_frame.loc[bad, "geometry"] = clipped_frame.loc[bad, "geometry"].apply(make_valid)
    clipped_frame.geometry = clipped_frame.geometry.apply(keep_line_geometry)
    keep = (~clipped_frame.geometry.isna()) & ~clipped_frame.geometry.is_empty
    discarded = int((~keep).sum())
    clipped_frame = clipped_frame.loc[keep].copy().reset_index(drop=True)
    if clipped_frame.empty:
        raise ValueError("No river line remains in the chosen display extent.")
    if not clipped_frame.geometry.is_valid.all():
        raise ValueError("Invalid river geometry remains after repair; not silently dropped.")
    return clipped_frame, {"full_intersection_records": full_intersections,
                           "post_clip_repaired_records": repair_count,
                           "empty_or_point_contacts_removed": discarded}


def regional_river_geometry(path: Path, uri: str, layer: str, source_crs, geo, args):
    """Read and cache ONLY the regional geometry; naming/style filters remain separate."""
    if geo.crs is None or not geo.crs.is_projected:
        raise ValueError("River display clipping requires the projected map CRS (metres).")
    bounds = np.asarray(geo.total_bounds, dtype=float)
    pad = args.river_buffer_km * 1000.0
    display_bounds = (bounds + np.array([-pad, -pad, pad, pad])).tolist()
    transformer = Transformer.from_crs(geo.crs, source_crs, always_xy=True)
    native_bounds = list(transformer.transform_bounds(*display_bounds, densify_pts=41))
    if not np.isfinite(native_bounds).all():
        raise ValueError("Non-finite native-CRS river read bounds.")
    mode = args.river_clip
    signature = {
        "schema": RIVER_CACHE_SCHEMA, "source": dataset_stamp(path), "layer": layer,
        "source_crs": str(source_crs), "map_crs": geo.crs.to_wkt(),
        "display_bounds": display_bounds, "native_read_bounds": native_bounds,
        "clip_mode": mode, "buffer_km": float(args.river_buffer_km),
        "country_geometry_digest": geometry_digest(geo) if mode == "country" else None,
        "shapely": shapely.__version__, "geos": shapely.geos_version_string,
        "pyogrio": pyogrio.__version__,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:32]
    cache_dir = getattr(args, "_river_cache_dir", None)
    cache_enabled = cache_dir is not None and not args.no_river_cache
    if cache_enabled:
        with Progress(f"Check regional cache: {path.name}"):
            cached = load_geometry_cache(cache_dir, key, signature, geo.crs)
        if cached is not None:
            return cached
    print(f"[RIVERS] Regional extent mode: {mode}; source: {path.name}; layer: {layer}", flush=True)
    with Progress(f"Read regional window from {path.name}"):
        river = gpd.read_file(uri, layer=layer, bbox=tuple(native_bounds), engine="pyogrio")
    read_rows = len(river)
    print(f"[RIVERS] Read {read_rows:,} records in native-CRS window", flush=True)
    if river.empty:
        raise ValueError("River source contains no features in the South Sudan display window.")
    keep = (~river.geometry.isna()) & ~river.geometry.is_empty
    source_empty = int((~keep).sum())
    river = river.loc[keep].copy().reset_index(drop=True)
    if river.empty or not river.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise ValueError("Selected layer must contain river LINE geometry, not basins/admin polygons.")
    with Progress("Reproject regional river lines"):
        river = river.to_crs(geo.crs)
    invalid = ~river.geometry.is_valid
    pre_repair = int(invalid.sum())
    if pre_repair:
        with Progress(f"Repair {pre_repair} invalid river records"):
            river.loc[invalid, "geometry"] = river.loc[invalid, "geometry"].apply(make_valid)
            river.geometry = river.geometry.apply(keep_line_geometry)
            keep = (~river.geometry.isna()) & ~river.geometry.is_empty
            if not keep.all():
                raise ValueError("An invalid source river did not repair to line geometry; inspect the data.")
    if mode == "bbox":
        mask = box(*display_bounds)  # Five coordinates, not the detailed ADM3 boundary.
    else:
        with Progress("Prepare exact national mask (zero outward buffer)"):
            national = getattr(args, "_national_union", None)
            if national is None:
                national = unary_union(geo.geometry)
            mask = national  # Exact country; padding applies ONLY to the data-read window.
    with Progress(f"Clip regional river lines in {args.clip_batch_size}-record batches ({mode})"):
        river, clip_meta = clip_river_batches(river, mask, args.clip_batch_size, mode)
    metadata = {"native_crs": str(source_crs), "read_bbox_native_crs": native_bounds,
                "read_window_rows": read_rows, "source_empty_records_removed": source_empty,
                "pre_clip_repaired_records": pre_repair, "clip_mode": mode,
                "clip_bounds_map_crs": display_bounds, "clip_map_crs": str(geo.crs),
                "clipped_regional_records": len(river), "geometry_simplified": False,
                "country_polygon_intersection_used": mode == "country",
                "adjacent_country_context_reaches_possible": mode == "bbox",
                "clip_note": ("Rectangle extent for mapping, not a national-river inventory; "
                              "no input river is simplified and no candidate flag is affected."
                              if mode == "bbox" else "Exact national union; no outward buffer. Read padding does not affect clipping."),
                "cache_hit": False, **clip_meta}
    if cache_enabled:
        with Progress("Save regional geometry cache"):
            save_geometry_cache(cache_dir, key, signature, river, metadata)
        metadata["cache_file"] = str(cache_dir / (key + ".gpkg"))
    return river, metadata




def read_rivers(path: Path, geo: gpd.GeoDataFrame, args) -> tuple[gpd.GeoDataFrame, dict]:
    uri = vector_uri(path)
    layer = choose_river_layer(uri, args.river_layer)
    info = pyogrio.read_info(uri, layer=layer)
    source_crs = info.get("crs")
    if not source_crs:
        raise ValueError("River CRS is missing; it will not be assumed to be EPSG:4326.")
    river, spatial_meta = regional_river_geometry(path, uri, layer, source_crs, geo, args)
    native_bbox = np.asarray(spatial_meta["read_bbox_native_crs"])
    read_rows = spatial_meta["read_window_rows"]
    river["source_record"] = np.arange(len(river))
    fields = {c.casefold(): c for c in river if c != "geometry"}
    id_column = fields.get("hyriv_id")
    river["river_id"] = (river[id_column].astype("string").str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
                         if id_column else river.source_record.astype("string"))
    name_columns = [fields[k] for k in ("name_en", "river_name", "name", "riv_name", "name_eng") if k in fields]
    source_name_values = {c: river[c].fillna("").astype(str).str.strip() for c in name_columns}
    river["river_name"] = ""
    river["name_source"] = "no name attribute"
    bad_names = {"", "nan", "none", "null", "unknown", "unnamed", "-99"}
    for c, values in source_name_values.items():
        usable = river.river_name.eq("") & ~values.str.lower().isin(bad_names)
        river.loc[usable, "river_name"] = values.loc[usable]
        river.loc[usable, "name_source"] = "source attribute: " + c
    name_column = ", ".join(name_columns) or None
    river["reviewed_major_override"] = pd.Series(pd.NA, index=river.index, dtype="boolean")
    mapping_matches = 0
    if args.river_names_csv:
        if not id_column:
            raise ValueError("--river-names-csv requires HYRIV_ID in the river data; coordinates are not used to guess names.")
        names = pd.read_csv(args.river_names_csv, dtype="string")
        if not {"HYRIV_ID", "river_name"}.issubset(names):
            raise ValueError("River names CSV must contain HYRIV_ID,river_name; optional major column.")
        names["HYRIV_ID"] = names.HYRIV_ID.str.strip().str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        if names.HYRIV_ID.duplicated().any() or names.HYRIV_ID.isna().any():
            raise ValueError("HYRIV_ID must be unique and nonempty in the reviewed river-name CSV.")
        names = names.set_index("HYRIV_ID")
        assigned = river.river_id.map(names.river_name)
        matched = assigned.notna() & assigned.str.strip().ne("")
        river.loc[matched, "river_name"] = assigned[matched]
        river.loc[matched, "name_source"] = "user-reviewed HYRIV_ID CSV"
        mapping_matches = int(matched.sum())
        if "major" in names:
            river["reviewed_major_override"] = boolean_values(river.river_id.map(names.major), "reviewed major", nullable=True)
    main_names = {normalized_name(n) for n in KNOWN_MAIN_NAMES}
    if args.major_names:
        main_names.update(normalized_name(n) for n in args.major_names.split(";") if n.strip())
    main_names.update(normalized_name(n) for n in RIVER_COLORS)
    river["major_by_name"] = river.river_name.map(canonical_river_name).map(normalized_name).isin(main_names)
    river["major_by_metric"] = False
    visible = pd.Series(True, index=river.index)
    if "ord_stra" in fields:
        order = pd.to_numeric(river[fields["ord_stra"]], errors="coerce")
        river["major_by_metric"] = order.ge(args.major_order)
        visible = order.ge(args.min_order) | order.isna()
        rule = f"Strahler order >= {args.major_order}; named main rivers also bold"
        detail = f"Only order >= {args.min_order} shown (plus major/name overrides; missing order retained)"
        metric = "ORD_STRA"
    elif "dis_av_cms" in fields:
        river["major_by_metric"] = pd.to_numeric(river[fields["dis_av_cms"]], errors="coerce").ge(args.major_discharge)
        rule = f"Mean discharge >= {args.major_discharge:g} m3/s; named main rivers also bold"
        detail, metric = "All available regional river lines shown", "DIS_AV_CMS"
    elif "scalerank" in fields:
        river["major_by_metric"] = pd.to_numeric(river[fields["scalerank"]], errors="coerce").le(args.major_scalerank)
        rule = f"Natural Earth scale rank <= {args.major_scalerank}; named main rivers also bold"
        detail, metric = "Generalized reference network, not a local hydrographic survey", "scalerank"
    else:
        rule, detail, metric = "Named main rivers / reviewed IDs only", "No river-order/discharge field: other lines remain thin", "names_only"
    river["is_major"] = river.major_by_name | river.major_by_metric
    override = river.reviewed_major_override.notna()
    river.loc[override, "is_major"] = river.loc[override, "reviewed_major_override"].astype(bool)
    visible = visible | river.is_major
    if override.any():
        rule += "; reviewed overrides applied"
    omitted = int((~visible).sum())
    river = river.loc[visible].copy().reset_index(drop=True)
    if river.empty:
        raise ValueError("No rivers remain after display filtering. Lower --min-order.")
    if not river.is_major.any():
        warnings.warn("No major rivers meet the current display rule. No arbitrary lines will be made major.")
    source_label = "HydroRIVERS" if "hydrorivers" in (str(path) + layer).lower() or id_column else (
        "Natural Earth 1:10m rivers/lake centerlines" if "ne_10m" in (str(path) + layer).lower() else path.name)
    metadata = {**spatial_meta, "source": str(path), "layer": layer, "source_label": source_label,
                "native_crs": source_crs, "read_bbox_native_crs": native_bbox.tolist(),
                "read_window_rows": read_rows, "shown_line_records": len(river),
                "major_line_records": int(river.is_major.sum()), "omitted_small_stream_records": omitted,
                "line_record_counts_are_not_number_of_distinct_rivers": True,
                "major_rule": rule, "detail": detail, "metric": metric,
                "major_order": args.major_order, "min_order": args.min_order,
                "major_discharge_m3s": args.major_discharge, "major_scalerank": args.major_scalerank,
                "name_field": name_column, "reviewed_name_matches": mapping_matches,
                "labels_inferred_from_coordinates": False,
                "available_names": sorted(set(river.loc[river.river_name.ne(""), "river_name"].tolist())),
                "unnamed_major_line_records": int((river.is_major & river.river_name.eq("")).sum()),
                "cartographic_emphasis_only": True, "buffer_km": args.river_buffer_km,
                "final_clip": ("Exact South Sudan mask; no outward buffer" if args.river_clip == "country" else "Regional rectangle (explicit override)")}
    return river, metadata


def download_ne(root: Path) -> Path:
    destination = root / "raw_data" / "rivers_reference" / "ne_10m_rivers_lake_centerlines.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and zipfile.is_zipfile(destination):
        return destination
    temporary = destination.with_suffix(".download")
    print("[DOWNLOAD] Natural Earth reference lines (~2 MB), not HydroRIVERS:", NE_URL, flush=True)
    try:
        request = Request(NE_URL, headers={"User-Agent": "13e-geospatial-research/1.0"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 20 * 1024 * 1024:
                    raise ValueError("Unexpectedly large reference download; aborted.")
                stream.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Downloaded reference is not a valid ZIP file.")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_river_source(root: Path, geo, args) -> tuple[gpd.GeoDataFrame, dict]:
    if args.rivers:
        path = resolve_path(args.rivers, root)
        candidates = river_candidates(path) if path.is_dir() and path.suffix.lower() != ".gdb" else [path]
    else:
        candidates = river_candidates(root / "raw_data")
    errors = []
    for path in candidates:
        try:
            river, meta = read_rivers(path, geo, args)
            meta["rejected_sources"] = errors
            print(f"[OK] Rivers: {path} | layer={meta['layer']} | shown segments={len(river)}", flush=True)
            return river, meta
        except Exception as exc:
            # Auto-discovery may encounter regional/non-line datasets. Never silently use polygons.
            errors.append({"path": str(path), "error": str(exc)})
            print(f"[RIVER SEARCH] Skipped {path.name}: {exc}", flush=True)
            if args.rivers and len(candidates) == 1:
                raise ValueError(f"Cannot use explicitly selected river dataset: {exc}") from exc
    if args.download_natural_earth:
        river, meta = read_rivers(download_ne(root), geo, args)
        meta["rejected_sources"] = errors
        meta["optional_reference_download_used"] = True
        return river, meta
    reasons = "\n".join(f"  {e['path']}: {e['error']}" for e in errors[:6])
    raise FileNotFoundError(
        "No usable local river LINE layer was found. The existing project usually stores it under "
        "raw_data/HydroRIVERS_v10_af/. A .gdb DIRECTORY is supported.\n"
        "Use --rivers PATH_TO_GDB_SHP_GPKG_GEOJSON_OR_ZIP; multi-layer datasets may need --river-layer NAME.\n"
        "Alternatively explicitly request --download-natural-earth for generalized reference lines.\n"
        "No fake river geometry or falsely successful maps were generated.\n" + reasons)


def line_parts(geometry):
    if geometry.geom_type == "LineString":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [p for g in geometry.geoms for p in line_parts(g)]
    return []


def draw_rivers(ax, river, metadata: dict, args) -> None:
    minor, major = river[~river.is_major], river[river.is_major]
    if not minor.empty:
        # Draw ALL narrow white casings before ALL dark cores. This keeps junctions
        # connected and avoids translucent blending with the flood polygons.
        # Points are display units, not geographic channel widths; major lines
        # and factor badges retain higher z-orders.
        if args.minor_casing_extra > 0:
            minor.plot(ax=ax, color=MINOR_CASING_COLOR,
                       linewidth=args.minor_width + args.minor_casing_extra,
                       alpha=MINOR_ALPHA, zorder=MINOR_CASING_ZORDER)
        minor.plot(ax=ax, color=RIVER_MINOR, linewidth=args.minor_width,
                   alpha=MINOR_ALPHA, zorder=MINOR_CORE_ZORDER)
    if not major.empty:
        # White casing keeps important channels visible over dark blue or red polygons.
        plain = major.loc[~major.named_colour]
        coloured = major.loc[major.named_colour]
        if not plain.empty:
            plain_width = max(.6, args.major_width*.48)
            plain.plot(ax=ax, color="white", linewidth=plain_width+.4, alpha=.65, zorder=5.35)
            plain.plot(ax=ax, color=RIVER_UNKNOWN, linewidth=plain_width, alpha=.65, zorder=5.4)
        if not coloured.empty:
            # Draw all casings first, then all coloured centres. The fine dark edge
            # keeps bright yellow readable on white; outer white separates blue/red fills.
            coloured.plot(ax=ax, color="white", linewidth=args.major_width + RIVER_CASING_EXTRA_PT,
                          alpha=1.0, zorder=5.45)
            coloured.plot(ax=ax, color=RIVER_EDGE_COLOR, linewidth=args.major_width + RIVER_EDGE_EXTRA_PT,
                          alpha=1.0, zorder=5.47)
            coloured.plot(ax=ax, color=coloured.line_color.tolist(), linewidth=args.major_width,
                          alpha=1.0, zorder=5.5)
    # Report layout default: river names are kept in the right-side key, not on the map.
    # To avoid clutter and keep one clean figure frame, map labels are suppressed by default.
    if getattr(args, "show_river_labels", False) is False:
        return
    named = major[major.named_colour]
    if named.empty:
        return  # HydroRIVERS commonly has reach IDs/order, not human-readable names.
    groups = []
    for name, group in named.groupby("display_name", sort=True):
        geom = unary_union(group.geometry)
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
        pieces = line_parts(geom)
        if pieces:
            line = max(pieces, key=lambda x: x.length)
            groups.append((line.length, name, line, river_label_colour(name, str(group.line_color.iloc[0]))))
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [t.get_window_extent(renderer).expanded(1.05, 1.15) for t in ax.texts if t.get_text()]
    axes_box = ax.get_window_extent()
    for _, name, line, river_color in sorted(groups, key=lambda t: (-t[0], t[1]))[:args.max_river_labels]:
        best = None
        for fraction in (.50, .32, .68, .18, .82):
            point = line.interpolate(fraction, normalized=True)
            for dx, dy in ((3, 5), (3, -9), (-3, 7)):
                artist = ax.annotate(name, (point.x, point.y), xytext=(dx, dy), textcoords="offset points",
                                     fontsize=8.0, color=river_color, fontstyle="italic", weight="bold",
                                     ha="left" if dx > 0 else "right", va="bottom", zorder=6.1,
                                     path_effects=[pe.withStroke(linewidth=2.3, foreground="white")])
                bb = artist.get_window_extent(renderer).expanded(1.06, 1.15)
                outside = not (axes_box.contains(bb.x0, bb.y0) and axes_box.contains(bb.x1, bb.y1))
                collisions = sum(bb.overlaps(old) for old in occupied)
                score = 100 * int(outside) + collisions
                if best is None or score < best[0]:
                    best = (score, point, dx, dy)
                artist.remove()
                if score == 0:
                    break
            if best[0] == 0:
                break
        _, point, dx, dy = best
        label = ax.annotate(name, (point.x, point.y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=8, color=river_color, fontstyle="italic", weight="bold",
                            ha="left" if dx > 0 else "right", va="bottom", zorder=6.1,
                            path_effects=[pe.withStroke(linewidth=2.3, foreground="white")])
        occupied.append(label.get_window_extent(renderer).expanded(1.06, 1.15))


def river_key(fig, metadata: dict, args) -> None:
    side = fig._named_river_sidebar
    side_text(side, 0, .99, "MAJOR RIVERS", fontsize=11, weight="bold", color=INK)
    is_ref = metadata.get("named_reference_overlay_used", False)
    is_generalized = metadata.get("reference_geometry_not_a_local_survey", False)
    description = ("Named reference centre-lines\n(separate, generalized geometry)" if is_ref else
                   "Generalized reference centre-lines\nNames from source attributes" if is_generalized else
                   "Names from source attributes\nor reviewed reach IDs")
    side_text(side, 0, .935, description, fontsize=9, color=MUTED, linespacing=1.45)
    entries = [(r["river_name"], r["colour_hex"], args.major_width)
               for r in metadata.get("named_river_colour_key", [])]
    if metadata.get("unnamed_or_other_major_line_records", 0):
        entries.append(("Other emphasized reaches", RIVER_UNKNOWN, max(.6, args.major_width*.48)))
    if metadata.get("minor_line_records", 0):
        entries.append(("Other shown river reaches", RIVER_MINOR, args.minor_width))
    step = min(.068, .67 / max(len(entries), 1))
    for i, (label, color, width) in enumerate(entries):
        y = .82 - i*step
        # A longer key sample with the SAME casing/edge as the mapped named line.
        is_named = label in {r["river_name"] for r in metadata.get("named_river_colour_key", [])}
        if label == "Other shown river reaches" and args.minor_casing_extra > 0:
            side.plot([.015, .19], [y-.013, y-.013], transform=side.transAxes,
                      color=MINOR_CASING_COLOR, linewidth=width + args.minor_casing_extra,
                      alpha=MINOR_ALPHA, solid_capstyle="round")
        if is_named:
            side.plot([.015, .19], [y-.013, y-.013], transform=side.transAxes,
                      color="white", linewidth=width + RIVER_CASING_EXTRA_PT,
                      solid_capstyle="round")
            side.plot([.015, .19], [y-.013, y-.013], transform=side.transAxes,
                      color=RIVER_EDGE_COLOR, linewidth=width + RIVER_EDGE_EXTRA_PT,
                      solid_capstyle="round")
        side.plot([.015, .19], [y-.013, y-.013], transform=side.transAxes,
                  color=color, linewidth=width, solid_capstyle="round")
        side_text(side, .235, y, label, fontsize=9.0 if len(entries)>10 else 9.5,
                  color=river_label_colour(label, color) if is_named else MUTED,
                  weight="bold" if is_named else "normal")
    side_text(side, 0, .075, "Colour = river identity\nWidth = display emphasis\nNot a tributary or risk classification.",
              fontsize=8.8, color=MUTED, linespacing=1.45)
    if not metadata.get("named_river_colour_key"):
        side_text(side, 0, .50, "River names unavailable.\nNo names have been inferred.", fontsize=10, color="#9A3412")


def derived_outlines(geo, level: str):
    code = level + "_pcode"
    if code not in geo:
        return None
    if geo[code].isna().any():
        raise ValueError(f"Missing parent {code} in ADM3 boundaries.")
    keep = [code, "geometry"]
    if level == "adm1" and "adm1_name" in geo:
        keep.insert(1, "adm1_name")
    return geo[keep].dissolve(by=code, as_index=False)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, help="Project root; automatically inferred from script path")
    p.add_argument("--input", type=Path, help="13d folder/ZIP/full CSV; v4 population output is preferred")
    p.add_argument("--out", type=Path, help="Default: processed_data/multivariate_flood_modeling/13e_humanitarian_maps_report_clean")
    p.add_argument("--scenario", choices=["population-first", "all", "main", "population-sensitivity", "baseline"], default="population-first",
                   help="Default: existing four-factor 13d scenario in main maps/; verification label unchanged")
    p.add_argument("--include-baseline", action="store_true", help="Also draw a baseline_comparison/ folder, when saved")
    p.add_argument("--named-rivers", type=Path, help="Named reference LINE layer; never names assigned by proximity")
    p.add_argument("--named-river-layer", help="Optional layer name in the named reference dataset")
    p.add_argument("--offline", action="store_true", help="Never download reference data; require local named rivers")
    p.add_argument("--allow-unnamed-rivers", action="store_true", help="Explicitly allow a generic river key if names unavailable")
    p.add_argument("--max-coloured-rivers", type=int, default=12, help="Maximum named major rivers given distinct colours (1..12)")
    p.add_argument("--adm3-geojson", type=Path, help="Default: raw_data/administrative_boundaries/ssd_admin3.geojson")
    p.add_argument("--rivers", type=Path, help="River dataset or folder. Supports .gdb, .shp, .gpkg, .geojson and .zip")
    p.add_argument("--river-layer", help="Explicit line layer name for a multi-layer river dataset")
    p.add_argument("--river-names-csv", type=Path, help="Reviewed HYRIV_ID,river_name[,major] table; never inferred from proximity")
    p.add_argument("--major-names", help="Additional exact main-river names separated by semicolons")
    p.add_argument("--major-order", type=int, default=6, help="Major river Strahler order >= this value, default 6")
    p.add_argument("--min-order", type=int, default=3, help="Minimum visible Strahler order, default 3; major/name overrides always retained")
    p.add_argument("--major-discharge", type=float, default=100, help="Fallback major threshold in m3/s only if ORD_STRA absent")
    p.add_argument("--major-scalerank", type=int, default=4, help="Natural Earth major scale rank <= this value")
    p.add_argument("--major-width", type=float, default=2.5, help="Major coloured line width in points, default 2.5; not channel width")
    p.add_argument("--minor-width", type=float, default=.60,
                   help="Other reaches: opaque dark core width in points, default 0.60")
    p.add_argument("--minor-casing-extra", type=float, default=.50,
                   help="Extra TOTAL white casing width in points, default 0.50 (0 disables); keep narrow in dense networks")
    p.add_argument("--river-buffer-km", type=float, default=12, help="Read-window padding in km (default 12); NEVER expands the country clipping mask")
    p.add_argument("--river-clip", choices=["bbox", "country"], default="country",
                   help="Default country: exact country-only clipping with cached batched predicates. bbox: optional regional overview")
    p.add_argument("--clip-batch-size", type=int, default=512, help="River clip records per progress batch (default 512)")
    p.add_argument("--no-river-cache", action="store_true", help="Do not read/write the generated regional geometry cache")
    p.add_argument("--download-natural-earth", action="store_true", help="Explicitly allow small reference download ONLY if no local river file works")
    p.add_argument("--show-river-labels", action="store_true", help="Optional: draw river names on the map itself. Default layout keeps names only in the right-side river key.")
    p.add_argument("--max-river-labels", type=int, default=8)
    p.add_argument("--no-state-labels", action="store_true")
    p.add_argument("--badge-size", type=float, default=11)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--formats", nargs="+", choices=["png", "svg"], default=["png", "svg"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 72 <= args.dpi <= 1200 or not 7 <= args.badge_size <= 24:
        raise ValueError("Use dpi 72..1200 and badge-size 7..24 points.")
    if not 1 <= args.min_order <= args.major_order <= 12:
        raise ValueError("Require 1 <= min-order <= major-order <= 12.")
    if not (0 < args.minor_width < args.major_width <= 8 and 0 <= args.river_buffer_km <= 100):
        raise ValueError("Require 0 < minor-width < major-width <= 8; river-buffer-km must be 0..100.")
    if not np.isfinite(args.minor_casing_extra) or not 0 <= args.minor_casing_extra <= 1.5:
        raise ValueError("--minor-casing-extra must be finite and between 0 and 1.5 points.")
    if args.max_river_labels < 0 or not np.isfinite(args.major_discharge) or args.major_discharge <= 0:
        raise ValueError("Invalid river label or discharge options.")
    if not 1 <= args.max_coloured_rivers <= 12:
        raise ValueError("--max-coloured-rivers must be between 1 and 12.")
    if args.offline and args.download_natural_earth:
        raise ValueError("--offline and --download-natural-earth contradict each other.")
    if not 16 <= args.clip_batch_size <= 10000:
        raise ValueError("--clip-batch-size must be between 16 and 10000.")
    args.formats = list(dict.fromkeys(args.formats))
    root = args.root.expanduser().resolve() if args.root else project_root()
    base = root / "processed_data" / "multivariate_flood_modeling"
    source = resolve_path(args.input, root) if args.input else pick_13d(base)
    out = resolve_path(args.out, root) if args.out else base / "13e_humanitarian_maps_report_clean"
    args._river_cache_dir = base / "_13e_river_cache"
    print(f"[RUN] {VERSION} | river-clip={args.river_clip} | output={out}", flush=True)
    if args.river_clip == "bbox":
        print("[NOTE] Rivers use a rectangular display extent, not an exact country cut. "
              "Adjacent-country context reaches may appear; screening data are unchanged.", flush=True)
    if out == source or (source.is_dir() and out.is_relative_to(source)):
        raise ValueError("13e output must not be inside the 13d input folder.")
    if out.is_relative_to(root / "raw_data"):
        raise ValueError("13e must not write outputs in raw_data.")
    if args.river_names_csv:
        args.river_names_csv = resolve_path(args.river_names_csv, root)
    boundary = resolve_path(args.adm3_geojson, root) if args.adm3_geojson else root / "raw_data/administrative_boundaries/ssd_admin3.geojson"
    with Progress("Read and validate saved 13d scenarios (read-only)"):
        items = read_scenarios(source, "all")
        ready = select_display_scenarios([(item, *validate_saved_result(item)) for item in items], args)
    reference_ids = set(ready[0][1].adm_id)
    if any(set(df.adm_id) != reference_ids for _, df, _, _ in ready):
        raise ValueError("13d scenarios use different ADM3 IDs; stop before mapping.")
    with Progress("Read, validate and project ADM3 boundaries"):
        geo, spatial_audit = load_geography(boundary, ready[0][1])
    geo_columns = [c for c in ("adm_id", "adm3_pcode", "adm2_pcode", "adm1_pcode", "adm1_name", "geometry") if c in geo]
    shape_only = geo[geo_columns].copy()
    with Progress("Prepare ADM2, ADM1 and national outlines once"):
        adm2, adm1 = derived_outlines(shape_only, "adm2"), derived_outlines(shape_only, "adm1")
        args._national_union = unary_union(shape_only.geometry)
        args._national_outline = gpd.GeoSeries([args._national_union], crs=shape_only.crs).boundary
    rivers, river_meta = assemble_named_rivers(root, shape_only, args)
    setup_style()
    out.mkdir(parents=True, exist_ok=True)
    shared = out / "river_data"
    shared.mkdir(exist_ok=True)
    # Use a dedicated file so repeat runs replace only the 13e-generated layer.
    river_out = shared / "01_rivers_display.gpkg"
    if river_out.exists():
        river_out.unlink()
    # GeoPackage column names are case-insensitive. A source RIVER_NAME plus our
    # canonical river_name would otherwise fail even though pandas permits both.
    # Export a stable canonical display schema; retain full attributes in the CSV audit.
    canonical = ["river_id", "river_name", "display_name", "name_source", "source_record",
                 "geometry_source", "is_major", "major_by_name", "major_by_metric",
                 "named_colour", "line_color", "is_named_reference_overlay", "geometry"]
    exported = rivers[[c for c in canonical if c in rivers]].copy()
    fields = {c.casefold(): c for c in rivers.columns if c != "geometry"}
    for raw_field in ("ord_stra", "dis_av_cms", "next_down"):
        if raw_field in fields:
            exported["source_" + raw_field] = rivers[fields[raw_field]]
    with Progress("Export the documented regional river display layer"):
        exported.to_file(river_out, layer="river_display", driver="GPKG", engine="pyogrio")
    rivers.drop(columns="geometry").to_csv(shared / "02_river_display_audit.csv", index=False)
    write_json(shared / "03_river_source_and_styling.json", river_meta)
    key_rows = [{**r, "name_sources": "; ".join(r["name_sources"]),
                 "geometry_sources": "; ".join(r["geometry_sources"])}
                for r in river_meta["named_river_colour_key"]]
    pd.DataFrame(key_rows, columns=["river_name", "colour_hex", "colour_name", "label_colour_hex",
                                    "line_width_points", "line_records", "name_sources", "geometry_sources",
                                    "separate_reference_geometry"]).to_csv(
        shared / "04_named_river_colour_key.csv", index=False)
    completed = []
    primary_has_population = ready[0][3]["population_factor_enabled"]
    for scenario_index, (item, df, meta, preservation) in enumerate(ready):
        # Presentation priority is NOT validation: preserve provisional labels.
        scenario_out = out if scenario_index == 0 else out / "baseline_comparison"
        scenario_out.mkdir(parents=True, exist_ok=True)
        print(f"[MAPS] {preservation['scenario']}: {len(df)} ADM3; {preservation['review_candidates']} candidates; "
              f"{preservation['worsening_candidates']} worsening candidates", flush=True)
        cutoffs = {k: float(v) for k, v in meta["numeric_thresholds"].items() if v is not None}
        args.quantile = float(meta["quantile"])  # Exact saved value; not a new tuning option.
        counts = factor_summary(df, cutoffs)
        view_summary, view_factors, view_membership = build_view_tables(df, counts)
        colliding = [c for c in geo_columns if c != "adm_id" and c in df]
        mapped = shape_only.merge(df.drop(columns=colliding), on="adm_id", validate="one_to_one")
        outputs, map_audit = plot_spatial_maps(mapped, adm2, adm1, df, cutoffs, counts,
                                               scenario_out / "maps", args, rivers, river_meta)
        # Verify every label belongs to the existing 13d candidate set.
        for name, expected in (("13_marker_placement_audit.csv", preservation["candidate_ids"]),
                               ("14_worsening_marker_placement_audit.csv", preservation["worsening_candidate_ids"]),
                               ("15_high_recent_marker_placement_audit.csv", preservation["high_recent_candidate_ids"])):
            audit_table = pd.read_csv(scenario_out / "maps" / name, dtype={"adm_id": "string", "factor_codes": "string"})
            if sorted(audit_table.adm_id.tolist()) != expected:
                raise ValueError(f"Map labels do not preserve the saved 13d selection: {name}")
            joined = audit_table.set_index("adm_id").factor_codes.reindex(expected)
            if not joined.equals(df.set_index("adm_id").factor_codes.reindex(expected).astype("string")):
                raise ValueError(f"Factor labels disagree with saved flags: {name}")
        if item["input_csv_path"]:
            after = hashlib.sha256(Path(item["input_csv_path"]).read_bytes()).hexdigest()
            if after != preservation["source_csv_sha256"]:
                raise ValueError("The 13d input changed during this run; do not mix outputs from different analyses.")
        counts.to_csv(scenario_out / "01_factor_counts_preserved.csv", index=False)
        df[["adm_id", "region_name", "priority_review_candidate", "candidate_via_recent_activity", "candidate_via_change_only", "increasing_flood_and_exposure", "factor_codes"]].to_csv(
            scenario_out / "02_selection_display_audit.csv", index=False)
        (scenario_out / "03_original_13d_methodology.json").write_bytes(item["meta_bytes"])
        view_summary.to_csv(scenario_out / "06_map_selection_summary.csv", index=False)
        view_factors.to_csv(scenario_out / "07_factor_counts_by_map.csv", index=False)
        df.loc[view_mask(df, "high_recent_activity")].to_csv(scenario_out / "08_high_recent_activity_candidates.csv", index=False)
        df.loc[view_mask(df, "high_change")].to_csv(scenario_out / "09_high_positive_change_candidates.csv", index=False)
        view_membership.to_csv(scenario_out / "10_paired_map_membership.csv", index=False)
        caption_file = write_figure_captions(scenario_out, df, cutoffs, args.quantile, river_meta)
        report = {"figure_captions_file": caption_file.name, "version": VERSION, "run_time_utc": datetime.now(timezone.utc).isoformat(),
                  "preservation": preservation, "geometry": spatial_audit, "map_quality": map_audit,
                  "river_source": river_meta, "numeric_thresholds_preserved": cutoffs,
                  "parent_outlines": "Dissolved from the SAME matched ADM3 geometry; no independent ADM1 code mismatch",
                  "map_files": ["maps/" + name for name in outputs],
                  "focused_maps": {"high_recent_activity": MAP_VIEWS["high_recent_activity"]["map_stem"],
                                   "high_positive_change": MAP_VIEWS["high_change"]["map_stem"]},
                  "map_selection_summary": view_summary.to_dict(orient="records"),
                  "map_factor_counts_file": "07_factor_counts_by_map.csv",
                  "paired_map_membership_file": "10_paired_map_membership.csv",
                  "recent_and_change_views_may_overlap": True,
                  "context_layers_deferred_to_13f": ["lakes", "wetlands", "terrain", "seasonal_rainfall"],
                  "limitations": ["Country-only is the default: exact ADM3 union, zero outward buffer; explicit bbox override is a regional overview.",
                                  "River lines are reference centerlines, not measured channel widths.",
                                  "Major styling is a display rule, not an official flood-risk classification.",
                                  "No new river factor, causal relationship, connectivity test or selection rule is added.",
                                  "Other displayed reaches are not a verified tributary/branch classification; the filtered display is not a drainage-density dataset.",
                                  "Map 12 excludes change-only candidates; map 10 retains the original combined selection.",
                                  "2019-2025 activity is historical, not live disaster severity; period change is not an annual trend slope.",
                                  "Population scenario/provisional status is inherited from 13d; not revalidated by 13e.",
                                  "The top-level maps/ folder denotes display preference, not validation or operational approval.",
                                  "Named reference geometry, when used, is independent of HydroRIVERS and must not be used for local distance measurement.",
                                  "River-name labels are drawn only from actual source attributes or a user-reviewed ID table.",
                                  "Circle anchors mark an administrative unit, not a facility or affected person's location.",
                                  "Review label positions on the real boundary map before publication."]}
        report["factor_colour_key_file"] = "05_factor_colour_key.json"
        write_json(scenario_out / "05_factor_colour_key.json", factor_colour_key(df))
        write_json(scenario_out / "04_13e_methodology_and_checks.json", report)
        completed.append({"scenario": preservation["scenario"], "output": str(scenario_out),
                          "candidate_count": preservation["review_candidates"], "worsening_count": preservation["worsening_candidates"],
                          "high_recent_count": preservation["high_recent_candidates"],
                          "map_count": 5, "exported_image_files": len(outputs), "flags_unchanged": True})
        print(f"[OK] Five maps in {scenario_out / 'maps'}", flush=True)
        print(f"[OK] Map 12 high recent activity: {preservation['high_recent_candidates']}; map 11 high change: {preservation['worsening_candidates']}; sets may overlap.", flush=True)
    write_json(out / "00_run_manifest.json", {"version": VERSION, "scenarios": completed,
        "rivers": river_meta["source"], "layer": river_meta["layer"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_13d": str(source), "original_13d_overwritten": False,
        "primary_display_includes_population4": primary_has_population,
        "primary_analysis_scenario": ready[0][3]["scenario"],
        "display_preference_is_not_population_validation": True})
    print("[OK] Factor colours unchanged: 1=health/red; 2=cropland/amber; 3=rangeland/green; 4=population/purple.")
    print(f"[OK] River styles: named colours unchanged; other reaches {RIVER_MINOR}, "
          f"{args.minor_width:.2f} pt opaque core + {args.minor_casing_extra:.2f} pt extra white casing. No inferred tributary labels.")
    print("[NOTE] Lakes, wetlands, terrain and seasonal rainfall are deferred to 13f; no such data are required by 13e.")
    for scenario in completed:
        if scenario["scenario"] == "population_sensitivity":
            print(f"[NOTE] Population 4 IS SHOWN, with PROVISIONAL status retained: {Path(scenario['output']) / 'maps'}")
    print(f"[OK] Complete: {out}", flush=True)
    if river_meta["unnamed_or_other_major_line_records"]:
        print("[NOTE] Unnamed/other major reaches remain neutral; no names assigned from location.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Run interrupted by the user. This is NOT a completed map run. "
              "13d inputs were not modified; completed regional caches can be reused.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
