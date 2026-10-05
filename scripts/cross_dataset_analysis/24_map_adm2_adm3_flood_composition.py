#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24_map_adm2_adm3_flood_composition_FINAL.py

JBG060 Capstone Data Challenge — South Sudan Flood Analysis

Purpose
-------
Create final nested ADM2–ADM3 choropleth maps using the final outputs from Scripts 21, 22, and 23.

The maps are designed to make cross-scale flood patterns visible:
1. ADM3 recent combined flood extent within ADM2 boundaries.
2. ADM3 unusual-only share within ADM2 boundaries.
3. ADM3 local amplification hotspots within ADM2 boundaries.
4. ADM2 cross-scale change classes with ADM3 local worsening highlighted.
5. ADM3 robust local amplification hotspots validated across Q70–Q80.

Inputs
------
Script 21:
processed_data/cross_dataset_analysis/03_flood_composition/
21_adm3_recurring_unusual_composition/tables/
    21_adm3_flood_mask_composition_summary.csv

Script 22:
processed_data/cross_dataset_analysis/03_flood_composition/
22_adm2_adm3_cross_scale_composition/tables/
    22_adm2_cross_scale_comparison.csv
    22_adm3_local_cross_scale_signals.csv

Script 23:
processed_data/cross_dataset_analysis/03_flood_composition/
23_cross_scale_hotspot_robustness/tables/
    23_robust_local_amplification_hotspots.csv

Administrative boundaries:
ADM2 and ADM3 vector files in .shp, .gpkg, or .geojson format.

Outputs
-------
processed_data/cross_dataset_analysis/03_flood_composition/
24_adm2_adm3_flood_composition_maps/

    figures/
        24_01_adm3_recent_combined_extent_map.png
        24_02_adm3_unusual_only_share_map.png
        24_03_adm3_local_amplification_hotspots_map.png
        24_04_cross_scale_change_classes_map.png

    tables/
        24_mapping_join_qc.csv

    qc/
        24_mapping_qc_summary.txt

Important interpretation
------------------------
ADM2 boundaries are used as contextual regional outlines.
ADM3 fills show local spatial variation.
The maps are descriptive visualizations of Scripts 21, 22, and 23 and do not create
new flood metrics.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = None

ADM2_BOUNDARY_FILE = None
ADM3_BOUNDARY_FILE = None

ADM3_SUMMARY_CSV = None
CROSS_SCALE_CSV = None
LOCAL_HOTSPOT_CSV = None
ROBUST_AMPLIFICATION_CSV = None

# Labels are intentionally sparse to avoid clutter.
LABEL_ADM2 = True
LABEL_TOP_ADM3 = True
TOP_ADM3_LABEL_N = 15

# Output quality.
DPI = 300

# =============================================================================
# PROJECT PATHS
# =============================================================================

def infer_project_root() -> Path:
    if PROJECT_ROOT:
        return Path(PROJECT_ROOT).expanduser().resolve()

    here = Path(__file__).resolve()

    for p in here.parents:
        if (p / "processed_data").exists():
            return p

    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "processed_data").exists():
            return p

    return cwd


ROOT = infer_project_root()

COMPOSITION_ROOT = (
    ROOT
    / "processed_data"
    / "cross_dataset_analysis"
    / "03_flood_composition"
)

ADM3_SUMMARY_DIR = (
    COMPOSITION_ROOT
    / "21_adm3_recurring_unusual_composition"
    / "tables"
)

CROSS_SCALE_DIR = (
    COMPOSITION_ROOT
    / "22_adm2_adm3_cross_scale_composition"
    / "tables"
)

ROBUSTNESS_DIR = (
    COMPOSITION_ROOT
    / "23_cross_scale_hotspot_robustness"
    / "tables"
)

OUTPUT_ROOT = (
    COMPOSITION_ROOT
    / "24_adm2_adm3_flood_composition_maps"
)

FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
QC_DIR = OUTPUT_ROOT / "qc"

for d in (FIG_DIR, TABLE_DIR, QC_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SAFE FILE SEARCH
# =============================================================================

def resolve_unique_file(manual_path, preferred_path, filename):
    if manual_path:
        p = Path(manual_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manual path does not exist: {p}")
        return p

    if preferred_path.exists():
        return preferred_path

    search_roots = [
        COMPOSITION_ROOT,
        ROOT / "processed_data",
        ROOT,
    ]

    candidates = []
    seen = set()

    for base in search_roots:
        if not base.exists():
            continue

        for p in base.rglob(filename):
            key = str(p.resolve())
            if key not in seen:
                candidates.append(p.resolve())
                seen.add(key)

    if not candidates:
        return None

    if len(candidates) > 1:
        msg = "\n".join(f"  - {p}" for p in sorted(candidates))
        raise RuntimeError(
            f"Multiple candidate files found for {filename}.\n"
            f"Please set the corresponding manual path in CONFIG.\n{msg}"
        )

    return candidates[0]


ADM3_SUMMARY_PATH = resolve_unique_file(
    ADM3_SUMMARY_CSV,
    ADM3_SUMMARY_DIR / "21_adm3_flood_mask_composition_summary.csv",
    "21_adm3_flood_mask_composition_summary.csv",
)

CROSS_SCALE_PATH = resolve_unique_file(
    CROSS_SCALE_CSV,
    CROSS_SCALE_DIR / "22_adm2_cross_scale_comparison.csv",
    "22_adm2_cross_scale_comparison.csv",
)

LOCAL_HOTSPOT_PATH = resolve_unique_file(
    LOCAL_HOTSPOT_CSV,
    CROSS_SCALE_DIR / "22_adm3_local_cross_scale_signals.csv",
    "22_adm3_local_cross_scale_signals.csv",
)

ROBUST_AMPLIFICATION_PATH = resolve_unique_file(
    ROBUST_AMPLIFICATION_CSV,
    ROBUSTNESS_DIR / "23_robust_local_amplification_hotspots.csv",
    "23_robust_local_amplification_hotspots.csv",
)


# =============================================================================
# BOUNDARY SEARCH
# =============================================================================

VECTOR_EXTENSIONS = {".shp", ".gpkg", ".geojson"}


def boundary_score(path: Path, level: str) -> tuple:
    s = str(path).lower()

    level_tokens = {
        "adm2": ("adm2", "admin2", "admin_2", "level2", "level_2"),
        "adm3": ("adm3", "admin3", "admin_3", "level3", "level_3"),
    }[level]

    has_level = any(t in s for t in level_tokens)
    south_sudan_hint = any(
        t in s for t in ("south_sudan", "southsudan", "ssd", "administrative")
    )

    return (
        0 if has_level else 1,
        0 if south_sudan_hint else 1,
        len(str(path)),
        str(path),
    )


def find_boundary(manual_path, level: str):
    if manual_path:
        p = Path(manual_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"{level.upper()} boundary file not found: {p}")
        return p

    candidates = []

    search_roots = [
        ROOT / "raw_data",
        ROOT / "processed_data",
        ROOT / "data",
        ROOT,
    ]

    seen = set()

    for base in search_roots:
        if not base.exists():
            continue

        for p in base.rglob("*"):
            if p.suffix.lower() not in VECTOR_EXTENSIONS:
                continue

            key = str(p.resolve())
            if key in seen:
                continue

            seen.add(key)

            s = str(p).lower()
            if level in s or f"admin{level[-1]}" in s or f"level{level[-1]}" in s:
                candidates.append(p.resolve())

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: boundary_score(p, level))

    # Do not silently guess if the first two files score equally.
    if len(candidates) > 1:
        first_score = boundary_score(candidates[0], level)[:2]
        second_score = boundary_score(candidates[1], level)[:2]

        if first_score == second_score:
            msg = "\n".join(f"  - {p}" for p in candidates[:20])
            raise RuntimeError(
                f"Multiple plausible {level.upper()} boundary files found.\n"
                f"Please set {level.upper()}_BOUNDARY_FILE in CONFIG.\n{msg}"
            )

    return candidates[0]


ADM2_BOUNDARY_PATH = find_boundary(ADM2_BOUNDARY_FILE, "adm2")
ADM3_BOUNDARY_PATH = find_boundary(ADM3_BOUNDARY_FILE, "adm3")


# =============================================================================
# COLUMN HELPERS
# =============================================================================

def pick_column(df, exact=(), contains=(), excludes=()):
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in exact:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if contains:
        for c in df.columns:
            lc = str(c).lower()
            if all(token.lower() in lc for token in contains):
                if not any(token.lower() in lc for token in excludes):
                    return c

    return None


def detect_boundary_columns(gdf, level):
    if level == "adm2":
        name = pick_column(
            gdf,
            exact=(
                "ADM2_NAME", "adm2_name", "NAME_2", "name_2",
                "admin2_name", "ADM2_EN", "ADM2NAME"
            ),
            contains=("adm2", "name"),
        )
        gid = pick_column(
            gdf,
            exact=(
                "ADM2_ID", "adm2_id", "GID_2", "gid_2",
                "ADM2_PCODE", "adm2_pcode", "ADM2CODE"
            ),
        )
        parent_name = None
        parent_id = None

    else:
        name = pick_column(
            gdf,
            exact=(
                "ADM3_NAME", "adm3_name", "NAME_3", "name_3",
                "admin3_name", "ADM3_EN", "ADM3NAME"
            ),
            contains=("adm3", "name"),
        )
        gid = pick_column(
            gdf,
            exact=(
                "ADM3_ID", "adm3_id", "GID_3", "gid_3",
                "ADM3_PCODE", "adm3_pcode", "ADM3CODE"
            ),
        )
        parent_name = pick_column(
            gdf,
            exact=(
                "ADM2_NAME", "adm2_name", "NAME_2", "name_2",
                "admin2_name", "ADM2_EN", "ADM2NAME"
            ),
            contains=("adm2", "name"),
        )
        parent_id = pick_column(
            gdf,
            exact=(
                "ADM2_ID", "adm2_id", "GID_2", "gid_2",
                "ADM2_PCODE", "adm2_pcode", "ADM2CODE"
            ),
        )

    return {
        "name": name,
        "id": gid,
        "parent_name": parent_name,
        "parent_id": parent_id,
    }


def clean_text(series):
    return (
        series.astype(str)
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )


# =============================================================================
# LOAD
# =============================================================================

print("=" * 105)
print("24 — ADM2–ADM3 FINAL NESTED FLOOD COMPOSITION MAPS")
print("=" * 105)
print(f"Project root       : {ROOT}")
print(f"ADM3 summary       : {ADM3_SUMMARY_PATH}")
print(f"Cross-scale table  : {CROSS_SCALE_PATH}")
print(f"Local hotspot table: {LOCAL_HOTSPOT_PATH}")
print(f"Robust hotspot table: {ROBUST_AMPLIFICATION_PATH}")
print(f"ADM2 boundary      : {ADM2_BOUNDARY_PATH}")
print(f"ADM3 boundary      : {ADM3_BOUNDARY_PATH}")
print(f"Output directory   : {OUTPUT_ROOT}")

if ADM3_SUMMARY_PATH is None:
    raise FileNotFoundError("Could not find Script 21 ADM3 summary.")
if CROSS_SCALE_PATH is None:
    raise FileNotFoundError("Could not find Script 22 cross-scale comparison table.")
if LOCAL_HOTSPOT_PATH is None:
    raise FileNotFoundError("Could not find Script 22 ADM3 hotspot table.")
if ROBUST_AMPLIFICATION_PATH is None:
    raise FileNotFoundError("Could not find Script 23 robust local amplification table.")
if ADM2_BOUNDARY_PATH is None:
    raise FileNotFoundError(
        "Could not find ADM2 boundary file. "
        "Set ADM2_BOUNDARY_FILE manually in CONFIG."
    )
if ADM3_BOUNDARY_PATH is None:
    raise FileNotFoundError(
        "Could not find ADM3 boundary file. "
        "Set ADM3_BOUNDARY_FILE manually in CONFIG."
    )

adm3_summary = pd.read_csv(ADM3_SUMMARY_PATH)
cross = pd.read_csv(CROSS_SCALE_PATH)
local = pd.read_csv(LOCAL_HOTSPOT_PATH)
robust_amp = pd.read_csv(ROBUST_AMPLIFICATION_PATH)

adm2_gdf = gpd.read_file(ADM2_BOUNDARY_PATH)
adm3_gdf = gpd.read_file(ADM3_BOUNDARY_PATH)

adm2_cols = detect_boundary_columns(adm2_gdf, "adm2")
adm3_cols = detect_boundary_columns(adm3_gdf, "adm3")

if adm2_cols["name"] is None and adm2_cols["id"] is None:
    raise ValueError(
        f"Could not identify ADM2 name or ID column.\n"
        f"Available columns: {list(adm2_gdf.columns)}"
    )

if adm3_cols["name"] is None and adm3_cols["id"] is None:
    raise ValueError(
        f"Could not identify ADM3 name or ID column.\n"
        f"Available columns: {list(adm3_gdf.columns)}"
    )


# =============================================================================
# CRS
# =============================================================================

if adm2_gdf.crs is None:
    raise ValueError("ADM2 boundary has no CRS.")
if adm3_gdf.crs is None:
    raise ValueError("ADM3 boundary has no CRS.")

if adm3_gdf.crs != adm2_gdf.crs:
    adm3_gdf = adm3_gdf.to_crs(adm2_gdf.crs)


# =============================================================================
# STANDARDIZE BOUNDARY KEYS
# =============================================================================

if adm2_cols["id"] is not None:
    adm2_gdf["_adm2_id_key"] = clean_text(adm2_gdf[adm2_cols["id"]])
if adm2_cols["name"] is not None:
    adm2_gdf["_adm2_name_key"] = clean_text(adm2_gdf[adm2_cols["name"]])

if adm3_cols["id"] is not None:
    adm3_gdf["_adm3_id_key"] = clean_text(adm3_gdf[adm3_cols["id"]])
if adm3_cols["name"] is not None:
    adm3_gdf["_adm3_name_key"] = clean_text(adm3_gdf[adm3_cols["name"]])
if adm3_cols["parent_id"] is not None:
    adm3_gdf["_adm2_id_key"] = clean_text(adm3_gdf[adm3_cols["parent_id"]])
if adm3_cols["parent_name"] is not None:
    adm3_gdf["_adm2_name_key"] = clean_text(adm3_gdf[adm3_cols["parent_name"]])


# =============================================================================
# PREPARE SCRIPT TABLE KEYS
# =============================================================================

for df in (adm3_summary, local, robust_amp):
    if "ADM3_ID_STD" in df.columns:
        df["_adm3_id_key"] = clean_text(df["ADM3_ID_STD"])
    if "ADM3_NAME_STD" in df.columns:
        df["_adm3_name_key"] = clean_text(df["ADM3_NAME_STD"])
    if "ADM2_ID_STD" in df.columns:
        df["_adm2_id_key"] = clean_text(df["ADM2_ID_STD"])
    if "ADM2_NAME_STD" in df.columns:
        df["_adm2_name_key"] = clean_text(df["ADM2_NAME_STD"])

if "ADM2_ID_STD" in cross.columns:
    cross["_adm2_id_key"] = clean_text(cross["ADM2_ID_STD"])
if "ADM2_NAME_STD" in cross.columns:
    cross["_adm2_name_key"] = clean_text(cross["ADM2_NAME_STD"])


# =============================================================================
# JOIN ADM3
# =============================================================================

use_adm3_id = (
    "_adm3_id_key" in adm3_gdf.columns
    and "_adm3_id_key" in adm3_summary.columns
)

if use_adm3_id:
    adm3_join_key = "_adm3_id_key"
else:
    if "_adm3_name_key" not in adm3_gdf.columns or "_adm3_name_key" not in adm3_summary.columns:
        raise ValueError("Cannot join ADM3 boundary to Script 21 table by ID or name.")

    # Use ADM2 parent name as a second key when available to protect against repeated ADM3 names.
    if "_adm2_name_key" in adm3_gdf.columns and "_adm2_name_key" in adm3_summary.columns:
        adm3_join_key = ["_adm2_name_key", "_adm3_name_key"]
    else:
        adm3_join_key = "_adm3_name_key"


summary_keep = [
    c for c in [
        "_adm3_id_key", "_adm3_name_key", "_adm2_id_key", "_adm2_name_key",
        "ADM3_ID_STD", "ADM3_NAME_STD", "ADM2_ID_STD", "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_unusual_only_share",
        "recent_recurring_only_share",
        "recent_overlap_share",
        "combined_change_recent_minus_early",
        "recent_composition_class",
    ]
    if c in adm3_summary.columns
]

adm3_map = adm3_gdf.merge(
    adm3_summary[summary_keep].drop_duplicates(
        subset=adm3_join_key if isinstance(adm3_join_key, list) else [adm3_join_key]
    ),
    on=adm3_join_key,
    how="left",
    validate="many_to_one",
)


# =============================================================================
# JOIN LOCAL HOTSPOT FLAGS
# =============================================================================

local_keep = [
    c for c in [
        "_adm3_id_key", "_adm3_name_key", "_adm2_id_key", "_adm2_name_key",
        "local_amplification_hotspot",
        "hidden_composition_mismatch",
        "local_unusual_hotspot",
        "local_strong_increase",
        "local_extent_minus_parent_adm2",
        "local_change_minus_parent_adm2",
        "local_signal_percentile_score",
        "change_cross_scale_class",
        "composition_cross_scale_class",
    ]
    if c in local.columns
]

adm3_map = adm3_map.merge(
    local[local_keep].drop_duplicates(
        subset=adm3_join_key if isinstance(adm3_join_key, list) else [adm3_join_key]
    ),
    on=adm3_join_key,
    how="left",
    validate="many_to_one",
    suffixes=("", "_local"),
)


# =============================================================================
# JOIN SCRIPT 23 ROBUSTNESS FLAGS
# =============================================================================

robust_keep = [
    c for c in [
        "_adm3_id_key", "_adm3_name_key", "_adm2_id_key", "_adm2_name_key",
        "local_amplification_scenario_count",
        "local_amplification_robustness",
        "basic_hotspot_scenario_count",
        "change_hotspot_scenario_count",
        "basic_hotspot_robustness",
        "change_hotspot_robustness",
    ]
    if c in robust_amp.columns
]

adm3_map = adm3_map.merge(
    robust_amp[robust_keep].drop_duplicates(
        subset=adm3_join_key if isinstance(adm3_join_key, list) else [adm3_join_key]
    ),
    on=adm3_join_key,
    how="left",
    validate="many_to_one",
    suffixes=("", "_robust"),
)

adm3_map["robust_local_amplification_flag"] = (
    adm3_map.get(
        "local_amplification_robustness",
        pd.Series(index=adm3_map.index, dtype=object),
    )
    == "Robust across Q70-Q80"
)


# =============================================================================
# JOIN ADM2 CROSS-SCALE CLASS
# =============================================================================

use_adm2_id = (
    "_adm2_id_key" in adm2_gdf.columns
    and "_adm2_id_key" in cross.columns
)

if use_adm2_id:
    adm2_join_key = "_adm2_id_key"
else:
    if "_adm2_name_key" not in adm2_gdf.columns or "_adm2_name_key" not in cross.columns:
        raise ValueError("Cannot join ADM2 boundary to Script 22 table by ID or name.")
    adm2_join_key = "_adm2_name_key"


cross_keep = [
    c for c in [
        "_adm2_id_key", "_adm2_name_key",
        "ADM2_ID_STD", "ADM2_NAME_STD",
        "adm2_recent_combined_mean",
        "adm2_recent_unusual_only_share",
        "adm2_combined_change",
        "composition_cross_scale_class",
        "change_cross_scale_class",
        "n_adm3_local_amplification",
        "share_adm3_local_amplification",
        "n_hidden_composition_mismatch",
    ]
    if c in cross.columns
]

adm2_map = adm2_gdf.merge(
    cross[cross_keep].drop_duplicates(
        subset=[adm2_join_key]
    ),
    on=adm2_join_key,
    how="left",
    validate="many_to_one",
)


# =============================================================================
# QC
# =============================================================================

adm3_matched = adm3_map["recent_combined_mean"].notna()
adm2_matched = adm2_map["adm2_recent_combined_mean"].notna()

qc = pd.DataFrame({
    "metric": [
        "ADM2 boundary polygons",
        "ADM2 polygons matched to Script 22",
        "ADM3 boundary polygons",
        "ADM3 polygons matched to Script 21",
        "ADM3 polygons unmatched",
        "Robust local amplification hotspots mapped",
    ],
    "value": [
        len(adm2_map),
        int(adm2_matched.sum()),
        len(adm3_map),
        int(adm3_matched.sum()),
        int((~adm3_matched).sum()),
        int(adm3_map["robust_local_amplification_flag"].sum()),
    ],
})

qc.to_csv(TABLE_DIR / "24_mapping_join_qc.csv", index=False)


# =============================================================================
# MAP HELPERS
# =============================================================================

def setup_map(ax, title):
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_axis_off()


def draw_adm2_outlines(ax, linewidth=1.0):
    adm2_map.boundary.plot(
        ax=ax,
        linewidth=linewidth,
        edgecolor="black",
        zorder=5,
    )


def label_adm2(ax):
    if not LABEL_ADM2:
        return

    labels = adm2_map[adm2_matched].copy()
    if labels.empty:
        return

    reps = labels.geometry.representative_point()

    name_col = "ADM2_NAME_STD" if "ADM2_NAME_STD" in labels.columns else adm2_cols["name"]

    for (_, row), p in zip(labels.iterrows(), reps):
        label = str(row[name_col])
        ax.text(
            p.x,
            p.y,
            label,
            fontsize=5.5,
            ha="center",
            va="center",
            zorder=7,
        )


def label_top_adm3(ax, metric):
    if not LABEL_TOP_ADM3:
        return

    tmp = adm3_map.dropna(subset=[metric]).nlargest(
        min(TOP_ADM3_LABEL_N, adm3_map[metric].notna().sum()),
        metric,
    )

    if tmp.empty:
        return

    reps = tmp.geometry.representative_point()

    name_col = "ADM3_NAME_STD" if "ADM3_NAME_STD" in tmp.columns else adm3_cols["name"]

    for (_, row), p in zip(tmp.iterrows(), reps):
        ax.annotate(
            str(row[name_col]),
            xy=(p.x, p.y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
            zorder=8,
        )


# =============================================================================
# FIGURE 1 — ADM3 RECENT COMBINED FLOOD EXTENT
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 13))

adm3_map.plot(
    column="recent_combined_mean",
    cmap="YlOrRd",
    linewidth=0.15,
    edgecolor="white",
    legend=True,
    legend_kwds={
        "label": "Recent mean annual combined flooded area (%)",
        "shrink": 0.65,
    },
    missing_kwds={
        "color": "lightgrey",
        "label": "No matched result",
    },
    ax=ax,
)

draw_adm2_outlines(ax, linewidth=1.0)
label_adm2(ax)
label_top_adm3(ax, "recent_combined_mean")

setup_map(
    ax,
    "South Sudan ADM3 Recent Combined Flood Extent\n"
    "ADM2 boundaries shown as regional outlines (2020–2025)",
)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "24_01_adm3_recent_combined_extent_map.png",
    dpi=DPI,
    bbox_inches="tight",
)
plt.close(fig)


# =============================================================================
# FIGURE 2 — ADM3 UNUSUAL-ONLY SHARE
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 13))

adm3_map.plot(
    column="recent_unusual_only_share",
    cmap="magma",
    vmin=0,
    vmax=1,
    linewidth=0.15,
    edgecolor="white",
    legend=True,
    legend_kwds={
        "label": "Unusual-only share of recent combined flood extent",
        "shrink": 0.65,
    },
    missing_kwds={
        "color": "lightgrey",
        "label": "No matched result",
    },
    ax=ax,
)

draw_adm2_outlines(ax, linewidth=1.0)
label_adm2(ax)

setup_map(
    ax,
    "South Sudan ADM3 Recent Unusual-Flood Composition\n"
    "ADM2 boundaries shown as regional outlines (2020–2025)",
)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "24_02_adm3_unusual_only_share_map.png",
    dpi=DPI,
    bbox_inches="tight",
)
plt.close(fig)


# =============================================================================
# FIGURE 3 — LOCAL AMPLIFICATION HOTSPOTS
# =============================================================================

if "local_amplification_hotspot" not in adm3_map.columns:
    raise ValueError(
        "Script 22 local hotspot table does not contain "
        "'local_amplification_hotspot'."
    )

hot = adm3_map[
    adm3_map["local_amplification_hotspot"].fillna(False).astype(bool)
].copy()

fig, ax = plt.subplots(figsize=(11, 13))

adm3_map.plot(
    color="whitesmoke",
    linewidth=0.12,
    edgecolor="white",
    ax=ax,
)

if len(hot):
    hot.plot(
        column="recent_combined_mean",
        cmap="YlOrRd",
        linewidth=0.35,
        edgecolor="black",
        legend=True,
        legend_kwds={
            "label": "Recent combined flood extent (%) among amplification hotspots",
            "shrink": 0.65,
        },
        ax=ax,
        zorder=4,
    )

draw_adm2_outlines(ax, linewidth=1.0)

if len(hot):
    reps = hot.geometry.representative_point()
    name_col = "ADM3_NAME_STD" if "ADM3_NAME_STD" in hot.columns else adm3_cols["name"]

    label_hot = hot.nlargest(
        min(20, len(hot)),
        "local_signal_percentile_score",
    )

    label_reps = label_hot.geometry.representative_point()

    for (_, row), p in zip(label_hot.iterrows(), label_reps):
        ax.annotate(
            str(row[name_col]),
            xy=(p.x, p.y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
            zorder=8,
        )

setup_map(
    ax,
    "ADM3 Local Flood Amplification Hotspots within ADM2 Regions\n"
    "Hotspots indicate local intensity/change signals stronger than the parent ADM2",
)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "24_03_adm3_local_amplification_hotspots_map.png",
    dpi=DPI,
    bbox_inches="tight",
)
plt.close(fig)


# =============================================================================
# FIGURE 4 — CROSS-SCALE CHANGE CLASSES
# =============================================================================

if "change_cross_scale_class" not in adm2_map.columns:
    raise ValueError(
        "Script 22 cross-scale table does not contain "
        "'change_cross_scale_class'."
    )

change_classes = [
    "Regional increase + Local worsening",
    "Regional increase + Local hotspot",
    "Regional increase + Mixed local response",
    "Regional neutral + Local hotspot",
    "Regional decline + Local worsening",
    "Regional decline + Local improving",
    "Regional decline + Mixed local response",
    "Broad cross-scale stability",
]

# Explicit categorical colors are appropriate here because colors encode named classes.
class_colors = {
    "Regional increase + Local worsening": "#8c2d04",
    "Regional increase + Local hotspot": "#d94801",
    "Regional increase + Mixed local response": "#f16913",
    "Regional neutral + Local hotspot": "#fdae6b",
    "Regional decline + Local worsening": "#6a51a3",
    "Regional decline + Local improving": "#238b45",
    "Regional decline + Mixed local response": "#74c476",
    "Broad cross-scale stability": "#bdbdbd",
}

fig, ax = plt.subplots(figsize=(11, 13))

adm2_map.plot(
    color="white",
    edgecolor="black",
    linewidth=0.8,
    ax=ax,
)

for cls in change_classes:
    subset = adm2_map[adm2_map["change_cross_scale_class"] == cls]
    if len(subset):
        subset.plot(
            color=class_colors[cls],
            edgecolor="black",
            linewidth=0.8,
            ax=ax,
        )

# Overlay ADM3 units classified as local worsening/hotspot signals.
local_flag_cols = [
    c for c in (
        "local_amplification_hotspot",
        "local_strong_increase",
    )
    if c in adm3_map.columns
]

if local_flag_cols:
    local_focus = np.zeros(len(adm3_map), dtype=bool)
    for c in local_flag_cols:
        local_focus |= adm3_map[c].fillna(False).astype(bool).to_numpy()

    focus = adm3_map.loc[local_focus]

    if len(focus):
        focus.boundary.plot(
            ax=ax,
            linewidth=0.55,
            edgecolor="black",
            zorder=7,
        )

label_adm2(ax)

legend_handles = [
    Patch(
        facecolor=class_colors[c],
        edgecolor="black",
        label=c,
    )
    for c in change_classes
    if (adm2_map["change_cross_scale_class"] == c).any()
]

legend_handles.append(
    Line2D(
        [0], [0],
        color="black",
        linewidth=1.5,
        label="ADM3 local worsening / amplification outline",
    )
)

ax.legend(
    handles=legend_handles,
    loc="lower left",
    fontsize=7,
    frameon=True,
)

setup_map(
    ax,
    "ADM2–ADM3 Cross-Scale Flood Change Classes\n"
    "ADM3 local worsening/amplification units outlined within parent ADM2 regions",
)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "24_04_cross_scale_change_classes_map.png",
    dpi=DPI,
    bbox_inches="tight",
)
plt.close(fig)


# =============================================================================
# FIGURE 5 — ROBUST LOCAL AMPLIFICATION HOTSPOTS
# =============================================================================

robust_hot = adm3_map[
    adm3_map["robust_local_amplification_flag"].fillna(False).astype(bool)
].copy()

fig, ax = plt.subplots(figsize=(11, 13))

adm3_map.plot(
    color="whitesmoke",
    linewidth=0.12,
    edgecolor="white",
    ax=ax,
)

if len(robust_hot):
    robust_hot.plot(
        column="recent_combined_mean",
        cmap="YlOrRd",
        linewidth=0.45,
        edgecolor="black",
        legend=True,
        legend_kwds={
            "label": "Recent combined flood extent (%) among robust amplification hotspots",
            "shrink": 0.65,
        },
        ax=ax,
        zorder=4,
    )

draw_adm2_outlines(ax, linewidth=1.0)

if len(robust_hot):
    name_col = (
        "ADM3_NAME_STD"
        if "ADM3_NAME_STD" in robust_hot.columns
        else adm3_cols["name"]
    )

    label_robust = robust_hot.nlargest(
        min(15, len(robust_hot)),
        "recent_combined_mean",
    )

    reps = label_robust.geometry.representative_point()

    for (_, row), p in zip(label_robust.iterrows(), reps):
        ax.annotate(
            str(row[name_col]),
            xy=(p.x, p.y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
            zorder=8,
        )

setup_map(
    ax,
    "Robust ADM3 Local Flood Amplification Hotspots\n"
    "Stable across Q70–Q80 sensitivity scenarios",
)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "24_05_robust_local_amplification_hotspots_map.png",
    dpi=DPI,
    bbox_inches="tight",
)
plt.close(fig)


# =============================================================================
# QC SUMMARY
# =============================================================================

qc_text = f"""24 — ADM2–ADM3 Final Nested Flood Composition Maps
================================================

Project root
------------
{ROOT}

Inputs
------
ADM3 summary:
{ADM3_SUMMARY_PATH}

Cross-scale table:
{CROSS_SCALE_PATH}

Local hotspot table:
{LOCAL_HOTSPOT_PATH}

Robust amplification table:
{ROBUST_AMPLIFICATION_PATH}

ADM2 boundary:
{ADM2_BOUNDARY_PATH}

ADM3 boundary:
{ADM3_BOUNDARY_PATH}

Detected boundary columns
-------------------------
ADM2 name: {adm2_cols["name"]}
ADM2 ID  : {adm2_cols["id"]}

ADM3 name       : {adm3_cols["name"]}
ADM3 ID         : {adm3_cols["id"]}
ADM3 parent name: {adm3_cols["parent_name"]}
ADM3 parent ID  : {adm3_cols["parent_id"]}

Join strategy
-------------
ADM3 join key: {adm3_join_key}
ADM2 join key: {adm2_join_key}

Join coverage
-------------
ADM2 boundary polygons                 : {len(adm2_map)}
ADM2 matched to Script 22              : {int(adm2_matched.sum())}
ADM2 unmatched                         : {int((~adm2_matched).sum())}

ADM3 boundary polygons                 : {len(adm3_map)}
ADM3 matched to Script 21              : {int(adm3_matched.sum())}
ADM3 unmatched                         : {int((~adm3_matched).sum())}

Local amplification hotspots mapped   : {int(adm3_map["local_amplification_hotspot"].fillna(False).sum()) if "local_amplification_hotspot" in adm3_map.columns else 0}
Robust amplification hotspots mapped  : {int(adm3_map["robust_local_amplification_flag"].fillna(False).sum())}

Interpretation
--------------
1. ADM3 fills visualize local flood metrics derived in Script 21.
2. ADM2 outlines provide regional context and Script 22 cross-scale classifications.
3. Local amplification hotspots indicate ADM3 units where local flood intensity
   or recent change is substantially stronger than the parent ADM2 signal.
4. Robust local amplification hotspots are those retained across the Q70, Q75,
   and Q80 sensitivity scenarios from Script 23.
5. These maps do not generate new flood observations; they spatially visualize
   existing Script 21, Script 22, and Script 23 results.
6. Unmatched polygons, if any, should be resolved before final publication.

DONE.
"""

(QC_DIR / "24_mapping_qc_summary.txt").write_text(
    qc_text,
    encoding="utf-8",
)


# =============================================================================
# FINAL CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 105)
print("MAPPING QC")
print("=" * 105)

print(f"ADM2 boundary polygons : {len(adm2_map):,}")
print(f"ADM2 matched           : {int(adm2_matched.sum()):,}")
print(f"ADM3 boundary polygons : {len(adm3_map):,}")
print(f"ADM3 matched           : {int(adm3_matched.sum()):,}")
print(f"ADM3 unmatched         : {int((~adm3_matched).sum()):,}")

if "local_amplification_hotspot" in adm3_map.columns:
    print(
        "Local amplification hotspots mapped: "
        f"{int(adm3_map['local_amplification_hotspot'].fillna(False).sum()):,}"
    )

print(
    "Robust local amplification hotspots mapped: "
    f"{int(adm3_map['robust_local_amplification_flag'].fillna(False).sum()):,}"
)

print("\nSaved figures:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}")

print("\nSaved QC:")
for f in sorted(QC_DIR.glob("*")):
    print(f"  {f.name}")

print("\nDONE.")
