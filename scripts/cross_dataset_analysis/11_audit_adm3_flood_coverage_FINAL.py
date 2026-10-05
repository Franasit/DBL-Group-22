#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_audit_adm3_flood_coverage.py

Purpose
-------
Audit ADM3 flood-data coverage BEFORE final temporal clustering.

This script checks whether a zero flood value means:
    1) flood was not detected although the expected source files exist, or
    2) the ADM3/year has incomplete or missing source coverage.

It identifies ADM3 units that require coverage review before temporal clustering.

Workflow position
-----------------
Script 10 -> Script 11 -> Script 12

Script 10 constructs ADM3 flood indicators and performs first-pass QC.
Script 11 audits raw source-file/tile coverage and zero-vs-no-detection cases.
Script 12 should perform ADM3 temporal-pattern analysis only after these QC
results have been reviewed.

IMPORTANT INTERPRETATION
------------------------
The compact recurring/unusual parquet files appear to contain POSITIVE FLOOD
EVENT PIXELS / POINTS, not a complete raster containing every valid non-flood
observation. Therefore:

    expected tile-year files exist + no flood event rows matched

means:
    "no detected flood record in the compact event dataset"

It does NOT by itself prove that every pixel/day was validly observed and truly
flood-free. Full observation-validity / cloud-quality information would be
needed to make that stronger claim.

The script deliberately does NOT copy ADM2 values into missing ADM3 units.
Parent ADM2 results may later be used as contextual / sensitivity information,
but should not be treated as observed ADM3 measurements.

Expected repository layout
--------------------------
DBL-Group-22/
├── raw_data/
│   ├── administrative_boundaries/
│   │   └── ssd_admin3.geojson
│   └── flood_masks/
│       ├── compact_recurring/
│       │   └── flood_events_hXXvXX_YYYY.parquet
│       └── compact_unusual/
│           └── flood_events_hXXvXX_YYYY.parquet
├── processed_data/
│   └── cross_dataset_analysis/
│       └── 02_adm3_flood/
│           ├── 10_build_adm3_flood_indicators/
│           └── 11_audit_adm3_flood_coverage/
└── scripts/
    └── cross_dataset_analysis/
        └── 11_audit_adm3_flood_coverage.py

Outputs
-------
All outputs are written to:
    processed_data/cross_dataset_analysis/02_adm3_flood/
        11_audit_adm3_flood_coverage/

Subfolders:
    tables/
        11_flood_source_file_inventory.csv
        11_adm3_tile_overlap.csv
        11_adm3_event_record_counts.csv
        11_adm3_year_coverage_audit.csv
        11_adm3_coverage_summary.csv
        11_adm3_zero_vs_nodata_review.csv
        11_adm3_script10_qc_crosscheck.csv

    qc/
        11_adm3_boundary_qc.csv
        11_audit_summary.txt

Interpretation:
    This script runs BEFORE ADM3 temporal clustering. It does not remove ADM3
    units and does not copy parent ADM2 values into ADM3. Its outputs should be
    reviewed before Script 12 temporal-pattern analysis.

Dependencies
------------
pandas, numpy, geopandas, shapely, pyproj, pyarrow
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import transform as shp_transform

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# CONFIG
# =============================================================================

# If the script is stored at:
#   <project>/scripts/cross_dataset_analysis/11_audit_adm3_flood_coverage.py
# then parents[2] is the repository root.
SCRIPT_PATH = Path(__file__).resolve()
AUTO_PROJECT_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) >= 3 else SCRIPT_PATH.parent

# Normally leave this as None. Use an exact Path only if the automatic root is
# not correct in your environment.
PROJECT_ROOT_OVERRIDE: Path | None = None
PROJECT_ROOT = PROJECT_ROOT_OVERRIDE or AUTO_PROJECT_ROOT

START_YEAR = 2000
END_YEAR = 2025
EXPECTED_YEARS = list(range(START_YEAR, END_YEAR + 1))
FLOOD_TYPES = ("recurring", "unusual")

# Canonical project paths used by the other cross-dataset scripts.
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
CROSS_DATASET_DIR = PROCESSED_DATA_DIR / "cross_dataset_analysis"
ADM3_ANALYSIS_DIR = CROSS_DATASET_DIR / "02_adm3_flood"

# Script 10 is the upstream ADM3 indicator-construction step.
SCRIPT10_DIR = ADM3_ANALYSIS_DIR / "10_build_adm3_flood_indicators"

# Script 11 writes only its own coverage-audit products here.
OUTPUT_DIR = ADM3_ANALYSIS_DIR / "11_audit_adm3_flood_coverage"
TABLE_DIR = OUTPUT_DIR / "tables"
QC_DIR = OUTPUT_DIR / "qc"

ADM3_PATH = RAW_DATA_DIR / "administrative_boundaries" / "ssd_admin3.geojson"
FLOOD_ROOT = RAW_DATA_DIR / "flood_masks"

# Optional Script 10 QC products. They are not required for the core tile audit,
# but, when present, they are used to cross-check which ADM3 units Script 10
# already flagged as all-zero or very sparse.
SCRIPT10_QC_FLAGS = (
    SCRIPT10_DIR / "qc" / "adm3_indicator_qc_flags.csv"
)
SCRIPT10_ZERO_AUDIT = (
    SCRIPT10_DIR / "qc" / "adm3_zero_flood_audit.csv"
)

# True = read every compact event parquet and spatially join positive flood
# records to ADM3. Keep True for the real audit.
READ_EVENT_ROWS = True

# Optional manual focus list. Usually leave empty; zero/no-data candidates are
# found automatically.
FOCUS_ADM3_PCODES: list[str] = []
FOCUS_ADM3_NAMES: list[str] = []


# =============================================================================
# PATH / COLUMN HELPERS
# =============================================================================


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {description}:\n  {path}\n\n"
            f"Detected project root:\n  {PROJECT_ROOT}\n\n"
            "If the repository layout is different, either move the file to the "
            "expected location or set PROJECT_ROOT_OVERRIDE near the top of this script."
        )
    return path


def find_type_dir(flood_root: Path, flood_type: str) -> Path:
    candidates = [
        flood_root / f"compact_{flood_type}",
        flood_root / flood_type,
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find {flood_type} flood directory. Checked:\n{checked}"
    )


def pick_column(columns, candidates, required=True):
    lower_map = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(
            f"Could not find any of columns {candidates}. "
            f"Available columns: {list(columns)}"
        )
    return None


def safe_read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet requires pyarrow or fastparquet. "
            "Install pyarrow in the project environment, e.g. `pip install pyarrow`."
        ) from exc


# =============================================================================
# SOURCE-FILE INVENTORY
# =============================================================================

FILE_RE = re.compile(r"flood_events_(h\d{2}v\d{2})_(\d{4})\.parquet$", re.I)


def parse_flood_filename(path: Path):
    m = FILE_RE.search(path.name)
    if not m:
        return None
    return m.group(1).lower(), int(m.group(2))


def build_file_inventory(flood_root: Path) -> pd.DataFrame:
    rows = []

    for flood_type in FLOOD_TYPES:
        type_dir = find_type_dir(flood_root, flood_type)

        for p in sorted(type_dir.glob("*.parquet")):
            parsed = parse_flood_filename(p)

            if parsed is None:
                rows.append(
                    {
                        "flood_type": flood_type,
                        "tile": None,
                        "year": None,
                        "path": str(p),
                        "filename_valid": False,
                        "file_size_mb": p.stat().st_size / (1024**2),
                    }
                )
                continue

            tile, year = parsed
            rows.append(
                {
                    "flood_type": flood_type,
                    "tile": tile,
                    "year": year,
                    "path": str(p),
                    "filename_valid": True,
                    "file_size_mb": p.stat().st_size / (1024**2),
                }
            )

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise RuntimeError(f"No parquet files were found under {flood_root}")

    return inventory


# =============================================================================
# ADM3 BOUNDARY QC
# =============================================================================


def audit_boundaries(adm3: gpd.GeoDataFrame) -> pd.DataFrame:
    pcode_col = pick_column(adm3.columns, ["adm3_pcode", "ADM3_PCODE", "pcode"])
    name_col = pick_column(adm3.columns, ["adm3_name", "ADM3_EN", "name"])
    adm2_name_col = pick_column(adm3.columns, ["adm2_name", "ADM2_EN"], required=False)
    adm2_pcode_col = pick_column(adm3.columns, ["adm2_pcode", "ADM2_PCODE"], required=False)

    pcode_as_text = adm3[pcode_col].astype(str)
    duplicate_counts = pcode_as_text.value_counts()

    rows = []
    for idx, r in adm3.iterrows():
        geom = r.geometry
        geom_missing = geom is None or geom.is_empty

        rows.append(
            {
                "adm3_index": idx,
                "adm3_pcode": str(r[pcode_col]),
                "adm3_name": r[name_col],
                "adm2_name": r[adm2_name_col] if adm2_name_col else None,
                "adm2_pcode": r[adm2_pcode_col] if adm2_pcode_col else None,
                "geometry_missing": geom_missing,
                "geometry_valid": False if geom_missing else bool(geom.is_valid),
                "duplicate_adm3_pcode": duplicate_counts.get(str(r[pcode_col]), 0) > 1,
                "area_sqkm_source": r.get("area_sqkm", np.nan),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MODIS TILE FOOTPRINTS
# =============================================================================

# MODIS sinusoidal grid constants.
MODIS_XMIN = -20015109.354
MODIS_YMAX = 10007554.677
MODIS_TILE_SIZE = 1111950.5196666666
MODIS_SINU = CRS.from_proj4(
    "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)
SINU_TO_WGS84 = Transformer.from_crs(MODIS_SINU, WGS84, always_xy=True)


def modis_tile_polygon_wgs84(tile: str):
    m = re.fullmatch(r"h(\d{2})v(\d{2})", tile.lower())
    if not m:
        raise ValueError(f"Unexpected MODIS tile name: {tile}")

    h, v = map(int, m.groups())
    x0 = MODIS_XMIN + h * MODIS_TILE_SIZE
    x1 = x0 + MODIS_TILE_SIZE
    y1 = MODIS_YMAX - v * MODIS_TILE_SIZE
    y0 = y1 - MODIS_TILE_SIZE

    poly_sinu = box(x0, y0, x1, y1)
    return shp_transform(SINU_TO_WGS84.transform, poly_sinu)


def build_tile_overlap(adm3: gpd.GeoDataFrame, tiles: list[str]) -> pd.DataFrame:
    pcode_col = pick_column(adm3.columns, ["adm3_pcode", "ADM3_PCODE", "pcode"])
    name_col = pick_column(adm3.columns, ["adm3_name", "ADM3_EN", "name"])
    adm2_name_col = pick_column(adm3.columns, ["adm2_name", "ADM2_EN"], required=False)

    tile_gdf = gpd.GeoDataFrame(
        {
            "tile": tiles,
            "geometry": [modis_tile_polygon_wgs84(t) for t in tiles],
        },
        crs="EPSG:4326",
    )

    keep_cols = [pcode_col, name_col]
    if adm2_name_col:
        keep_cols.append(adm2_name_col)
    keep_cols.append("geometry")

    work = adm3[keep_cols].copy()
    work = work.rename(columns={pcode_col: "adm3_pcode", name_col: "adm3_name"})
    work["adm3_pcode"] = work["adm3_pcode"].astype(str)

    if adm2_name_col:
        work = work.rename(columns={adm2_name_col: "adm2_name"})
    else:
        work["adm2_name"] = None

    # An ADM3 can overlap more than one MODIS tile, so use polygon intersection.
    joined = gpd.sjoin(work, tile_gdf, how="left", predicate="intersects")

    return pd.DataFrame(
        joined[["adm3_pcode", "adm3_name", "adm2_name", "tile"]].drop_duplicates()
    )


# =============================================================================
# POSITIVE EVENT RECORDS -> ADM3
# =============================================================================


def detect_xy_columns(df: pd.DataFrame):
    lat = pick_column(
        df.columns,
        ["lat", "latitude", "y", "center_lat", "pixel_lat"],
        required=False,
    )
    lon = pick_column(
        df.columns,
        ["lon", "long", "longitude", "x", "center_lon", "pixel_lon"],
        required=False,
    )

    if lat is None or lon is None:
        raise KeyError(
            "Could not detect latitude/longitude columns in parquet. "
            f"Available columns: {list(df.columns)}"
        )

    return lat, lon


def detect_date_column(df: pd.DataFrame):
    return pick_column(
        df.columns,
        ["date", "acquisition_date", "datetime", "time"],
        required=False,
    )


def count_flood_records_by_adm3(
    inventory: pd.DataFrame,
    adm3: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Count positive compact flood-event rows by ADM3/year/type/tile."""

    pcode_col = pick_column(adm3.columns, ["adm3_pcode", "ADM3_PCODE", "pcode"])
    name_col = pick_column(adm3.columns, ["adm3_name", "ADM3_EN", "name"])

    boundary = adm3[[pcode_col, name_col, "geometry"]].copy().rename(
        columns={pcode_col: "adm3_pcode", name_col: "adm3_name"}
    )
    boundary["adm3_pcode"] = boundary["adm3_pcode"].astype(str)
    boundary = boundary.to_crs("EPSG:4326")

    all_rows = []

    valid_inventory = inventory[
        inventory["filename_valid"] & inventory["year"].between(START_YEAR, END_YEAR)
    ].copy()

    for i, rec in valid_inventory.reset_index(drop=True).iterrows():
        path = Path(rec["path"])
        flood_type = str(rec["flood_type"])
        tile = str(rec["tile"])
        year = int(rec["year"])

        print(
            f"    [{i + 1:03d}/{len(valid_inventory):03d}] "
            f"{flood_type:9s} {tile} {year}: {path.name}"
        )

        df = safe_read_parquet(path)
        if df.empty:
            continue

        lat_col, lon_col = detect_xy_columns(df)
        date_col = detect_date_column(df)

        work = df.copy()
        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
        work = work.dropna(subset=[lat_col, lon_col])

        if work.empty:
            continue

        points = gpd.GeoDataFrame(
            work,
            geometry=gpd.points_from_xy(work[lon_col], work[lat_col]),
            crs="EPSG:4326",
        )

        # 'within' prevents a point exactly on a shared ADM3 boundary from being
        # counted in two polygons. Grid centres should normally fall inside one ADM3.
        joined = gpd.sjoin(
            points,
            boundary[["adm3_pcode", "adm3_name", "geometry"]],
            how="inner",
            predicate="within",
        )

        if joined.empty:
            continue

        if date_col is not None:
            joined = joined.assign(
                _date=pd.to_datetime(joined[date_col], errors="coerce").dt.date
            )

            grouped = (
                joined.groupby(["adm3_pcode", "adm3_name"], dropna=False)
                .agg(
                    flood_record_count=("geometry", "size"),
                    flood_observation_dates=("_date", lambda s: s.dropna().nunique()),
                )
                .reset_index()
            )
        else:
            grouped = (
                joined.groupby(["adm3_pcode", "adm3_name"], dropna=False)
                .size()
                .reset_index(name="flood_record_count")
            )
            grouped["flood_observation_dates"] = np.nan

        grouped["year"] = year
        grouped["flood_type"] = flood_type
        grouped["tile"] = tile
        all_rows.append(grouped)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "adm3_pcode",
                "adm3_name",
                "year",
                "flood_type",
                "tile",
                "flood_record_count",
                "flood_observation_dates",
            ]
        )

    return pd.concat(all_rows, ignore_index=True)


# =============================================================================
# COMPLETE ADM3 x YEAR AUDIT PANEL
# =============================================================================


def build_adm3_year_audit(
    adm3: gpd.GeoDataFrame,
    tile_overlap: pd.DataFrame,
    inventory: pd.DataFrame,
    event_counts: pd.DataFrame,
) -> pd.DataFrame:
    pcode_col = pick_column(adm3.columns, ["adm3_pcode", "ADM3_PCODE", "pcode"])
    name_col = pick_column(adm3.columns, ["adm3_name", "ADM3_EN", "name"])
    adm2_name_col = pick_column(adm3.columns, ["adm2_name", "ADM2_EN"], required=False)
    adm2_pcode_col = pick_column(adm3.columns, ["adm2_pcode", "ADM2_PCODE"], required=False)

    meta_cols = [pcode_col, name_col]
    if adm2_name_col:
        meta_cols.append(adm2_name_col)
    if adm2_pcode_col:
        meta_cols.append(adm2_pcode_col)

    meta = adm3[meta_cols].drop_duplicates().rename(
        columns={
            pcode_col: "adm3_pcode",
            name_col: "adm3_name",
            **({adm2_name_col: "adm2_name"} if adm2_name_col else {}),
            **({adm2_pcode_col: "adm2_pcode"} if adm2_pcode_col else {}),
        }
    )
    meta["adm3_pcode"] = meta["adm3_pcode"].astype(str)

    if "adm2_name" not in meta.columns:
        meta["adm2_name"] = None
    if "adm2_pcode" not in meta.columns:
        meta["adm2_pcode"] = None

    # Build the complete panel explicitly: every ADM3 x every year.
    panel = (
        meta.assign(_join_key=1)
        .merge(
            pd.DataFrame({"year": EXPECTED_YEARS, "_join_key": 1}),
            on="_join_key",
            how="inner",
        )
        .drop(columns="_join_key")
    )

    overlap_map = (
        tile_overlap.dropna(subset=["tile"])
        .assign(adm3_pcode=lambda x: x["adm3_pcode"].astype(str))
        .groupby("adm3_pcode")["tile"]
        .apply(lambda s: sorted(set(map(str, s))))
        .to_dict()
    )

    valid_inventory = inventory[
        inventory["filename_valid"] & inventory["year"].between(START_YEAR, END_YEAR)
    ].copy()

    file_keys = set(
        zip(
            valid_inventory["flood_type"].astype(str),
            valid_inventory["tile"].astype(str),
            valid_inventory["year"].astype(int),
        )
    )

    event_lookup = defaultdict(lambda: {ft: 0 for ft in FLOOD_TYPES})
    date_lookup = defaultdict(lambda: {ft: 0 for ft in FLOOD_TYPES})

    if not event_counts.empty:
        event_counts = event_counts.copy()
        event_counts["adm3_pcode"] = event_counts["adm3_pcode"].astype(str)

        aggregated = (
            event_counts.groupby(
                ["adm3_pcode", "year", "flood_type"],
                as_index=False,
            )
            .agg(
                flood_record_count=("flood_record_count", "sum"),
                flood_observation_dates=("flood_observation_dates", "sum"),
            )
        )

        for _, r in aggregated.iterrows():
            key = (str(r["adm3_pcode"]), int(r["year"]))
            flood_type = str(r["flood_type"])
            event_lookup[key][flood_type] = int(r["flood_record_count"])

            if pd.notna(r["flood_observation_dates"]):
                date_lookup[key][flood_type] = int(r["flood_observation_dates"])

    output_rows = []

    for _, r in panel.iterrows():
        pcode = str(r["adm3_pcode"])
        year = int(r["year"])
        tiles = overlap_map.get(pcode, [])

        row = r.to_dict()
        row["overlapping_tiles"] = ";".join(tiles)
        row["n_overlapping_tiles"] = len(tiles)

        any_partial = False
        all_types_complete = True
        any_file_available = False

        for flood_type in FLOOD_TYPES:
            expected = len(tiles)
            available = sum((flood_type, tile, year) in file_keys for tile in tiles)

            row[f"{flood_type}_expected_tile_files"] = expected
            row[f"{flood_type}_available_tile_files"] = available
            row[f"{flood_type}_file_coverage_fraction"] = (
                available / expected if expected > 0 else np.nan
            )
            row[f"{flood_type}_flood_records"] = event_lookup[(pcode, year)][flood_type]
            row[f"{flood_type}_flood_dates"] = date_lookup[(pcode, year)][flood_type]

            if available > 0:
                any_file_available = True
            if expected == 0 or available < expected:
                all_types_complete = False
            if 0 < available < expected:
                any_partial = True

        total_records = sum(row[f"{ft}_flood_records"] for ft in FLOOD_TYPES)
        row["combined_flood_records"] = int(total_records)
        row["flood_detected"] = bool(total_records > 0)

        if len(tiles) == 0:
            status = "NO_TILE_OVERLAP"
        elif all_types_complete and total_records > 0:
            status = "PROCESSED_TILE_FILES__FLOOD_DETECTED"
        elif all_types_complete and total_records == 0:
            status = "PROCESSED_TILE_FILES__NO_FLOOD_RECORD"
        elif any_partial or any_file_available:
            status = "PARTIAL_TILE_YEAR_FILES"
        else:
            status = "NO_TILE_YEAR_FILES"

        row["qc_status"] = status
        output_rows.append(row)

    return pd.DataFrame(output_rows)


# =============================================================================
# ADM3-LEVEL SUMMARY
# =============================================================================


def summarize_adm3(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["adm3_pcode", "adm3_name", "adm2_name", "adm2_pcode"]

    for (pcode, name, adm2_name, adm2_pcode), g in audit.groupby(
        group_cols,
        dropna=False,
    ):
        complete = g["qc_status"].isin(
            [
                "PROCESSED_TILE_FILES__FLOOD_DETECTED",
                "PROCESSED_TILE_FILES__NO_FLOOD_RECORD",
            ]
        )
        flood = g["flood_detected"].fillna(False)
        no_record = g["qc_status"].eq("PROCESSED_TILE_FILES__NO_FLOOD_RECORD")
        partial = g["qc_status"].eq("PARTIAL_TILE_YEAR_FILES")
        no_files = g["qc_status"].eq("NO_TILE_YEAR_FILES")
        no_overlap = g["qc_status"].eq("NO_TILE_OVERLAP")

        if complete.all() and flood.sum() == 0:
            interpretation = "26Y_PROCESSED__ZERO_DETECTED_FLOOD_RECORDS"
        elif complete.all() and flood.sum() > 0:
            interpretation = "26Y_PROCESSED__FLOOD_DETECTED"
        elif partial.any() or no_files.any() or no_overlap.any():
            interpretation = "INCOMPLETE_SOURCE_COVERAGE"
        else:
            interpretation = "REVIEW"

        rows.append(
            {
                "adm3_pcode": str(pcode),
                "adm3_name": name,
                "adm2_name": adm2_name,
                "adm2_pcode": adm2_pcode,
                "years_expected": len(EXPECTED_YEARS),
                "years_complete_tile_files": int(complete.sum()),
                "years_with_detected_flood": int(flood.sum()),
                "years_complete_no_flood_record": int(no_record.sum()),
                "years_partial_tile_files": int(partial.sum()),
                "years_no_tile_year_files": int(no_files.sum()),
                "years_no_tile_overlap": int(no_overlap.sum()),
                "total_recurring_flood_records": int(g["recurring_flood_records"].sum()),
                "total_unusual_flood_records": int(g["unusual_flood_records"].sum()),
                "total_combined_flood_records": int(g["combined_flood_records"].sum()),
                "audit_interpretation": interpretation,
                "scientific_note": (
                    "Compact event files distinguish detected flood from absence of a "
                    "detected event record; they do not by themselves prove full "
                    "valid-observation / clear-sky coverage."
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["audit_interpretation", "adm3_name"],
        kind="stable",
    )


# =============================================================================
# OPTIONAL CROSS-CHECK WITH SCRIPT 10 QC
# =============================================================================


def load_script10_qc_flags() -> pd.DataFrame | None:
    """
    Load Script 10 ADM3 QC flags when available.

    This is a diagnostic cross-check only. Script 11 independently audits
    source-file/tile coverage from the raw flood-mask inputs.
    """
    if not SCRIPT10_QC_FLAGS.exists():
        return None

    df = pd.read_csv(SCRIPT10_QC_FLAGS)

    pcode_col = pick_column(
        df.columns,
        ["ADM3_PCODE", "adm3_pcode", "ADM3_ID", "adm3_id"],
        required=False,
    )
    name_col = pick_column(
        df.columns,
        ["ADM3_NAME", "adm3_name", "ADM3_EN", "name"],
        required=False,
    )

    if pcode_col is None and name_col is None:
        return None

    out = df.copy()

    if pcode_col is not None:
        out["_script10_key"] = out[pcode_col].astype(str)
        out["_script10_match_method"] = "pcode_or_id"
    else:
        out["_script10_key"] = out[name_col].astype(str)
        out["_script10_match_method"] = "name"

    return out


def build_script10_crosscheck(
    summary: pd.DataFrame,
    script10_qc: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Compare Script 11 coverage interpretation with Script 10 diagnostic flags.

    No exclusion decision is made here.
    """
    out = summary.copy()

    if script10_qc is None:
        out["script10_qc_available"] = False
        out["script10_requires_review"] = np.nan
        out["script10_qc_flags"] = np.nan
        return out

    out["script10_qc_available"] = True

    # Prefer pcode-like matching when the Script 10 table supports it.
    method = script10_qc["_script10_match_method"].iloc[0]

    if method == "pcode_or_id":
        lookup = script10_qc.drop_duplicates("_script10_key").set_index(
            "_script10_key"
        )

        out["_script11_key"] = out["adm3_pcode"].astype(str)

        # If no pcode-style matches are found, safely fall back to ADM3 name.
        matched = out["_script11_key"].isin(lookup.index).sum()

        if matched == 0:
            name_col = pick_column(
                script10_qc.columns,
                ["ADM3_NAME", "adm3_name", "ADM3_EN", "name"],
                required=False,
            )
            if name_col is not None:
                script10_qc = script10_qc.copy()
                script10_qc["_script10_key"] = script10_qc[name_col].astype(str)
                lookup = script10_qc.drop_duplicates("_script10_key").set_index(
                    "_script10_key"
                )
                out["_script11_key"] = out["adm3_name"].astype(str)
                method = "name_fallback"
    else:
        lookup = script10_qc.drop_duplicates("_script10_key").set_index(
            "_script10_key"
        )
        out["_script11_key"] = out["adm3_name"].astype(str)

    review_col = pick_column(
        script10_qc.columns,
        ["requires_review"],
        required=False,
    )
    flags_col = pick_column(
        script10_qc.columns,
        ["qc_flags"],
        required=False,
    )

    if review_col is not None:
        out["script10_requires_review"] = out["_script11_key"].map(
            lookup[review_col].to_dict()
        )
    else:
        out["script10_requires_review"] = np.nan

    if flags_col is not None:
        out["script10_qc_flags"] = out["_script11_key"].map(
            lookup[flags_col].to_dict()
        )
    else:
        out["script10_qc_flags"] = np.nan

    out["script10_match_method"] = method
    out = out.drop(columns=["_script11_key"])

    return out


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    adm3_path = require_path(ADM3_PATH, "ADM3 boundary file")
    flood_root = require_path(FLOOD_ROOT, "flood_masks directory")

    print("=" * 100)
    print("11 ADM3 FLOOD COVERAGE AUDIT")
    print("=" * 100)
    print(f"Script       : {SCRIPT_PATH}")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"ADM3 boundary: {adm3_path}")
    print(f"Flood root   : {flood_root}")
    print(f"Output dir   : {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # 1. Boundary QC
    # -------------------------------------------------------------------------
    print("\n[1/7] Reading and auditing ADM3 boundaries...")
    adm3 = gpd.read_file(adm3_path)

    if adm3.crs is None:
        raise ValueError("ADM3 boundary CRS is missing.")

    adm3 = adm3.to_crs("EPSG:4326")

    boundary_qc = audit_boundaries(adm3)
    boundary_qc.to_csv(QC_DIR / "11_adm3_boundary_qc.csv", index=False)

    print(f"      ADM3 units              : {len(adm3):,}")
    print(f"      Invalid geometries      : {(~boundary_qc['geometry_valid']).sum():,}")
    print(f"      Duplicate ADM3 pcodes   : {boundary_qc['duplicate_adm3_pcode'].sum():,}")

    # -------------------------------------------------------------------------
    # 2. Flood source-file inventory
    # -------------------------------------------------------------------------
    print("\n[2/7] Building flood source-file inventory...")
    inventory = build_file_inventory(flood_root)
    inventory.to_csv(TABLE_DIR / "11_flood_source_file_inventory.csv", index=False)

    valid_inventory = inventory[
        inventory["filename_valid"] & inventory["year"].between(START_YEAR, END_YEAR)
    ].copy()

    tiles = sorted(valid_inventory["tile"].dropna().astype(str).unique().tolist())

    print(f"      Valid parquet files     : {len(valid_inventory):,}")
    print(f"      Tiles                   : {tiles}")
    if not valid_inventory.empty:
        print(
            f"      Years                   : {int(valid_inventory['year'].min())} - "
            f"{int(valid_inventory['year'].max())}"
        )
        print("      Files by type:")
        print(valid_inventory.groupby("flood_type").size().to_string())

    # -------------------------------------------------------------------------
    # 3. ADM3 x MODIS tile overlap
    # -------------------------------------------------------------------------
    print("\n[3/7] Intersecting ADM3 boundaries with MODIS tile footprints...")
    tile_overlap = build_tile_overlap(adm3, tiles)
    tile_overlap.to_csv(TABLE_DIR / "11_adm3_tile_overlap.csv", index=False)

    tile_counts = (
        tile_overlap.groupby("adm3_pcode")["tile"]
        .apply(lambda s: int(s.notna().sum()))
    )
    n_no_tile = int(tile_counts.eq(0).sum())
    print(f"      ADM3 with no tile intersection: {n_no_tile}")

    # -------------------------------------------------------------------------
    # 4. Spatial join of positive flood-event records
    # -------------------------------------------------------------------------
    if READ_EVENT_ROWS:
        print("\n[4/7] Spatially joining compact positive flood-event records to ADM3...")
        event_counts = count_flood_records_by_adm3(inventory, adm3)
        event_counts.to_csv(TABLE_DIR / "11_adm3_event_record_counts.csv", index=False)
        print(f"      ADM3-year-type-tile event groups: {len(event_counts):,}")
    else:
        print("\n[4/7] Skipping event-row spatial joins (READ_EVENT_ROWS=False).")
        event_counts = pd.DataFrame()

    # -------------------------------------------------------------------------
    # 5. Complete 512 x 26 audit panel
    # -------------------------------------------------------------------------
    print("\n[5/7] Building complete ADM3 x year coverage panel...")
    audit = build_adm3_year_audit(adm3, tile_overlap, inventory, event_counts)
    audit.to_csv(TABLE_DIR / "11_adm3_year_coverage_audit.csv", index=False)

    expected_rows = len(adm3) * len(EXPECTED_YEARS)
    print(f"      Actual audit rows       : {len(audit):,}")
    print(f"      Expected rows           : {expected_rows:,}")

    if len(audit) != expected_rows:
        raise RuntimeError(
            f"ADM3-year panel is incomplete: expected {expected_rows}, got {len(audit)}."
        )

    print("      QC statuses:")
    print(audit["qc_status"].value_counts(dropna=False).to_string())

    # -------------------------------------------------------------------------
    # 6. ADM3-level coverage summary + zero-vs-no-data review
    # -------------------------------------------------------------------------
    print("\n[6/7] Summarising ADM3 coverage and zero-vs-no-data candidates...")
    summary = summarize_adm3(audit)
    summary.to_csv(TABLE_DIR / "11_adm3_coverage_summary.csv", index=False)

    review = summary[
        summary["audit_interpretation"].isin(
            [
                "26Y_PROCESSED__ZERO_DETECTED_FLOOD_RECORDS",
                "INCOMPLETE_SOURCE_COVERAGE",
                "REVIEW",
            ]
        )
    ].copy()

    if FOCUS_ADM3_PCODES:
        review = pd.concat(
            [review, summary[summary["adm3_pcode"].isin(map(str, FOCUS_ADM3_PCODES))]],
            ignore_index=True,
        ).drop_duplicates("adm3_pcode")

    if FOCUS_ADM3_NAMES:
        review = pd.concat(
            [review, summary[summary["adm3_name"].isin(FOCUS_ADM3_NAMES)]],
            ignore_index=True,
        ).drop_duplicates("adm3_pcode")

    review.to_csv(TABLE_DIR / "11_adm3_zero_vs_nodata_review.csv", index=False)

    # -------------------------------------------------------------------------
    # 7. Optional cross-check against Script 10 QC flags
    # -------------------------------------------------------------------------
    print("\n[7/7] Cross-checking against Script 10 QC flags...")
    script10_qc = load_script10_qc_flags()
    crosscheck = build_script10_crosscheck(summary, script10_qc)

    crosscheck.to_csv(
        TABLE_DIR / "11_adm3_script10_qc_crosscheck.csv",
        index=False,
    )

    if script10_qc is None:
        print("      Script 10 QC flags not found; raw coverage audit remains valid.")
    else:
        matched = int(crosscheck["script10_qc_flags"].notna().sum())
        print(f"      ADM3 matched to Script 10 QC: {matched:,} / {len(crosscheck):,}")

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    zero26 = summary[
        summary["audit_interpretation"] == "26Y_PROCESSED__ZERO_DETECTED_FLOOD_RECORDS"
    ]
    incomplete = summary[
        summary["audit_interpretation"] == "INCOMPLETE_SOURCE_COVERAGE"
    ]
    flood_any = summary[summary["years_with_detected_flood"] > 0]

    invalid_geometries = int((~boundary_qc["geometry_valid"]).sum())
    duplicate_pcodes = int(boundary_qc["duplicate_adm3_pcode"].sum())

    script10_section = ""
    if script10_qc is not None:
        n_script10_review = int(
            pd.to_numeric(
                crosscheck["script10_requires_review"],
                errors="coerce",
            ).fillna(0).astype(bool).sum()
        )
        n_crosscheck_matched = int(
            crosscheck["script10_qc_flags"].notna().sum()
        )
        script10_section = f"""
Script 10 QC cross-check
------------------------
Script 10 QC table: {SCRIPT10_QC_FLAGS}
ADM3 matched to Script 10 QC: {n_crosscheck_matched}
ADM3 flagged for review by Script 10: {n_script10_review}

Important:
Script 10 flags are screening diagnostics. Script 11 source-file/tile coverage
results should be used to interpret all-zero and very sparse flood series before
Script 12 temporal-pattern analysis.
"""

    summary_text = f"""11 ADM3 FLOOD COVERAGE AUDIT SUMMARY
{'=' * 78}
Project root: {PROJECT_ROOT}
ADM3 boundary file: {adm3_path}
Flood root: {flood_root}
Output directory: {OUTPUT_DIR}

Boundary / panel checks
-----------------------
ADM3 units: {len(adm3)}
Invalid ADM3 geometries: {invalid_geometries}
Duplicate ADM3 pcodes: {duplicate_pcodes}
Expected ADM3-year rows ({START_YEAR}-{END_YEAR}): {expected_rows}
Actual ADM3-year audit rows: {len(audit)}

Flood source files
------------------
Valid parquet files in analysis period: {len(valid_inventory)}
Tiles found: {', '.join(tiles)}

ADM3-level coverage results
---------------------------
ADM3 with >=1 detected flood record: {len(flood_any)}
ADM3 with all 26 years of expected tile files but zero detected flood records: {len(zero26)}
ADM3 with incomplete tile/file coverage: {len(incomplete)}

Interpretation rule
-------------------
26Y_PROCESSED__ZERO_DETECTED_FLOOD_RECORDS means that the expected compact
recurring/unusual tile-year files exist for all 26 years and no positive flood
event rows were spatially matched to that ADM3.

This supports the statement:
    "no detected flood record in the compact event dataset"

It does NOT by itself prove:
    "true flood-free conditions under complete valid observation"

For that stronger conclusion, a full observation-validity / quality / cloud mask
would be required. Parent ADM2 values should not be copied into missing ADM3 units
as if they were observed ADM3 measurements.
{script10_section}
"""

    (QC_DIR / "11_audit_summary.txt").write_text(summary_text, encoding="utf-8")

    print("\n" + summary_text)

    if len(zero26):
        print("\nADM3 with 26Y processed source files but zero detected flood records:")
        print(
            zero26[["adm3_pcode", "adm3_name", "adm2_name"]]
            .sort_values("adm3_name")
            .to_string(index=False)
        )

    if len(incomplete):
        print("\nADM3 with incomplete source coverage — review these first:")
        print(
            incomplete[
                [
                    "adm3_pcode",
                    "adm3_name",
                    "adm2_name",
                    "years_complete_tile_files",
                    "years_partial_tile_files",
                    "years_no_tile_year_files",
                    "years_no_tile_overlap",
                ]
            ]
            .sort_values("adm3_name")
            .to_string(index=False)
        )

    print("\nSaved all outputs to:")
    print(f"  {OUTPUT_DIR}")
    print("\nDONE")


if __name__ == "__main__":
    main()
