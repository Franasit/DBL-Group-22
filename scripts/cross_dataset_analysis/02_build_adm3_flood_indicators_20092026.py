#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_build_adm3_flood_indicators.py

South Sudan Flood Analysis
JBG060 Capstone Data Challenge

Purpose
-------
Build ADM3-level daily and annual flood indicators from two flood-mask
collections:

    raw_data/flood_masks/compact_recurring
    raw_data/flood_masks/compact_unusual

Administrative boundary:

    raw_data/administrative_boundaries/ssd_admin3.geojson


Outputs
-------
processed_data/flood_analysis/adm3/

    adm3_daily_flood_indicators_by_type.csv
    adm3_daily_flood_indicators_combined.csv

    adm3_annual_flood_indicators_by_type.csv
    adm3_annual_flood_indicators_combined.csv

    adm3_annual_flood_indicators_all.csv

    adm3_boundary_summary.csv
    flood_file_inventory.csv
    flood_processing_summary.csv


Important interpretation
------------------------
annual_unique_flooded_area_km2:
    Spatial flood footprint.
    A pixel flooded multiple times during the same year is counted once.

annual_flood_area_days_km2:
    Sum of daily flooded area.
    Represents both spatial extent and duration.

flood_days:
    Number of dates on which any flooded pixel was observed in the ADM3.

flood_event_count:
    Number of separate clusters of flood-observation days.

annual_flooded_percent:
    annual_unique_flooded_area_km2 / ADM3 area * 100.


Three categories are produced:
    recurring
    unusual
    combined

For "combined", duplicate pixels appearing in both source collections on the
same date are removed before aggregation.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 1. PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(
    "/Users/shibai/Documents/GitHub/DBL-Group-22"
)

FLOOD_ROOT = (
    PROJECT_ROOT
    / "raw_data"
    / "flood_masks"
)

FLOOD_DIRS = {
    "recurring": FLOOD_ROOT / "compact_recurring",
    "unusual": FLOOD_ROOT / "compact_unusual",
}

ADM3_BOUNDARY_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "administrative_boundaries"
    / "ssd_admin3.geojson"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# OUTPUT FILES
# ============================================================================

DAILY_BY_TYPE_OUTPUT = (
    OUTPUT_DIR
    / "adm3_daily_flood_indicators_by_type.csv"
)

DAILY_COMBINED_OUTPUT = (
    OUTPUT_DIR
    / "adm3_daily_flood_indicators_combined.csv"
)

ANNUAL_BY_TYPE_OUTPUT = (
    OUTPUT_DIR
    / "adm3_annual_flood_indicators_by_type.csv"
)

ANNUAL_COMBINED_OUTPUT = (
    OUTPUT_DIR
    / "adm3_annual_flood_indicators_combined.csv"
)

ANNUAL_ALL_OUTPUT = (
    OUTPUT_DIR
    / "adm3_annual_flood_indicators_all.csv"
)

BOUNDARY_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "adm3_boundary_summary.csv"
)

FILE_INVENTORY_OUTPUT = (
    OUTPUT_DIR
    / "flood_file_inventory.csv"
)

PROCESSING_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "flood_processing_summary.csv"
)


# ============================================================================
# ANALYSIS SETTINGS
# ============================================================================

WGS84 = "EPSG:4326"
EQUAL_AREA_CRS = "EPSG:6933"

EVENT_GAP_DAYS = 1
COORD_DECIMALS = 6


# ============================================================================
# 2. GENERAL HELPERS
# ============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


# ============================================================================
# 3. FLOOD FILE DISCOVERY
# ============================================================================

def extract_year_tile(path: Path):
    name = path.stem.lower()

    year_match = re.search(
        r"(?:19|20)\d{2}",
        name,
    )

    tile_match = re.search(
        r"h\d{2}v\d{2}",
        name,
    )

    year = (
        int(year_match.group())
        if year_match
        else None
    )

    tile = (
        tile_match.group()
        if tile_match
        else None
    )

    return year, tile


def is_copy_filename(path: Path) -> bool:
    return bool(
        re.search(
            r"\s*\(\d+\)$",
            path.stem,
        )
    )


def deduplicate_source_files(
    files: list[Path],
) -> list[Path]:

    grouped = {}
    unmatched = []

    for path in files:

        year, tile = extract_year_tile(path)

        if year is None or tile is None:
            unmatched.append(path)
            continue

        key = (year, tile)

        if key not in grouped:
            grouped[key] = path
            continue

        existing = grouped[key]

        if (
            is_copy_filename(existing)
            and not is_copy_filename(path)
        ):
            grouped[key] = path

    return (
        sorted(grouped.values())
        + sorted(unmatched)
    )


def discover_flood_files() -> pd.DataFrame:

    records = []

    for flood_type, folder in FLOOD_DIRS.items():

        require_path(
            folder,
            f"{flood_type} flood directory",
        )

        files = sorted(
            folder.rglob("*.parquet")
        )

        files = deduplicate_source_files(
            files
        )

        print(
            f"{flood_type}: "
            f"{len(files)} parquet files"
        )

        for path in files:

            year, tile = extract_year_tile(
                path
            )

            records.append(
                {
                    "flood_type": flood_type,
                    "year": year,
                    "tile": tile,
                    "path": str(path),
                    "filename": path.name,
                }
            )

    inventory = pd.DataFrame(records)

    if inventory.empty:
        raise FileNotFoundError(
            "\nNo flood parquet files were found."
        )

    inventory = inventory.sort_values(
        [
            "year",
            "tile",
            "flood_type",
        ],
        na_position="last",
    ).reset_index(drop=True)

    inventory.to_csv(
        FILE_INVENTORY_OUTPUT,
        index=False,
    )

    return inventory


# ============================================================================
# 4. ADM3 BOUNDARY
# ============================================================================

def identify_adm3_columns(
    gdf: gpd.GeoDataFrame,
):
    """
    Try to preserve:
        ADM1
        ADM2
        ADM3
    hierarchy where available.
    """

    column_map = {
        str(c).upper(): c
        for c in gdf.columns
    }

    id_candidates = [
        "GID_3",
        "ADM3_PCODE",
        "ADM3_ID",
        "ADM3_CODE",
    ]

    name_candidates = [
        "NAME_3",
        "ADM3_EN",
        "ADM3_NAME",
        "NAME",
    ]

    adm2_id_candidates = [
        "GID_2",
        "ADM2_PCODE",
        "ADM2_ID",
    ]

    adm2_name_candidates = [
        "NAME_2",
        "ADM2_EN",
        "ADM2_NAME",
    ]

    adm1_id_candidates = [
        "GID_1",
        "ADM1_PCODE",
        "ADM1_ID",
    ]

    adm1_name_candidates = [
        "NAME_1",
        "ADM1_EN",
        "ADM1_NAME",
    ]

    def find_first(candidates):
        for candidate in candidates:
            if candidate in column_map:
                return column_map[candidate]
        return None

    id_col = find_first(id_candidates)
    name_col = find_first(name_candidates)

    adm2_id_col = find_first(adm2_id_candidates)
    adm2_name_col = find_first(adm2_name_candidates)

    adm1_id_col = find_first(adm1_id_candidates)
    adm1_name_col = find_first(adm1_name_candidates)

    gdf = gdf.copy()

    if id_col is None:
        gdf["GENERATED_ADM3_ID"] = [
            f"ADM3_{i:04d}"
            for i in range(
                1,
                len(gdf) + 1,
            )
        ]
        id_col = "GENERATED_ADM3_ID"

    if name_col is None:
        gdf["GENERATED_ADM3_NAME"] = (
            gdf[id_col].astype(str)
        )
        name_col = "GENERATED_ADM3_NAME"

    if adm2_id_col is None:
        gdf["ADM2_PARENT_ID"] = ""
        adm2_id_col = "ADM2_PARENT_ID"

    if adm2_name_col is None:
        gdf["ADM2_PARENT_NAME"] = ""
        adm2_name_col = "ADM2_PARENT_NAME"

    if adm1_id_col is None:
        gdf["ADM1_PARENT_ID"] = ""
        adm1_id_col = "ADM1_PARENT_ID"

    if adm1_name_col is None:
        gdf["ADM1_PARENT_NAME"] = ""
        adm1_name_col = "ADM1_PARENT_NAME"

    return (
        gdf,
        id_col,
        name_col,
        adm2_id_col,
        adm2_name_col,
        adm1_id_col,
        adm1_name_col,
    )


def load_adm3_boundary():

    require_path(
        ADM3_BOUNDARY_FILE,
        "South Sudan ADM3 boundary",
    )

    print(
        f"\nLoading ADM3 boundary:\n"
        f"{ADM3_BOUNDARY_FILE}"
    )

    adm3 = gpd.read_file(
        ADM3_BOUNDARY_FILE
    )

    if adm3.empty:
        raise ValueError(
            "ADM3 boundary is empty."
        )

    if adm3.crs is None:

        print(
            "Boundary CRS missing. "
            "Assuming EPSG:4326."
        )

        adm3 = adm3.set_crs(
            WGS84
        )

    adm3 = adm3.to_crs(
        WGS84
    )

    (
        adm3,
        id_col,
        name_col,
        adm2_id_col,
        adm2_name_col,
        adm1_id_col,
        adm1_name_col,
    ) = identify_adm3_columns(
        adm3
    )

    adm3 = adm3[
        adm3.geometry.notna()
        & ~adm3.geometry.is_empty
    ].copy()

    try:
        adm3["geometry"] = (
            adm3.geometry.make_valid()
        )

    except Exception:
        adm3["geometry"] = (
            adm3.geometry.buffer(0)
        )

    adm3_equal_area = adm3.to_crs(
        EQUAL_AREA_CRS
    )

    adm3["adm3_area_km2"] = (
        adm3_equal_area.geometry.area
        / 1_000_000.0
    )

    adm3 = adm3[
        [
            adm1_id_col,
            adm1_name_col,
            adm2_id_col,
            adm2_name_col,
            id_col,
            name_col,
            "adm3_area_km2",
            "geometry",
        ]
    ].copy()

    adm3 = adm3.rename(
        columns={
            adm1_id_col: "ADM1_ID",
            adm1_name_col: "ADM1_NAME",
            adm2_id_col: "ADM2_ID",
            adm2_name_col: "ADM2_NAME",
            id_col: "ADM3_ID",
            name_col: "ADM3_NAME",
        }
    )

    for col in [
        "ADM1_ID",
        "ADM1_NAME",
        "ADM2_ID",
        "ADM2_NAME",
        "ADM3_ID",
        "ADM3_NAME",
    ]:
        adm3[col] = (
            adm3[col]
            .astype(str)
        )

    minx, miny, maxx, maxy = (
        adm3.total_bounds
    )

    print(
        "\nADM3 boundary bounds:"
    )

    print(
        f"Longitude: "
        f"{minx:.4f} to {maxx:.4f}"
    )

    print(
        f"Latitude : "
        f"{miny:.4f} to {maxy:.4f}"
    )

    print(
        f"ADM3 count: {len(adm3)}"
    )

    (
        adm3[
            [
                "ADM1_ID",
                "ADM1_NAME",
                "ADM2_ID",
                "ADM2_NAME",
                "ADM3_ID",
                "ADM3_NAME",
                "adm3_area_km2",
            ]
        ]
        .sort_values(
            [
                "ADM2_NAME",
                "ADM3_NAME",
            ]
        )
        .to_csv(
            BOUNDARY_SUMMARY_OUTPUT,
            index=False,
        )
    )

    return adm3


# ============================================================================
# 5. FLOOD DATA HELPERS
# ============================================================================

def normalize_flood_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rename = {}

    for col in df.columns:

        low = str(col).strip().lower()

        if low in {
            "date",
            "datetime",
            "timestamp",
            "time",
            "observation_date",
        }:
            rename[col] = "date"

        elif low in {
            "lat",
            "latitude",
            "y",
        }:
            rename[col] = "lat"

        elif low in {
            "lon",
            "lng",
            "long",
            "longitude",
            "x",
        }:
            rename[col] = "lon"

    return df.rename(
        columns=rename
    )


def read_flood_file(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_parquet(
        path
    )

    df = normalize_flood_columns(
        df
    )

    required = {
        "date",
        "lat",
        "lon",
    }

    missing = required - set(
        df.columns
    )

    if missing:

        raise ValueError(
            f"\nMissing required columns "
            f"in {path.name}: "
            f"{sorted(missing)}\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    df = df[
        [
            "date",
            "lat",
            "lon",
        ]
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce",
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date",
            "lat",
            "lon",
        ]
    )

    df = df[
        df["lat"].between(
            -90,
            90,
        )
        & df["lon"].between(
            -180,
            180,
        )
    ].copy()

    df["date"] = (
        df["date"]
        .dt.normalize()
    )

    df["lat"] = (
        df["lat"]
        .round(
            COORD_DECIMALS
        )
    )

    df["lon"] = (
        df["lon"]
        .round(
            COORD_DECIMALS
        )
    )

    df = df.drop_duplicates(
        subset=[
            "date",
            "lat",
            "lon",
        ]
    )

    df["year"] = (
        df["date"]
        .dt.year
        .astype(int)
    )

    return df


# ============================================================================
# 6. GRID RESOLUTION / PIXEL AREA
# ============================================================================

def estimate_coordinate_spacing(
    values,
):

    values = pd.Series(
        values
    ).dropna()

    unique = np.sort(
        np.unique(
            np.round(
                values.astype(float),
                7,
            )
        )
    )

    if len(unique) < 2:
        return None

    diffs = np.diff(
        unique
    )

    diffs = diffs[
        np.isfinite(diffs)
        & (diffs > 1e-7)
    ]

    if len(diffs) == 0:
        return None

    rounded = np.round(
        diffs,
        6,
    )

    counts = Counter(
        rounded
    )

    common = counts.most_common(
        30
    )

    if not common:
        return float(
            np.median(diffs)
        )

    max_frequency = common[0][1]

    candidates = [
        float(diff)
        for diff, frequency
        in common
        if frequency
        >= max(
            2,
            max_frequency * 0.10,
        )
    ]

    if candidates:
        return min(candidates)

    return float(
        np.min(diffs)
    )


def estimate_grid_resolution(
    df: pd.DataFrame,
):

    dlat = estimate_coordinate_spacing(
        df["lat"]
    )

    dlon = estimate_coordinate_spacing(
        df["lon"]
    )

    return dlat, dlon


def grid_cell_area_km2(
    latitude,
    dlat_deg,
    dlon_deg,
):

    earth_radius_km = 6371.0088

    latitude = np.asarray(
        latitude,
        dtype=float,
    )

    lat_lower = np.radians(
        latitude
        - abs(dlat_deg) / 2
    )

    lat_upper = np.radians(
        latitude
        + abs(dlat_deg) / 2
    )

    dlon_rad = math.radians(
        abs(dlon_deg)
    )

    area = (
        earth_radius_km ** 2
        * dlon_rad
        * np.abs(
            np.sin(lat_upper)
            - np.sin(lat_lower)
        )
    )

    return area


# ============================================================================
# 7. SPATIAL SANITY CHECK
# ============================================================================

def bbox_overlap(
    df: pd.DataFrame,
    adm3: gpd.GeoDataFrame,
):

    flood_min_lon = df["lon"].min()
    flood_max_lon = df["lon"].max()

    flood_min_lat = df["lat"].min()
    flood_max_lat = df["lat"].max()

    adm_min_lon, adm_min_lat, \
        adm_max_lon, adm_max_lat = (
            adm3.total_bounds
        )

    overlap = (
        flood_max_lon >= adm_min_lon
        and flood_min_lon <= adm_max_lon
        and flood_max_lat >= adm_min_lat
        and flood_min_lat <= adm_max_lat
    )

    return overlap


def print_flood_bounds(
    df: pd.DataFrame,
):

    print(
        "Flood coordinate bounds:"
    )

    print(
        f"  Longitude: "
        f"{df['lon'].min():.6f} "
        f"to "
        f"{df['lon'].max():.6f}"
    )

    print(
        f"  Latitude : "
        f"{df['lat'].min():.6f} "
        f"to "
        f"{df['lat'].max():.6f}"
    )


# ============================================================================
# 8. SPATIAL JOIN
# ============================================================================

def spatially_assign_adm3(
    df: pd.DataFrame,
    adm3: gpd.GeoDataFrame,
):

    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["lon"],
            df["lat"],
        ),
        crs=WGS84,
    )

    joined = gpd.sjoin(
        points,
        adm3[
            [
                "ADM1_ID",
                "ADM1_NAME",
                "ADM2_ID",
                "ADM2_NAME",
                "ADM3_ID",
                "ADM3_NAME",
                "adm3_area_km2",
                "geometry",
            ]
        ],
        how="inner",
        predicate="intersects",
    )

    if "index_right" in joined.columns:
        joined = joined.drop(
            columns="index_right"
        )

    return joined


# ============================================================================
# 9. DAILY + ANNUAL INTERMEDIATE AGGREGATION
# ============================================================================

def aggregate_joined_points(
    joined: gpd.GeoDataFrame,
):

    group_base = [
        "ADM1_ID",
        "ADM1_NAME",
        "ADM2_ID",
        "ADM2_NAME",
        "ADM3_ID",
        "ADM3_NAME",
    ]

    daily = (
        joined
        .groupby(
            group_base
            + [
                "date",
                "year",
            ],
            as_index=False,
        )
        .agg(
            flooded_pixel_count=(
                "pixel_area_km2",
                "size",
            ),
            daily_flooded_area_km2=(
                "pixel_area_km2",
                "sum",
            ),
        )
    )

    annual_unique_cells = (
        joined[
            group_base
            + [
                "year",
                "lat",
                "lon",
                "pixel_area_km2",
            ]
        ]
        .drop_duplicates(
            subset=[
                "ADM3_ID",
                "year",
                "lat",
                "lon",
            ]
        )
    )

    annual_unique = (
        annual_unique_cells
        .groupby(
            group_base
            + [
                "year",
            ],
            as_index=False,
        )
        .agg(
            annual_unique_flooded_pixel_count=(
                "pixel_area_km2",
                "size",
            ),
            annual_unique_flooded_area_km2=(
                "pixel_area_km2",
                "sum",
            ),
        )
    )

    return (
        daily,
        annual_unique,
    )


# ============================================================================
# 10. PROCESS ONE SOURCE FILE
# ============================================================================

def process_single_file(
    path: Path,
    flood_type: str,
    adm3: gpd.GeoDataFrame,
):

    print_subheader(
        f"Processing {flood_type}: "
        f"{path.name}"
    )

    df = read_flood_file(
        path
    )

    input_rows = len(df)

    print(
        f"Valid unique rows: "
        f"{input_rows:,}"
    )

    print_flood_bounds(
        df
    )

    if not bbox_overlap(
        df,
        adm3,
    ):

        print(
            "WARNING: flood bounding box "
            "does not overlap South Sudan ADM3."
        )

        return (
            None,
            None,
            {
                "flood_type": flood_type,
                "filename": path.name,
                "status": "NO_BBOX_OVERLAP",
                "rows_input": input_rows,
            },
        )

    dlat, dlon = (
        estimate_grid_resolution(
            df
        )
    )

    if (
        dlat is None
        or dlon is None
    ):
        raise ValueError(
            f"Could not estimate grid "
            f"resolution for {path.name}"
        )

    print(
        f"Estimated grid resolution: "
        f"{dlat:.6f}° × "
        f"{dlon:.6f}°"
    )

    df["pixel_area_km2"] = (
        grid_cell_area_km2(
            latitude=df["lat"],
            dlat_deg=dlat,
            dlon_deg=dlon,
        )
    )

    median_pixel_area = (
        df["pixel_area_km2"]
        .median()
    )

    print(
        f"Median pixel area: "
        f"{median_pixel_area:.6f} km²"
    )

    joined = spatially_assign_adm3(
        df,
        adm3,
    )

    rows_inside = len(
        joined
    )

    print(
        f"Rows inside ADM3: "
        f"{rows_inside:,}"
    )

    if joined.empty:

        return (
            None,
            None,
            {
                "flood_type": flood_type,
                "filename": path.name,
                "status": "NO_SPATIAL_MATCH",
                "rows_input": input_rows,
                "rows_inside_adm3": 0,
                "grid_dlat_deg": dlat,
                "grid_dlon_deg": dlon,
                "median_pixel_area_km2":
                    median_pixel_area,
            },
        )

    daily, annual_unique = (
        aggregate_joined_points(
            joined
        )
    )

    daily["flood_type"] = (
        flood_type
    )

    annual_unique["flood_type"] = (
        flood_type
    )

    summary = {
        "flood_type": flood_type,
        "filename": path.name,
        "status": "OK",
        "rows_input": input_rows,
        "rows_inside_adm3":
            rows_inside,
        "share_inside_adm3":
            rows_inside / input_rows,
        "grid_dlat_deg": dlat,
        "grid_dlon_deg": dlon,
        "median_pixel_area_km2":
            median_pixel_area,
        "min_date":
            joined["date"].min(),
        "max_date":
            joined["date"].max(),
        "adm3_count":
            joined["ADM3_ID"].nunique(),
    }

    return (
        daily,
        annual_unique,
        summary,
    )


# ============================================================================
# 11. PROCESS COMBINED RECURRING + UNUSUAL
# ============================================================================

def process_combined_group(
    paths: list[Path],
    adm3: gpd.GeoDataFrame,
    group_label: str,
):

    if not paths:
        return None, None

    frames = []

    for path in paths:

        df = read_flood_file(
            path
        )

        frames.append(
            df
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    before = len(
        combined
    )

    combined = (
        combined
        .drop_duplicates(
            subset=[
                "date",
                "lat",
                "lon",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    after = len(
        combined
    )

    print(
        f"\nCombined {group_label}: "
        f"{before:,} → "
        f"{after:,} rows after "
        f"cross-type deduplication."
    )

    if combined.empty:
        return None, None

    if not bbox_overlap(
        combined,
        adm3,
    ):
        return None, None

    dlat, dlon = (
        estimate_grid_resolution(
            combined
        )
    )

    if (
        dlat is None
        or dlon is None
    ):
        return None, None

    combined["pixel_area_km2"] = (
        grid_cell_area_km2(
            combined["lat"],
            dlat,
            dlon,
        )
    )

    joined = spatially_assign_adm3(
        combined,
        adm3,
    )

    if joined.empty:
        return None, None

    daily, annual_unique = (
        aggregate_joined_points(
            joined
        )
    )

    daily["flood_type"] = (
        "combined"
    )

    annual_unique["flood_type"] = (
        "combined"
    )

    return (
        daily,
        annual_unique,
    )


# ============================================================================
# 12. EVENT COUNT
# ============================================================================

def count_flood_events(
    dates,
):

    dates = pd.to_datetime(
        pd.Series(
            dates
        )
        .dropna()
        .unique()
    )

    if len(dates) == 0:
        return 0

    dates = pd.Series(
        sorted(dates)
    )

    gaps = (
        dates
        .diff()
        .dt.days
    )

    events = (
        1
        + (
            gaps
            > EVENT_GAP_DAYS
        ).sum()
    )

    return int(events)


# ============================================================================
# 13. MERGE DAILY PARTS
# ============================================================================

def merge_daily_parts(
    parts: list[pd.DataFrame],
):

    df = pd.concat(
        parts,
        ignore_index=True,
    )

    group_cols = [
        "ADM1_ID",
        "ADM1_NAME",
        "ADM2_ID",
        "ADM2_NAME",
        "ADM3_ID",
        "ADM3_NAME",
        "date",
        "year",
        "flood_type",
    ]

    df = (
        df
        .groupby(
            group_cols,
            as_index=False,
        )
        .agg(
            flooded_pixel_count=(
                "flooded_pixel_count",
                "sum",
            ),
            daily_flooded_area_km2=(
                "daily_flooded_area_km2",
                "sum",
            ),
        )
    )

    return (
        df
        .sort_values(
            [
                "flood_type",
                "ADM2_NAME",
                "ADM3_NAME",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================================
# 14. MERGE ANNUAL UNIQUE FOOTPRINTS
# ============================================================================

def merge_annual_unique_parts(
    parts: list[pd.DataFrame],
):

    df = pd.concat(
        parts,
        ignore_index=True,
    )

    df = (
        df
        .groupby(
            [
                "ADM1_ID",
                "ADM1_NAME",
                "ADM2_ID",
                "ADM2_NAME",
                "ADM3_ID",
                "ADM3_NAME",
                "year",
                "flood_type",
            ],
            as_index=False,
        )
        .agg(
            annual_unique_flooded_pixel_count=(
                "annual_unique_flooded_pixel_count",
                "sum",
            ),
            annual_unique_flooded_area_km2=(
                "annual_unique_flooded_area_km2",
                "sum",
            ),
        )
    )

    return df


# ============================================================================
# 15. BUILD COMPLETE ADM3 × YEAR × TYPE GRID
# ============================================================================

def build_complete_grid(
    adm3: gpd.GeoDataFrame,
    years: list[int],
    flood_types: list[str],
):

    adm_table = (
        adm3[
            [
                "ADM1_ID",
                "ADM1_NAME",
                "ADM2_ID",
                "ADM2_NAME",
                "ADM3_ID",
                "ADM3_NAME",
                "adm3_area_km2",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    year_table = pd.DataFrame(
        {
            "year": years
        }
    )

    type_table = pd.DataFrame(
        {
            "flood_type":
                flood_types
        }
    )

    adm_table["_key"] = 1
    year_table["_key"] = 1
    type_table["_key"] = 1

    grid = (
        adm_table
        .merge(
            year_table,
            on="_key",
        )
        .merge(
            type_table,
            on="_key",
        )
        .drop(
            columns="_key"
        )
    )

    return grid


# ============================================================================
# 16. BUILD FINAL ANNUAL TABLE
# ============================================================================

def build_annual_indicators(
    daily: pd.DataFrame,
    annual_unique: pd.DataFrame,
    adm3: gpd.GeoDataFrame,
    years: list[int],
    flood_types: list[str],
):

    group_keys = [
        "ADM1_ID",
        "ADM1_NAME",
        "ADM2_ID",
        "ADM2_NAME",
        "ADM3_ID",
        "ADM3_NAME",
        "year",
        "flood_type",
    ]

    annual_daily = (
        daily
        .groupby(
            group_keys,
            as_index=False,
        )
        .agg(
            flood_days=(
                "date",
                "nunique",
            ),
            annual_flood_area_days_km2=(
                "daily_flooded_area_km2",
                "sum",
            ),
            max_daily_flooded_area_km2=(
                "daily_flooded_area_km2",
                "max",
            ),
            mean_flooded_area_on_flood_days_km2=(
                "daily_flooded_area_km2",
                "mean",
            ),
            median_flooded_area_on_flood_days_km2=(
                "daily_flooded_area_km2",
                "median",
            ),
            total_flood_pixel_observations=(
                "flooded_pixel_count",
                "sum",
            ),
        )
    )

    event_counts = (
        daily
        .groupby(
            group_keys
        )["date"]
        .apply(
            count_flood_events
        )
        .reset_index(
            name="flood_event_count"
        )
    )

    annual_daily = (
        annual_daily
        .merge(
            event_counts,
            on=group_keys,
            how="left",
        )
    )

    annual = (
        annual_daily
        .merge(
            annual_unique,
            on=group_keys,
            how="outer",
        )
    )

    full_grid = (
        build_complete_grid(
            adm3=adm3,
            years=years,
            flood_types=flood_types,
        )
    )

    annual = (
        full_grid
        .merge(
            annual,
            on=group_keys,
            how="left",
        )
    )

    zero_columns = [
        "flood_days",
        "annual_flood_area_days_km2",
        "max_daily_flooded_area_km2",
        "mean_flooded_area_on_flood_days_km2",
        "median_flooded_area_on_flood_days_km2",
        "total_flood_pixel_observations",
        "flood_event_count",
        "annual_unique_flooded_pixel_count",
        "annual_unique_flooded_area_km2",
    ]

    for col in zero_columns:

        annual[col] = (
            annual[col]
            .fillna(0)
        )

    integer_columns = [
        "flood_days",
        "total_flood_pixel_observations",
        "flood_event_count",
        "annual_unique_flooded_pixel_count",
    ]

    for col in integer_columns:

        annual[col] = (
            annual[col]
            .round()
            .astype(int)
        )

    annual["flood_occurred"] = (
        annual["flood_days"]
        > 0
    ).astype(int)

    annual["annual_flooded_ratio"] = (
        annual[
            "annual_unique_flooded_area_km2"
        ]
        / annual[
            "adm3_area_km2"
        ]
    )

    annual["annual_flooded_percent"] = (
        annual[
            "annual_flooded_ratio"
        ]
        * 100
    )

    annual["mean_event_duration_days"] = (
        np.where(
            annual[
                "flood_event_count"
            ] > 0,
            annual[
                "flood_days"
            ]
            / annual[
                "flood_event_count"
            ],
            0,
        )
    )

    annual = (
        annual
        .sort_values(
            [
                "flood_type",
                "ADM2_NAME",
                "ADM3_NAME",
                "year",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return annual


# ============================================================================
# 17. MAIN
# ============================================================================

def main():

    print_header(
        "ADM3 FLOOD INDICATOR BUILDER"
    )

    print(
        f"Project root:\n"
        f"{PROJECT_ROOT}"
    )

    print(
        f"\nADM3 boundary:\n"
        f"{ADM3_BOUNDARY_FILE}"
    )

    print(
        f"\nOutput folder:\n"
        f"{OUTPUT_DIR}"
    )

    # ----------------------------------------------------------------------
    # Load ADM3 boundary
    # ----------------------------------------------------------------------

    adm3 = load_adm3_boundary()

    # ----------------------------------------------------------------------
    # Discover flood files
    # ----------------------------------------------------------------------

    print_header(
        "DISCOVERING FLOOD FILES"
    )

    inventory = discover_flood_files()

    print(
        "\nFile inventory:"
    )

    print(
        inventory[
            [
                "flood_type",
                "year",
                "tile",
                "filename",
            ]
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------------------
    # Containers
    # ----------------------------------------------------------------------

    daily_type_parts = []
    unique_type_parts = []

    daily_combined_parts = []
    unique_combined_parts = []

    processing_records = []

    discovered_years = set()

    # ----------------------------------------------------------------------
    # Process source datasets
    # ----------------------------------------------------------------------

    print_header(
        "PROCESSING RECURRING AND UNUSUAL FLOOD MASKS"
    )

    total_files = len(
        inventory
    )

    for i, row in inventory.iterrows():

        print(
            f"\n[{i + 1}/{total_files}]"
        )

        path = Path(
            row["path"]
        )

        flood_type = str(
            row["flood_type"]
        )

        try:

            daily, annual_unique, summary = (
                process_single_file(
                    path=path,
                    flood_type=flood_type,
                    adm3=adm3,
                )
            )

            processing_records.append(
                summary
            )

            if daily is not None:

                daily_type_parts.append(
                    daily
                )

                discovered_years.update(
                    daily[
                        "year"
                    ]
                    .astype(int)
                    .unique()
                    .tolist()
                )

            if annual_unique is not None:

                unique_type_parts.append(
                    annual_unique
                )

        except Exception as exc:

            print(
                f"\nERROR: "
                f"{path.name}"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            processing_records.append(
                {
                    "flood_type":
                        flood_type,
                    "filename":
                        path.name,
                    "status":
                        "ERROR",
                    "error":
                        str(exc),
                }
            )

    # ----------------------------------------------------------------------
    # Build combined source
    # ----------------------------------------------------------------------

    print_header(
        "BUILDING COMBINED FLOOD MASK"
    )

    grouped_paths = defaultdict(
        list
    )

    for _, row in inventory.iterrows():

        key = (
            row["year"],
            row["tile"],
        )

        grouped_paths[key].append(
            Path(
                row["path"]
            )
        )

    group_items = sorted(
        grouped_paths.items(),
        key=lambda x: (
            x[0][0]
            if x[0][0] is not None
            else 9999,
            str(x[0][1]),
        ),
    )

    for index, (key, paths) in enumerate(
        group_items,
        start=1,
    ):

        year, tile = key

        label = (
            f"year={year}, "
            f"tile={tile}"
        )

        print(
            f"\n[{index}/{len(group_items)}] "
            f"{label}"
        )

        try:

            daily, annual_unique = (
                process_combined_group(
                    paths=paths,
                    adm3=adm3,
                    group_label=label,
                )
            )

            if daily is not None:

                daily_combined_parts.append(
                    daily
                )

                discovered_years.update(
                    daily[
                        "year"
                    ]
                    .astype(int)
                    .unique()
                    .tolist()
                )

            if annual_unique is not None:

                unique_combined_parts.append(
                    annual_unique
                )

        except Exception as exc:

            print(
                f"Combined group failed: "
                f"{label}"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    # ----------------------------------------------------------------------
    # Validate success
    # ----------------------------------------------------------------------

    if not daily_type_parts:

        raise RuntimeError(
            "\nNo recurring/unusual flood files "
            "were successfully spatially joined "
            "to ADM3."
        )

    if not daily_combined_parts:

        raise RuntimeError(
            "\nNo combined ADM3 flood data were produced."
        )

    if not discovered_years:

        raise RuntimeError(
            "\nNo valid flood years were found."
        )

    # ----------------------------------------------------------------------
    # Processing summary
    # ----------------------------------------------------------------------

    processing_df = pd.DataFrame(
        processing_records
    )

    processing_df.to_csv(
        PROCESSING_SUMMARY_OUTPUT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Merge daily
    # ----------------------------------------------------------------------

    print_header(
        "MERGING DAILY RESULTS"
    )

    daily_by_type = (
        merge_daily_parts(
            daily_type_parts
        )
    )

    daily_combined = (
        merge_daily_parts(
            daily_combined_parts
        )
    )

    daily_by_type.to_csv(
        DAILY_BY_TYPE_OUTPUT,
        index=False,
    )

    daily_combined.to_csv(
        DAILY_COMBINED_OUTPUT,
        index=False,
    )

    print(
        f"Daily by-type rows: "
        f"{len(daily_by_type):,}"
    )

    print(
        f"Daily combined rows: "
        f"{len(daily_combined):,}"
    )

    # ----------------------------------------------------------------------
    # Merge annual footprint
    # ----------------------------------------------------------------------

    annual_unique_by_type = (
        merge_annual_unique_parts(
            unique_type_parts
        )
    )

    annual_unique_combined = (
        merge_annual_unique_parts(
            unique_combined_parts
        )
    )

    # ----------------------------------------------------------------------
    # Years
    # ----------------------------------------------------------------------

    min_year = min(
        discovered_years
    )

    max_year = max(
        discovered_years
    )

    years = list(
        range(
            min_year,
            max_year + 1,
        )
    )

    print(
        f"\nDetected year range: "
        f"{min_year}-{max_year}"
    )

    # ----------------------------------------------------------------------
    # Annual by type
    # ----------------------------------------------------------------------

    print_header(
        "BUILDING ANNUAL ADM3 INDICATORS"
    )

    annual_by_type = (
        build_annual_indicators(
            daily=daily_by_type,
            annual_unique=
                annual_unique_by_type,
            adm3=adm3,
            years=years,
            flood_types=[
                "recurring",
                "unusual",
            ],
        )
    )

    annual_combined = (
        build_annual_indicators(
            daily=daily_combined,
            annual_unique=
                annual_unique_combined,
            adm3=adm3,
            years=years,
            flood_types=[
                "combined",
            ],
        )
    )

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------

    annual_by_type.to_csv(
        ANNUAL_BY_TYPE_OUTPUT,
        index=False,
    )

    annual_combined.to_csv(
        ANNUAL_COMBINED_OUTPUT,
        index=False,
    )

    annual_all = pd.concat(
        [
            annual_by_type,
            annual_combined,
        ],
        ignore_index=True,
    )

    annual_all = (
        annual_all
        .sort_values(
            [
                "flood_type",
                "ADM2_NAME",
                "ADM3_NAME",
                "year",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    annual_all.to_csv(
        ANNUAL_ALL_OUTPUT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------

    print_header(
        "QUALITY CHECK"
    )

    print(
        "\nMaximum annual flooded percent:"
    )

    diagnostic = (
        annual_all
        .groupby(
            "flood_type"
        )[
            "annual_flooded_percent"
        ]
        .max()
        .round(2)
    )

    print(
        diagnostic
    )

    suspicious = (
        annual_all[
            annual_all[
                "annual_flooded_percent"
            ]
            > 105
        ]
    )

    if len(suspicious) > 0:

        print(
            "\nWARNING:"
            "\nSome ADM3-year observations have "
            "annual flooded percentage > 105%."
        )

        print(
            suspicious[
                [
                    "ADM2_NAME",
                    "ADM3_NAME",
                    "year",
                    "flood_type",
                    "annual_flooded_percent",
                ]
            ]
            .sort_values(
                "annual_flooded_percent",
                ascending=False,
            )
            .head(30)
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    print_header(
        "FINISHED"
    )

    print(
        f"ADM3 regions: "
        f"{adm3['ADM3_ID'].nunique()}"
    )

    print(
        f"Years: "
        f"{min_year}-{max_year}"
    )

    print(
        "\nFlood categories:"
        "\n  recurring"
        "\n  unusual"
        "\n  combined"
    )

    print(
        "\nSaved:"
    )

    for path in [
        DAILY_BY_TYPE_OUTPUT,
        DAILY_COMBINED_OUTPUT,
        ANNUAL_BY_TYPE_OUTPUT,
        ANNUAL_COMBINED_OUTPUT,
        ANNUAL_ALL_OUTPUT,
        BOUNDARY_SUMMARY_OUTPUT,
        FILE_INVENTORY_OUTPUT,
        PROCESSING_SUMMARY_OUTPUT,
    ]:
        print(
            f"  {path}"
        )

    print(
        "\nKey file for later trend analysis:"
    )

    print(
        f"  {ANNUAL_ALL_OUTPUT}"
    )

    print(
        "\nNext:"
        "\n03_analyze_adm2_flood_trends.py"
        "\n04_analyze_adm3_flood_trends.py"
    )


if __name__ == "__main__":
    main()