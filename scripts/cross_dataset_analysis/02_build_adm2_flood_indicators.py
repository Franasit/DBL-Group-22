"""
02_build_adm2_flood_indicators.py

Purpose
-------
Build daily ADM2-level flood indicators for South Sudan from the
compact_unusual and compact_recurring flood-event parquet datasets.

Study period
------------
2000-2025

Main outputs
------------
1. ADM2_daily_flood_indicators.csv
2. ADM2_flood_summary_2000_2025.csv
3. ADM2_flood_spatial_qc.csv

Main daily indicators
---------------------
- unusual_flood_occurrence
- unusual_flood_pixel_count
- unusual_flooded_area_km2_est

- recurring_flood_occurrence
- recurring_flood_pixel_count
- recurring_flooded_area_km2_est

- flood_occurrence
- flood_pixel_count
- flooded_area_km2_est
- flooded_fraction_est

Important
---------
The flooded-area variables are estimated under the assumption that
each unique date-lat-lon record represents one flooded grid cell.

The script estimates grid-cell size from the coordinate spacing in
the flood dataset.

If this assumption is not appropriate for the source dataset,
use flood_occurrence and flood_pixel_count as the primary indicators.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd


# ================================================================
# 1. PROJECT PATHS
# ================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
OUTPUT_DIR = PROCESSED_DATA_DIR / "cross_dataset_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# Flood data
# ------------------------------------------------

FLOOD_BASE_DIR = RAW_DATA_DIR / "flood_masks"

UNUSUAL_DIR = FLOOD_BASE_DIR / "compact_unusual"
RECURRING_DIR = FLOOD_BASE_DIR / "compact_recurring"


# ------------------------------------------------
# Administrative boundary
# ------------------------------------------------

# If you know the exact ADM2 shapefile path, you can set it here.
#
# Example:
#
# BOUNDARY_FILE = (
#     RAW_DATA_DIR
#     / "administrative_boundary"
#     / "ssd_admbnda_adm2.shp"
# )
#
# Otherwise leave it as None and the script will search automatically.

BOUNDARY_FILE = None


# ================================================================
# 2. ANALYSIS SETTINGS
# ================================================================

START_YEAR = 2000
END_YEAR = 2025

START_DATE = f"{START_YEAR}-01-01"
END_DATE = f"{END_YEAR}-12-31"

POINT_CRS = "EPSG:4326"

# Equal-area projection used for calculating ADM2 polygon area.
AREA_CRS = "EPSG:6933"


# ================================================================
# 3. OUTPUT FILES
# ================================================================

DAILY_OUTPUT = (
    OUTPUT_DIR
    / "ADM2_daily_flood_indicators.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "ADM2_flood_summary_2000_2025.csv"
)

QC_OUTPUT = (
    OUTPUT_DIR
    / "ADM2_flood_spatial_qc.csv"
)


# ================================================================
# 4. HELPER FUNCTIONS
# ================================================================


def find_adm2_boundary():
    """
    Try to locate an ADM2 shapefile automatically.

    Preference is given to filenames containing:
    - adm2
    - admin2
    - level2
    """

    if BOUNDARY_FILE is not None:

        boundary_path = Path(BOUNDARY_FILE)

        if not boundary_path.exists():
            raise FileNotFoundError(
                f"Specified boundary file does not exist:\n"
                f"{boundary_path}"
            )

        return boundary_path


    search_roots = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
    ]

    shapefiles = []

    for root in search_roots:

        if root.exists():
            shapefiles.extend(
                root.rglob("*.shp")
            )


    if not shapefiles:
        raise FileNotFoundError(
            "\nNo shapefile was found under:\n"
            f"{RAW_DATA_DIR}\n"
            f"or\n"
            f"{PROCESSED_DATA_DIR}\n\n"
            "Please set BOUNDARY_FILE manually."
        )


    keywords = [
        "adm2",
        "admin2",
        "level2",
        "level_2",
    ]


    adm2_candidates = []

    for shp in shapefiles:

        name_lower = shp.name.lower()

        if any(
            keyword in name_lower
            for keyword in keywords
        ):
            adm2_candidates.append(shp)


    if len(adm2_candidates) == 1:
        return adm2_candidates[0]


    if len(adm2_candidates) > 1:

        print("\nMultiple possible ADM2 shapefiles found:")

        for shp in adm2_candidates:
            print(f"  - {shp}")

        raise RuntimeError(
            "\nMultiple ADM2 shapefiles were found.\n"
            "Please set BOUNDARY_FILE manually near the top "
            "of this script."
        )


    print("\nShapefiles found:")

    for shp in shapefiles:
        print(f"  - {shp}")

    raise RuntimeError(
        "\nShapefiles exist, but no obvious ADM2 file "
        "could be identified.\n"
        "Please set BOUNDARY_FILE manually."
    )


def find_column(columns, candidates):
    """
    Find the first matching column name.

    Matching is case-insensitive.
    """

    column_lookup = {
        col.lower(): col
        for col in columns
    }

    for candidate in candidates:

        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]

    return None


def prepare_adm2_boundary(boundary_path):
    """
    Read and standardise the ADM2 boundary layer.
    """

    print("\n" + "=" * 80)
    print("READING ADM2 BOUNDARY")
    print("=" * 80)

    print(f"Boundary file:\n{boundary_path}")

    adm2 = gpd.read_file(boundary_path)

    print(f"\nBoundary rows: {len(adm2):,}")
    print(f"Original CRS: {adm2.crs}")

    if adm2.crs is None:
        raise ValueError(
            "ADM2 boundary has no CRS information."
        )


    # ------------------------------------------------
    # Detect ADM2 name field
    # ------------------------------------------------

    name_candidates = [
        "ADM2_EN",
        "ADM2_NAME",
        "ADM2NAME",
        "NAME_2",
        "NAME2",
        "admin2Name",
        "admin2_name",
        "county",
        "County",
    ]

    adm2_name_col = find_column(
        adm2.columns,
        name_candidates
    )


    if adm2_name_col is None:

        raise ValueError(
            "\nCould not automatically identify "
            "the ADM2 name column.\n\n"
            f"Available columns:\n"
            f"{adm2.columns.tolist()}"
        )


    # ------------------------------------------------
    # Detect ADM2 code field
    # ------------------------------------------------

    code_candidates = [
        "ADM2_PCODE",
        "ADM2_CODE",
        "ADM2CODE",
        "GID_2",
        "HASC_2",
        "ID_2",
        "admin2Pcod",
        "admin2_code",
    ]

    adm2_code_col = find_column(
        adm2.columns,
        code_candidates
    )


    print(
        f"\nADM2 name column: {adm2_name_col}"
    )

    print(
        "ADM2 code column: "
        f"{adm2_code_col}"
    )


    # ------------------------------------------------
    # Standardise columns
    # ------------------------------------------------

    keep_columns = [
        adm2_name_col,
        "geometry",
    ]


    if adm2_code_col is not None:
        keep_columns.insert(
            0,
            adm2_code_col
        )


    adm2 = adm2[
        keep_columns
    ].copy()


    rename_dict = {
        adm2_name_col: "adm2_name"
    }


    if adm2_code_col is not None:

        rename_dict[
            adm2_code_col
        ] = "adm2_code"


    adm2 = adm2.rename(
        columns=rename_dict
    )


    # ------------------------------------------------
    # Create fallback code if necessary
    # ------------------------------------------------

    if "adm2_code" not in adm2.columns:

        warnings.warn(
            "No ADM2 code column found. "
            "ADM2 name will be used as the code."
        )

        adm2["adm2_code"] = (
            adm2["adm2_name"]
            .astype(str)
        )


    adm2["adm2_name"] = (
        adm2["adm2_name"]
        .astype(str)
        .str.strip()
    )

    adm2["adm2_code"] = (
        adm2["adm2_code"]
        .astype(str)
        .str.strip()
    )


    # ------------------------------------------------
    # Remove invalid geometries
    # ------------------------------------------------

    adm2 = adm2[
        adm2.geometry.notna()
    ].copy()

    adm2 = adm2[
        ~adm2.geometry.is_empty
    ].copy()


    invalid_geometry_count = int(
        (~adm2.geometry.is_valid).sum()
    )

    if invalid_geometry_count > 0:

        print(
            f"\nFixing {invalid_geometry_count} "
            "invalid ADM2 geometries..."
        )

        adm2["geometry"] = (
            adm2.geometry.buffer(0)
        )


    # ------------------------------------------------
    # Calculate ADM2 area
    # ------------------------------------------------

    adm2_area = (
        adm2
        .to_crs(AREA_CRS)
    )

    adm2["adm2_area_km2"] = (
        adm2_area.geometry.area
        / 1_000_000
    )


    # ------------------------------------------------
    # Convert to flood-point CRS
    # ------------------------------------------------

    adm2 = adm2.to_crs(
        POINT_CRS
    )


    print(
        f"\nADM2 regions: "
        f"{len(adm2):,}"
    )

    print(
        "\nADM2 area summary (km²):"
    )

    print(
        adm2[
            "adm2_area_km2"
        ].describe()
    )


    return adm2


def extract_year_tile(file_path):
    """
    Extract tile and year from a flood parquet filename.

    Example:
    flood_events_h20v08_2025.parquet
    """

    match = re.search(
        r"(h\d+v\d+)_(\d{4})",
        file_path.name
    )

    if match is None:
        return None, None

    tile = match.group(1)
    year = int(match.group(2))

    return year, tile


def build_file_index(folder):
    """
    Build a dictionary:

    (year, tile) -> parquet path

    Duplicate copies such as '(1)' are ignored.
    """

    if not folder.exists():

        raise FileNotFoundError(
            f"Flood folder not found:\n{folder}"
        )


    files = sorted(
        folder.glob("*.parquet")
    )


    index = {}


    for file_path in files:

        year, tile = extract_year_tile(
            file_path
        )

        if year is None:
            continue

        if not (
            START_YEAR
            <= year
            <= END_YEAR
        ):
            continue


        key = (
            year,
            tile
        )


        # Prefer filename without "(1)", "(2)", etc.
        if key in index:

            current = index[key]

            current_has_copy_suffix = bool(
                re.search(
                    r"\(\d+\)",
                    current.stem
                )
            )

            new_has_copy_suffix = bool(
                re.search(
                    r"\(\d+\)",
                    file_path.stem
                )
            )


            if (
                current_has_copy_suffix
                and not new_has_copy_suffix
            ):
                index[key] = file_path

            else:

                print(
                    "Duplicate year-tile file ignored:"
                )

                print(
                    f"  {file_path.name}"
                )

        else:

            index[key] = file_path


    return index


def estimate_coordinate_spacing(values):
    """
    Estimate original grid spacing from coordinate values.

    Uses the smallest meaningful positive difference between
    unique coordinates.

    Returns None if spacing cannot be estimated.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]


    if len(values) < 2:
        return None


    unique_values = np.unique(
        np.round(
            values,
            8
        )
    )


    if len(unique_values) < 2:
        return None


    unique_values.sort()

    diffs = np.diff(
        unique_values
    )

    diffs = diffs[
        diffs > 1e-8
    ]


    if len(diffs) == 0:
        return None


    # Use a low percentile instead of absolute min
    # to reduce the influence of numerical noise.
    spacing = float(
        np.percentile(
            diffs,
            5
        )
    )


    return spacing


def add_estimated_pixel_area(
    df,
    lat_spacing,
    lon_spacing
):
    """
    Estimate the area represented by each flood grid point.

    Uses an approximate geographic conversion:

    1 degree latitude ~= 111.32 km
    1 degree longitude ~= 111.32 * cos(latitude) km

    This is appropriate as an approximate indicator,
    not as an exact hydrological area measurement.
    """

    if (
        lat_spacing is None
        or lon_spacing is None
    ):

        df[
            "pixel_area_km2_est"
        ] = np.nan

        return df


    latitude_height_km = (
        111.32
        * lat_spacing
    )


    longitude_width_km = (
        111.32
        * lon_spacing
        * np.cos(
            np.radians(
                df["lat"].astype(float)
            )
        )
    )


    df[
        "pixel_area_km2_est"
    ] = (
        latitude_height_km
        * longitude_width_km
    )


    return df


def load_flood_file(
    file_path,
    dataset_type
):
    """
    Read one flood parquet file and standardise columns.
    """

    use_columns = [
        "date",
        "lat",
        "lon",
        "tile",
        "cloud_frac",
    ]


    df = pd.read_parquet(
        file_path,
        columns=use_columns
    )


    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce"
    )


    df = df.dropna(
        subset=[
            "date",
            "lat",
            "lon",
        ]
    ).copy()


    df["dataset_type"] = (
        dataset_type
    )


    return df


def spatial_join_flood_to_adm2(
    flood_df,
    adm2
):
    """
    Spatially assign flood observations to ADM2 polygons.
    """

    flood_gdf = gpd.GeoDataFrame(
        flood_df,
        geometry=gpd.points_from_xy(
            flood_df["lon"],
            flood_df["lat"]
        ),
        crs=POINT_CRS
    )


    joined = gpd.sjoin(
        flood_gdf,
        adm2[
            [
                "adm2_code",
                "adm2_name",
                "adm2_area_km2",
                "geometry",
            ]
        ],
        how="left",
        predicate="within"
    )


    return joined


def aggregate_joined(
    joined,
    prefix
):
    """
    Aggregate spatially matched flood observations
    to daily ADM2 level.
    """

    matched = joined[
        joined["adm2_code"].notna()
    ].copy()


    if matched.empty:

        return pd.DataFrame(
            columns=[
                "date",
                "adm2_code",
                "adm2_name",
                f"{prefix}_flood_pixel_count",
                f"{prefix}_flooded_area_km2_est",
                f"{prefix}_cloud_frac_mean",
            ]
        )


    grouped = (
        matched
        .groupby(
            [
                "date",
                "adm2_code",
                "adm2_name",
            ],
            as_index=False
        )
        .agg(
            flood_pixel_count=(
                "lat",
                "size"
            ),
            flooded_area_km2_est=(
                "pixel_area_km2_est",
                "sum"
            ),
            cloud_frac_mean=(
                "cloud_frac",
                "mean"
            ),
        )
    )


    grouped = grouped.rename(
        columns={
            "flood_pixel_count":
                f"{prefix}_flood_pixel_count",

            "flooded_area_km2_est":
                f"{prefix}_flooded_area_km2_est",

            "cloud_frac_mean":
                f"{prefix}_cloud_frac_mean",
        }
    )


    return grouped


# ================================================================
# 5. READ ADMINISTRATIVE BOUNDARY
# ================================================================

boundary_path = find_adm2_boundary()

adm2 = prepare_adm2_boundary(
    boundary_path
)


# ================================================================
# 6. INDEX FLOOD FILES
# ================================================================

print("\n" + "=" * 80)
print("INDEXING FLOOD FILES")
print("=" * 80)


unusual_index = build_file_index(
    UNUSUAL_DIR
)

recurring_index = build_file_index(
    RECURRING_DIR
)


print(
    f"\nUnique unusual year-tile files: "
    f"{len(unusual_index)}"
)

print(
    f"Unique recurring year-tile files: "
    f"{len(recurring_index)}"
)


expected_keys = []

for year in range(
    START_YEAR,
    END_YEAR + 1
):

    for tile in [
        "h20v08",
        "h21v08",
    ]:

        expected_keys.append(
            (year, tile)
        )


missing_unusual = [
    key
    for key in expected_keys
    if key not in unusual_index
]

missing_recurring = [
    key
    for key in expected_keys
    if key not in recurring_index
]


if missing_unusual:

    print(
        "\nWARNING: Missing unusual files:"
    )

    for item in missing_unusual:
        print(f"  {item}")


if missing_recurring:

    print(
        "\nWARNING: Missing recurring files:"
    )

    for item in missing_recurring:
        print(f"  {item}")


# ================================================================
# 7. PROCESS FLOOD DATA
# ================================================================

unusual_results = []
recurring_results = []
combined_results = []

qc_results = []


all_keys = sorted(
    set(unusual_index.keys())
    | set(recurring_index.keys())
)


print("\n" + "=" * 80)
print("BUILDING ADM2 FLOOD INDICATORS")
print("=" * 80)


for key in all_keys:

    year, tile = key

    print(
        f"\nProcessing year={year}, tile={tile}"
    )


    unusual_df = None
    recurring_df = None


    # ------------------------------------------------
    # Read unusual
    # ------------------------------------------------

    if key in unusual_index:

        unusual_file = (
            unusual_index[key]
        )

        print(
            f"  Unusual: "
            f"{unusual_file.name}"
        )

        unusual_df = load_flood_file(
            unusual_file,
            "unusual"
        )


    # ------------------------------------------------
    # Read recurring
    # ------------------------------------------------

    if key in recurring_index:

        recurring_file = (
            recurring_index[key]
        )

        print(
            f"  Recurring: "
            f"{recurring_file.name}"
        )

        recurring_df = load_flood_file(
            recurring_file,
            "recurring"
        )


    # ------------------------------------------------
    # Estimate grid resolution using all points
    # ------------------------------------------------

    spacing_sources = []

    if unusual_df is not None:
        spacing_sources.append(
            unusual_df[
                ["lat", "lon"]
            ]
        )

    if recurring_df is not None:
        spacing_sources.append(
            recurring_df[
                ["lat", "lon"]
            ]
        )


    if spacing_sources:

        coordinates = pd.concat(
            spacing_sources,
            ignore_index=True
        )

        lat_spacing = (
            estimate_coordinate_spacing(
                coordinates["lat"].values
            )
        )

        lon_spacing = (
            estimate_coordinate_spacing(
                coordinates["lon"].values
            )
        )

    else:

        lat_spacing = None
        lon_spacing = None


    print(
        f"  Estimated latitude spacing: "
        f"{lat_spacing}"
    )

    print(
        f"  Estimated longitude spacing: "
        f"{lon_spacing}"
    )


    # ============================================================
    # A. UNUSUAL FLOODS
    # ============================================================

    if unusual_df is not None:

        unusual_df = (
            add_estimated_pixel_area(
                unusual_df,
                lat_spacing,
                lon_spacing
            )
        )


        joined_unusual = (
            spatial_join_flood_to_adm2(
                unusual_df,
                adm2
            )
        )


        unusual_total = len(
            joined_unusual
        )

        unusual_matched = int(
            joined_unusual[
                "adm2_code"
            ].notna().sum()
        )


        unusual_match_rate = (
            unusual_matched
            / unusual_total
            if unusual_total > 0
            else np.nan
        )


        unusual_agg = aggregate_joined(
            joined_unusual,
            "unusual"
        )


        unusual_results.append(
            unusual_agg
        )


        qc_results.append({

            "year":
                year,

            "tile":
                tile,

            "dataset_type":
                "unusual",

            "input_rows":
                unusual_total,

            "matched_rows":
                unusual_matched,

            "unmatched_rows":
                unusual_total
                - unusual_matched,

            "match_rate":
                unusual_match_rate,

            "latitude_spacing":
                lat_spacing,

            "longitude_spacing":
                lon_spacing,
        })


        del joined_unusual


    # ============================================================
    # B. RECURRING FLOODS
    # ============================================================

    if recurring_df is not None:

        recurring_df = (
            add_estimated_pixel_area(
                recurring_df,
                lat_spacing,
                lon_spacing
            )
        )


        joined_recurring = (
            spatial_join_flood_to_adm2(
                recurring_df,
                adm2
            )
        )


        recurring_total = len(
            joined_recurring
        )

        recurring_matched = int(
            joined_recurring[
                "adm2_code"
            ].notna().sum()
        )


        recurring_match_rate = (
            recurring_matched
            / recurring_total
            if recurring_total > 0
            else np.nan
        )


        recurring_agg = (
            aggregate_joined(
                joined_recurring,
                "recurring"
            )
        )


        recurring_results.append(
            recurring_agg
        )


        qc_results.append({

            "year":
                year,

            "tile":
                tile,

            "dataset_type":
                "recurring",

            "input_rows":
                recurring_total,

            "matched_rows":
                recurring_matched,

            "unmatched_rows":
                recurring_total
                - recurring_matched,

            "match_rate":
                recurring_match_rate,

            "latitude_spacing":
                lat_spacing,

            "longitude_spacing":
                lon_spacing,
        })


        del joined_recurring


    # ============================================================
    # C. COMBINED FLOOD LAYER
    # ============================================================

    combined_parts = []

    if unusual_df is not None:

        combined_parts.append(
            unusual_df[
                [
                    "date",
                    "lat",
                    "lon",
                    "tile",
                    "cloud_frac",
                    "pixel_area_km2_est",
                ]
            ]
        )


    if recurring_df is not None:

        combined_parts.append(
            recurring_df[
                [
                    "date",
                    "lat",
                    "lon",
                    "tile",
                    "cloud_frac",
                    "pixel_area_km2_est",
                ]
            ]
        )


    if combined_parts:

        combined_df = pd.concat(
            combined_parts,
            ignore_index=True
        )


        # ------------------------------------------------
        # Critical:
        # remove overlap between unusual and recurring
        # ------------------------------------------------

        rows_before_dedup = len(
            combined_df
        )


        combined_df = (
            combined_df
            .sort_values(
                [
                    "date",
                    "lat",
                    "lon",
                ]
            )
            .drop_duplicates(
                subset=[
                    "date",
                    "lat",
                    "lon",
                ],
                keep="first"
            )
            .copy()
        )


        rows_after_dedup = len(
            combined_df
        )


        overlap_removed = (
            rows_before_dedup
            - rows_after_dedup
        )


        joined_combined = (
            spatial_join_flood_to_adm2(
                combined_df,
                adm2
            )
        )


        combined_total = len(
            joined_combined
        )


        combined_matched = int(
            joined_combined[
                "adm2_code"
            ].notna().sum()
        )


        combined_match_rate = (
            combined_matched
            / combined_total
            if combined_total > 0
            else np.nan
        )


        combined_agg = (
            aggregate_joined(
                joined_combined,
                "combined"
            )
        )


        combined_results.append(
            combined_agg
        )


        qc_results.append({

            "year":
                year,

            "tile":
                tile,

            "dataset_type":
                "combined",

            "input_rows":
                rows_before_dedup,

            "rows_after_overlap_removal":
                rows_after_dedup,

            "overlap_rows_removed":
                overlap_removed,

            "matched_rows":
                combined_matched,

            "unmatched_rows":
                combined_total
                - combined_matched,

            "match_rate":
                combined_match_rate,

            "latitude_spacing":
                lat_spacing,

            "longitude_spacing":
                lon_spacing,
        })


        del combined_df
        del joined_combined


    del unusual_df
    del recurring_df


# ================================================================
# 8. COMBINE ALL YEAR-TILE RESULTS
# ================================================================

print("\n" + "=" * 80)
print("COMBINING DAILY RESULTS")
print("=" * 80)


def combine_result_list(
    result_list,
    prefix
):
    """
    Combine year-tile aggregates.

    A date-ADM2 pair may occur in both spatial tiles,
    so values are summed again after concatenation.
    """

    if not result_list:

        return pd.DataFrame()


    df = pd.concat(
        result_list,
        ignore_index=True
    )


    count_col = (
        f"{prefix}_flood_pixel_count"
    )

    area_col = (
        f"{prefix}_flooded_area_km2_est"
    )

    cloud_col = (
        f"{prefix}_cloud_frac_mean"
    )


    df = (
        df
        .groupby(
            [
                "date",
                "adm2_code",
                "adm2_name",
            ],
            as_index=False
        )
        .agg(
            **{
                count_col:
                    (
                        count_col,
                        "sum"
                    ),

                area_col:
                    (
                        area_col,
                        "sum"
                    ),

                cloud_col:
                    (
                        cloud_col,
                        "mean"
                    ),
            }
        )
    )


    return df


unusual_daily = combine_result_list(
    unusual_results,
    "unusual"
)

recurring_daily = combine_result_list(
    recurring_results,
    "recurring"
)

combined_daily = combine_result_list(
    combined_results,
    "combined"
)


# ================================================================
# 9. BUILD COMPLETE DATE x ADM2 TABLE
# ================================================================

print(
    "\nBuilding complete "
    "date × ADM2 panel..."
)


all_dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)


adm2_lookup = (
    adm2[
        [
            "adm2_code",
            "adm2_name",
            "adm2_area_km2",
        ]
    ]
    .drop_duplicates()
    .copy()
)


date_df = pd.DataFrame({
    "date": all_dates
})


date_df["_key"] = 1
adm2_lookup["_key"] = 1


panel = date_df.merge(
    adm2_lookup,
    on="_key"
).drop(
    columns="_key"
)


print(
    f"Complete panel rows: "
    f"{len(panel):,}"
)


# ================================================================
# 10. MERGE FLOOD INDICATORS
# ================================================================

merge_keys = [
    "date",
    "adm2_code",
    "adm2_name",
]


if not unusual_daily.empty:

    panel = panel.merge(
        unusual_daily,
        on=merge_keys,
        how="left"
    )


if not recurring_daily.empty:

    panel = panel.merge(
        recurring_daily,
        on=merge_keys,
        how="left"
    )


if not combined_daily.empty:

    panel = panel.merge(
        combined_daily,
        on=merge_keys,
        how="left"
    )


# ================================================================
# 11. CREATE FINAL INDICATORS
# ================================================================

count_columns = [
    "unusual_flood_pixel_count",
    "recurring_flood_pixel_count",
    "combined_flood_pixel_count",
]

area_columns = [
    "unusual_flooded_area_km2_est",
    "recurring_flooded_area_km2_est",
    "combined_flooded_area_km2_est",
]


for col in count_columns:

    if col not in panel.columns:
        panel[col] = 0

    panel[col] = (
        panel[col]
        .fillna(0)
        .astype("int64")
    )


for col in area_columns:

    if col not in panel.columns:
        panel[col] = 0.0

    panel[col] = (
        panel[col]
        .fillna(0.0)
    )


# ------------------------------------------------
# Occurrence variables
# ------------------------------------------------

panel[
    "unusual_flood_occurrence"
] = (
    panel[
        "unusual_flood_pixel_count"
    ] > 0
).astype(int)


panel[
    "recurring_flood_occurrence"
] = (
    panel[
        "recurring_flood_pixel_count"
    ] > 0
).astype(int)


panel[
    "flood_occurrence"
] = (
    panel[
        "combined_flood_pixel_count"
    ] > 0
).astype(int)


# ------------------------------------------------
# Rename combined indicators
# ------------------------------------------------

panel = panel.rename(
    columns={
        "combined_flood_pixel_count":
            "flood_pixel_count",

        "combined_flooded_area_km2_est":
            "flooded_area_km2_est",

        "combined_cloud_frac_mean":
            "cloud_frac_mean",
    }
)


# ------------------------------------------------
# Flooded fraction
# ------------------------------------------------

panel[
    "flooded_fraction_est"
] = (
    panel[
        "flooded_area_km2_est"
    ]
    /
    panel[
        "adm2_area_km2"
    ]
)


# Keep fractions in sensible range.
# Values > 1 indicate that point counting / area estimation
# should be inspected.

panel[
    "flooded_fraction_over_1"
] = (
    panel[
        "flooded_fraction_est"
    ] > 1
).astype(int)


# ================================================================
# 12. FINAL COLUMN ORDER
# ================================================================

final_columns = [

    "date",

    "adm2_code",
    "adm2_name",
    "adm2_area_km2",

    "flood_occurrence",
    "flood_pixel_count",
    "flooded_area_km2_est",
    "flooded_fraction_est",

    "unusual_flood_occurrence",
    "unusual_flood_pixel_count",
    "unusual_flooded_area_km2_est",

    "recurring_flood_occurrence",
    "recurring_flood_pixel_count",
    "recurring_flooded_area_km2_est",

    "cloud_frac_mean",

    "flooded_fraction_over_1",
]


existing_final_columns = [
    col
    for col in final_columns
    if col in panel.columns
]


panel = panel[
    existing_final_columns
].copy()


panel = panel.sort_values(
    [
        "date",
        "adm2_code",
    ]
).reset_index(
    drop=True
)


# ================================================================
# 13. ADM2 LONG-TERM SUMMARY
# ================================================================

summary = (
    panel
    .groupby(
        [
            "adm2_code",
            "adm2_name",
        ],
        as_index=False
    )
    .agg(

        adm2_area_km2=(
            "adm2_area_km2",
            "first"
        ),

        flood_days=(
            "flood_occurrence",
            "sum"
        ),

        unusual_flood_days=(
            "unusual_flood_occurrence",
            "sum"
        ),

        recurring_flood_days=(
            "recurring_flood_occurrence",
            "sum"
        ),

        mean_flood_pixel_count=(
            "flood_pixel_count",
            "mean"
        ),

        max_flood_pixel_count=(
            "flood_pixel_count",
            "max"
        ),

        mean_flooded_area_km2_est=(
            "flooded_area_km2_est",
            "mean"
        ),

        max_flooded_area_km2_est=(
            "flooded_area_km2_est",
            "max"
        ),

        mean_flooded_fraction_est=(
            "flooded_fraction_est",
            "mean"
        ),

        max_flooded_fraction_est=(
            "flooded_fraction_est",
            "max"
        ),
    )
)


summary = summary.sort_values(
    "flood_days",
    ascending=False
).reset_index(
    drop=True
)


# ================================================================
# 14. SPATIAL QC SUMMARY
# ================================================================

qc_df = pd.DataFrame(
    qc_results
)


# ================================================================
# 15. SAVE OUTPUTS
# ================================================================

panel.to_csv(
    DAILY_OUTPUT,
    index=False
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

qc_df.to_csv(
    QC_OUTPUT,
    index=False
)


# ================================================================
# 16. PRINT FINAL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("ADM2 FLOOD INDICATORS COMPLETE")
print("=" * 80)


print(
    f"\nStudy period: "
    f"{START_YEAR}-{END_YEAR}"
)

print(
    f"ADM2 regions: "
    f"{panel['adm2_code'].nunique()}"
)

print(
    f"Daily panel rows: "
    f"{len(panel):,}"
)

print(
    f"Total flood-positive ADM2-days: "
    f"{panel['flood_occurrence'].sum():,}"
)


print(
    "\nFlood occurrence distribution:"
)

print(
    panel[
        "flood_occurrence"
    ].value_counts()
)


print(
    "\nTop 10 ADM2 regions "
    "by flood days:"
)

print(
    summary[
        [
            "adm2_name",
            "flood_days",
            "unusual_flood_days",
            "recurring_flood_days",
            "max_flooded_area_km2_est",
            "max_flooded_fraction_est",
        ]
    ].head(10)
)


if not qc_df.empty:

    print(
        "\nSpatial matching rates:"
    )

    print(
        qc_df
        .groupby(
            "dataset_type"
        )[
            "match_rate"
        ]
        .describe()
    )


    low_match = qc_df[
        qc_df[
            "match_rate"
        ] < 0.95
    ]


    if not low_match.empty:

        print(
            "\nWARNING:"
            " Some files have spatial "
            "matching rates below 95%."
        )

        print(
            low_match[
                [
                    "year",
                    "tile",
                    "dataset_type",
                    "input_rows",
                    "matched_rows",
                    "unmatched_rows",
                    "match_rate",
                ]
            ]
        )


fraction_problem_count = int(
    panel[
        "flooded_fraction_over_1"
    ].sum()
)


print(
    "\nADM2-days with estimated "
    "flooded fraction > 100%: "
    f"{fraction_problem_count:,}"
)


if fraction_problem_count > 0:

    print(
        "\nWARNING:"
        " Estimated flooded area exceeded "
        "ADM2 area for some observations."
    )

    print(
        "This means the pixel-area assumption "
        "or grid spacing estimation should be "
        "reviewed before using flooded_area "
        "as a modelling target."
    )


print(
    "\nSaved daily indicators to:"
)

print(
    DAILY_OUTPUT
)


print(
    "\nSaved long-term ADM2 summary to:"
)

print(
    SUMMARY_OUTPUT
)


print(
    "\nSaved spatial QC to:"
)

print(
    QC_OUTPUT
)


print(
    "\nRecommended next step:"
)

print(
    "Merge ADM2_daily_flood_indicators.csv "
    "with ADM2_daily_rainfall_runoff.csv "
    "using date + adm2_code."
)