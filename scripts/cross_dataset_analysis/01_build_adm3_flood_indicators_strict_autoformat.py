"""
01_build_adm3_flood_indicators_strict.py

STRICT ADM3 flood-indicator builder for South Sudan, 2000-2025.

Main methodological rule
------------------------
Annual flood extent is calculated from UNIQUE flood pixels across the whole year:

    ADM3_ID + year + lat + lon

A pixel flooded repeatedly in the same year contributes once to annual spatial
extent, while repeat occurrence is represented separately by flood days,
event count, and cumulative flood-area-days.

The script also unions recurring + unusual pixels before producing combined
daily/annual indicators, so overlapping pixels are not double-counted.

Expected project structure
--------------------------
DBL-Group-22/
├── raw_data/
│   ├── administrative_boundaries/
│   │   └── ssd_admin3.geojson
│   └── ... flood-mask directories somewhere below raw_data ...
├── processed_data/
│   └── flood_analysis/
│       └── adm3/
└── scripts/
    └── cross_dataset_analysis/

The script automatically searches PROJECT_ROOT for directories named:
    recurring
    unusual

and selects a matching pair containing CSV flood-mask files.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(
    "/Users/shibai/Documents/GitHub/DBL-Group-22"
)

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

START_YEAR = 2000
END_YEAR = 2025

FLOOD_ROOT = (
    PROJECT_ROOT
    / "raw_data"
    / "flood_masks"
)

RECURRING_DIR = (
    FLOOD_ROOT
    / "compact_recurring"
)

UNUSUAL_DIR = (
    FLOOD_ROOT
    / "compact_unusual"
)

# Optional manual boundary-field override.
ADM0_FIELD = None
ADM1_FIELD = None
ADM2_FIELD = None
ADM3_FIELD = None


# ============================================================
# 2. PATH HELPERS
# ============================================================

def validate_project_paths() -> None:
    print("=" * 78)
    print("PATH CHECK")
    print("=" * 78)

    print(f"Project root:\n{PROJECT_ROOT}")
    print(f"\nADM3 boundary:\n{ADM3_BOUNDARY_FILE}")
    print(f"Boundary exists: {ADM3_BOUNDARY_FILE.exists()}")

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"\nPROJECT_ROOT does not exist:\n{PROJECT_ROOT}"
        )

    if not ADM3_BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            "\nADM3 boundary file does not exist:\n"
            f"{ADM3_BOUNDARY_FILE}\n\n"
            "Expected file:\n"
            "raw_data/administrative_boundaries/ssd_admin3.geojson"
        )


SUPPORTED_TABLE_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".csv.bz2",
    ".csv.xz",
    ".csv.zip",
    ".txt",
    ".tsv",
    ".dat",
    ".gz",
    ".bz2",
    ".xz",
    ".zip",
    ".parquet",
    ".pq",
    ".feather",
    ".pkl",
    ".pickle",
)


def is_candidate_data_file(path: Path) -> bool:
    """
    Treat every non-hidden regular file as a possible flood-mask table.

    This deliberately does NOT require a specific extension because the
    compact flood-mask collection may use compressed or extensionless files.
    """
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.name not in {".DS_Store"}
    )


def table_file_count(folder: Path) -> int:
    return sum(
        1
        for path in folder.rglob("*")
        if is_candidate_data_file(path)
    )


def read_table(path: Path) -> pd.DataFrame:
    """
    Robust flood-mask reader.

    Supported directly:
        CSV / TXT / TSV / DAT
        gzip / bz2 / xz / zip compressed text
        Parquet
        Feather
        Pickle

    If the extension is unfamiliar, the function falls back to
    pandas.read_csv() and lets pandas infer compression where possible.
    """

    name = path.name.lower()

    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)

    if name.endswith(".feather"):
        return pd.read_feather(path)

    if name.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)

    if name.endswith(".tsv"):
        return pd.read_csv(
            path,
            sep="\t",
            compression="infer",
        )

    if name.endswith(
        (
            ".csv",
            ".csv.gz",
            ".csv.bz2",
            ".csv.xz",
            ".csv.zip",
            ".txt",
            ".dat",
            ".gz",
            ".bz2",
            ".xz",
            ".zip",
        )
    ):
        # First try normal comma-separated parsing.
        try:
            df = pd.read_csv(
                path,
                compression="infer",
            )

            if len(df.columns) > 1:
                return df
        except Exception:
            pass

        # Then try automatic delimiter detection.
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            compression="infer",
        )

    # Extensionless / unknown extension:
    # try common tabular readers in a safe order.
    errors = []

    try:
        df = pd.read_csv(
            path,
            compression="infer",
        )
        if len(df.columns) > 1:
            return df
    except Exception as exc:
        errors.append(f"read_csv: {exc}")

    try:
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            compression="infer",
        )
    except Exception as exc:
        errors.append(f"auto-delimited text: {exc}")

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        errors.append(f"parquet: {exc}")

    try:
        return pd.read_pickle(path)
    except Exception as exc:
        errors.append(f"pickle: {exc}")

    raise ValueError(
        "Could not read flood-mask file:\n"
        f"{path}\n\n"
        + "\n".join(errors)
    )


def find_named_dirs(root: Path, name: str) -> list[Path]:
    matches = []

    for path in root.rglob("*"):
        if (
            path.is_dir()
            and path.name.lower() == name.lower()
        ):
            matches.append(path)

    return sorted(matches)


def score_pair(recurring: Path, unusual: Path) -> tuple[int, int, int]:
    """
    Higher score for:
    - same parent directory
    - many CSVs
    - directory names under raw_data
    """
    same_parent = int(recurring.parent == unusual.parent)
    n_csv = csv_count(recurring) + csv_count(unusual)

    raw_bonus = int(
        "raw_data" in {p.lower() for p in recurring.parts}
        and "raw_data" in {p.lower() for p in unusual.parts}
    )

    return (
        same_parent,
        raw_bonus,
        n_csv,
    )


def discover_flood_dirs() -> tuple[Path, Path]:
    """
    Use the project's confirmed flood-mask directories.
    """

    recurring = Path(RECURRING_DIR)
    unusual = Path(UNUSUAL_DIR)

    print("\nFlood-mask folders:")
    print(f"Recurring: {recurring}")
    print(f"Unusual:   {unusual}")

    if not recurring.exists():
        raise FileNotFoundError(
            "\nRecurring flood-mask directory does not exist:\n"
            f"{recurring}"
        )

    if not unusual.exists():
        raise FileNotFoundError(
            "\nUnusual flood-mask directory does not exist:\n"
            f"{unusual}"
        )

    recurring_count = table_file_count(recurring)
    unusual_count = table_file_count(unusual)

    print(f"Recurring flood-mask files: {recurring_count}")
    print(f"Unusual flood-mask files:   {unusual_count}")

    def show_sample_files(folder: Path, label: str) -> None:
        files = [
            p for p in folder.rglob("*")
            if is_candidate_data_file(p)
        ]

        print(f"\n{label} sample files:")
        for p in files[:10]:
            print(f"  - {p.name}")

    show_sample_files(
        recurring,
        "Recurring",
    )

    show_sample_files(
        unusual,
        "Unusual",
    )

    if recurring_count == 0:
        sample_files = [
            p for p in recurring.rglob("*")
            if p.is_file()
        ][:20]

        raise FileNotFoundError(
            "\nNo supported flood-mask files found in:\n"
            f"{recurring}\n\n"
            "First files found in that directory:\n"
            + "\n".join(str(p) for p in sample_files)
            + "\n\nSupported formats: "
            + ", ".join(SUPPORTED_TABLE_SUFFIXES)
        )

    if unusual_count == 0:
        sample_files = [
            p for p in unusual.rglob("*")
            if p.is_file()
        ][:20]

        raise FileNotFoundError(
            "\nNo supported flood-mask files found in:\n"
            f"{unusual}\n\n"
            "First files found in that directory:\n"
            + "\n".join(str(p) for p in sample_files)
            + "\n\nSupported formats: "
            + ", ".join(SUPPORTED_TABLE_SUFFIXES)
        )

    return recurring, unusual


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def detect_field(
    columns,
    candidates,
):
    lower_map = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def detect_adm_fields(
    gdf: gpd.GeoDataFrame,
):
    columns = list(gdf.columns)

    adm0 = ADM0_FIELD or detect_field(
        columns,
        [
            "ADM0_NAME",
            "ADM0_EN",
            "NAME_0",
            "COUNTRY",
            "ADM0_PCODE",
        ],
    )

    adm1 = ADM1_FIELD or detect_field(
        columns,
        [
            "ADM1_NAME",
            "ADM1_EN",
            "NAME_1",
            "STATE",
            "ADM1_PCODE",
        ],
    )

    adm2 = ADM2_FIELD or detect_field(
        columns,
        [
            "ADM2_NAME",
            "ADM2_EN",
            "NAME_2",
            "COUNTY",
            "ADM2_PCODE",
        ],
    )

    adm3 = ADM3_FIELD or detect_field(
        columns,
        [
            "ADM3_NAME",
            "ADM3_EN",
            "NAME_3",
            "PAYAM",
            "ADM3_PCODE",
        ],
    )

    print("\nDetected administrative fields:")
    print(f"ADM0: {adm0}")
    print(f"ADM1: {adm1}")
    print(f"ADM2: {adm2}")
    print(f"ADM3: {adm3}")

    if adm3 is None:
        raise ValueError(
            "\nCould not detect the ADM3 name field.\n"
            f"Available columns:\n{columns}\n\n"
            "Set ADM3_FIELD manually near the top of the script."
        )

    return adm0, adm1, adm2, adm3


def find_table_files(
    folder: Path,
) -> list[Path]:
    """
    Return every non-hidden file in the flood-mask directory.

    Actual format validation happens in read_table().
    """
    return sorted(
        path
        for path in folder.rglob("*")
        if is_candidate_data_file(path)
    )


def extract_date_from_filename(
    filename: str,
):
    """
    Recognises:
        YYYY-MM-DD
        YYYY_MM_DD
        YYYYMMDD

    If only a year is visible in the filename, returns Jan 1 of that year.
    This is sufficient for annual spatial extent, but event/flood-day metrics
    are only truly temporal if the source files represent dates.
    """

    stem = Path(filename).stem

    full_patterns = [
        r"(?<!\d)(20\d{2})[-_](\d{1,2})[-_](\d{1,2})(?!\d)",
        r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)",
    ]

    for pattern in full_patterns:
        match = re.search(
            pattern,
            stem,
        )

        if match:
            year, month, day = map(
                int,
                match.groups(),
            )

            try:
                return pd.Timestamp(
                    year=year,
                    month=month,
                    day=day,
                )
            except ValueError:
                pass

    year_match = re.search(
        r"(?<!\d)(20\d{2})(?!\d)",
        stem,
    )

    if year_match:
        year = int(year_match.group(1))

        if START_YEAR <= year <= END_YEAR:
            return pd.Timestamp(
                year=year,
                month=1,
                day=1,
            )

    return pd.NaT


def standardize_flood_dataframe(
    df: pd.DataFrame,
):
    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    lat_candidates = [
        "lat",
        "latitude",
        "y",
    ]

    lon_candidates = [
        "lon",
        "longitude",
        "long",
        "lng",
        "x",
    ]

    lat_col = next(
        (
            lower_map[c]
            for c in lat_candidates
            if c in lower_map
        ),
        None,
    )

    lon_col = next(
        (
            lower_map[c]
            for c in lon_candidates
            if c in lower_map
        ),
        None,
    )

    if lat_col is None or lon_col is None:
        raise ValueError(
            "\nCould not identify lat/lon columns.\n"
            f"Columns found:\n{list(df.columns)}"
        )

    df = df.rename(
        columns={
            lat_col: "lat",
            lon_col: "lon",
        }
    ).copy()

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
            "lat",
            "lon",
        ]
    )

    return df


def estimate_grid_resolution(
    df: pd.DataFrame,
):
    unique_lat = np.sort(
        df["lat"]
        .dropna()
        .unique()
    )

    unique_lon = np.sort(
        df["lon"]
        .dropna()
        .unique()
    )

    lat_diff = np.diff(unique_lat)
    lon_diff = np.diff(unique_lon)

    lat_diff = lat_diff[
        lat_diff > 0
    ]

    lon_diff = lon_diff[
        lon_diff > 0
    ]

    lat_res = (
        np.median(lat_diff)
        if len(lat_diff) > 0
        else np.nan
    )

    lon_res = (
        np.median(lon_diff)
        if len(lon_diff) > 0
        else np.nan
    )

    return lat_res, lon_res


def approximate_pixel_area_km2(
    latitude,
    lat_res,
    lon_res,
):
    """
    Approximate lat/lon-cell area in km².
    """

    lat_height_km = (
        abs(lat_res)
        * 111.32
    )

    lon_width_km = (
        abs(lon_res)
        * 111.32
        * np.cos(
            np.radians(latitude)
        )
    )

    return (
        lat_height_km
        * lon_width_km
    )


def calculate_event_count(
    dates,
):
    """
    Consecutive dates belong to one event.
    """

    dates = (
        pd.Series(
            pd.to_datetime(dates)
        )
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(dates) == 0:
        return 0

    if len(dates) == 1:
        return 1

    gaps = (
        dates
        .diff()
        .dt.days
    )

    return int(
        1
        + (
            gaps.iloc[1:] > 1
        ).sum()
    )


# ============================================================
# 4. LOAD ADM3 BOUNDARY
# ============================================================

def load_adm3_boundary():
    print("\n" + "=" * 78)
    print("LOAD ADM3 BOUNDARY")
    print("=" * 78)

    adm3 = gpd.read_file(
        ADM3_BOUNDARY_FILE
    )

    if adm3.empty:
        raise ValueError(
            "ADM3 boundary is empty."
        )

    print(
        f"Boundary rows: {len(adm3):,}"
    )

    print(
        f"Boundary CRS: {adm3.crs}"
    )

    if adm3.crs is None:
        raise ValueError(
            "ADM3 boundary has no CRS."
        )

    adm3 = (
        adm3
        .to_crs("EPSG:4326")
        .reset_index(drop=True)
    )

    (
        adm0_field,
        adm1_field,
        adm2_field,
        adm3_field,
    ) = detect_adm_fields(adm3)

    adm3["ADM3_ID"] = (
        np.arange(len(adm3))
        + 1
    )

    adm3["ADM3_NAME"] = (
        adm3[adm3_field]
        .astype(str)
        .str.strip()
    )

    if adm2_field is not None:
        adm3["ADM2_NAME"] = (
            adm3[adm2_field]
            .astype(str)
            .str.strip()
        )
    else:
        adm3["ADM2_NAME"] = np.nan

    if adm1_field is not None:
        adm3["ADM1_NAME"] = (
            adm3[adm1_field]
            .astype(str)
            .str.strip()
        )
    else:
        adm3["ADM1_NAME"] = np.nan

    if adm0_field is not None:
        adm3["ADM0_NAME"] = (
            adm3[adm0_field]
            .astype(str)
            .str.strip()
        )
    else:
        adm3["ADM0_NAME"] = "South Sudan"

    # Africa Albers Equal Area
    adm3_equal_area = (
        adm3
        .to_crs(
            "ESRI:102022"
        )
    )

    adm3["total_area_km2"] = (
        adm3_equal_area
        .geometry
        .area
        / 1_000_000
    )

    boundary_summary = adm3[
        [
            "ADM3_ID",
            "ADM0_NAME",
            "ADM1_NAME",
            "ADM2_NAME",
            "ADM3_NAME",
            "total_area_km2",
        ]
    ].copy()

    boundary_summary.to_csv(
        OUTPUT_DIR
        / "adm3_boundary_summary.csv",
        index=False,
    )

    print(
        f"\nADM3 units: "
        f"{boundary_summary['ADM3_ID'].nunique():,}"
    )

    print(
        "Median ADM3 area: "
        f"{boundary_summary['total_area_km2'].median():,.2f} km²"
    )

    return adm3, boundary_summary


# ============================================================
# 5. FLOOD FILE INVENTORY
# ============================================================

def build_inventory(
    recurring_dir: Path,
    unusual_dir: Path,
):
    print("\n" + "=" * 78)
    print("BUILD FLOOD FILE INVENTORY")
    print("=" * 78)

    rows = []

    for flood_type, folder in [
        (
            "recurring",
            recurring_dir,
        ),
        (
            "unusual",
            unusual_dir,
        ),
    ]:
        files = find_table_files(
            folder
        )

        print(
            f"{flood_type}: "
            f"{len(files):,} flood-mask files"
        )

        for file in files:
            date = extract_date_from_filename(
                file.name
            )

            rows.append(
                {
                    "flood_type":
                        flood_type,

                    "file_name":
                        file.name,

                    "file_path":
                        str(file),

                    "date":
                        date,

                    "year":
                        (
                            int(date.year)
                            if pd.notna(date)
                            else np.nan
                        ),
                }
            )

    inventory = pd.DataFrame(
        rows
    )

    if inventory.empty:
        raise RuntimeError(
            "No supported flood-mask files were found."
        )

    bad_dates = inventory[
        inventory["date"].isna()
    ]

    if len(bad_dates) > 0:
        print(
            "\nWARNING: Could not extract date/year from "
            f"{len(bad_dates)} file(s)."
        )

        print(
            bad_dates[
                [
                    "flood_type",
                    "file_name",
                ]
            ].head(20).to_string(
                index=False
            )
        )

    inventory = (
        inventory
        .sort_values(
            [
                "flood_type",
                "date",
                "file_name",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    inventory.to_csv(
        OUTPUT_DIR
        / "flood_file_inventory.csv",
        index=False,
    )

    return inventory


# ============================================================
# 6. GRID RESOLUTION
# ============================================================

def estimate_resolution_from_inventory(
    inventory: pd.DataFrame,
):
    print("\n" + "=" * 78)
    print("ESTIMATE FLOOD GRID RESOLUTION")
    print("=" * 78)

    for file_path in inventory[
        "file_path"
    ]:
        try:
            sample = read_table(
                Path(file_path)
            )

            sample = standardize_flood_dataframe(
                sample
            )

            if len(sample) < 2:
                continue

            lat_res, lon_res = (
                estimate_grid_resolution(
                    sample
                )
            )

            if (
                not np.isnan(lat_res)
                and
                not np.isnan(lon_res)
            ):
                median_lat = (
                    sample["lat"].median()
                )

                pixel_area = (
                    approximate_pixel_area_km2(
                        median_lat,
                        lat_res,
                        lon_res,
                    )
                )

                print(
                    "Estimated resolution: "
                    f"{lat_res:.8f}° × "
                    f"{lon_res:.8f}°"
                )

                print(
                    "Approximate median flood-pixel area: "
                    f"{pixel_area:.6f} km²"
                )

                return (
                    lat_res,
                    lon_res,
                )

        except Exception:
            continue

    raise RuntimeError(
        "Could not estimate flood-mask grid resolution."
    )


# ============================================================
# 7. PROCESS FLOOD FILES
# ============================================================

def process_flood_files(
    inventory,
    adm3,
    lat_res,
    lon_res,
):
    print("\n" + "=" * 78)
    print("PROCESS FLOOD MASKS")
    print("=" * 78)

    boundary_cols = [
        "ADM3_ID",
        "ADM0_NAME",
        "ADM1_NAME",
        "ADM2_NAME",
        "ADM3_NAME",
        "total_area_km2",
        "geometry",
    ]

    daily_by_type_list = []
    pixel_records_list = []
    processing_log = []

    total_files = len(
        inventory
    )

    for idx, row in (
        inventory
        .reset_index(drop=True)
        .iterrows()
    ):
        flood_type = row[
            "flood_type"
        ]

        file_path = Path(
            row[
                "file_path"
            ]
        )

        date = row[
            "date"
        ]

        year = row[
            "year"
        ]

        print(
            f"[{idx + 1}/{total_files}] "
            f"{flood_type} | "
            f"{file_path.name}"
        )

        try:
            if pd.isna(date):
                raise ValueError(
                    "Date/year could not be "
                    "extracted from filename."
                )

            if not (
                START_YEAR
                <= int(year)
                <= END_YEAR
            ):
                processing_log.append(
                    {
                        "file_name":
                            file_path.name,

                        "flood_type":
                            flood_type,

                        "date":
                            date,

                        "year":
                            year,

                        "status":
                            "OUTSIDE_STUDY_PERIOD",
                    }
                )

                continue

            df = read_table(
                file_path
            )

            df = standardize_flood_dataframe(
                df
            )

            n_original = len(
                df
            )

            # One coordinate once per source file.
            df = (
                df
                .drop_duplicates(
                    subset=[
                        "lat",
                        "lon",
                    ]
                )
                .copy()
            )

            n_unique = len(
                df
            )

            if df.empty:
                processing_log.append(
                    {
                        "file_name":
                            file_path.name,

                        "flood_type":
                            flood_type,

                        "date":
                            date,

                        "year":
                            int(year),

                        "status":
                            "EMPTY_FILE",

                        "rows_raw":
                            n_original,

                        "unique_pixels":
                            0,

                        "matched_pixels":
                            0,

                        "intersected_adm3":
                            0,
                    }
                )

                continue

            df[
                "pixel_area_km2"
            ] = (
                approximate_pixel_area_km2(
                    df["lat"],
                    lat_res,
                    lon_res,
                )
            )

            flood_gdf = (
                gpd.GeoDataFrame(
                    df,
                    geometry=
                    gpd.points_from_xy(
                        df["lon"],
                        df["lat"],
                    ),
                    crs="EPSG:4326",
                )
            )

            # Use "intersects" rather than "within" so a point exactly
            # on a polygon boundary is not automatically discarded.
            joined = gpd.sjoin(
                flood_gdf,
                adm3[
                    boundary_cols
                ],
                how="inner",
                predicate="intersects",
            )

            if joined.empty:
                processing_log.append(
                    {
                        "file_name":
                            file_path.name,

                        "flood_type":
                            flood_type,

                        "date":
                            date,

                        "year":
                            int(year),

                        "status":
                            "NO_INTERSECTION",

                        "rows_raw":
                            n_original,

                        "unique_pixels":
                            n_unique,

                        "matched_pixels":
                            0,

                        "intersected_adm3":
                            0,
                    }
                )

                continue

            # If a point lies exactly on an administrative boundary,
            # intersects may match >1 polygon. Keep one deterministic
            # assignment per source pixel.
            joined = (
                joined
                .sort_values(
                    [
                        "lat",
                        "lon",
                        "ADM3_ID",
                    ]
                )
                .drop_duplicates(
                    subset=[
                        "lat",
                        "lon",
                    ],
                    keep="first",
                )
                .copy()
            )

            pixel_keep = joined[
                [
                    "ADM3_ID",
                    "ADM0_NAME",
                    "ADM1_NAME",
                    "ADM2_NAME",
                    "ADM3_NAME",
                    "total_area_km2",
                    "lat",
                    "lon",
                    "pixel_area_km2",
                ]
            ].copy()

            pixel_keep[
                "date"
            ] = pd.Timestamp(
                date
            )

            pixel_keep[
                "year"
            ] = int(
                year
            )

            pixel_keep[
                "flood_type"
            ] = flood_type

            pixel_records_list.append(
                pixel_keep
            )

            daily = (
                joined
                .groupby(
                    [
                        "ADM3_ID",
                        "ADM0_NAME",
                        "ADM1_NAME",
                        "ADM2_NAME",
                        "ADM3_NAME",
                        "total_area_km2",
                    ],
                    as_index=False,
                )
                .agg(
                    flooded_area_km2=(
                        "pixel_area_km2",
                        "sum",
                    ),

                    flood_pixel_count=(
                        "pixel_area_km2",
                        "size",
                    ),
                )
            )

            daily[
                "flooded_percent"
            ] = (
                daily[
                    "flooded_area_km2"
                ]
                /
                daily[
                    "total_area_km2"
                ]
                * 100
            )

            daily[
                "date"
            ] = pd.Timestamp(
                date
            )

            daily[
                "year"
            ] = int(
                year
            )

            daily[
                "flood_type"
            ] = flood_type

            daily_by_type_list.append(
                daily
            )

            processing_log.append(
                {
                    "file_name":
                        file_path.name,

                    "flood_type":
                        flood_type,

                    "date":
                        date,

                    "year":
                        int(year),

                    "status":
                        "OK",

                    "rows_raw":
                        n_original,

                    "unique_pixels":
                        n_unique,

                    "matched_pixels":
                        len(joined),

                    "intersected_adm3":
                        joined[
                            "ADM3_ID"
                        ].nunique(),
                }
            )

        except Exception as exc:
            print(
                f"  ERROR: {exc}"
            )

            processing_log.append(
                {
                    "file_name":
                        file_path.name,

                    "flood_type":
                        flood_type,

                    "date":
                        date,

                    "year":
                        year,

                    "status":
                        "ERROR",

                    "error":
                        str(exc),
                }
            )

    processing_summary = pd.DataFrame(
        processing_log
    )

    processing_summary.to_csv(
        OUTPUT_DIR
        / "flood_processing_summary.csv",
        index=False,
    )

    if not pixel_records_list:
        raise RuntimeError(
            "\nNo flood pixels were assigned to ADM3 boundaries.\n"
            "Check flood-mask coordinates, CRS, and South Sudan boundary."
        )

    pixels_by_type = pd.concat(
        pixel_records_list,
        ignore_index=True,
    )

    daily_by_type = pd.concat(
        daily_by_type_list,
        ignore_index=True,
    )

    # Stable keys for exact coordinate de-duplication.
    pixels_by_type[
        "lat_key"
    ] = (
        pixels_by_type[
            "lat"
        ].round(8)
    )

    pixels_by_type[
        "lon_key"
    ] = (
        pixels_by_type[
            "lon"
        ].round(8)
    )

    daily_by_type = (
        daily_by_type
        .sort_values(
            [
                "ADM3_ID",
                "date",
                "flood_type",
            ]
        )
        .reset_index(drop=True)
    )

    daily_by_type.to_csv(
        OUTPUT_DIR
        / "adm3_daily_flood_indicators_by_type.csv",
        index=False,
    )

    return (
        pixels_by_type,
        daily_by_type,
        processing_summary,
    )


# ============================================================
# 8. STRICT DAILY COMBINED
# ============================================================

def build_daily_combined(
    pixels_by_type,
):
    print("\n" + "=" * 78)
    print("BUILD STRICT COMBINED DAILY INDICATORS")
    print("=" * 78)

    # Union recurring + unusual pixels:
    # same ADM3/date/coordinate counts once.
    daily_unique_pixels = (
        pixels_by_type
        .drop_duplicates(
            subset=[
                "ADM3_ID",
                "date",
                "lat_key",
                "lon_key",
            ]
        )
        .copy()
    )

    daily_combined = (
        daily_unique_pixels
        .groupby(
            [
                "ADM3_ID",
                "ADM0_NAME",
                "ADM1_NAME",
                "ADM2_NAME",
                "ADM3_NAME",
                "date",
                "year",
                "total_area_km2",
            ],
            as_index=False,
        )
        .agg(
            flooded_area_km2=(
                "pixel_area_km2",
                "sum",
            ),

            flood_pixel_count=(
                "pixel_area_km2",
                "size",
            ),
        )
    )

    daily_combined[
        "flooded_percent"
    ] = (
        daily_combined[
            "flooded_area_km2"
        ]
        /
        daily_combined[
            "total_area_km2"
        ]
        * 100
    )

    daily_combined[
        "flood_type"
    ] = "combined"

    daily_combined = (
        daily_combined
        .sort_values(
            [
                "ADM3_ID",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    daily_combined.to_csv(
        OUTPUT_DIR
        / "adm3_daily_flood_indicators_combined.csv",
        index=False,
    )

    return daily_combined


# ============================================================
# 9. STRICT ANNUAL BY TYPE
# ============================================================

def build_annual_by_type(
    pixels_by_type,
    daily_by_type,
):
    print("\n" + "=" * 78)
    print("BUILD STRICT ANNUAL INDICATORS BY TYPE")
    print("=" * 78)

    # Annual unique pixel footprint.
    annual_unique_pixels = (
        pixels_by_type
        .drop_duplicates(
            subset=[
                "ADM3_ID",
                "year",
                "flood_type",
                "lat_key",
                "lon_key",
            ]
        )
        .copy()
    )

    annual_extent = (
        annual_unique_pixels
        .groupby(
            [
                "ADM3_ID",
                "ADM0_NAME",
                "ADM1_NAME",
                "ADM2_NAME",
                "ADM3_NAME",
                "year",
                "flood_type",
                "total_area_km2",
            ],
            as_index=False,
        )
        .agg(
            annual_unique_flooded_area_km2=(
                "pixel_area_km2",
                "sum",
            ),

            annual_unique_flood_pixel_count=(
                "pixel_area_km2",
                "size",
            ),
        )
    )

    annual_extent[
        "annual_flooded_percent"
    ] = (
        annual_extent[
            "annual_unique_flooded_area_km2"
        ]
        /
        annual_extent[
            "total_area_km2"
        ]
        * 100
    )

    temporal_rows = []

    for (
        adm3_id,
        year,
        flood_type,
    ), group in daily_by_type.groupby(
        [
            "ADM3_ID",
            "year",
            "flood_type",
        ]
    ):
        dates = (
            group[
                "date"
            ]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )

        temporal_rows.append(
            {
                "ADM3_ID":
                    adm3_id,

                "year":
                    year,

                "flood_type":
                    flood_type,

                "flood_days":
                    len(dates),

                "flood_event_count":
                    calculate_event_count(
                        dates
                    ),

                "cumulative_flood_area_days":
                    group[
                        "flooded_area_km2"
                    ].sum(),

                "mean_daily_flooded_area_km2":
                    group[
                        "flooded_area_km2"
                    ].mean(),

                "max_daily_flooded_area_km2":
                    group[
                        "flooded_area_km2"
                    ].max(),

                "max_daily_flooded_percent":
                    group[
                        "flooded_percent"
                    ].max(),
            }
        )

    temporal = pd.DataFrame(
        temporal_rows
    )

    annual = annual_extent.merge(
        temporal,
        on=[
            "ADM3_ID",
            "year",
            "flood_type",
        ],
        how="left",
    )

    annual = (
        annual
        .sort_values(
            [
                "ADM3_ID",
                "year",
                "flood_type",
            ]
        )
        .reset_index(drop=True)
    )

    annual.to_csv(
        OUTPUT_DIR
        / "adm3_annual_flood_indicators_by_type.csv",
        index=False,
    )

    return annual


# ============================================================
# 10. STRICT ANNUAL COMBINED
# ============================================================

def build_annual_combined(
    pixels_by_type,
    daily_combined,
):
    print("\n" + "=" * 78)
    print("BUILD STRICT COMBINED ANNUAL INDICATORS")
    print("=" * 78)

    # Critical strict definition:
    # union all recurring + unusual coordinates across entire year.
    annual_unique_pixels = (
        pixels_by_type
        .drop_duplicates(
            subset=[
                "ADM3_ID",
                "year",
                "lat_key",
                "lon_key",
            ]
        )
        .copy()
    )

    annual_extent = (
        annual_unique_pixels
        .groupby(
            [
                "ADM3_ID",
                "ADM0_NAME",
                "ADM1_NAME",
                "ADM2_NAME",
                "ADM3_NAME",
                "year",
                "total_area_km2",
            ],
            as_index=False,
        )
        .agg(
            annual_unique_flooded_area_km2=(
                "pixel_area_km2",
                "sum",
            ),

            annual_unique_flood_pixel_count=(
                "pixel_area_km2",
                "size",
            ),
        )
    )

    annual_extent[
        "annual_flooded_percent"
    ] = (
        annual_extent[
            "annual_unique_flooded_area_km2"
        ]
        /
        annual_extent[
            "total_area_km2"
        ]
        * 100
    )

    temporal_rows = []

    for (
        adm3_id,
        year,
    ), group in daily_combined.groupby(
        [
            "ADM3_ID",
            "year",
        ]
    ):
        dates = (
            group[
                "date"
            ]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )

        temporal_rows.append(
            {
                "ADM3_ID":
                    adm3_id,

                "year":
                    year,

                "flood_days":
                    len(dates),

                "flood_event_count":
                    calculate_event_count(
                        dates
                    ),

                "cumulative_flood_area_days":
                    group[
                        "flooded_area_km2"
                    ].sum(),

                "mean_daily_flooded_area_km2":
                    group[
                        "flooded_area_km2"
                    ].mean(),

                "max_daily_flooded_area_km2":
                    group[
                        "flooded_area_km2"
                    ].max(),

                "max_daily_flooded_percent":
                    group[
                        "flooded_percent"
                    ].max(),
            }
        )

    temporal = pd.DataFrame(
        temporal_rows
    )

    annual = annual_extent.merge(
        temporal,
        on=[
            "ADM3_ID",
            "year",
        ],
        how="left",
    )

    annual[
        "flood_type"
    ] = "combined"

    annual = (
        annual
        .sort_values(
            [
                "ADM3_ID",
                "year",
            ]
        )
        .reset_index(drop=True)
    )

    annual.to_csv(
        OUTPUT_DIR
        / "adm3_annual_flood_indicators_combined.csv",
        index=False,
    )

    return annual


# ============================================================
# 11. COMPLETE ADM3 x YEAR PANEL
# ============================================================

def build_complete_panel(
    boundary_summary,
    annual_combined,
):
    print("\n" + "=" * 78)
    print("BUILD COMPLETE ADM3 × YEAR PANEL")
    print("=" * 78)

    years = pd.DataFrame(
        {
            "year":
                range(
                    START_YEAR,
                    END_YEAR + 1,
                )
        }
    )

    adm3_base = boundary_summary[
        [
            "ADM3_ID",
            "ADM0_NAME",
            "ADM1_NAME",
            "ADM2_NAME",
            "ADM3_NAME",
            "total_area_km2",
        ]
    ].copy()

    adm3_base[
        "_join_key"
    ] = 1

    years[
        "_join_key"
    ] = 1

    complete = (
        adm3_base
        .merge(
            years,
            on="_join_key",
            how="inner",
        )
        .drop(
            columns=[
                "_join_key"
            ]
        )
    )

    measure_cols = [
        "ADM3_ID",
        "year",
        "annual_unique_flooded_area_km2",
        "annual_unique_flood_pixel_count",
        "annual_flooded_percent",
        "flood_days",
        "flood_event_count",
        "cumulative_flood_area_days",
        "mean_daily_flooded_area_km2",
        "max_daily_flooded_area_km2",
        "max_daily_flooded_percent",
    ]

    complete = complete.merge(
        annual_combined[
            measure_cols
        ],
        on=[
            "ADM3_ID",
            "year",
        ],
        how="left",
    )

    zero_cols = [
        "annual_unique_flooded_area_km2",
        "annual_unique_flood_pixel_count",
        "annual_flooded_percent",
        "flood_days",
        "flood_event_count",
        "cumulative_flood_area_days",
        "mean_daily_flooded_area_km2",
        "max_daily_flooded_area_km2",
        "max_daily_flooded_percent",
    ]

    complete[
        zero_cols
    ] = (
        complete[
            zero_cols
        ]
        .fillna(0)
    )

    complete[
        "flood_type"
    ] = "combined"

    complete = (
        complete
        .sort_values(
            [
                "ADM3_ID",
                "year",
            ]
        )
        .reset_index(drop=True)
    )

    complete.to_csv(
        OUTPUT_DIR
        / "adm3_annual_complete_panel_combined.csv",
        index=False,
    )

    return complete


# ============================================================
# 12. QUALITY CHECKS
# ============================================================

def run_quality_checks(
    boundary_summary,
    inventory,
    processing_summary,
    daily_combined,
    annual_combined,
    complete_panel,
):
    print("\n" + "=" * 78)
    print("STRICT QUALITY CHECK")
    print("=" * 78)

    n_adm3 = (
        boundary_summary[
            "ADM3_ID"
        ].nunique()
    )

    n_years = (
        END_YEAR
        - START_YEAR
        + 1
    )

    expected_rows = (
        n_adm3
        * n_years
    )

    print(
        f"ADM3 boundary units: "
        f"{n_adm3:,}"
    )

    print(
        f"Study years: "
        f"{n_years}"
    )

    print(
        f"Flood-mask files: "
        f"{len(inventory):,}"
    )

    print(
        f"Expected complete-panel rows: "
        f"{expected_rows:,}"
    )

    print(
        f"Actual complete-panel rows: "
        f"{len(complete_panel):,}"
    )

    print(
        "ADM3 units with >=1 flood observation: "
        f"{annual_combined['ADM3_ID'].nunique():,}"
    )

    print(
        "\nMaximum strict annual flooded percent: "
        f"{complete_panel['annual_flooded_percent'].max():.4f}%"
    )

    print(
        "Maximum daily flooded percent: "
        f"{daily_combined['flooded_percent'].max():.4f}%"
    )

    print(
        "\nProcessing status:"
    )

    print(
        processing_summary[
            "status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    annual_over_100 = complete_panel[
        complete_panel[
            "annual_flooded_percent"
        ] > 100.5
    ].copy()

    daily_over_100 = daily_combined[
        daily_combined[
            "flooded_percent"
        ] > 100.5
    ].copy()

    print(
        "\nAnnual records > 100.5%: "
        f"{len(annual_over_100):,}"
    )

    print(
        "Daily records > 100.5%: "
        f"{len(daily_over_100):,}"
    )

    if len(
        annual_over_100
    ) > 0:
        annual_over_100.to_csv(
            OUTPUT_DIR
            / "WARNING_annual_percent_over_100.csv",
            index=False,
        )

    if len(
        daily_over_100
    ) > 0:
        daily_over_100.to_csv(
            OUTPUT_DIR
            / "WARNING_daily_percent_over_100.csv",
            index=False,
        )

    if len(
        complete_panel
    ) != expected_rows:
        print(
            "\nWARNING: complete panel row count "
            "does not equal ADM3 count × 26 years."
        )
    else:
        print(
            "\nComplete ADM3 × year panel: PASS"
        )


# ============================================================
# 13. MAIN
# ============================================================

def main():
    validate_project_paths()

    (
        recurring_dir,
        unusual_dir,
    ) = discover_flood_dirs()

    (
        adm3,
        boundary_summary,
    ) = load_adm3_boundary()

    inventory = build_inventory(
        recurring_dir,
        unusual_dir,
    )

    (
        lat_res,
        lon_res,
    ) = estimate_resolution_from_inventory(
        inventory
    )

    (
        pixels_by_type,
        daily_by_type,
        processing_summary,
    ) = process_flood_files(
        inventory=inventory,
        adm3=adm3,
        lat_res=lat_res,
        lon_res=lon_res,
    )

    daily_combined = build_daily_combined(
        pixels_by_type
    )

    annual_by_type = build_annual_by_type(
        pixels_by_type=pixels_by_type,
        daily_by_type=daily_by_type,
    )

    annual_combined = build_annual_combined(
        pixels_by_type=pixels_by_type,
        daily_combined=daily_combined,
    )

    annual_all = pd.concat(
        [
            annual_by_type,
            annual_combined,
        ],
        ignore_index=True,
        sort=False,
    )

    annual_all = (
        annual_all
        .sort_values(
            [
                "ADM3_ID",
                "year",
                "flood_type",
            ]
        )
        .reset_index(drop=True)
    )

    annual_all.to_csv(
        OUTPUT_DIR
        / "adm3_annual_flood_indicators_all.csv",
        index=False,
    )

    complete_panel = build_complete_panel(
        boundary_summary=boundary_summary,
        annual_combined=annual_combined,
    )

    run_quality_checks(
        boundary_summary=boundary_summary,
        inventory=inventory,
        processing_summary=processing_summary,
        daily_combined=daily_combined,
        annual_combined=annual_combined,
        complete_panel=complete_panel,
    )

    print("\n" + "=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)

    output_files = [
        "adm3_boundary_summary.csv",
        "flood_file_inventory.csv",
        "flood_processing_summary.csv",
        "adm3_daily_flood_indicators_by_type.csv",
        "adm3_daily_flood_indicators_combined.csv",
        "adm3_annual_flood_indicators_by_type.csv",
        "adm3_annual_flood_indicators_combined.csv",
        "adm3_annual_flood_indicators_all.csv",
        "adm3_annual_complete_panel_combined.csv",
    ]

    for filename in output_files:
        path = (
            OUTPUT_DIR
            / filename
        )

        print(
            f"{filename}: "
            f"{'OK' if path.exists() else 'MISSING'}"
        )

    print(
        "\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nSTRICT ADM3 flood processing completed."
    )


if __name__ == "__main__":
    main()
