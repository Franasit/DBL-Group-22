"""
plot_adm2_top5_hotspots_large.py

Purpose
-------
Visualize ADM2 hydrological patterns using rainfall and runoff.

All ADM2 areas are colored according to their hydrological values.
Only the top 5 ADM2 areas are labelled and highlighted with black borders.

Outputs
-------
1. ADM2_top5_mean_daily_rainfall.png
2. ADM2_top5_mean_daily_runoff.png
3. ADM2_top5_hydrological_hotspots.csv
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

BOUNDARY_DIR = (
    PROJECT_ROOT
    / "raw_data"
    / "administrative_boundaries"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "cross_dataset_analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "figures"
    / "cross_dataset_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILE
# ============================================================

HYDROLOGY_FILE = (
    PROCESSED_DIR
    / "ADM2_daily_rainfall_runoff.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize_name(series):
    """
    Normalize names to improve ADM2 matching.
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )


def detect_column(
    dataframe,
    candidates,
    label,
    required=True,
):
    """
    Detect a column using common candidate names.
    """

    columns_lower = {
        col.lower(): col
        for col in dataframe.columns
    }

    # Exact match
    for candidate in candidates:
        if candidate.lower() in columns_lower:
            detected = columns_lower[
                candidate.lower()
            ]

            print(
                f"Detected {label} column: "
                f"{detected}"
            )

            return detected

    # Partial match
    for column in dataframe.columns:

        column_lower = column.lower()

        for candidate in candidates:

            candidate_lower = candidate.lower()

            if candidate_lower in column_lower:
                print(
                    f"Detected {label} column: "
                    f"{column}"
                )

                return column

    if required:
        raise KeyError(
            f"\nCould not detect {label} column.\n"
            f"Available columns:\n"
            f"{list(dataframe.columns)}"
        )

    return None


def find_adm2_boundary_file(boundary_dir):
    """
    Recursively search for an ADM2 boundary file.

    Supported formats:
    - .shp
    - .geojson
    - .gpkg
    """

    supported_extensions = {
        ".shp",
        ".geojson",
        ".gpkg",
    }

    if not boundary_dir.exists():
        raise FileNotFoundError(
            "Administrative boundary directory "
            "does not exist:\n"
            f"{boundary_dir}"
        )

    boundary_files = [
        path
        for path in boundary_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in supported_extensions
        )
    ]

    if not boundary_files:
        raise FileNotFoundError(
            "\nNo administrative boundary file "
            "was found.\n\n"
            f"Searched recursively in:\n"
            f"{boundary_dir}\n\n"
            "Supported formats:\n"
            "  - .shp\n"
            "  - .geojson\n"
            "  - .gpkg"
        )

    print(
        "\nAdministrative boundary files detected:"
    )

    for path in boundary_files:
        print(
            " -",
            path.relative_to(
                PROJECT_ROOT
            ),
        )

    # --------------------------------------------------------
    # Prefer filenames containing ADM2
    # --------------------------------------------------------

    adm2_candidates = [
        path
        for path in boundary_files
        if (
            "adm2" in str(path).lower()
            or "admin2" in str(path).lower()
            or "level2" in str(path).lower()
        )
    ]

    if adm2_candidates:

        print(
            "\nADM2 candidate(s):"
        )

        for path in adm2_candidates:
            print(
                " -",
                path.relative_to(
                    PROJECT_ROOT
                ),
            )

        return adm2_candidates[0]

    # --------------------------------------------------------
    # Otherwise inspect file columns
    # --------------------------------------------------------

    print(
        "\nNo filename explicitly contains ADM2."
    )

    print(
        "Inspecting boundary file columns..."
    )

    possible_adm2_columns = [
        "adm2_name",
        "adm2_en",
        "adm2",
        "admin2",
        "admin2_name",
        "name_2",
        "county",
        "county_name",
        "shapename",
    ]

    for path in boundary_files:

        try:

            sample = gpd.read_file(
                path,
                rows=5,
            )

            sample_columns = [
                col.lower()
                for col in sample.columns
            ]

            for candidate in possible_adm2_columns:

                if candidate.lower() in sample_columns:

                    print(
                        "\nAutomatically identified "
                        "ADM2 boundary file:"
                    )

                    print(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    )

                    return path

        except Exception as exc:

            print(
                f"Warning: could not inspect "
                f"{path.name}: {exc}"
            )

    raise FileNotFoundError(
        "\nBoundary files were found, "
        "but no ADM2 boundary file "
        "could be identified automatically."
    )


def create_top5_map(
    boundary_gdf,
    value_column,
    title,
    legend_label,
    output_file,
):
    """
    Plot all ADM2 areas using indicator values.

    All areas are colored.
    Only the top 5 are labelled and highlighted.
    """

    valid_gdf = (
        boundary_gdf
        .dropna(
            subset=[value_column]
        )
        .copy()
    )

    if valid_gdf.empty:

        print(
            f"Warning: no valid values found "
            f"for {value_column}."
        )

        return

    # ========================================================
    # SELECT TOP 5
    # ========================================================

    top5 = (
        valid_gdf
        .sort_values(
            value_column,
            ascending=False,
        )
        .head(5)
        .copy()
    )

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 16)
    )

    # ========================================================
    # PLOT ALL ADM2 AREAS WITH COLOR
    # ========================================================

    valid_gdf.plot(
        ax=ax,
        column=value_column,
        legend=True,
        edgecolor="white",
        linewidth=0.6,
        legend_kwds={
            "label": legend_label,
            "shrink": 0.65,
        },
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "label": "No data",
        },
    )

    # ========================================================
    # HIGHLIGHT TOP 5 WITH BLACK BORDER
    # ========================================================

    top5.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=2.0,
    )

    # ========================================================
    # LABEL ONLY TOP 5
    # ========================================================

    for _, row in top5.iterrows():

        point = (
            row.geometry
            .representative_point()
        )

        adm2_label = str(
            row.get(
                "_adm2_display",
                "ADM2",
            )
        )

        value = row[
            value_column
        ]

        label = (
            f"{adm2_label}\n"
            f"{value:,.2f}"
        )

        ax.annotate(
            label,
            xy=(
                point.x,
                point.y,
            ),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle":
                    "round,pad=0.25",
                "facecolor":
                    "white",
                "alpha":
                    0.85,
                "edgecolor":
                    "black",
                "linewidth":
                    0.5,
            },
        )

    # ========================================================
    # TITLE AND STYLE
    # ========================================================

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=18,
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nSaved figure:"
    )

    print(
        output_file
    )


# ============================================================
# START
# ============================================================

print_section(
    "ADM2 HYDROLOGICAL HOTSPOT ANALYSIS"
)

print(
    "Project root:"
)

print(
    PROJECT_ROOT
)


# ============================================================
# 1. LOAD HYDROLOGY DATA
# ============================================================

print_section(
    "1. LOADING ADM2 HYDROLOGY DATA"
)

if not HYDROLOGY_FILE.exists():

    print(
        "\nExpected file not found:"
    )

    print(
        HYDROLOGY_FILE
    )

    print(
        "\nSearching processed_data "
        "for ADM2_daily_rainfall_runoff.csv..."
    )

    matches = list(
        (
            PROJECT_ROOT
            / "processed_data"
        ).rglob(
            "ADM2_daily_rainfall_runoff.csv"
        )
    )

    if not matches:

        raise FileNotFoundError(
            "\nCould not find "
            "ADM2_daily_rainfall_runoff.csv "
            "inside processed_data."
        )

    HYDROLOGY_FILE = matches[0]

    print(
        "\nAutomatically found:"
    )

    print(
        HYDROLOGY_FILE
    )


hydrology = pd.read_csv(
    HYDROLOGY_FILE
)

print(
    "\nHydrology file:"
)

print(
    HYDROLOGY_FILE
)

print(
    f"\nRows: "
    f"{len(hydrology):,}"
)

print(
    f"Columns: "
    f"{len(hydrology.columns)}"
)

print(
    "\nHydrology columns:"
)

for column in hydrology.columns:
    print(
        " -",
        column,
    )


# ============================================================
# 2. DETECT HYDROLOGY COLUMNS
# ============================================================

print_section(
    "2. DETECTING HYDROLOGY COLUMNS"
)

hydrology_adm2_col = detect_column(
    hydrology,
    [
        "ADM2_NAME",
        "ADM2_EN",
        "ADM2",
        "adm2_name",
        "admin2",
        "admin2_name",
        "county",
        "county_name",
        "NAME_2",
        "shapeName",
    ],
    "hydrology ADM2",
)

rainfall_col = detect_column(
    hydrology,
    [
        "rainfall_mm",
        "daily_rainfall_mm",
        "mean_rainfall_mm",
        "total_precipitation_mm",
        "precipitation_mm",
        "rainfall",
        "precipitation",
    ],
    "rainfall",
)

runoff_col = detect_column(
    hydrology,
    [
        "runoff_mm",
        "daily_runoff_mm",
        "mean_runoff_mm",
        "total_runoff_mm",
        "runoff",
    ],
    "runoff",
)


# ============================================================
# 3. CLEAN DATA
# ============================================================

print_section(
    "3. CLEANING HYDROLOGY DATA"
)

hydrology[
    rainfall_col
] = pd.to_numeric(
    hydrology[
        rainfall_col
    ],
    errors="coerce",
)

hydrology[
    runoff_col
] = pd.to_numeric(
    hydrology[
        runoff_col
    ],
    errors="coerce",
)

hydrology[
    "_adm2_key"
] = normalize_name(
    hydrology[
        hydrology_adm2_col
    ]
)


# ============================================================
# 4. AGGREGATE HYDROLOGY BY ADM2
# ============================================================

print_section(
    "4. AGGREGATING HYDROLOGY BY ADM2"
)

adm2_summary = (
    hydrology
    .groupby(
        "_adm2_key",
        as_index=False,
    )
    .agg(
        mean_daily_rainfall_mm=(
            rainfall_col,
            "mean",
        ),
        max_daily_rainfall_mm=(
            rainfall_col,
            "max",
        ),
        mean_daily_runoff_mm=(
            runoff_col,
            "mean",
        ),
        max_daily_runoff_mm=(
            runoff_col,
            "max",
        ),
        rainfall_observation_count=(
            rainfall_col,
            "count",
        ),
        runoff_observation_count=(
            runoff_col,
            "count",
        ),
    )
)


# ============================================================
# ADD DISPLAY NAME
# ============================================================

display_names = (
    hydrology[
        [
            "_adm2_key",
            hydrology_adm2_col,
        ]
    ]
    .drop_duplicates(
        subset=[
            "_adm2_key"
        ]
    )
    .rename(
        columns={
            hydrology_adm2_col:
                "_adm2_display"
        }
    )
)

adm2_summary = (
    adm2_summary
    .merge(
        display_names,
        on="_adm2_key",
        how="left",
    )
)


print(
    f"\nNumber of ADM2 units: "
    f"{len(adm2_summary):,}"
)


# ============================================================
# DISPLAY TOP 5 TABLES
# ============================================================

print(
    "\nTop 5 ADM2 by "
    "mean daily rainfall:"
)

print(
    adm2_summary
    .nlargest(
        5,
        "mean_daily_rainfall_mm",
    )
    [
        [
            "_adm2_display",
            "mean_daily_rainfall_mm",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nTop 5 ADM2 by "
    "mean daily runoff:"
)

print(
    adm2_summary
    .nlargest(
        5,
        "mean_daily_runoff_mm",
    )
    [
        [
            "_adm2_display",
            "mean_daily_runoff_mm",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 5. FIND ADM2 BOUNDARY
# ============================================================

print_section(
    "5. FINDING ADM2 BOUNDARY"
)

boundary_file = (
    find_adm2_boundary_file(
        BOUNDARY_DIR
    )
)

print(
    "\nUsing ADM2 boundary file:"
)

print(
    boundary_file
)


# ============================================================
# 6. LOAD ADM2 BOUNDARY
# ============================================================

print_section(
    "6. LOADING ADM2 BOUNDARY"
)

adm2_boundaries = gpd.read_file(
    boundary_file
)

print(
    f"\nBoundary polygons: "
    f"{len(adm2_boundaries):,}"
)

print(
    "\nBoundary columns:"
)

for column in adm2_boundaries.columns:

    print(
        " -",
        column,
    )


# ============================================================
# 7. DETECT BOUNDARY ADM2 NAME
# ============================================================

print_section(
    "7. DETECTING ADM2 BOUNDARY NAME"
)

boundary_adm2_col = detect_column(
    adm2_boundaries,
    [
        "ADM2_NAME",
        "ADM2_EN",
        "ADM2",
        "adm2_name",
        "admin2",
        "admin2_name",
        "county",
        "county_name",
        "NAME_2",
        "shapeName",
    ],
    "boundary ADM2",
)


adm2_boundaries[
    "_adm2_key"
] = normalize_name(
    adm2_boundaries[
        boundary_adm2_col
    ]
)


# ============================================================
# 8. MERGE HYDROLOGY WITH BOUNDARIES
# ============================================================

print_section(
    "8. MERGING HYDROLOGY WITH ADM2 BOUNDARIES"
)

merged = (
    adm2_boundaries
    .merge(
        adm2_summary,
        on="_adm2_key",
        how="left",
    )
)


matched_count = (
    merged[
        "mean_daily_rainfall_mm"
    ]
    .notna()
    .sum()
)

print(
    f"\nMatched ADM2 polygons: "
    f"{matched_count:,} / "
    f"{len(merged):,}"
)


# ============================================================
# SHOW UNMATCHED AREAS
# ============================================================

unmatched = merged[
    merged[
        "mean_daily_rainfall_mm"
    ].isna()
]

if not unmatched.empty:

    print(
        "\nBoundary ADM2 units without "
        "hydrology data:"
    )

    unmatched_names = (
        unmatched[
            boundary_adm2_col
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    for name in unmatched_names[:30]:

        print(
            " -",
            name,
        )

    if len(
        unmatched_names
    ) > 30:

        print(
            f"... plus "
            f"{len(unmatched_names) - 30} "
            f"more."
        )


# ============================================================
# CRS INFORMATION
# ============================================================

print(
    "\nBoundary CRS:"
)

print(
    merged.crs
)


# ============================================================
# 9. CREATE RAINFALL MAP
# ============================================================

print_section(
    "9. CREATING ADM2 RAINFALL MAP"
)

rainfall_output = (
    OUTPUT_DIR
    / "ADM2_top5_mean_daily_rainfall.png"
)

create_top5_map(
    boundary_gdf=merged,
    value_column=(
        "mean_daily_rainfall_mm"
    ),
    title=(
        "ADM2 Mean Daily Rainfall "
        "with Top 5 Hotspots"
    ),
    legend_label=(
        "Mean daily rainfall (mm)"
    ),
    output_file=(
        rainfall_output
    ),
)


# ============================================================
# 10. CREATE RUNOFF MAP
# ============================================================

print_section(
    "10. CREATING ADM2 RUNOFF MAP"
)

runoff_output = (
    OUTPUT_DIR
    / "ADM2_top5_mean_daily_runoff.png"
)

create_top5_map(
    boundary_gdf=merged,
    value_column=(
        "mean_daily_runoff_mm"
    ),
    title=(
        "ADM2 Mean Daily Runoff "
        "with Top 5 Hotspots"
    ),
    legend_label=(
        "Mean daily runoff (mm)"
    ),
    output_file=(
        runoff_output
    ),
)


# ============================================================
# 11. SAVE TOP 5 TABLE
# ============================================================

print_section(
    "11. SAVING TOP 5 HOTSPOT TABLE"
)

rainfall_top5 = (
    adm2_summary
    .nlargest(
        5,
        "mean_daily_rainfall_mm",
    )
    .copy()
)

rainfall_top5[
    "hotspot_type"
] = "rainfall"


runoff_top5 = (
    adm2_summary
    .nlargest(
        5,
        "mean_daily_runoff_mm",
    )
    .copy()
)

runoff_top5[
    "hotspot_type"
] = "runoff"


top5_table = pd.concat(
    [
        rainfall_top5,
        runoff_top5,
    ],
    ignore_index=True,
)


top5_output = (
    PROCESSED_DIR
    / "ADM2_top5_hydrological_hotspots.csv"
)


top5_table.to_csv(
    top5_output,
    index=False,
)


print(
    "\nSaved:"
)

print(
    top5_output
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_section(
    "COMPLETED"
)

print(
    "ADM2 hydrological hotspot analysis "
    "completed successfully."
)

print(
    "\nGenerated figures:"
)

print(
    f"1. {rainfall_output}"
)

print(
    f"2. {runoff_output}"
)

print(
    "\nGenerated table:"
)

print(
    f"3. {top5_output}"
)

print(
    "\nMap behavior:"
)

print(
    "- All ADM2 areas are colored "
    "according to rainfall/runoff values."
)

print(
    "- Only the top 5 areas are labelled."
)

print(
    "- Top 5 areas are highlighted "
    "with black borders."
)

print(
    "\nRecommended next step:"
)

print(
    "Build ADM2 flood occurrence and "
    "flooded-area indicators, then merge "
    "them with ADM2_daily_rainfall_runoff.csv."
)