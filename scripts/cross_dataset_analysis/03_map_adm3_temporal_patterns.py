"""
03_map_adm3_temporal_patterns.py

South Sudan Flood Analysis
ADM3 Temporal Pattern Maps

Inputs
------
1. raw_data/administrative_boundaries/ssd_admin3.geojson
2. processed_data/flood_analysis/adm3/temporal_patterns/
       adm3_temporal_pattern_summary.csv

Outputs
-------
processed_data/flood_analysis/adm3/temporal_patterns/maps/

    01_adm3_temporal_pattern_classification.png
    02_adm3_trend_slope.png
    03_adm3_recent_minus_early.png
    04_adm3_long_term_mean.png

Purpose
-------
Visualise the ADM3 temporal flood patterns produced by
02_analyze_adm3_temporal_patterns.py.

The script produces one map per figure (no subplots).
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================================
# 1. PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(
    "/Users/shibai/Documents/GitHub/DBL-Group-22"
)

ADM3_BOUNDARY_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "administrative_boundaries"
    / "ssd_admin3.geojson"
)

TEMPORAL_SUMMARY_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm3"
    / "temporal_patterns"
    / "adm3_temporal_pattern_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm3"
    / "temporal_patterns"
    / "maps"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# 2. MAP SETTINGS
# ============================================================================

FIGSIZE = (
    10,
    12,
)

DPI = 300

BOUNDARY_LINEWIDTH = 0.25

TITLE_SIZE = 15

LEGEND_FONT_SIZE = 9


# ============================================================================
# 3. HELPERS
# ============================================================================

def require_file(
    path: Path,
    description: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


def detect_field(
    columns,
    candidates,
):

    lower_map = {
        str(column).lower():
            column
        for column in columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    return None


def load_boundary():
    print("=" * 78)
    print("LOAD ADM3 BOUNDARY")
    print("=" * 78)

    require_file(
        ADM3_BOUNDARY_FILE,
        "ADM3 boundary",
    )

    gdf = gpd.read_file(
        ADM3_BOUNDARY_FILE
    )

    if gdf.empty:

        raise ValueError(
            "ADM3 boundary file is empty."
        )

    if gdf.crs is None:

        raise ValueError(
            "ADM3 boundary has no CRS."
        )

    gdf = (
        gdf
        .to_crs(
            "EPSG:4326"
        )
        .reset_index(drop=True)
    )

    print(
        f"Boundary rows: "
        f"{len(gdf):,}"
    )

    print(
        f"CRS: "
        f"{gdf.crs}"
    )

    return gdf


def load_temporal_summary():
    print("\n" + "=" * 78)
    print("LOAD ADM3 TEMPORAL SUMMARY")
    print("=" * 78)

    require_file(
        TEMPORAL_SUMMARY_FILE,
        "ADM3 temporal pattern summary",
    )

    df = pd.read_csv(
        TEMPORAL_SUMMARY_FILE
    )

    required = [
        "ADM3_ID",
        "ADM3_NAME",
        "temporal_pattern",
        "trend_slope",
        "recent_minus_early",
        "long_term_mean",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nTemporal summary is missing columns:\n"
            + "\n".join(
                missing
            )
        )

    print(
        f"Temporal summary rows: "
        f"{len(df):,}"
    )

    print(
        "Unique ADM3 IDs: "
        f"{df['ADM3_ID'].nunique():,}"
    )

    return df


def prepare_boundary_for_merge(
    boundary: gpd.GeoDataFrame,
):
    """
    Reconstruct ADM3_ID using the same deterministic row-order logic
    used by 01_build_adm3_flood_indicators_strict_autoformat.py.

    The original strict script assigned:
        ADM3_ID = np.arange(len(adm3)) + 1

    after loading/resetting the same GeoJSON boundary.
    """

    boundary = (
        boundary
        .reset_index(drop=True)
        .copy()
    )

    boundary[
        "ADM3_ID"
    ] = (
        np.arange(
            len(boundary)
        )
        + 1
    )

    # Also retain boundary names for validation.
    adm3_name_field = detect_field(
        boundary.columns,
        [
            "adm3_name",
            "ADM3_NAME",
            "name_3",
            "NAME_3",
            "payam",
            "PAYAM",
        ]
    )

    if adm3_name_field is not None:

        boundary[
            "BOUNDARY_ADM3_NAME"
        ] = (
            boundary[
                adm3_name_field
            ]
            .astype(str)
            .str.strip()
        )

    return boundary


def merge_boundary_summary(
    boundary,
    summary,
):
    print("\n" + "=" * 78)
    print("MERGE BOUNDARY + TEMPORAL SUMMARY")
    print("=" * 78)

    if summary[
        "ADM3_ID"
    ].duplicated().any():

        duplicates = (
            summary[
                summary[
                    "ADM3_ID"
                ].duplicated(
                    keep=False
                )
            ]
            .sort_values(
                "ADM3_ID"
            )
        )

        raise ValueError(
            "\nTemporal summary contains duplicate ADM3_ID values.\n"
            f"{duplicates[['ADM3_ID', 'ADM3_NAME']].head(20)}"
        )

    merged = boundary.merge(
        summary,
        on="ADM3_ID",
        how="left",
        validate="one_to_one",
    )

    missing_summary = (
        merged[
            "temporal_pattern"
        ]
        .isna()
        .sum()
    )

    print(
        f"Merged ADM3 polygons: "
        f"{len(merged):,}"
    )

    print(
        f"Polygons without temporal summary: "
        f"{missing_summary:,}"
    )

    if missing_summary > 0:

        print(
            "\nWARNING: Some ADM3 polygons did not match "
            "the temporal summary."
        )

    # Name-level validation where possible.
    if (
        "BOUNDARY_ADM3_NAME"
        in merged.columns
        and
        "ADM3_NAME"
        in merged.columns
    ):

        left = (
            merged[
                "BOUNDARY_ADM3_NAME"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        right = (
            merged[
                "ADM3_NAME"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        comparable = (
            merged[
                "ADM3_NAME"
            ]
            .notna()
        )

        mismatch_count = (
            (
                left != right
            )
            &
            comparable
        ).sum()

        print(
            f"ADM3 name mismatches: "
            f"{int(mismatch_count):,}"
        )

        if mismatch_count > 0:

            print(
                "\nWARNING: ADM3_ID matches exist but some names differ."
            )

            print(
                merged.loc[
                    (
                        left != right
                    )
                    &
                    comparable,
                    [
                        "ADM3_ID",
                        "BOUNDARY_ADM3_NAME",
                        "ADM3_NAME",
                    ]
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )

    return merged


def clean_axis(
    ax,
):
    ax.set_axis_off()

    for spine in ax.spines.values():
        spine.set_visible(
            False
        )


def add_source_note(
    fig,
    text,
):
    fig.text(
        0.5,
        0.025,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
    )


# ============================================================================
# 4. CATEGORICAL MAP
# ============================================================================

def plot_temporal_pattern_map(
    gdf,
):
    print(
        "\nCreating temporal-pattern classification map..."
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    gdf.plot(
        ax=ax,
        column="temporal_pattern",
        categorical=True,
        legend=True,
        linewidth=BOUNDARY_LINEWIDTH,
        edgecolor="white",
        missing_kwds={
            "label":
                "No temporal result",
        },
    )

    ax.set_title(
        "South Sudan ADM3 Flood Temporal Patterns\n2000–2025",
        fontsize=TITLE_SIZE,
        pad=14,
    )

    clean_axis(
        ax
    )

    legend = ax.get_legend()

    if legend is not None:

        legend.set_title(
            "Temporal pattern"
        )

        legend.set_bbox_to_anchor(
            (
                1.02,
                0.5,
            )
        )

        legend.set_loc(
            "center left"
        )

        for text in legend.get_texts():

            text.set_fontsize(
                LEGEND_FONT_SIZE
            )

    add_source_note(
        fig,
        (
            "Classification based on long-term exposure, "
            "early–recent change, trend, persistence and variability."
        ),
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            0.82,
            1,
        ]
    )

    output = (
        OUTPUT_DIR
        / "01_adm3_temporal_pattern_classification.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================================
# 5. TREND-SLOPE MAP
# ============================================================================

def plot_trend_slope_map(
    gdf,
):
    print(
        "Creating trend-slope map..."
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    gdf.plot(
        ax=ax,
        column="trend_slope",
        legend=True,
        linewidth=BOUNDARY_LINEWIDTH,
        edgecolor="white",
        legend_kwds={
            "label":
                (
                    "Annual change in flooded area "
                    "(percentage points/year)"
                ),
            "shrink":
                0.7,
        },
    )

    ax.set_title(
        "ADM3 Flood Trend Slope\n2000–2025",
        fontsize=TITLE_SIZE,
        pad=14,
    )

    clean_axis(
        ax
    )

    add_source_note(
        fig,
        (
            "Positive values indicate increasing annual flooded percentage; "
            "negative values indicate decreasing annual flooded percentage."
        ),
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    output = (
        OUTPUT_DIR
        / "02_adm3_trend_slope.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================================
# 6. RECENT MINUS EARLY MAP
# ============================================================================

def plot_recent_minus_early_map(
    gdf,
):
    print(
        "Creating recent-minus-early map..."
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    gdf.plot(
        ax=ax,
        column="recent_minus_early",
        legend=True,
        linewidth=BOUNDARY_LINEWIDTH,
        edgecolor="white",
        legend_kwds={
            "label":
                (
                    "Recent mean − early mean "
                    "(percentage points)"
                ),
            "shrink":
                0.7,
        },
    )

    ax.set_title(
        "ADM3 Change in Flood Exposure\nRecent Period vs Early Period",
        fontsize=TITLE_SIZE,
        pad=14,
    )

    clean_axis(
        ax
    )

    add_source_note(
        fig,
        (
            "Positive values indicate higher mean flood exposure "
            "in the recent period than in the early period."
        ),
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    output = (
        OUTPUT_DIR
        / "03_adm3_recent_minus_early.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================================
# 7. LONG-TERM MEAN MAP
# ============================================================================

def plot_long_term_mean_map(
    gdf,
):
    print(
        "Creating long-term mean map..."
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    gdf.plot(
        ax=ax,
        column="long_term_mean",
        legend=True,
        linewidth=BOUNDARY_LINEWIDTH,
        edgecolor="white",
        legend_kwds={
            "label":
                (
                    "Mean annual flooded area "
                    "(% of ADM3)"
                ),
            "shrink":
                0.7,
        },
    )

    ax.set_title(
        "ADM3 Long-Term Mean Flood Exposure\n2000–2025",
        fontsize=TITLE_SIZE,
        pad=14,
    )

    clean_axis(
        ax
    )

    add_source_note(
        fig,
        (
            "Long-term mean of annual flooded percentage "
            "for each ADM3."
        ),
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    output = (
        OUTPUT_DIR
        / "04_adm3_long_term_mean.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================================
# 8. SAVE MAP DATA
# ============================================================================

def save_map_data(
    gdf,
):
    columns = [
        column
        for column in [
            "ADM3_ID",
            "ADM0_NAME",
            "ADM1_NAME",
            "ADM2_NAME",
            "ADM3_NAME",
            "temporal_pattern",
            "long_term_mean",
            "early_mean",
            "recent_mean",
            "recent_minus_early",
            "previous_5y_mean",
            "recent_5y_mean",
            "recent_5y_change",
            "trend_slope",
            "trend_pvalue",
            "trend_r2",
            "flood_year_count",
            "flood_year_share",
            "max_flooded_percent",
            "peak_year",
            "peak_value",
        ]
        if column in gdf.columns
    ]

    gdf[
        columns
    ].to_csv(
        OUTPUT_DIR
        / "adm3_temporal_pattern_map_data.csv",
        index=False,
    )


# ============================================================================
# 9. SUMMARY
# ============================================================================

def print_summary(
    gdf,
):
    print("\n" + "=" * 78)
    print("MAP SUMMARY")
    print("=" * 78)

    print(
        f"ADM3 polygons: "
        f"{len(gdf):,}"
    )

    print(
        "\nTemporal pattern counts:"
    )

    print(
        gdf[
            "temporal_pattern"
        ]
        .fillna(
            "No temporal result"
        )
        .value_counts()
        .to_string()
    )

    print(
        "\nTrend slope range:"
    )

    print(
        f"Min: "
        f"{gdf['trend_slope'].min():.4f}"
    )

    print(
        f"Max: "
        f"{gdf['trend_slope'].max():.4f}"
    )

    print(
        "\nRecent-minus-early range:"
    )

    print(
        f"Min: "
        f"{gdf['recent_minus_early'].min():.4f}"
    )

    print(
        f"Max: "
        f"{gdf['recent_minus_early'].max():.4f}"
    )

    print(
        "\nLong-term mean range:"
    )

    print(
        f"Min: "
        f"{gdf['long_term_mean'].min():.4f}"
    )

    print(
        f"Max: "
        f"{gdf['long_term_mean'].max():.4f}"
    )


# ============================================================================
# 10. MAIN
# ============================================================================

def main():
    boundary = load_boundary()

    summary = load_temporal_summary()

    boundary = prepare_boundary_for_merge(
        boundary
    )

    merged = merge_boundary_summary(
        boundary,
        summary,
    )

    save_map_data(
        merged
    )

    plot_temporal_pattern_map(
        merged
    )

    plot_trend_slope_map(
        merged
    )

    plot_recent_minus_early_map(
        merged
    )

    plot_long_term_mean_map(
        merged
    )

    print_summary(
        merged
    )

    print("\n" + "=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)

    output_files = [
        "01_adm3_temporal_pattern_classification.png",
        "02_adm3_trend_slope.png",
        "03_adm3_recent_minus_early.png",
        "04_adm3_long_term_mean.png",
        "adm3_temporal_pattern_map_data.csv",
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
        "\nADM3 temporal-pattern mapping completed."
    )


if __name__ == "__main__":
    main()
