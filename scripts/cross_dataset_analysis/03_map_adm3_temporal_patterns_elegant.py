"""
03_map_adm3_temporal_patterns_clean.py

South Sudan Flood Analysis
Clean / presentation-quality ADM3 temporal pattern maps

Inputs
------
1. raw_data/administrative_boundaries/ssd_admin3.geojson
2. processed_data/flood_analysis/adm3/temporal_patterns/
       adm3_temporal_pattern_summary.csv

Outputs
-------
processed_data/flood_analysis/adm3/temporal_patterns/maps_clean/

    01_adm3_temporal_pattern_classification_clean.png
    02_adm3_trend_slope_clean.png
    03_adm3_recent_minus_early_clean.png
    04_adm3_long_term_mean_clean.png
    adm3_temporal_pattern_map_data.csv

Design principles
-----------------
- Warm colours = worsening / hotspot conditions
- Blue = declining flood exposure
- Neutral cool/grey = stable conditions
- Diverging scales for signed change metrics
- Larger map, less empty space
- Cleaner legends and titles
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

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
    / "maps_clean"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# 2. VISUAL SETTINGS
# ============================================================================

FIGSIZE = (11, 9)
DPI = 320

BOUNDARY_COLOR = "#F7F4EF"
BOUNDARY_LINEWIDTH = 0.28

OUTLINE_COLOR = "#5A5A5A"
OUTLINE_LINEWIDTH = 0.45

TITLE_SIZE = 17
SUBTITLE_SIZE = 10
NOTE_SIZE = 8.5
LEGEND_FONT_SIZE = 9.5


# Semantic categorical colours
PATTERN_COLORS = {
    "Persistent hotspot": "#8E1B1B",
    "Emerging hotspot": "#E67E22",
    "Increasing": "#B04A3A",
    "Episodic / volatile": "#D9A441",
    "Declining": "#2F6B9A",
    "Stable moderate": "#72A7B9",
    "Stable low": "#D8D6D0",
}

PATTERN_ORDER = [
    "Persistent hotspot",
    "Emerging hotspot",
    "Increasing",
    "Episodic / volatile",
    "Declining",
    "Stable moderate",
    "Stable low",
]


# ============================================================================
# 3. HELPERS
# ============================================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


def detect_field(columns, candidates):
    lower_map = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def load_boundary() -> gpd.GeoDataFrame:
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
        .to_crs("EPSG:4326")
        .reset_index(drop=True)
    )

    print(f"Boundary rows: {len(gdf):,}")
    print(f"CRS: {gdf.crs}")

    return gdf


def load_temporal_summary() -> pd.DataFrame:
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
            + "\n".join(missing)
        )

    print(
        f"Temporal summary rows: {len(df):,}"
    )

    print(
        f"Unique ADM3 IDs: {df['ADM3_ID'].nunique():,}"
    )

    return df


def prepare_boundary_for_merge(
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    boundary = (
        boundary
        .reset_index(drop=True)
        .copy()
    )

    # Must match the ID creation logic in the strict builder.
    boundary["ADM3_ID"] = (
        np.arange(len(boundary))
        + 1
    )

    adm3_name_field = detect_field(
        boundary.columns,
        [
            "adm3_name",
            "ADM3_NAME",
            "name_3",
            "NAME_3",
            "payam",
            "PAYAM",
        ],
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
    boundary: gpd.GeoDataFrame,
    summary: pd.DataFrame,
) -> gpd.GeoDataFrame:

    print("\n" + "=" * 78)
    print("MERGE BOUNDARY + TEMPORAL SUMMARY")
    print("=" * 78)

    if summary["ADM3_ID"].duplicated().any():
        raise ValueError(
            "Temporal summary contains duplicate ADM3_ID values."
        )

    merged = boundary.merge(
        summary,
        on="ADM3_ID",
        how="left",
        validate="one_to_one",
    )

    missing_summary = (
        merged["temporal_pattern"]
        .isna()
        .sum()
    )

    print(
        f"Merged ADM3 polygons: {len(merged):,}"
    )

    print(
        f"Polygons without temporal summary: {missing_summary:,}"
    )

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
            merged["ADM3_NAME"].notna()
        )

        mismatch_count = (
            (
                left != right
            )
            &
            comparable
        ).sum()

        print(
            f"ADM3 name mismatches: {int(mismatch_count):,}"
        )

    return merged


def clean_axis(ax) -> None:
    ax.set_axis_off()

    ax.set_aspect(
        "equal"
    )


def add_country_outline(
    gdf: gpd.GeoDataFrame,
    ax,
) -> None:

    country = gdf.dissolve()

    country.boundary.plot(
        ax=ax,
        color=OUTLINE_COLOR,
        linewidth=OUTLINE_LINEWIDTH,
        zorder=5,
    )


def add_note(
    fig,
    text: str,
) -> None:

    fig.text(
        0.5,
        0.025,
        text,
        ha="center",
        va="bottom",
        fontsize=NOTE_SIZE,
    )


def save_figure(
    fig,
    filename: str,
) -> None:

    output = (
        OUTPUT_DIR
        / filename
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


# ============================================================================
# 4. CATEGORICAL TEMPORAL PATTERN MAP
# ============================================================================

def plot_temporal_pattern_map(
    gdf: gpd.GeoDataFrame,
) -> None:

    print(
        "\nCreating clean temporal-pattern classification map..."
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # Draw stable low background first
    gdf.plot(
        ax=ax,
        color="#F7F7F7",
        edgecolor=BOUNDARY_COLOR,
        linewidth=BOUNDARY_LINEWIDTH,
    )

    for pattern in PATTERN_ORDER:

        subset = gdf[
            gdf[
                "temporal_pattern"
            ]
            ==
            pattern
        ]

        if subset.empty:
            continue

        subset.plot(
            ax=ax,
            color=PATTERN_COLORS[
                pattern
            ],
            edgecolor=BOUNDARY_COLOR,
            linewidth=BOUNDARY_LINEWIDTH,
        )

    add_country_outline(
        gdf,
        ax,
    )

    ax.set_title(
        "ADM3 Flood Temporal Patterns in South Sudan (2000–2025)",
        fontsize=TITLE_SIZE,
        pad=14,
        weight="semibold",
    )

    clean_axis(
        ax
    )

    present_patterns = [
        pattern
        for pattern in PATTERN_ORDER
        if (
            gdf[
                "temporal_pattern"
            ]
            ==
            pattern
        ).any()
    ]

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=10,
            markerfacecolor=
                PATTERN_COLORS[
                    pattern
                ],
            markeredgecolor="none",
            label=pattern,
        )
        for pattern
        in present_patterns
    ]

    ax.legend(
        handles=legend_handles,
        title="Temporal pattern",
        loc="center left",
        bbox_to_anchor=(
            1.01,
            0.5,
        ),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=10.5,
    )

    add_note(
        fig,
        (
            "Warm tones highlight persistent or worsening flood patterns; "
            "cool tones indicate declining or stable conditions."
        ),
    )

    plt.tight_layout(
        rect=[
            0.01,
            0.05,
            0.83,
            0.97,
        ]
    )

    save_figure(
        fig,
        "01_adm3_temporal_pattern_classification_clean.png",
    )


# ============================================================================
# 5. TREND SLOPE MAP
# ============================================================================

def plot_trend_slope_map(
    gdf: gpd.GeoDataFrame,
) -> None:

    print(
        "Creating clean trend-slope map..."
    )

    values = (
        pd.to_numeric(
            gdf[
                "trend_slope"
            ],
            errors="coerce",
        )
        .dropna()
    )

    vmax = float(
        np.nanpercentile(
            np.abs(values),
            98,
        )
    )

    if vmax == 0:
        vmax = 1.0

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    gdf.plot(
        ax=ax,
        column="trend_slope",
        cmap="RdBu_r",
        norm=norm,
        legend=True,
        edgecolor=BOUNDARY_COLOR,
        linewidth=BOUNDARY_LINEWIDTH,
        legend_kwds={
            "label":
                "Annual change in flooded percentage points",
            "shrink":
                0.72,
        },
    )

    add_country_outline(
        gdf,
        ax,
    )

    ax.set_title(
        "ADM3 Flood Trend Slope (2000–2025)",
        fontsize=TITLE_SIZE,
        pad=14,
        weight="semibold",
    )

    clean_axis(
        ax
    )

    add_note(
        fig,
        (
            "Red = increasing flood exposure; "
            "blue = decreasing flood exposure; "
            "white = little long-term change."
        ),
    )

    plt.tight_layout(
        rect=[
            0.01,
            0.05,
            0.97,
            0.97,
        ]
    )

    save_figure(
        fig,
        "02_adm3_trend_slope_clean.png",
    )


# ============================================================================
# 6. RECENT MINUS EARLY MAP
# ============================================================================

def plot_recent_minus_early_map(
    gdf: gpd.GeoDataFrame,
) -> None:

    print(
        "Creating clean recent-minus-early map..."
    )

    values = (
        pd.to_numeric(
            gdf[
                "recent_minus_early"
            ],
            errors="coerce",
        )
        .dropna()
    )

    vmax = float(
        np.nanpercentile(
            np.abs(values),
            98,
        )
    )

    if vmax == 0:
        vmax = 1.0

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    gdf.plot(
        ax=ax,
        column="recent_minus_early",
        cmap="RdBu_r",
        norm=norm,
        legend=True,
        edgecolor=BOUNDARY_COLOR,
        linewidth=BOUNDARY_LINEWIDTH,
        legend_kwds={
            "label":
                "Recent mean − early mean (percentage points)",
            "shrink":
                0.72,
        },
    )

    add_country_outline(
        gdf,
        ax,
    )

    ax.set_title(
        "Change in ADM3 Flood Exposure: Recent vs Early Period",
        fontsize=TITLE_SIZE,
        pad=14,
        weight="semibold",
    )

    clean_axis(
        ax
    )

    add_note(
        fig,
        (
            "Red = higher flood exposure in the recent period; "
            "blue = lower recent exposure."
        ),
    )

    plt.tight_layout(
        rect=[
            0.01,
            0.05,
            0.97,
            0.97,
        ]
    )

    save_figure(
        fig,
        "03_adm3_recent_minus_early_clean.png",
    )


# ============================================================================
# 7. LONG-TERM MEAN MAP
# ============================================================================

def plot_long_term_mean_map(
    gdf: gpd.GeoDataFrame,
) -> None:

    print(
        "Creating clean long-term mean map..."
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    gdf.plot(
        ax=ax,
        column="long_term_mean",
        cmap="OrRd",
        legend=True,
        edgecolor=BOUNDARY_COLOR,
        linewidth=BOUNDARY_LINEWIDTH,
        legend_kwds={
            "label":
                "Mean annual flooded area (% of ADM3)",
            "shrink":
                0.72,
        },
    )

    add_country_outline(
        gdf,
        ax,
    )

    ax.set_title(
        "ADM3 Long-Term Mean Flood Exposure (2000–2025)",
        fontsize=TITLE_SIZE,
        pad=14,
        weight="semibold",
    )

    clean_axis(
        ax
    )

    add_note(
        fig,
        (
            "Higher values indicate a larger mean share of the ADM3 "
            "affected by flooding across 2000–2025."
        ),
    )

    plt.tight_layout(
        rect=[
            0.01,
            0.05,
            0.97,
            0.97,
        ]
    )

    save_figure(
        fig,
        "04_adm3_long_term_mean_clean.png",
    )


# ============================================================================
# 8. SAVE MAP DATA
# ============================================================================

def save_map_data(
    gdf: gpd.GeoDataFrame,
) -> None:

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
    gdf: gpd.GeoDataFrame,
) -> None:

    print("\n" + "=" * 78)
    print("MAP SUMMARY")
    print("=" * 78)

    print(
        f"ADM3 polygons: {len(gdf):,}"
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
        f"Min: {gdf['trend_slope'].min():.4f}"
    )

    print(
        f"Max: {gdf['trend_slope'].max():.4f}"
    )

    print(
        "\nRecent-minus-early range:"
    )

    print(
        f"Min: {gdf['recent_minus_early'].min():.4f}"
    )

    print(
        f"Max: {gdf['recent_minus_early'].max():.4f}"
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
        "01_adm3_temporal_pattern_classification_clean.png",
        "02_adm3_trend_slope_clean.png",
        "03_adm3_recent_minus_early_clean.png",
        "04_adm3_long_term_mean_clean.png",
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
        "\nClean ADM3 temporal-pattern mapping completed."
    )


if __name__ == "__main__":
    main()
