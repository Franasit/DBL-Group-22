"""
05_map_adm2_adm3_cross_scale_patterns.py

South Sudan Flood Analysis
Cross-scale ADM2–ADM3 mapping based on V5 multi-label analysis.

========================================================================
INPUTS
========================================================================

V5 cross-scale results:
processed_data/flood_analysis/cross_scale_adm2_adm3_v5/

    02_adm3_cross_scale_comparison_v5.csv
    04_adm2_cross_scale_summary_v5.csv

Administrative boundaries:
raw_data/administrative_boundaries/

    ssd_admin3.geojson
    ssd_admin2.geojson

========================================================================
OUTPUTS
========================================================================

processed_data/flood_analysis/cross_scale_adm2_adm3_v5/maps/

    01_primary_cross_scale_pattern_map.png
    02_regional_chronic_local_hotspot_map.png
    03_regional_decline_local_worsening_map.png
    04_adm2_internal_heterogeneity_entropy_map.png
    05_regional_neutral_local_hotspot_map.png

    adm3_cross_scale_map_data.csv
    adm2_heterogeneity_map_data.csv

========================================================================
DESIGN
========================================================================

- ADM3 polygons display local cross-scale relationships.
- ADM2 outlines are overlaid for regional context.
- Warm colours highlight hotspot / worsening relationships.
- Blue highlights decline.
- Neutral grey is used for mixed / background areas.
- Heterogeneity map uses ADM2-level ADM3 pattern entropy.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

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

ADM2_BOUNDARY_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "administrative_boundaries"
    / "ssd_admin2.geojson"
)

V5_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "cross_scale_adm2_adm3_v5"
)

ADM3_CROSS_SCALE_FILE = (
    V5_DIR
    / "02_adm3_cross_scale_comparison_v5.csv"
)

ADM2_SUMMARY_FILE = (
    V5_DIR
    / "04_adm2_cross_scale_summary_v5.csv"
)

OUTPUT_DIR = (
    V5_DIR
    / "maps"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# 2. VISUAL SETTINGS
# ============================================================================

FIGSIZE = (12, 9)
DPI = 320

ADM3_EDGE_COLOR = "#F4F1EC"
ADM3_EDGE_WIDTH = 0.22

ADM2_EDGE_COLOR = "#555555"
ADM2_EDGE_WIDTH = 0.70

COUNTRY_EDGE_COLOR = "#2F2F2F"
COUNTRY_EDGE_WIDTH = 0.90

BACKGROUND_GREY = "#E8E6E1"
NO_DATA_GREY = "#D0D0D0"

TITLE_SIZE = 17
SUBTITLE_SIZE = 10
NOTE_SIZE = 8.5
LEGEND_SIZE = 9


# ============================================================================
# 3. PRIMARY CROSS-SCALE PALETTE
# ============================================================================

PRIMARY_COLORS = {
    "Regional chronic + local hotspot":
        "#8C1D18",

    "Regional emerging + local worsening":
        "#C94C2E",

    "Regional accelerating + local increase":
        "#D66A35",

    "Regional resurgence + local worsening":
        "#E59B42",

    "Regional volatile + local hotspot":
        "#B87C4C",

    "Regional decline + local worsening":
        "#7B3294",

    "Regional chronic + local decline":
        "#3B6EA8",

    "Regional acceleration + local decline":
        "#6B8FB3",

    "Regional decline + local decline":
        "#2E6F95",

    "Regional low + local emerging":
        "#D89F4A",

    "Regional low + local hotspot":
        "#C88D3C",

    "Regional low + local stable":
        "#A8C2BD",

    "Regional neutral + local hotspot":
        "#A65C85",

    "Regional neutral + local stable":
        "#CFCFCB",

    "Regional/local volatility alignment":
        "#9C7B6C",

    "Local volatile / mixed regional context":
        "#B8A99A",

    "Multi-signal / mixed":
        "#BDBDBD",
}


# Order used in legend
PRIMARY_ORDER = [
    "Regional chronic + local hotspot",
    "Regional emerging + local worsening",
    "Regional accelerating + local increase",
    "Regional resurgence + local worsening",
    "Regional volatile + local hotspot",
    "Regional decline + local worsening",
    "Regional chronic + local decline",
    "Regional acceleration + local decline",
    "Regional decline + local decline",
    "Regional low + local emerging",
    "Regional low + local hotspot",
    "Regional low + local stable",
    "Regional neutral + local hotspot",
    "Regional neutral + local stable",
    "Regional/local volatility alignment",
    "Local volatile / mixed regional context",
    "Multi-signal / mixed",
]


# ============================================================================
# 4. HELPERS
# ============================================================================

def require_file(path: Path, description: str) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


def normalize_text(value) -> str:

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def detect_column(columns, candidates):

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


def clean_axis(ax) -> None:

    ax.set_axis_off()
    ax.set_aspect("equal")


def add_country_outline(
    gdf: gpd.GeoDataFrame,
    ax,
) -> None:

    country = gdf.dissolve()

    country.boundary.plot(
        ax=ax,
        color=COUNTRY_EDGE_COLOR,
        linewidth=COUNTRY_EDGE_WIDTH,
        zorder=20,
    )


def add_adm2_outline(
    adm2: gpd.GeoDataFrame,
    ax,
) -> None:

    adm2.boundary.plot(
        ax=ax,
        color=ADM2_EDGE_COLOR,
        linewidth=ADM2_EDGE_WIDTH,
        zorder=15,
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
        pad_inches=0.12,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def add_note(
    fig,
    text: str,
) -> None:

    fig.text(
        0.5,
        0.018,
        text,
        ha="center",
        va="bottom",
        fontsize=NOTE_SIZE,
    )


# ============================================================================
# 5. LOAD BOUNDARIES
# ============================================================================

def load_boundaries():

    print("=" * 80)
    print("LOAD ADMINISTRATIVE BOUNDARIES")
    print("=" * 80)

    require_file(
        ADM3_BOUNDARY_FILE,
        "ADM3 boundary",
    )

    require_file(
        ADM2_BOUNDARY_FILE,
        "ADM2 boundary",
    )

    adm3 = (
        gpd.read_file(
            ADM3_BOUNDARY_FILE
        )
        .to_crs("EPSG:4326")
        .reset_index(drop=True)
    )

    adm2 = (
        gpd.read_file(
            ADM2_BOUNDARY_FILE
        )
        .to_crs("EPSG:4326")
        .reset_index(drop=True)
    )

    # ADM3 ID must match strict flood-processing pipeline.
    adm3[
        "ADM3_ID"
    ] = (
        np.arange(
            len(adm3)
        )
        + 1
    )

    print(
        f"ADM3 polygons: {len(adm3):,}"
    )

    print(
        f"ADM2 polygons: {len(adm2):,}"
    )

    return adm3, adm2


# ============================================================================
# 6. LOAD V5 CROSS-SCALE DATA
# ============================================================================

def load_v5_tables():

    print("\n" + "=" * 80)
    print("LOAD V5 CROSS-SCALE TABLES")
    print("=" * 80)

    require_file(
        ADM3_CROSS_SCALE_FILE,
        "ADM3 V5 cross-scale table",
    )

    require_file(
        ADM2_SUMMARY_FILE,
        "ADM2 V5 cross-scale summary",
    )

    adm3_cross = pd.read_csv(
        ADM3_CROSS_SCALE_FILE
    )

    adm2_summary = pd.read_csv(
        ADM2_SUMMARY_FILE
    )

    required_adm3 = [
        "ADM3_ID",
        "ADM2_NAME",
        "ADM3_NAME",
        "ADM3_PATTERN",
        "primary_cross_scale_pattern",
        "flag_regional_chronic_local_hotspot",
        "flag_regional_decline_local_worsening",
        "flag_regional_neutral_local_hotspot",
    ]

    missing = [
        col
        for col in required_adm3
        if col not in adm3_cross.columns
    ]

    if missing:

        raise ValueError(
            "\nV5 ADM3 cross-scale table is missing required columns:\n"
            + "\n".join(missing)
        )

    required_adm2 = [
        "ADM2_NAME",
        "adm3_pattern_entropy",
    ]

    missing2 = [
        col
        for col in required_adm2
        if col not in adm2_summary.columns
    ]

    if missing2:

        raise ValueError(
            "\nV5 ADM2 summary is missing required columns:\n"
            + "\n".join(missing2)
        )

    print(
        f"ADM3 V5 rows: {len(adm3_cross):,}"
    )

    print(
        f"ADM2 summary rows: {len(adm2_summary):,}"
    )

    return (
        adm3_cross,
        adm2_summary,
    )


# ============================================================================
# 7. MERGE ADM3 BOUNDARY + V5 RESULTS
# ============================================================================

def merge_adm3(
    adm3_boundary,
    adm3_cross,
):

    print("\n" + "=" * 80)
    print("MERGE ADM3 MAP DATA")
    print("=" * 80)

    if (
        adm3_cross[
            "ADM3_ID"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate ADM3_ID detected in V5 cross-scale table."
        )

    merged = adm3_boundary.merge(
        adm3_cross,
        on="ADM3_ID",
        how="left",
        validate="one_to_one",
    )

    missing = int(
        merged[
            "primary_cross_scale_pattern"
        ]
        .isna()
        .sum()
    )

    print(
        f"ADM3 map polygons: {len(merged):,}"
    )

    print(
        f"ADM3 polygons without V5 result: {missing:,}"
    )

    if missing > 0:

        raise ValueError(
            "Some ADM3 polygons have no V5 cross-scale result."
        )

    return merged


# ============================================================================
# 8. MERGE ADM2 BOUNDARY + HETEROGENEITY
# ============================================================================

def merge_adm2(
    adm2_boundary,
    adm2_summary,
):

    print("\n" + "=" * 80)
    print("MERGE ADM2 HETEROGENEITY DATA")
    print("=" * 80)

    boundary_name_col = detect_column(
        adm2_boundary.columns,
        [
            "adm2_name",
            "ADM2_NAME",
            "NAME_2",
            "name_2",
            "county",
        ],
    )

    if boundary_name_col is None:

        raise ValueError(
            "\nCould not detect ADM2 name column in boundary.\n"
            f"Available columns:\n{list(adm2_boundary.columns)}"
        )

    adm2_boundary = (
        adm2_boundary.copy()
    )

    adm2_boundary[
        "ADM2_KEY"
    ] = (
        adm2_boundary[
            boundary_name_col
        ]
        .map(
            normalize_text
        )
    )

    adm2_summary = (
        adm2_summary.copy()
    )

    adm2_summary[
        "ADM2_KEY"
    ] = (
        adm2_summary[
            "ADM2_NAME"
        ]
        .map(
            normalize_text
        )
    )

    if (
        adm2_summary[
            "ADM2_KEY"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate ADM2 names detected in V5 summary."
        )

    merged = adm2_boundary.merge(
        adm2_summary,
        on="ADM2_KEY",
        how="left",
        validate="one_to_one",
    )

    missing = int(
        merged[
            "adm3_pattern_entropy"
        ]
        .isna()
        .sum()
    )

    print(
        f"ADM2 map polygons: {len(merged):,}"
    )

    print(
        f"ADM2 polygons without heterogeneity result: {missing:,}"
    )

    if missing > 0:

        print(
            "WARNING: Some ADM2 boundary names did not match the V5 summary."
        )

    return merged


# ============================================================================
# 9. MAP 1 — PRIMARY CROSS-SCALE RELATIONSHIP
# ============================================================================

def plot_primary_relationship(
    adm3_gdf,
    adm2_gdf,
):

    print(
        "\nCreating primary cross-scale relationship map..."
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # neutral base
    adm3_gdf.plot(
        ax=ax,
        color=BACKGROUND_GREY,
        edgecolor=ADM3_EDGE_COLOR,
        linewidth=ADM3_EDGE_WIDTH,
        zorder=1,
    )

    present = []

    for category in PRIMARY_ORDER:

        subset = adm3_gdf[
            adm3_gdf[
                "primary_cross_scale_pattern"
            ]
            ==
            category
        ]

        if subset.empty:
            continue

        present.append(
            category
        )

        subset.plot(
            ax=ax,
            color=PRIMARY_COLORS[
                category
            ],
            edgecolor=ADM3_EDGE_COLOR,
            linewidth=ADM3_EDGE_WIDTH,
            zorder=2,
        )

    add_adm2_outline(
        adm2_gdf,
        ax,
    )

    add_country_outline(
        adm3_gdf,
        ax,
    )

    ax.set_title(
        "ADM2–ADM3 Cross-Scale Flood Pattern Relationships",
        fontsize=TITLE_SIZE,
        pad=14,
        weight="semibold",
    )

    clean_axis(
        ax
    )

    legend_handles = [
        Patch(
            facecolor=PRIMARY_COLORS[
                category
            ],
            edgecolor="none",
            label=category,
        )
        for category in present
    ]

    ax.legend(
        handles=legend_handles,
        title="Primary cross-scale relationship",
        loc="center left",
        bbox_to_anchor=(
            1.01,
            0.5,
        ),
        frameon=False,
        fontsize=LEGEND_SIZE,
        title_fontsize=10,
    )

    add_note(
        fig,
        (
            "ADM2 outlines provide regional context; ADM3 colours show the "
            "dominant cross-scale relationship derived from the V5 multi-label analysis."
        ),
    )

    plt.tight_layout(
        rect=[
            0.01,
            0.05,
            0.80,
            0.97,
        ]
    )

    save_figure(
        fig,
        "01_primary_cross_scale_pattern_map.png",
    )


# ============================================================================
# 10. GENERIC BINARY HIGHLIGHT MAP
# ============================================================================

def plot_binary_highlight(
    adm3_gdf,
    adm2_gdf,
    flag_column,
    highlight_color,
    title,
    subtitle,
    filename,
):

    print(
        f"Creating {filename}..."
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # Background
    adm3_gdf.plot(
        ax=ax,
        color=BACKGROUND_GREY,
        edgecolor=ADM3_EDGE_COLOR,
        linewidth=ADM3_EDGE_WIDTH,
        zorder=1,
    )

    selected = (
        adm3_gdf[
            adm3_gdf[
                flag_column
            ]
            ==
            1
        ]
    )

    selected.plot(
        ax=ax,
        color=highlight_color,
        edgecolor="#FFFFFF",
        linewidth=0.35,
        zorder=4,
    )

    add_adm2_outline(
        adm2_gdf,
        ax,
    )

    add_country_outline(
        adm3_gdf,
        ax,
    )

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        pad=14,
        weight="semibold",
    )

    clean_axis(
        ax
    )

    legend_handles = [
        Patch(
            facecolor=highlight_color,
            edgecolor="none",
            label=f"Highlighted ADM3 (n={len(selected)})",
        ),
        Patch(
            facecolor=BACKGROUND_GREY,
            edgecolor="none",
            label="Other ADM3",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        fontsize=LEGEND_SIZE,
    )

    add_note(
        fig,
        subtitle,
    )

    plt.tight_layout(
        rect=[
            0.01,
            0.05,
            0.98,
            0.97,
        ]
    )

    save_figure(
        fig,
        filename,
    )


# ============================================================================
# 11. MAP 4 — ADM2 INTERNAL HETEROGENEITY
# ============================================================================

def plot_adm2_entropy(
    adm2_gdf,
):

    print(
        "Creating ADM2 heterogeneity / entropy map..."
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    values = pd.to_numeric(
        adm2_gdf[
            "adm3_pattern_entropy"
        ],
        errors="coerce",
    )

    valid = (
        values
        .dropna()
    )

    if valid.empty:

        raise ValueError(
            "No valid ADM2 entropy values available."
        )

    vmin = 0.0
    vmax = max(
        1.0,
        float(
            valid.max()
        ),
    )

    adm2_gdf.plot(
        ax=ax,
        column="adm3_pattern_entropy",
        cmap="YlOrBr",
        vmin=vmin,
        vmax=vmax,
        legend=True,
        edgecolor="#FFFFFF",
        linewidth=0.45,
        missing_kwds={
            "color":
                NO_DATA_GREY,

            "label":
                "No data",
        },
        legend_kwds={
            "label":
                "ADM3 pattern entropy (0 = homogeneous, 1 = highly heterogeneous)",

            "shrink":
                0.75,
        },
    )

    add_country_outline(
        adm2_gdf,
        ax,
    )

    ax.set_title(
        "Within-ADM2 Heterogeneity of ADM3 Flood Temporal Patterns",
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
            "Higher entropy indicates a greater diversity of ADM3 temporal "
            "patterns within the same ADM2 region."
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
        "04_adm2_internal_heterogeneity_entropy_map.png",
    )


# ============================================================================
# 12. SAVE MAP DATA
# ============================================================================

def save_map_data(
    adm3_gdf,
    adm2_gdf,
):

    adm3_columns = [
        col
        for col in [
            "ADM3_ID",
            "ADM2_NAME",
            "ADM3_NAME",
            "ADM3_PATTERN",
            "primary_cross_scale_pattern",
            "flag_regional_chronic_local_hotspot",
            "flag_regional_decline_local_worsening",
            "flag_regional_neutral_local_hotspot",
            "flag_regional_accelerating_local_increase",
            "flag_regional_volatile_local_hotspot",
            "continuous_trend_opposite",
        ]
        if col in adm3_gdf.columns
    ]

    adm3_gdf[
        adm3_columns
    ].to_csv(
        OUTPUT_DIR
        / "adm3_cross_scale_map_data.csv",
        index=False,
    )

    adm2_columns = [
        col
        for col in [
            "ADM2_NAME",
            "ADM2_FINAL_PATTERN",
            "n_adm3",
            "adm3_dominant_pattern",
            "adm3_pattern_entropy",
            "adm3_hotspot_share",
            "adm3_worsening_share",
            "adm3_declining_share",
        ]
        if col in adm2_gdf.columns
    ]

    adm2_gdf[
        adm2_columns
    ].to_csv(
        OUTPUT_DIR
        / "adm2_heterogeneity_map_data.csv",
        index=False,
    )


# ============================================================================
# 13. TERMINAL SUMMARY
# ============================================================================

def print_summary(
    adm3_gdf,
    adm2_gdf,
):

    print("\n" + "=" * 80)
    print("MAP SUMMARY")
    print("=" * 80)

    print(
        "\nPrimary cross-scale pattern counts:"
    )

    print(
        adm3_gdf[
            "primary_cross_scale_pattern"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nKey highlighted cases:"
    )

    for col in [
        "flag_regional_chronic_local_hotspot",
        "flag_regional_decline_local_worsening",
        "flag_regional_neutral_local_hotspot",
        "flag_regional_accelerating_local_increase",
        "flag_regional_volatile_local_hotspot",
    ]:

        if col in adm3_gdf.columns:

            print(
                f"{col}: "
                f"{int(adm3_gdf[col].sum()):,}"
            )

    entropy = pd.to_numeric(
        adm2_gdf[
            "adm3_pattern_entropy"
        ],
        errors="coerce",
    )

    print(
        "\nADM2 entropy summary:"
    )

    print(
        entropy.describe().to_string()
    )


# ============================================================================
# 14. MAIN
# ============================================================================

def main():

    adm3_boundary, adm2_boundary = (
        load_boundaries()
    )

    (
        adm3_cross,
        adm2_summary,
    ) = (
        load_v5_tables()
    )

    adm3_map = (
        merge_adm3(
            adm3_boundary,
            adm3_cross,
        )
    )

    adm2_map = (
        merge_adm2(
            adm2_boundary,
            adm2_summary,
        )
    )

    # ------------------------------------------------------------------------
    # Map 1
    # ------------------------------------------------------------------------

    plot_primary_relationship(
        adm3_map,
        adm2_map,
    )

    # ------------------------------------------------------------------------
    # Map 2
    # ------------------------------------------------------------------------

    plot_binary_highlight(
        adm3_gdf=adm3_map,
        adm2_gdf=adm2_map,
        flag_column=
            "flag_regional_chronic_local_hotspot",
        highlight_color=
            "#8C1D18",
        title=
            "Regional Chronic Flooding with Local ADM3 Hotspots",
        subtitle=
            (
                "Highlighted ADM3 units are local hotspots located within ADM2 "
                "regions identified as chronic flood areas."
            ),
        filename=
            "02_regional_chronic_local_hotspot_map.png",
    )

    # ------------------------------------------------------------------------
    # Map 3
    # ------------------------------------------------------------------------

    plot_binary_highlight(
        adm3_gdf=adm3_map,
        adm2_gdf=adm2_map,
        flag_column=
            "flag_regional_decline_local_worsening",
        highlight_color=
            "#7B3294",
        title=
            "Regional Decline with Local Flood Worsening",
        subtitle=
            (
                "Highlighted ADM3 units worsen despite a declining regional "
                "ADM2 signal, revealing local trajectories hidden by aggregation."
            ),
        filename=
            "03_regional_decline_local_worsening_map.png",
    )

    # ------------------------------------------------------------------------
    # Map 4
    # ------------------------------------------------------------------------

    plot_adm2_entropy(
        adm2_map,
    )

    # ------------------------------------------------------------------------
    # Map 5
    # ------------------------------------------------------------------------

    plot_binary_highlight(
        adm3_gdf=adm3_map,
        adm2_gdf=adm2_map,
        flag_column=
            "flag_regional_neutral_local_hotspot",
        highlight_color=
            "#A65C85",
        title=
            "Localized Flood Hotspots within Regionally Neutral ADM2 Areas",
        subtitle=
            (
                "Highlighted ADM3 units show hotspot behaviour even where the "
                "parent ADM2 has no dominant special temporal pattern."
            ),
        filename=
            "05_regional_neutral_local_hotspot_map.png",
    )

    # ------------------------------------------------------------------------
    # Save data
    # ------------------------------------------------------------------------

    save_map_data(
        adm3_map,
        adm2_map,
    )

    print_summary(
        adm3_map,
        adm2_map,
    )

    # ------------------------------------------------------------------------
    # Output QC
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    output_files = [
        "01_primary_cross_scale_pattern_map.png",
        "02_regional_chronic_local_hotspot_map.png",
        "03_regional_decline_local_worsening_map.png",
        "04_adm2_internal_heterogeneity_entropy_map.png",
        "05_regional_neutral_local_hotspot_map.png",
        "adm3_cross_scale_map_data.csv",
        "adm2_heterogeneity_map_data.csv",
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
        "\nCross-scale mapping completed."
    )


if __name__ == "__main__":
    main()
