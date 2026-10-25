#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_visualize_adm2_temporal_patterns.py

South Sudan Flood Analysis
JBG060 Capstone Data Challenge

Purpose
-------
Create presentation-ready visualisations that directly follow the data-driven
temporal-pattern workflow established in Scripts 03 and 04.

Workflow
--------
Script 03:
    discovers six ADM2 temporal flood patterns.

Script 04:
    selects one centroid-nearest observed ADM2 representative for each pattern.

Script 05:
    visualises those six automatically selected representatives and the
    early-versus-recent change across all ADM2 areas.

This script does NOT manually assign hotspot labels or manually choose
representative ADM2 areas.

Inputs
------
1. Annual ADM2 combined flood indicators:
   processed_data/cross_dataset_analysis/01_adm2_flood/indicators/tables/
       adm2_annual_flood_indicators_combined.csv

2. Script 04 representative selection table:
   processed_data/cross_dataset_analysis/01_adm2_flood/
       04_representative_temporal_patterns/tables/
       01_representative_pattern_selection.csv

Outputs
-------
processed_data/cross_dataset_analysis/01_adm2_flood/
    05_visualize_adm2_temporal_patterns/

    tables/
        01_adm2_early_vs_recent_summary.csv
        02_representatives_used_for_visualization.csv

    figures/
        01_adm2_early_vs_recent_slope_chart.png
        02_six_pattern_representative_trajectories.png
        03_six_pattern_early_recent_comparison.png
        04_all_adm2_temporal_pattern_map.png
        05_representative_adm2_location_map.png
        06_adm2_early_recent_change_map.png
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import geopandas as gpd
except ImportError:
    gpd = None

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/shibai/Documents/GitHub/DBL-Group-22"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "cross_dataset_analysis"
)

ADM2_ROOT = (
    PROCESSED_ROOT
    / "01_adm2_flood"
)

ANNUAL_FILE = (
    ADM2_ROOT
    / "indicators"
    / "tables"
    / "adm2_annual_flood_indicators_combined.csv"
)

ADM2_BOUNDARY_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "administrative_boundaries"
    / "ssd_admin2.geojson"
)

PATTERN_MEMBERSHIP_FILE = (
    ADM2_ROOT
    / "temporal_patterns"
    / "tables"
    / "03_pattern_membership.csv"
)

# Script 04 output. This is the authoritative source for the six
# data-driven representative ADM2 areas used by this visualization script.
REPRESENTATIVE_FILE = (
    ADM2_ROOT
    / "04_representative_temporal_patterns"
    / "tables"
    / "01_representative_pattern_selection.csv"
)

# Script 05 has its own output folder so its results are not mixed with
# Script 04 tables/figures.
OUTPUT_ROOT = (
    ADM2_ROOT
    / "05_visualize_adm2_temporal_patterns"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. STUDY PERIODS
# =============================================================================

STUDY_START_YEAR = 2000
STUDY_END_YEAR = 2025

EARLY_START = 2000
EARLY_END = 2007

RECENT_START = 2017
RECENT_END = 2025


# =============================================================================
# 3. REPRESENTATIVE ADM2 AREAS
# =============================================================================

# Representatives are NOT hard-coded here.
# They are read from Script 04 so Script 05 always reflects the same
# six data-driven temporal patterns and centroid-nearest representatives.


# =============================================================================
# 4. HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


def load_representatives() -> pd.DataFrame:
    """Load the six representatives selected by Script 04."""
    require_file(REPRESENTATIVE_FILE)

    reps = pd.read_csv(REPRESENTATIVE_FILE)

    # Support the current Script 04 output while keeping column validation
    # explicit and easy to diagnose.
    pattern_candidates = ["pattern_name", "Pattern", "temporal_pattern"]
    name_candidates = [
        "representative_adm2_name",
        "representative_ADM2_NAME",
        "ADM2_NAME",
        "representative_name",
    ]

    pattern_col = next((c for c in pattern_candidates if c in reps.columns), None)
    name_col = next((c for c in name_candidates if c in reps.columns), None)

    if pattern_col is None or name_col is None:
        raise ValueError(
            "\nScript 04 representative table does not contain the expected "
            "pattern/representative columns.\n"
            f"Available columns: {list(reps.columns)}"
        )

    reps = reps.rename(
        columns={
            pattern_col: "pattern_name",
            name_col: "representative_adm2_name",
        }
    ).copy()

    if "cluster" not in reps.columns:
        reps["cluster"] = np.arange(1, len(reps) + 1)

    reps = (
        reps.dropna(subset=["pattern_name", "representative_adm2_name"])
        .drop_duplicates(subset=["pattern_name"])
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    if len(reps) != 6:
        print(
            f"Warning: expected 6 temporal patterns from Script 04, "
            f"but found {len(reps)}."
        )

    output_file = TABLE_DIR / "02_representatives_used_for_visualization.csv"
    reps.to_csv(output_file, index=False)
    print(f"Representatives loaded from Script 04:\n{REPRESENTATIVE_FILE}")
    print(f"Visualization copy saved:\n{output_file}")

    return reps


def load_annual_data() -> pd.DataFrame:
    require_file(ANNUAL_FILE)

    df = pd.read_csv(ANNUAL_FILE)

    required = {
        "ADM2_ID",
        "ADM2_NAME",
        "year",
        "annual_flooded_percent",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(sorted(missing))
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["annual_flooded_percent"] = pd.to_numeric(
        df["annual_flooded_percent"],
        errors="coerce",
    )

    if "flood_type" in df.columns:
        df = df[
            df["flood_type"]
            .astype(str)
            .str.lower()
            .eq("combined")
        ].copy()

    df = df[
        df["year"].between(
            STUDY_START_YEAR,
            STUDY_END_YEAR,
        )
    ].copy()

    return df


# =============================================================================
# 5. BUILD EARLY VS RECENT SUMMARY
# =============================================================================

def build_early_recent_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("BUILDING EARLY VS RECENT SUMMARY")

    records = []

    for adm2_name, subset in df.groupby("ADM2_NAME"):

        early = subset[
            subset["year"].between(
                EARLY_START,
                EARLY_END,
            )
        ]

        recent = subset[
            subset["year"].between(
                RECENT_START,
                RECENT_END,
            )
        ]

        if early.empty or recent.empty:
            continue

        early_mean = early["annual_flooded_percent"].mean()
        recent_mean = recent["annual_flooded_percent"].mean()

        records.append(
            {
                "ADM2_NAME": adm2_name,
                "early_mean_flooded_percent": early_mean,
                "recent_mean_flooded_percent": recent_mean,
                "recent_minus_early_pp": recent_mean - early_mean,
            }
        )

    summary = pd.DataFrame(records)

    summary = summary.sort_values(
        "recent_minus_early_pp",
        ascending=False,
    ).reset_index(drop=True)

    output_file = (
        TABLE_DIR
        / "01_adm2_early_vs_recent_summary.csv"
    )

    summary.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved:\n{output_file}")

    return summary


# =============================================================================
# 6. FIGURE 1
#    EARLY VS RECENT SLOPE CHART
# =============================================================================

def plot_early_vs_recent_slope_chart(
    summary: pd.DataFrame,
) -> None:

    print_header("GENERATING EARLY VS RECENT SLOPE CHART")

    if summary.empty:
        print("No data available.")
        return

    # Keep all ADM2 areas, ordered by magnitude of change.
    plot_df = summary.copy()

    # Dynamic height so 79 ADM2 names remain readable.
    fig_height = max(
        10,
        len(plot_df) * 0.28,
    )

    fig, ax = plt.subplots(
        figsize=(11, fig_height)
    )

    y = np.arange(len(plot_df))

    # Neutral connecting line
    for i, row in plot_df.iterrows():
        ax.plot(
            [
                row["early_mean_flooded_percent"],
                row["recent_mean_flooded_percent"],
            ],
            [i, i],
            linewidth=1.1,
            alpha=0.55,
        )

    # Early and recent points
    ax.scatter(
        plot_df["early_mean_flooded_percent"],
        y,
        s=34,
        label=f"Early ({EARLY_START}-{EARLY_END})",
        zorder=3,
    )

    ax.scatter(
        plot_df["recent_mean_flooded_percent"],
        y,
        s=34,
        marker="s",
        label=f"Recent ({RECENT_START}-{RECENT_END})",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        plot_df["ADM2_NAME"],
        fontsize=8,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Mean annual flooded area (%)"
    )

    ax.set_title(
        "ADM2 Flood Change: Early vs Recent Period\n"
        f"{EARLY_START}-{EARLY_END} compared with {RECENT_START}-{RECENT_END}",
        fontsize=14,
        pad=12,
    )

    ax.grid(
        axis="x",
        alpha=0.22,
    )

    ax.legend(
        loc="lower right",
        frameon=True,
    )

    # Remove unnecessary spines for cleaner appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "01_adm2_early_vs_recent_slope_chart.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved:\n{output_file}")


# =============================================================================
# 7. FIGURE 2
#    SIX DATA-DRIVEN REPRESENTATIVE TRAJECTORIES
# =============================================================================

def plot_six_pattern_trajectories(
    df: pd.DataFrame,
    reps: pd.DataFrame,
) -> None:

    print_header("GENERATING SIX-PATTERN REPRESENTATIVE TRAJECTORIES")

    n = len(reps)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 4.2 * nrows),
        sharex=True,
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, (_, rep) in zip(axes, reps.iterrows()):
        adm2_name = str(rep["representative_adm2_name"])
        pattern_name = str(rep["pattern_name"])

        subset = (
            df[df["ADM2_NAME"] == adm2_name]
            .sort_values("year")
            .copy()
        )

        if subset.empty:
            ax.set_title(f"{adm2_name}\n{pattern_name}\nNo annual data found")
            continue

        early = subset[
            subset["year"].between(EARLY_START, EARLY_END)
        ]
        recent = subset[
            subset["year"].between(RECENT_START, RECENT_END)
        ]

        early_mean = early["annual_flooded_percent"].mean()
        recent_mean = recent["annual_flooded_percent"].mean()
        change_pp = recent_mean - early_mean
        sign = "+" if change_pp >= 0 else ""

        ax.plot(
            subset["year"],
            subset["annual_flooded_percent"],
            marker="o",
            linewidth=1.8,
            markersize=4,
        )

        # No universal 2020 line is imposed here. Script 03 change points are
        # ADM2-specific; a common hard-coded breakpoint would overstate the
        # evidence in a presentation figure.

        ax.set_title(
            f"{pattern_name}\nRepresentative: {adm2_name}",
            fontsize=11,
        )
        ax.set_ylabel("Annual flooded area (%)")
        ax.grid(alpha=0.20)

        annotation = (
            f"Early {early_mean:.2f}%  →  Recent {recent_mean:.2f}%\n"
            f"Change: {sign}{change_pp:.2f} pp"
        )
        ax.text(
            0.03,
            0.95,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.80,
                "edgecolor": "none",
            },
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes:
        ax.set_xlabel("Year")

    # Hide unused panels if the upstream table contains fewer than 6 rows.
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Representative ADM2 Flood Trajectories for the Six Temporal Patterns\n"
        "Representatives selected automatically in Script 04",
        fontsize=15,
        y=1.01,
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "02_six_pattern_representative_trajectories.png"
    )
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved:\n{output_file}")


# =============================================================================
# 8. FIGURE 3
#    EARLY VS RECENT COMPARISON FOR THE SIX REPRESENTATIVES
# =============================================================================

def plot_six_pattern_early_recent(
    df: pd.DataFrame,
    reps: pd.DataFrame,
) -> None:

    print_header("GENERATING SIX-PATTERN EARLY VS RECENT COMPARISON")

    records = []

    for _, rep in reps.iterrows():
        adm2_name = str(rep["representative_adm2_name"])
        pattern_name = str(rep["pattern_name"])

        subset = df[df["ADM2_NAME"] == adm2_name].copy()
        early = subset[subset["year"].between(EARLY_START, EARLY_END)]
        recent = subset[subset["year"].between(RECENT_START, RECENT_END)]

        if early.empty or recent.empty:
            continue

        records.append(
            {
                "pattern_name": pattern_name,
                "ADM2_NAME": adm2_name,
                "early_mean": early["annual_flooded_percent"].mean(),
                "recent_mean": recent["annual_flooded_percent"].mean(),
            }
        )

    plot_df = pd.DataFrame(records)

    if plot_df.empty:
        print("No representative comparison data available.")
        return

    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 1.05)))

    for i, row in plot_df.iterrows():
        ax.plot(
            [row["early_mean"], row["recent_mean"]],
            [i, i],
            linewidth=1.2,
            alpha=0.60,
        )

    ax.scatter(
        plot_df["early_mean"],
        y,
        s=55,
        label=f"Early ({EARLY_START}-{EARLY_END})",
        zorder=3,
    )
    ax.scatter(
        plot_df["recent_mean"],
        y,
        s=55,
        marker="s",
        label=f"Recent ({RECENT_START}-{RECENT_END})",
        zorder=3,
    )

    labels = [
        f"{row.pattern_name}\n{row.ADM2_NAME}"
        for row in plot_df.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean annual flooded area (%)")
    ax.set_title(
        "Early-to-Recent Change in the Six Temporal-Pattern Representatives",
        fontsize=14,
        pad=12,
    )
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "03_six_pattern_early_recent_comparison.png"
    )
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved:\n{output_file}")



# =============================================================================
# 9. SPATIAL VISUALISATION HELPERS
# =============================================================================

def load_adm2_boundary():
    """Load the same official ADM2 polygon layer used by Script 01."""
    if gpd is None:
        raise ImportError(
            "\ngeopandas is required for the three ADM2 maps in Script 05.\n"
            "Install it in the project environment before running this script."
        )

    require_file(ADM2_BOUNDARY_FILE)
    boundary = gpd.read_file(ADM2_BOUNDARY_FILE)

    if boundary.empty:
        raise ValueError(f"ADM2 boundary file is empty:\n{ADM2_BOUNDARY_FILE}")

    if boundary.crs is None:
        print("Warning: ADM2 boundary CRS is missing; plotting in source coordinates.")

    return boundary


def identify_boundary_key_columns(boundary) -> tuple[str | None, str | None]:
    """Identify ADM2 ID/name fields without silently guessing arbitrary columns."""
    id_candidates = [
        "ADM2_ID", "adm2_id", "ADM2_PCODE", "ADM2_CODE",
        "admin2Pcod", "shapeID", "GID_2",
    ]
    name_candidates = [
        "ADM2_NAME", "adm2_name", "ADM2_EN", "admin2Name",
        "ADMIN2NAME", "County_EN", "COUNTYNAME", "county",
        "NAME_2", "shapeName", "name",
    ]

    id_col = next((c for c in id_candidates if c in boundary.columns), None)
    name_col = next((c for c in name_candidates if c in boundary.columns), None)

    if id_col is None and name_col is None:
        raise ValueError(
            "Could not identify an ADM2 ID or name field in the boundary layer.\n"
            f"Available columns: {list(boundary.columns)}"
        )

    return id_col, name_col


def _clean_join_key(series: pd.Series) -> pd.Series:
    """Normalize join keys while preserving them as strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def join_table_to_boundary(
    boundary,
    table: pd.DataFrame,
    *,
    table_id_col: str = "ADM2_ID",
    table_name_col: str = "ADM2_NAME",
):
    """Join ADM2 results to polygons, preferring ADM2_ID and falling back to name."""
    id_col, name_col = identify_boundary_key_columns(boundary)
    gdf = boundary.copy()
    tab = table.copy()

    join_method = None

    if id_col is not None and table_id_col in tab.columns:
        gdf["_join_key"] = _clean_join_key(gdf[id_col])
        tab["_join_key"] = _clean_join_key(tab[table_id_col])
        merged = gdf.merge(tab, on="_join_key", how="left", suffixes=("", "_result"))
        matched = int(merged[table_id_col].notna().sum()) if table_id_col in merged.columns else 0
        if matched > 0:
            join_method = f"ADM2 ID ({id_col} ↔ {table_id_col})"
        else:
            merged = None
    else:
        merged = None

    if merged is None:
        if name_col is None or table_name_col not in tab.columns:
            raise ValueError(
                "ADM2_ID join was unavailable/unsuccessful and no ADM2 name "
                "columns were available for a controlled fallback."
            )

        gdf["_join_key"] = (
            gdf[name_col].astype(str).str.strip().str.casefold()
        )
        tab["_join_key"] = (
            tab[table_name_col].astype(str).str.strip().str.casefold()
        )
        merged = gdf.merge(tab, on="_join_key", how="left", suffixes=("", "_result"))
        join_method = f"ADM2 name ({name_col} ↔ {table_name_col})"

    print(f"Spatial join method: {join_method}")
    return merged


def load_pattern_membership() -> pd.DataFrame:
    """Load all 79 ADM2 pattern assignments produced by Script 03."""
    require_file(PATTERN_MEMBERSHIP_FILE)
    membership = pd.read_csv(PATTERN_MEMBERSHIP_FILE)

    required = {"ADM2_ID", "ADM2_NAME", "cluster", "pattern_name"}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError(
            "Missing columns from Script 03 pattern membership:\n"
            + "\n".join(sorted(missing))
        )

    return membership[list(required)].drop_duplicates(subset=["ADM2_ID"])


# =============================================================================
# 10. FIGURE 4
#     ALL ADM2 TEMPORAL-PATTERN MAP
# =============================================================================

def plot_all_adm2_temporal_pattern_map(boundary, membership: pd.DataFrame) -> None:
    print_header("GENERATING ALL-ADM2 TEMPORAL-PATTERN MAP")

    gdf = join_table_to_boundary(boundary, membership)

    matched = int(gdf["pattern_name"].notna().sum())
    print(f"ADM2 polygons with temporal-pattern assignment: {matched}/{len(gdf)}")

    fig, ax = plt.subplots(figsize=(11, 12))

    # Unmatched polygons remain visible for QC rather than disappearing.
    gdf.plot(ax=ax, facecolor="0.92", edgecolor="0.65", linewidth=0.45)

    valid = gdf[gdf["pattern_name"].notna()].copy()
    if not valid.empty:
        valid.plot(
            column="pattern_name",
            categorical=True,
            legend=True,
            ax=ax,
            edgecolor="white",
            linewidth=0.45,
            legend_kwds={
                "title": "Temporal pattern",
                "loc": "lower left",
                "bbox_to_anchor": (0.0, -0.02),
                "fontsize": 8,
                "title_fontsize": 9,
            },
        )

    ax.set_title(
        "ADM2 Temporal Flood Patterns in South Sudan\n"
        "Pattern membership discovered in Script 03",
        fontsize=14,
        pad=12,
    )
    ax.set_axis_off()
    fig.tight_layout()

    out = FIGURE_DIR / "04_all_adm2_temporal_pattern_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{out}")


# =============================================================================
# 11. FIGURE 5
#     SIX REPRESENTATIVE ADM2 LOCATIONS
# =============================================================================

def plot_representative_adm2_location_map(boundary, reps: pd.DataFrame) -> None:
    print_header("GENERATING REPRESENTATIVE ADM2 LOCATION MAP")

    # Normalize Script 04 representative fields for the spatial join.
    table = reps.copy()
    id_candidates = [
        "representative_ADM2_ID",
        "representative_adm2_id",
        "ADM2_ID",
    ]
    id_col = next((c for c in id_candidates if c in table.columns), None)

    table["ADM2_NAME"] = table["representative_adm2_name"]
    if id_col is not None:
        table["ADM2_ID"] = table[id_col]

    gdf = join_table_to_boundary(boundary, table)

    selected = gdf[gdf["pattern_name"].notna()].copy()
    if selected.empty:
        raise ValueError("None of the Script 04 representatives matched the ADM2 boundary.")

    fig, ax = plt.subplots(figsize=(10, 12))
    boundary.plot(ax=ax, facecolor="0.94", edgecolor="0.65", linewidth=0.45)

    selected.plot(
        column="pattern_name",
        categorical=True,
        ax=ax,
        edgecolor="black",
        linewidth=1.1,
        legend=True,
        legend_kwds={
            "title": "Representative temporal pattern",
            "loc": "lower left",
            "bbox_to_anchor": (0.0, -0.02),
            "fontsize": 8,
            "title_fontsize": 9,
        },
    )

    # Labels are placed at representative points of polygons, which are more
    # reliable than raw centroids for irregular/multipart administrative areas.
    label_points = selected.copy()
    if label_points.crs is not None and label_points.crs.is_geographic:
        projected = label_points.to_crs(3857)
        pts = projected.geometry.representative_point()
        pts = gpd.GeoSeries(pts, crs=3857).to_crs(label_points.crs)
    else:
        pts = label_points.geometry.representative_point()

    for (_, row), point in zip(label_points.iterrows(), pts):
        name = row.get("ADM2_NAME_result", row.get("ADM2_NAME", ""))
        cluster = row.get("cluster", "")
        ax.annotate(
            f"P{int(cluster)} {name}" if pd.notna(cluster) else str(name),
            xy=(point.x, point.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5,
            weight="bold",
        )

    ax.set_title(
        "Locations of the Six Representative ADM2 Areas\n"
        "Representatives selected in Script 04",
        fontsize=14,
        pad=12,
    )
    ax.set_axis_off()
    fig.tight_layout()

    out = FIGURE_DIR / "05_representative_adm2_location_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{out}")


# =============================================================================
# 12. FIGURE 6
#     ADM2 EARLY-TO-RECENT CHANGE MAP
# =============================================================================

def plot_adm2_early_recent_change_map(boundary, summary: pd.DataFrame) -> None:
    print_header("GENERATING ADM2 EARLY-TO-RECENT CHANGE MAP")

    gdf = join_table_to_boundary(boundary, summary)
    value_col = "recent_minus_early_pp"

    matched = int(gdf[value_col].notna().sum())
    print(f"ADM2 polygons with early/recent change value: {matched}/{len(gdf)}")

    fig, ax = plt.subplots(figsize=(10, 12))
    gdf.plot(ax=ax, facecolor="0.92", edgecolor="0.65", linewidth=0.45)

    valid = gdf[gdf[value_col].notna()].copy()
    if not valid.empty:
        # Symmetric limits make zero the visual reference for increase/decrease.
        vmax = float(np.nanmax(np.abs(valid[value_col].to_numpy(dtype=float))))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        valid.plot(
            column=value_col,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            legend=True,
            ax=ax,
            edgecolor="white",
            linewidth=0.45,
            legend_kwds={
                "label": "Recent minus early flooded area (percentage points)",
                "shrink": 0.72,
            },
        )

    ax.set_title(
        "ADM2 Change in Mean Annual Flooded Area\n"
        f"{RECENT_START}-{RECENT_END} minus {EARLY_START}-{EARLY_END}",
        fontsize=14,
        pad=12,
    )
    ax.set_axis_off()
    fig.tight_layout()

    out = FIGURE_DIR / "06_adm2_early_recent_change_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{out}")


# =============================================================================
# 13. MAIN
# =============================================================================

def main() -> None:

    print_header("ADM2 TEMPORAL-PATTERN VISUALISATION")

    print(f"Annual input:\n{ANNUAL_FILE}")
    print(f"\nScript 04 representatives:\n{REPRESENTATIVE_FILE}")
    print(f"\nOutput root:\n{OUTPUT_ROOT}")

    df = load_annual_data()
    reps = load_representatives()
    summary = build_early_recent_summary(df)

    plot_early_vs_recent_slope_chart(summary)
    plot_six_pattern_trajectories(df, reps)
    plot_six_pattern_early_recent(df, reps)

    # Spatial visualisations use the same ADM2 boundary as Script 01 and
    # existing Script 03/04/05 outputs; no flood-mask indicators are recomputed.
    boundary = load_adm2_boundary()
    membership = load_pattern_membership()
    plot_all_adm2_temporal_pattern_map(boundary, membership)
    plot_representative_adm2_location_map(boundary, reps)
    plot_adm2_early_recent_change_map(boundary, summary)

    print_header("FINISHED")

    print(
        "\nGenerated:\n"
        "  tables/01_adm2_early_vs_recent_summary.csv\n"
        "  tables/02_representatives_used_for_visualization.csv\n"
        "  figures/01_adm2_early_vs_recent_slope_chart.png\n"
        "  figures/02_six_pattern_representative_trajectories.png\n"
        "  figures/03_six_pattern_early_recent_comparison.png\n"
        "  figures/04_all_adm2_temporal_pattern_map.png\n"
        "  figures/05_representative_adm2_location_map.png\n"
        "  figures/06_adm2_early_recent_change_map.png\n"
    )

    print(f"Saved under:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
