#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_analyze_adm2_flood_trends.py

South Sudan Flood Analysis
JBG060 Capstone Data Challenge

Purpose
-------
Analyse long-term ADM2 flood patterns using:

    processed_data/flood_analysis/adm2/
        adm2_annual_flood_indicators_combined.csv

Main outputs
------------
1. adm2_long_term_hotspot_ranking.csv
2. adm2_flood_persistence_ranking.csv
3. adm2_early_vs_recent_change.csv
4. adm2_trend_analysis.csv
5. adm2_hotspot_classification.csv

Figures
-------
1. adm2_top10_hotspots_mean_flooded_percent.png
2. adm2_top10_flood_persistence.png
3. adm2_early_vs_recent_comparison.png
4. adm2_selected_hotspot_timeseries.png

Interpretation
--------------
- long-term hotspot:
    high long-term mean annual flooded percentage

- persistent hotspot:
    high flood-day persistence and/or many flood years

- emerging hotspot:
    recent flood extent substantially exceeds early-period flood extent

- declining hotspot:
    recent flood extent is substantially lower than early-period flood extent
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/shibai/Documents/GitHub/DBL-Group-22"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm2"
    / "adm2_annual_flood_indicators_combined.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm2"
    / "trend_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. ANALYSIS SETTINGS
# =============================================================================

STUDY_START_YEAR = 2000
STUDY_END_YEAR = 2025

EARLY_START = 2000
EARLY_END = 2007

RECENT_START = 2017
RECENT_END = 2025

LATEST_START = 2021
LATEST_END = 2025

TOP_N = 10

QUANTILE_HIGH = 0.75
QUANTILE_LOW = 0.25


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{path}\n"
            "\nPlease run 01_build_adm2_flood_indicators.py first."
        )


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return np.nan

    return float(
        series.mean()
    )


def linear_trend(
    years: pd.Series,
    values: pd.Series,
):
    """
    Fit:

        y = intercept + slope * year

    Returns:
        slope,
        intercept,
        r_squared
    """

    x = pd.to_numeric(
        years,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    y = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    y_hat = (
        intercept
        + slope * x
    )

    ss_res = np.sum(
        (y - y_hat) ** 2
    )

    ss_tot = np.sum(
        (y - y.mean()) ** 2
    )

    if ss_tot == 0:
        r_squared = 0.0

    else:
        r_squared = (
            1
            - ss_res / ss_tot
        )

    return (
        float(slope),
        float(intercept),
        float(r_squared),
    )


# =============================================================================
# 4. LOAD DATA
# =============================================================================

def load_data() -> pd.DataFrame:

    require_file(
        INPUT_FILE
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        "ADM2_ID",
        "ADM2_NAME",
        "year",
        "adm2_area_km2",
        "flood_days",
        "flood_event_count",
        "annual_unique_flooded_area_km2",
        "annual_flood_area_days_km2",
        "annual_flooded_percent",
        "mean_event_duration_days",
        "flood_occurred",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    numeric_columns = [
        "adm2_area_km2",
        "flood_days",
        "flood_event_count",
        "annual_unique_flooded_area_km2",
        "annual_flood_area_days_km2",
        "annual_flooded_percent",
        "mean_event_duration_days",
        "flood_occurred",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df[
        df["year"].between(
            STUDY_START_YEAR,
            STUDY_END_YEAR,
        )
    ].copy()

    if "flood_type" in df.columns:

        df = df[
            df["flood_type"]
            .astype(str)
            .str.lower()
            .eq("combined")
        ].copy()

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"ADM2 count: "
        f"{df['ADM2_ID'].nunique():,}"
    )

    print(
        f"Year range: "
        f"{int(df['year'].min())}-"
        f"{int(df['year'].max())}"
    )

    return df


# =============================================================================
# 5. LONG-TERM HOTSPOT RANKING
# =============================================================================

def build_long_term_hotspot_ranking(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "LONG-TERM ADM2 HOTSPOT RANKING"
    )

    summary = (
        df
        .groupby(
            [
                "ADM2_ID",
                "ADM2_NAME",
            ],
            as_index=False,
        )
        .agg(
            adm2_area_km2=(
                "adm2_area_km2",
                "first",
            ),

            flood_year_count=(
                "flood_occurred",
                "sum",
            ),

            mean_annual_flooded_percent=(
                "annual_flooded_percent",
                "mean",
            ),

            median_annual_flooded_percent=(
                "annual_flooded_percent",
                "median",
            ),

            max_annual_flooded_percent=(
                "annual_flooded_percent",
                "max",
            ),

            mean_flood_days_per_year=(
                "flood_days",
                "mean",
            ),

            max_flood_days=(
                "flood_days",
                "max",
            ),

            mean_flood_events_per_year=(
                "flood_event_count",
                "mean",
            ),

            mean_event_duration_days=(
                "mean_event_duration_days",
                "mean",
            ),

            mean_annual_flood_area_days_km2=(
                "annual_flood_area_days_km2",
                "mean",
            ),
        )
    )

    n_years = (
        STUDY_END_YEAR
        - STUDY_START_YEAR
        + 1
    )

    summary[
        "flood_year_share"
    ] = (
        summary[
            "flood_year_count"
        ]
        / n_years
    )

    summary[
        "hotspot_rank"
    ] = (
        summary[
            "mean_annual_flooded_percent"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    summary = (
        summary
        .sort_values(
            "hotspot_rank"
        )
        .reset_index(
            drop=True
        )
    )

    output = (
        OUTPUT_DIR
        / "adm2_long_term_hotspot_ranking.csv"
    )

    summary.to_csv(
        output,
        index=False,
    )

    print(
        summary[
            [
                "hotspot_rank",
                "ADM2_NAME",
                "flood_year_count",
                "mean_annual_flooded_percent",
                "mean_flood_days_per_year",
                "max_annual_flooded_percent",
            ]
        ]
        .head(TOP_N)
        .to_string(
            index=False
        )
    )

    return summary


# =============================================================================
# 6. FLOOD PERSISTENCE RANKING
# =============================================================================

def build_persistence_ranking(
    long_term: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "FLOOD PERSISTENCE RANKING"
    )

    persistence = (
        long_term.copy()
    )

    persistence[
        "persistence_rank"
    ] = (
        persistence[
            "mean_flood_days_per_year"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    persistence = (
        persistence
        .sort_values(
            "persistence_rank"
        )
        .reset_index(
            drop=True
        )
    )

    output = (
        OUTPUT_DIR
        / "adm2_flood_persistence_ranking.csv"
    )

    persistence.to_csv(
        output,
        index=False,
    )

    print(
        persistence[
            [
                "persistence_rank",
                "ADM2_NAME",
                "mean_flood_days_per_year",
                "flood_year_count",
                "mean_annual_flooded_percent",
            ]
        ]
        .head(TOP_N)
        .to_string(
            index=False
        )
    )

    return persistence


# =============================================================================
# 7. EARLY VS RECENT COMPARISON
# =============================================================================

def build_early_recent_change(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "EARLY VS RECENT FLOOD CHANGE"
    )

    records = []

    for (
        adm2_id,
        adm2_name,
    ), group in df.groupby(
        [
            "ADM2_ID",
            "ADM2_NAME",
        ]
    ):

        early = group[
            group["year"].between(
                EARLY_START,
                EARLY_END,
            )
        ]

        recent = group[
            group["year"].between(
                RECENT_START,
                RECENT_END,
            )
        ]

        latest = group[
            group["year"].between(
                LATEST_START,
                LATEST_END,
            )
        ]

        records.append(
            {
                "ADM2_ID":
                    adm2_id,

                "ADM2_NAME":
                    adm2_name,

                "early_mean_flooded_percent":
                    safe_mean(
                        early[
                            "annual_flooded_percent"
                        ]
                    ),

                "recent_mean_flooded_percent":
                    safe_mean(
                        recent[
                            "annual_flooded_percent"
                        ]
                    ),

                "latest_5y_mean_flooded_percent":
                    safe_mean(
                        latest[
                            "annual_flooded_percent"
                        ]
                    ),

                "early_mean_flood_days":
                    safe_mean(
                        early[
                            "flood_days"
                        ]
                    ),

                "recent_mean_flood_days":
                    safe_mean(
                        recent[
                            "flood_days"
                        ]
                    ),

                "latest_5y_mean_flood_days":
                    safe_mean(
                        latest[
                            "flood_days"
                        ]
                    ),
            }
        )

    change = pd.DataFrame(
        records
    )

    change[
        "flooded_percent_change_pp"
    ] = (
        change[
            "recent_mean_flooded_percent"
        ]
        - change[
            "early_mean_flooded_percent"
        ]
    )

    change[
        "flood_days_change"
    ] = (
        change[
            "recent_mean_flood_days"
        ]
        - change[
            "early_mean_flood_days"
        ]
    )

    change[
        "recent_rank"
    ] = (
        change[
            "recent_mean_flooded_percent"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    change[
        "early_rank"
    ] = (
        change[
            "early_mean_flooded_percent"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    change[
        "rank_improvement"
    ] = (
        change[
            "early_rank"
        ]
        - change[
            "recent_rank"
        ]
    )

    change = (
        change
        .sort_values(
            "flooded_percent_change_pp",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    output = (
        OUTPUT_DIR
        / "adm2_early_vs_recent_change.csv"
    )

    change.to_csv(
        output,
        index=False,
    )

    print(
        "\nLargest increases:"
    )

    print(
        change[
            [
                "ADM2_NAME",
                "early_mean_flooded_percent",
                "recent_mean_flooded_percent",
                "flooded_percent_change_pp",
                "early_rank",
                "recent_rank",
            ]
        ]
        .head(TOP_N)
        .to_string(
            index=False
        )
    )

    print(
        "\nLargest decreases:"
    )

    print(
        change[
            [
                "ADM2_NAME",
                "early_mean_flooded_percent",
                "recent_mean_flooded_percent",
                "flooded_percent_change_pp",
                "early_rank",
                "recent_rank",
            ]
        ]
        .tail(TOP_N)
        .sort_values(
            "flooded_percent_change_pp"
        )
        .to_string(
            index=False
        )
    )

    return change


# =============================================================================
# 8. LINEAR TREND ANALYSIS
# =============================================================================

def build_trend_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "LINEAR TREND ANALYSIS"
    )

    records = []

    for (
        adm2_id,
        adm2_name,
    ), group in df.groupby(
        [
            "ADM2_ID",
            "ADM2_NAME",
        ]
    ):

        group = (
            group
            .sort_values(
                "year"
            )
        )

        flooded_slope, _, flooded_r2 = (
            linear_trend(
                group["year"],
                group[
                    "annual_flooded_percent"
                ],
            )
        )

        flood_days_slope, _, flood_days_r2 = (
            linear_trend(
                group["year"],
                group[
                    "flood_days"
                ],
            )
        )

        area_days_slope, _, area_days_r2 = (
            linear_trend(
                group["year"],
                group[
                    "annual_flood_area_days_km2"
                ],
            )
        )

        records.append(
            {
                "ADM2_ID":
                    adm2_id,

                "ADM2_NAME":
                    adm2_name,

                "flooded_percent_trend_pp_per_year":
                    flooded_slope,

                "flooded_percent_trend_r2":
                    flooded_r2,

                "flood_days_trend_days_per_year":
                    flood_days_slope,

                "flood_days_trend_r2":
                    flood_days_r2,

                "flood_area_days_trend_km2days_per_year":
                    area_days_slope,

                "flood_area_days_trend_r2":
                    area_days_r2,
            }
        )

    trends = pd.DataFrame(
        records
    )

    trends[
        "trend_rank"
    ] = (
        trends[
            "flooded_percent_trend_pp_per_year"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    trends = (
        trends
        .sort_values(
            "trend_rank"
        )
        .reset_index(
            drop=True
        )
    )

    output = (
        OUTPUT_DIR
        / "adm2_trend_analysis.csv"
    )

    trends.to_csv(
        output,
        index=False,
    )

    print(
        trends[
            [
                "trend_rank",
                "ADM2_NAME",
                "flooded_percent_trend_pp_per_year",
                "flooded_percent_trend_r2",
                "flood_days_trend_days_per_year",
            ]
        ]
        .head(TOP_N)
        .to_string(
            index=False
        )
    )

    return trends


# =============================================================================
# 9. HOTSPOT CLASSIFICATION
# =============================================================================

def build_hotspot_classification(
    long_term: pd.DataFrame,
    change: pd.DataFrame,
    trends: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "HOTSPOT CLASSIFICATION"
    )

    merged = (
        long_term
        .merge(
            change,
            on=[
                "ADM2_ID",
                "ADM2_NAME",
            ],
            how="left",
        )
        .merge(
            trends,
            on=[
                "ADM2_ID",
                "ADM2_NAME",
            ],
            how="left",
        )
    )

    high_extent_threshold = (
        merged[
            "mean_annual_flooded_percent"
        ]
        .quantile(
            QUANTILE_HIGH
        )
    )

    high_persistence_threshold = (
        merged[
            "mean_flood_days_per_year"
        ]
        .quantile(
            QUANTILE_HIGH
        )
    )

    emerging_threshold = (
        merged[
            "flooded_percent_change_pp"
        ]
        .quantile(
            QUANTILE_HIGH
        )
    )

    declining_threshold = (
        merged[
            "flooded_percent_change_pp"
        ]
        .quantile(
            QUANTILE_LOW
        )
    )

    def classify(row):

        high_extent = (
            row[
                "mean_annual_flooded_percent"
            ]
            >= high_extent_threshold
        )

        high_persistence = (
            row[
                "mean_flood_days_per_year"
            ]
            >= high_persistence_threshold
        )

        emerging = (
            row[
                "flooded_percent_change_pp"
            ]
            >= emerging_threshold
            and row[
                "flooded_percent_trend_pp_per_year"
            ] > 0
        )

        declining = (
            row[
                "flooded_percent_change_pp"
            ]
            <= declining_threshold
            and row[
                "flooded_percent_trend_pp_per_year"
            ] < 0
        )

        if (
            emerging
            and (
                high_extent
                or high_persistence
            )
        ):
            return (
                "High-risk emerging hotspot"
            )

        if emerging:
            return (
                "Emerging hotspot"
            )

        if (
            high_extent
            and high_persistence
        ):
            return (
                "Persistent regional hotspot"
            )

        if high_extent:
            return (
                "High-extent hotspot"
            )

        if high_persistence:
            return (
                "Persistent flood area"
            )

        if declining:
            return (
                "Declining hotspot"
            )

        return (
            "Lower relative hotspot level"
        )

    merged[
        "hotspot_class"
    ] = merged.apply(
        classify,
        axis=1,
    )

    merged[
        "high_extent_threshold"
    ] = high_extent_threshold

    merged[
        "high_persistence_threshold"
    ] = high_persistence_threshold

    merged[
        "emerging_change_threshold_pp"
    ] = emerging_threshold

    merged[
        "declining_change_threshold_pp"
    ] = declining_threshold

    output = (
        OUTPUT_DIR
        / "adm2_hotspot_classification.csv"
    )

    merged.to_csv(
        output,
        index=False,
    )

    print(
        "\nClassification counts:"
    )

    print(
        merged[
            "hotspot_class"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nThresholds:"
    )

    print(
        f"High extent >= "
        f"{high_extent_threshold:.2f}%"
    )

    print(
        f"High persistence >= "
        f"{high_persistence_threshold:.2f} "
        f"flood days/year"
    )

    print(
        f"Emerging change >= "
        f"{emerging_threshold:.2f} "
        f"percentage points"
    )

    return merged


# =============================================================================
# 10. FIGURE 1 - LONG-TERM HOTSPOTS
# =============================================================================

def plot_top_hotspots(
    long_term: pd.DataFrame,
):

    top = (
        long_term
        .nsmallest(
            TOP_N,
            "hotspot_rank",
        )
        .sort_values(
            "mean_annual_flooded_percent"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.barh(
        top["ADM2_NAME"],
        top[
            "mean_annual_flooded_percent"
        ],
    )

    ax.set_xlabel(
        "Mean annual flooded area (%)"
    )

    ax.set_ylabel(
        "ADM2"
    )

    ax.set_title(
        f"Top {TOP_N} ADM2 Long-Term Flood Hotspots "
        f"({STUDY_START_YEAR}-{STUDY_END_YEAR})"
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    for bar in bars:

        value = (
            bar.get_width()
        )

        ax.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.2f}%",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "adm2_top10_hotspots_mean_flooded_percent.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 11. FIGURE 2 - FLOOD PERSISTENCE
# =============================================================================

def plot_top_persistence(
    persistence: pd.DataFrame,
):

    top = (
        persistence
        .nsmallest(
            TOP_N,
            "persistence_rank",
        )
        .sort_values(
            "mean_flood_days_per_year"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.barh(
        top["ADM2_NAME"],
        top[
            "mean_flood_days_per_year"
        ],
    )

    ax.set_xlabel(
        "Mean flood days per year"
    )

    ax.set_ylabel(
        "ADM2"
    )

    ax.set_title(
        f"Top {TOP_N} ADM2 Flood Persistence Areas "
        f"({STUDY_START_YEAR}-{STUDY_END_YEAR})"
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    for bar in bars:

        value = (
            bar.get_width()
        )

        ax.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.2f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "adm2_top10_flood_persistence.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 12. FIGURE 3 - EARLY VS RECENT COMPARISON
# =============================================================================

def plot_early_vs_recent_comparison(
    change: pd.DataFrame,
):
    """
    Show the ADM2 areas with the strongest increase in flood extent.

    Selection:
        Top N ADM2 by recent-minus-early flooded percentage change.

    Display order:
        Largest increase at the top.

    Each ADM2 shows:
        - Early-period mean annual flooded area
        - Recent-period mean annual flooded area
        - Change in percentage points
    """

    # Select Top N based strictly on the size of the increase.
    top = (
        change
        .nlargest(
            TOP_N,
            "flooded_percent_change_pp",
        )
        .copy()
    )

    # Horizontal bar charts draw the first row at the bottom.
    # Sort ascending here so the largest increase appears at the top.
    top = (
        top
        .sort_values(
            "flooded_percent_change_pp",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(top)
    )

    bar_height = 0.36

    fig, ax = plt.subplots(
        figsize=(11.5, 7)
    )

    early_bars = ax.barh(
        y - bar_height / 2,
        top[
            "early_mean_flooded_percent"
        ],
        height=bar_height,
        label=f"Early period ({EARLY_START}-{EARLY_END})",
    )

    recent_bars = ax.barh(
        y + bar_height / 2,
        top[
            "recent_mean_flooded_percent"
        ],
        height=bar_height,
        label=f"Recent period ({RECENT_START}-{RECENT_END})",
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        top[
            "ADM2_NAME"
        ]
    )

    ax.set_xlabel(
        "Mean annual flooded area (%)"
    )

    ax.set_ylabel(
        "ADM2"
    )

    ax.set_title(
        "Where Has Flood Extent Increased Most?\n"
        f"Change in mean annual flooded area: "
        f"{EARLY_START}-{EARLY_END} vs "
        f"{RECENT_START}-{RECENT_END}"
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    # -------------------------------------------------------------------------
    # Value labels for early-period bars
    # -------------------------------------------------------------------------

    for bar in early_bars:

        value = (
            bar.get_width()
        )

        ax.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.2f}%",
            va="center",
            fontsize=8,
        )

    # -------------------------------------------------------------------------
    # Value labels for recent-period bars
    # -------------------------------------------------------------------------

    for bar in recent_bars:

        value = (
            bar.get_width()
        )

        ax.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.2f}%",
            va="center",
            fontsize=8,
        )

    # -------------------------------------------------------------------------
    # Percentage-point change labels
    # -------------------------------------------------------------------------

    max_recent = (
        top[
            "recent_mean_flooded_percent"
        ]
        .max()
    )

    max_change_label_x = (
        max_recent * 1.08
    )

    for i, row in top.iterrows():

        change_pp = (
            row[
                "flooded_percent_change_pp"
            ]
        )

        sign = (
            "+"
            if change_pp >= 0
            else ""
        )

        ax.text(
            max_change_label_x,
            i,
            f"{sign}{change_pp:.2f} pp",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Give enough room for the change labels.
    ax.set_xlim(
        0,
        max_recent * 1.30,
    )

    # Move legend outside the plot so it cannot cover Canal/Pigi
    # or any other change label.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
        borderaxespad=0.0,
        frameon=True,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "adm2_early_vs_recent_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 13. FIGURE 4 - SELECTED TIME SERIES
# =============================================================================

def plot_selected_timeseries(
    df: pd.DataFrame,
    long_term: pd.DataFrame,
    change: pd.DataFrame,
):

    selected = []

    selected.extend(
        long_term
        .nsmallest(
            4,
            "hotspot_rank",
        )[
            "ADM2_NAME"
        ]
        .tolist()
    )

    selected.extend(
        change
        .nlargest(
            4,
            "flooded_percent_change_pp",
        )[
            "ADM2_NAME"
        ]
        .tolist()
    )

    selected = list(
        dict.fromkeys(
            selected
        )
    )[:6]

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for name in selected:

        subset = (
            df[
                df[
                    "ADM2_NAME"
                ]
                == name
            ]
            .sort_values(
                "year"
            )
        )

        ax.plot(
            subset[
                "year"
            ],
            subset[
                "annual_flooded_percent"
            ],
            marker="o",
            linewidth=1.5,
            label=name,
        )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Annual flooded area (%)"
    )

    ax.set_title(
        "Selected ADM2 Flood-Extent Time Series"
    )

    ax.grid(
        alpha=0.3,
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "adm2_selected_hotspot_timeseries.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================

def print_final_summary(
    long_term: pd.DataFrame,
    change: pd.DataFrame,
    classification: pd.DataFrame,
):

    print_header(
        "FINAL SUMMARY"
    )

    print(
        "\nTop 5 long-term hotspots:"
    )

    print(
        long_term[
            [
                "hotspot_rank",
                "ADM2_NAME",
                "mean_annual_flooded_percent",
                "mean_flood_days_per_year",
            ]
        ]
        .head(5)
        .round(2)
        .to_string(
            index=False
        )
    )

    print(
        "\nTop 5 strongest recent increases:"
    )

    print(
        change[
            [
                "ADM2_NAME",
                "early_mean_flooded_percent",
                "recent_mean_flooded_percent",
                "flooded_percent_change_pp",
            ]
        ]
        .head(5)
        .round(2)
        .to_string(
            index=False
        )
    )

    print(
        "\nHigh-risk / emerging classifications:"
    )

    priority = classification[
        classification[
            "hotspot_class"
        ].isin(
            [
                "High-risk emerging hotspot",
                "Emerging hotspot",
                "Persistent regional hotspot",
                "High-extent hotspot",
            ]
        )
    ].copy()

    priority = (
        priority
        .sort_values(
            [
                "mean_annual_flooded_percent",
                "flooded_percent_change_pp",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    print(
        priority[
            [
                "ADM2_NAME",
                "hotspot_class",
                "mean_annual_flooded_percent",
                "recent_mean_flooded_percent",
                "flooded_percent_change_pp",
                "flooded_percent_trend_pp_per_year",
            ]
        ]
        .head(20)
        .round(2)
        .to_string(
            index=False
        )
    )

    print(
        "\nSaved outputs to:"
    )

    print(
        OUTPUT_DIR
    )


# =============================================================================
# 15. MAIN
# =============================================================================

def main():

    print_header(
        "ADM2 FLOOD TREND ANALYSIS"
    )

    print(
        f"Input:\n{INPUT_FILE}"
    )

    print(
        f"\nOutput:\n{OUTPUT_DIR}"
    )

    df = load_data()

    long_term = (
        build_long_term_hotspot_ranking(
            df
        )
    )

    persistence = (
        build_persistence_ranking(
            long_term
        )
    )

    change = (
        build_early_recent_change(
            df
        )
    )

    trends = (
        build_trend_analysis(
            df
        )
    )

    classification = (
        build_hotspot_classification(
            long_term=long_term,
            change=change,
            trends=trends,
        )
    )

    print_header(
        "GENERATING FIGURES"
    )

    plot_top_hotspots(
        long_term
    )

    plot_top_persistence(
        persistence
    )

    plot_early_vs_recent_comparison(
        change
    )

    plot_selected_timeseries(
        df=df,
        long_term=long_term,
        change=change,
    )

    print_final_summary(
        long_term=long_term,
        change=change,
        classification=classification,
    )


if __name__ == "__main__":
    main()
