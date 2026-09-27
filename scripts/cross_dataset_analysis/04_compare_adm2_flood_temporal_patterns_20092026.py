#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_compare_adm2_flood_temporal_patterns.py

South Sudan Flood Analysis
JBG060 Capstone Data Challenge

Purpose
-------
Compare representative ADM2 flood temporal patterns between 2000 and 2025.

This script is designed to complement:

    01_build_adm2_flood_indicators.py
    03_analyze_adm2_flood_trends.py

Rather than ranking ADM2 regions again, this script selects representative
areas showing different types of flood behaviour:

1. Long-term + emerging hotspot
2. Long-term hotspot
3. Emerging hotspot
4. Persistent flood area
5. Declining / historically important hotspot

Representative ADM2 areas
-------------------------
Rubkona
    Long-term + emerging hotspot

Mayom
    Long-term hotspot

Guit
    Emerging hotspot

Mayendit
    Emerging hotspot

Ayod
    Persistent flood area

Canal/Pigi
    Emerging hotspot

Luakpiny/Nasir
    Declining / historically important hotspot

Twic East
    Persistent + increasing hotspot

Outputs
-------
1. adm2_representative_temporal_patterns_combined.png
2. adm2_representative_temporal_patterns_panels.png
3. adm2_representative_temporal_patterns_summary.csv
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


# =============================================================================
# 1. PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/shibai/Documents/GitHub/DBL-Group-22"
)

ANNUAL_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm2"
    / "adm2_annual_flood_indicators_combined.csv"
)

TREND_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm2"
    / "trend_analysis"
)

LONG_TERM_FILE = (
    TREND_DIR
    / "adm2_long_term_hotspot_ranking.csv"
)

CHANGE_FILE = (
    TREND_DIR
    / "adm2_early_vs_recent_change.csv"
)

TREND_FILE = (
    TREND_DIR
    / "adm2_trend_analysis.csv"
)

OUTPUT_DIR = (
    TREND_DIR
    / "temporal_patterns"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. STUDY PERIOD
# =============================================================================

STUDY_START_YEAR = 2000
STUDY_END_YEAR = 2025

EARLY_START = 2000
EARLY_END = 2007

RECENT_START = 2017
RECENT_END = 2025

LATEST_START = 2021
LATEST_END = 2025


# =============================================================================
# 3. REPRESENTATIVE ADM2 AREAS
# =============================================================================

REPRESENTATIVE_AREAS = {
    "Rubkona":
        "Long-term + emerging",

    "Mayom":
        "Long-term hotspot",

    "Guit":
        "Emerging hotspot",

    "Mayendit":
        "Emerging hotspot",

    "Ayod":
        "Persistent flood area",

    "Canal/Pigi":
        "Emerging hotspot",

    "Luakpiny/Nasir":
        "Declining / historically important",

    "Twic East":
        "Persistent + increasing",
}


# =============================================================================
# 4. GENERAL HELPERS
# =============================================================================

def print_header(
    title: str,
) -> None:

    print(
        "\n"
        + "=" * 80
    )

    print(
        title
    )

    print(
        "=" * 80
    )


def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{path}"
        )


# =============================================================================
# 5. LOAD DATA
# =============================================================================

def load_annual_data() -> pd.DataFrame:

    require_file(
        ANNUAL_FILE
    )

    df = pd.read_csv(
        ANNUAL_FILE
    )

    required = {
        "ADM2_ID",
        "ADM2_NAME",
        "year",
        "annual_flooded_percent",
        "flood_days",
        "flood_event_count",
        "annual_flood_area_days_km2",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "\nMissing columns from annual file:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    numeric_columns = [
        "annual_flooded_percent",
        "flood_days",
        "flood_event_count",
        "annual_flood_area_days_km2",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if "flood_type" in df.columns:

        df = df[
            df[
                "flood_type"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "combined"
            )
        ].copy()

    df = df[
        df[
            "year"
        ].between(
            STUDY_START_YEAR,
            STUDY_END_YEAR,
        )
    ].copy()

    return df


def load_optional_analysis_files():

    long_term = None
    change = None
    trends = None

    if LONG_TERM_FILE.exists():

        long_term = pd.read_csv(
            LONG_TERM_FILE
        )

    if CHANGE_FILE.exists():

        change = pd.read_csv(
            CHANGE_FILE
        )

    if TREND_FILE.exists():

        trends = pd.read_csv(
            TREND_FILE
        )

    return (
        long_term,
        change,
        trends,
    )


# =============================================================================
# 6. CHECK REPRESENTATIVE AREAS
# =============================================================================

def validate_representative_areas(
    df: pd.DataFrame,
) -> None:

    available = set(
        df[
            "ADM2_NAME"
        ]
        .astype(str)
        .unique()
    )

    requested = set(
        REPRESENTATIVE_AREAS.keys()
    )

    missing = (
        requested
        - available
    )

    if missing:

        print(
            "\nWARNING:"
        )

        print(
            "The following representative ADM2 names "
            "were not found:"
        )

        for name in sorted(
            missing
        ):

            print(
                f"  - {name}"
            )

        print(
            "\nAvailable similar names should be "
            "checked in the annual dataset."
        )


# =============================================================================
# 7. BUILD REPRESENTATIVE SUMMARY
# =============================================================================

def build_representative_summary(
    df: pd.DataFrame,
    long_term: pd.DataFrame | None,
    change: pd.DataFrame | None,
    trends: pd.DataFrame | None,
) -> pd.DataFrame:

    print_header(
        "BUILDING REPRESENTATIVE ADM2 SUMMARY"
    )

    records = []

    for (
        adm2_name,
        pattern_type,
    ) in REPRESENTATIVE_AREAS.items():

        subset = df[
            df[
                "ADM2_NAME"
            ]
            == adm2_name
        ].copy()

        if subset.empty:
            continue

        early = subset[
            subset[
                "year"
            ].between(
                EARLY_START,
                EARLY_END,
            )
        ]

        recent = subset[
            subset[
                "year"
            ].between(
                RECENT_START,
                RECENT_END,
            )
        ]

        latest = subset[
            subset[
                "year"
            ].between(
                LATEST_START,
                LATEST_END,
            )
        ]

        record = {
            "ADM2_NAME":
                adm2_name,

            "pattern_type":
                pattern_type,

            "long_term_mean_flooded_percent":
                subset[
                    "annual_flooded_percent"
                ].mean(),

            "early_mean_flooded_percent":
                early[
                    "annual_flooded_percent"
                ].mean(),

            "recent_mean_flooded_percent":
                recent[
                    "annual_flooded_percent"
                ].mean(),

            "latest_5y_mean_flooded_percent":
                latest[
                    "annual_flooded_percent"
                ].mean(),

            "recent_minus_early_pp":
                (
                    recent[
                        "annual_flooded_percent"
                    ].mean()
                    -
                    early[
                        "annual_flooded_percent"
                    ].mean()
                ),

            "mean_flood_days_per_year":
                subset[
                    "flood_days"
                ].mean(),

            "max_annual_flooded_percent":
                subset[
                    "annual_flooded_percent"
                ].max(),

            "max_flood_year":
                subset.loc[
                    subset[
                        "annual_flooded_percent"
                    ].idxmax(),
                    "year",
                ],
        }

        if long_term is not None:

            match = long_term[
                long_term[
                    "ADM2_NAME"
                ]
                == adm2_name
            ]

            if not match.empty:

                if (
                    "hotspot_rank"
                    in match.columns
                ):

                    record[
                        "long_term_hotspot_rank"
                    ] = (
                        match.iloc[0][
                            "hotspot_rank"
                        ]
                    )

        if change is not None:

            match = change[
                change[
                    "ADM2_NAME"
                ]
                == adm2_name
            ]

            if not match.empty:

                if (
                    "early_rank"
                    in match.columns
                ):

                    record[
                        "early_rank"
                    ] = (
                        match.iloc[0][
                            "early_rank"
                        ]
                    )

                if (
                    "recent_rank"
                    in match.columns
                ):

                    record[
                        "recent_rank"
                    ] = (
                        match.iloc[0][
                            "recent_rank"
                        ]
                    )

        if trends is not None:

            match = trends[
                trends[
                    "ADM2_NAME"
                ]
                == adm2_name
            ]

            if not match.empty:

                if (
                    "flooded_percent_trend_pp_per_year"
                    in match.columns
                ):

                    record[
                        "trend_pp_per_year"
                    ] = (
                        match.iloc[0][
                            "flooded_percent_trend_pp_per_year"
                        ]
                    )

                if (
                    "flooded_percent_trend_r2"
                    in match.columns
                ):

                    record[
                        "trend_r2"
                    ] = (
                        match.iloc[0][
                            "flooded_percent_trend_r2"
                        ]
                    )

        records.append(
            record
        )

    summary = pd.DataFrame(
        records
    )

    output_file = (
        OUTPUT_DIR
        / "adm2_representative_temporal_patterns_summary.csv"
    )

    summary.to_csv(
        output_file,
        index=False,
    )

    display_columns = [
        "ADM2_NAME",
        "pattern_type",
        "long_term_mean_flooded_percent",
        "early_mean_flooded_percent",
        "recent_mean_flooded_percent",
        "recent_minus_early_pp",
        "mean_flood_days_per_year",
    ]

    existing_display_columns = [
        col
        for col in display_columns
        if col in summary.columns
    ]

    print(
        summary[
            existing_display_columns
        ]
        .round(
            2
        )
        .to_string(
            index=False
        )
    )

    return summary


# =============================================================================
# 8. FIGURE 1
#    COMBINED REPRESENTATIVE TIME SERIES
# =============================================================================

def plot_combined_temporal_patterns(
    df: pd.DataFrame,
) -> None:

    print_header(
        "GENERATING COMBINED TEMPORAL PATTERN FIGURE"
    )

    fig, ax = plt.subplots(
        figsize=(
            13,
            8,
        )
    )

    for (
        adm2_name,
        pattern_type,
    ) in REPRESENTATIVE_AREAS.items():

        subset = (
            df[
                df[
                    "ADM2_NAME"
                ]
                == adm2_name
            ]
            .sort_values(
                "year"
            )
        )

        if subset.empty:
            continue

        label = (
            f"{adm2_name} "
            f"({pattern_type})"
        )

        ax.plot(
            subset[
                "year"
            ],
            subset[
                "annual_flooded_percent"
            ],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=label,
        )

    # Mark approximately where the recent flood expansion
    # becomes visually prominent in several ADM2 areas.
    ax.axvline(
        x=2020,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )

    ymax = ax.get_ylim()[1]

    ax.text(
        2020.2,
        ymax * 0.95,
        "2020",
        fontsize=9,
        va="top",
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Annual flooded area (%)"
    )

    ax.set_title(
        "Representative ADM2 Flood Trajectories "
        "(2000-2025)\n"
        "Examples of long-term, emerging, persistent "
        "and declining flood patterns"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "adm2_representative_temporal_patterns_combined.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved:\n{output_file}"
    )


# =============================================================================
# 9. FIGURE 2
#    SMALL-MULTIPLE PANELS
# =============================================================================

def plot_temporal_pattern_panels(
    df: pd.DataFrame,
) -> None:

    print_header(
        "GENERATING SMALL-MULTIPLE TEMPORAL PATTERN FIGURE"
    )

    areas = list(
        REPRESENTATIVE_AREAS.items()
    )

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(
            14,
            15,
        ),
        sharex=True,
    )

    axes = axes.flatten()

    for ax, (
        adm2_name,
        pattern_type,
    ) in zip(
        axes,
        areas,
    ):

        subset = (
            df[
                df[
                    "ADM2_NAME"
                ]
                == adm2_name
            ]
            .sort_values(
                "year"
            )
        )

        if subset.empty:

            ax.set_title(
                f"{adm2_name}\n"
                "No data found"
            )

            continue

        ax.plot(
            subset[
                "year"
            ],
            subset[
                "annual_flooded_percent"
            ],
            marker="o",
            linewidth=1.6,
            markersize=4,
        )

        ax.axvline(
            x=2020,
            linestyle="--",
            linewidth=0.9,
            alpha=0.5,
        )

        ax.set_title(
            f"{adm2_name}\n"
            f"{pattern_type}",
            fontsize=11,
        )

        ax.set_ylabel(
            "Flooded area (%)"
        )

        ax.grid(
            alpha=0.25,
        )

        # -------------------------------------------------------------
        # Add early and recent period averages
        # -------------------------------------------------------------

        early = subset[
            subset[
                "year"
            ].between(
                EARLY_START,
                EARLY_END,
            )
        ]

        recent = subset[
            subset[
                "year"
            ].between(
                RECENT_START,
                RECENT_END,
            )
        ]

        early_mean = (
            early[
                "annual_flooded_percent"
            ].mean()
        )

        recent_mean = (
            recent[
                "annual_flooded_percent"
            ].mean()
        )

        change_pp = (
            recent_mean
            - early_mean
        )

        sign = (
            "+"
            if change_pp >= 0
            else ""
        )

        annotation = (
            f"Early: {early_mean:.2f}%\n"
            f"Recent: {recent_mean:.2f}%\n"
            f"Change: {sign}{change_pp:.2f} pp"
        )

        ax.text(
            0.02,
            0.96,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={
                "boxstyle":
                    "round,pad=0.3",

                "facecolor":
                    "white",

                "alpha":
                    0.75,

                "edgecolor":
                    "none",
            },
        )

    for ax in axes[-2:]:

        ax.set_xlabel(
            "Year"
        )

    fig.suptitle(
        "Representative ADM2 Flood Temporal Patterns "
        "(2000-2025)\n"
        "Comparison of different long-term and recent "
        "flood trajectories",
        fontsize=15,
        y=1.01,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "adm2_representative_temporal_patterns_panels.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved:\n{output_file}"
    )


# =============================================================================
# 10. PRINT INTERPRETATION TABLE
# =============================================================================

def print_interpretation_summary(
    summary: pd.DataFrame,
) -> None:

    print_header(
        "REPRESENTATIVE TEMPORAL PATTERNS"
    )

    for _, row in summary.iterrows():

        name = (
            row[
                "ADM2_NAME"
            ]
        )

        pattern = (
            row[
                "pattern_type"
            ]
        )

        early = (
            row[
                "early_mean_flooded_percent"
            ]
        )

        recent = (
            row[
                "recent_mean_flooded_percent"
            ]
        )

        change = (
            row[
                "recent_minus_early_pp"
            ]
        )

        sign = (
            "+"
            if change >= 0
            else ""
        )

        print(
            f"\n{name}"
        )

        print(
            f"  Pattern: "
            f"{pattern}"
        )

        print(
            f"  Early mean: "
            f"{early:.2f}%"
        )

        print(
            f"  Recent mean: "
            f"{recent:.2f}%"
        )

        print(
            f"  Change: "
            f"{sign}{change:.2f} pp"
        )


# =============================================================================
# 11. MAIN
# =============================================================================

def main():

    print_header(
        "ADM2 FLOOD TEMPORAL PATTERN COMPARISON"
    )

    print(
        f"Annual input:\n"
        f"{ANNUAL_FILE}"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    df = (
        load_annual_data()
    )

    (
        long_term,
        change,
        trends,
    ) = load_optional_analysis_files()

    # -------------------------------------------------------------------------
    # Validate representative ADM2 names
    # -------------------------------------------------------------------------

    validate_representative_areas(
        df
    )

    # -------------------------------------------------------------------------
    # Build summary table
    # -------------------------------------------------------------------------

    summary = (
        build_representative_summary(
            df=df,
            long_term=long_term,
            change=change,
            trends=trends,
        )
    )

    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------

    plot_combined_temporal_patterns(
        df
    )

    plot_temporal_pattern_panels(
        df
    )

    # -------------------------------------------------------------------------
    # Print interpretation summary
    # -------------------------------------------------------------------------

    print_interpretation_summary(
        summary
    )

    # -------------------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------------------

    print_header(
        "FINISHED"
    )

    print(
        "\nGenerated:"
    )

    print(
        "  - "
        "adm2_representative_temporal_patterns_summary.csv"
    )

    print(
        "  - "
        "adm2_representative_temporal_patterns_combined.png"
    )

    print(
        "  - "
        "adm2_representative_temporal_patterns_panels.png"
    )

    print(
        f"\nSaved under:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()