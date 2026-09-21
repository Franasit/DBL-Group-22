#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_visualize_adm2_flood_change_clean.py

South Sudan Flood Analysis
JBG060 Capstone Data Challenge

Purpose
-------
Create clearer, presentation-ready ADM2 flood temporal visualisations.

This script replaces the visually crowded multi-line comparison with:

1. Early vs Recent slope chart
   - compares 2000-2007 mean flooded area with 2017-2025 mean flooded area
   - makes increases and decreases easy to see

2. Clean 4-panel representative trajectories
   - Rubkona: sharp emerging hotspot
   - Mayom: long-term hotspot
   - Twic East: persistent + increasing
   - Luakpiny/Nasir: declining / historically important

Input
-----
processed_data/flood_analysis/adm2/adm2_annual_flood_indicators_combined.csv

Outputs
-------
processed_data/flood_analysis/adm2/trend_analysis/temporal_patterns_clean/

1. adm2_early_vs_recent_slope_chart.png
2. adm2_representative_4panel_clean.png
3. adm2_early_vs_recent_summary.csv
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

ANNUAL_FILE = (
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
    / "temporal_patterns_clean"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

REPRESENTATIVE_4 = {
    "Rubkona": "Sharp emerging hotspot",
    "Mayom": "Long-term hotspot",
    "Twic East": "Persistent + increasing",
    "Luakpiny/Nasir": "Declining / historically important",
}


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
        OUTPUT_DIR
        / "adm2_early_vs_recent_summary.csv"
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
        OUTPUT_DIR
        / "adm2_early_vs_recent_slope_chart.png"
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
#    CLEAN 4-PANEL REPRESENTATIVE TRAJECTORIES
# =============================================================================

def plot_representative_4panel(
    df: pd.DataFrame,
) -> None:

    print_header("GENERATING CLEAN 4-PANEL FIGURE")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 9),
        sharex=True,
    )

    axes = axes.flatten()

    for ax, (adm2_name, pattern_type) in zip(
        axes,
        REPRESENTATIVE_4.items(),
    ):

        subset = (
            df[
                df["ADM2_NAME"] == adm2_name
            ]
            .sort_values("year")
            .copy()
        )

        if subset.empty:
            ax.set_title(
                f"{adm2_name}\nNo data found"
            )
            continue

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

        early_mean = early["annual_flooded_percent"].mean()
        recent_mean = recent["annual_flooded_percent"].mean()
        change_pp = recent_mean - early_mean

        sign = "+" if change_pp >= 0 else ""

        # One simple line per panel.
        ax.plot(
            subset["year"],
            subset["annual_flooded_percent"],
            marker="o",
            linewidth=1.8,
            markersize=4,
        )

        # Light 2020 reference line.
        ax.axvline(
            x=2020,
            linestyle="--",
            linewidth=0.9,
            alpha=0.35,
        )

        ax.set_title(
            f"{adm2_name}\n{pattern_type}",
            fontsize=11,
        )

        ax.set_ylabel(
            "Flooded area (%)"
        )

        ax.grid(
            alpha=0.20,
        )

        # Simplified annotation
        annotation = (
            f"{early_mean:.2f}% → {recent_mean:.2f}% "
            f"({sign}{change_pp:.2f} pp)"
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

    for ax in axes[-2:]:
        ax.set_xlabel("Year")

    fig.suptitle(
        "Representative ADM2 Flood Trajectories (2000-2025)\n"
        "Four contrasting temporal flood patterns",
        fontsize=15,
        y=1.01,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "adm2_representative_4panel_clean.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved:\n{output_file}")


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:

    print_header(
        "CLEAN ADM2 FLOOD TEMPORAL VISUALISATION"
    )

    print(f"Annual input:\n{ANNUAL_FILE}")
    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    df = load_annual_data()

    summary = build_early_recent_summary(df)

    plot_early_vs_recent_slope_chart(summary)

    plot_representative_4panel(df)

    print_header("FINISHED")

    print(
        "\nGenerated:\n"
        "  - adm2_early_vs_recent_summary.csv\n"
        "  - adm2_early_vs_recent_slope_chart.png\n"
        "  - adm2_representative_4panel_clean.png\n"
    )

    print(f"Saved under:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
