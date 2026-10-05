#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_compare_adm2_flood_temporal_patterns.py

South Sudan Flood Analysis
JBG060 Capstone Data Challenge

Purpose
-------
Interpret the six ADM2 temporal flood patterns discovered by Script 03 using
representative annual flood trajectories from 2000-2025.

This script does NOT create new clusters and does NOT assign manual hotspot
labels. It reads the pattern membership produced by Script 03 and selects one
representative ADM2 for each pattern reproducibly: the ADM2 closest to its
pattern centroid in the same standardized 10-feature temporal space used for
pattern discovery.

Detected change points are also read from Script 03. A vertical marker is drawn
only when a representative ADM2 has a detected change point; no fixed year is
inserted manually.

Inputs
------
1. Annual ADM2 flood indicators from Script 01
2. Optional trend tables from Script 02
3. Temporal features, pattern membership and change points from Script 03

Outputs
-------
processed_data/cross_dataset_analysis/01_adm2_flood/04_representative_temporal_patterns/

    tables/
        01_representative_pattern_selection.csv
        02_representative_temporal_patterns_summary.csv
        03_representative_annual_trajectories.csv

    figures/
        01_representative_pattern_trajectories_combined.png
        02_representative_pattern_trajectories_panels.png
        03_representative_pattern_change_comparison.png

Methodological role
-------------------
Script 03: discover multivariate temporal flood patterns.
Script 04: illustrate and interpret those patterns with representative annual
           trajectories.

The script is therefore an interpretation/visualization stage rather than a
new clustering or flood-risk classification stage.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path("/Users/shibai/Documents/GitHub/DBL-Group-22")
PROCESSED_ROOT = PROJECT_ROOT / "processed_data" / "cross_dataset_analysis"

ADM2_ROOT = PROCESSED_ROOT / "01_adm2_flood"

ANNUAL_FILE = (
    ADM2_ROOT
    / "indicators"
    / "tables"
    / "adm2_annual_flood_indicators_combined.csv"
)

TREND_TABLE_DIR = ADM2_ROOT / "trends" / "tables"
LONG_TERM_FILE = TREND_TABLE_DIR / "adm2_long_term_hotspot_ranking.csv"
CHANGE_FILE = TREND_TABLE_DIR / "adm2_early_vs_recent_change.csv"
TREND_FILE = TREND_TABLE_DIR / "adm2_trend_analysis.csv"

TEMPORAL_TABLE_DIR = ADM2_ROOT / "temporal_patterns" / "tables"
TEMPORAL_FEATURE_FILE = TEMPORAL_TABLE_DIR / "01_adm2_temporal_features.csv"
PATTERN_MEMBERSHIP_FILE = TEMPORAL_TABLE_DIR / "03_pattern_membership.csv"
CHANGE_POINT_FILE = TEMPORAL_TABLE_DIR / "06_change_points.csv"

OUTPUT_ROOT = ADM2_ROOT / "04_representative_temporal_patterns"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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
# 3. TEMPORAL FEATURES USED BY SCRIPT 03
# =============================================================================

# Keep this list aligned with Script 03. These are shape/change features rather
# than absolute flood-severity indicators.
CLUSTER_FEATURES = [
    "trend_slope",
    "recent_slope",
    "acceleration",
    "change_pp",
    "recent_5y_change_pp",
    "flood_year_fraction",
    "longest_consecutive_run",
    "max_minus_median",
    "post_peak_decline",
    "cv_flood",
]


# =============================================================================
# 4. GENERAL HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"\nRequired file not found:\n{path}")


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def safe_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.max()) if values.notna().any() else np.nan


# =============================================================================
# 5. LOAD INPUT DATA
# =============================================================================

def load_annual_data() -> pd.DataFrame:
    require_file(ANNUAL_FILE)
    df = pd.read_csv(ANNUAL_FILE)

    required = {
        "ADM2_ID",
        "ADM2_NAME",
        "year",
        "annual_flooded_percent",
        "flood_days",
        "flood_event_count",
        "annual_flood_area_days_km2",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing columns from annual indicator file:\n"
            + "\n".join(sorted(missing))
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in [
        "annual_flooded_percent",
        "flood_days",
        "flood_event_count",
        "annual_flood_area_days_km2",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "flood_type" in df.columns:
        df = df[df["flood_type"].astype(str).str.lower().eq("combined")].copy()

    df = df[df["year"].between(STUDY_START_YEAR, STUDY_END_YEAR)].copy()
    return df


def load_script03_outputs():
    for path in [
        TEMPORAL_FEATURE_FILE,
        PATTERN_MEMBERSHIP_FILE,
        CHANGE_POINT_FILE,
    ]:
        require_file(path)

    features = pd.read_csv(TEMPORAL_FEATURE_FILE)
    membership = pd.read_csv(PATTERN_MEMBERSHIP_FILE)
    change_points = pd.read_csv(CHANGE_POINT_FILE)

    required_membership = {"ADM2_ID", "ADM2_NAME", "cluster", "pattern_name"}
    missing = required_membership - set(membership.columns)
    if missing:
        raise ValueError(
            "Missing columns from Script 03 pattern membership:\n"
            + "\n".join(sorted(missing))
        )

    missing_features = set(CLUSTER_FEATURES) - set(features.columns)
    if missing_features:
        raise ValueError(
            "Script 03 temporal-feature table does not contain the expected "
            "clustering features:\n"
            + "\n".join(sorted(missing_features))
        )

    return features, membership, change_points


def load_optional_script02_outputs():
    long_term = pd.read_csv(LONG_TERM_FILE) if LONG_TERM_FILE.exists() else None
    change = pd.read_csv(CHANGE_FILE) if CHANGE_FILE.exists() else None
    trends = pd.read_csv(TREND_FILE) if TREND_FILE.exists() else None
    return long_term, change, trends


# =============================================================================
# 6. SELECT ONE REPRESENTATIVE ADM2 PER SCRIPT 03 PATTERN
# =============================================================================

def select_pattern_representatives(
    features: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Select the ADM2 closest to each pattern centroid.

    The same 10 temporal features used in Script 03 are standardized globally.
    For each pattern, a centroid is calculated in standardized feature space.
    The member with the smallest Euclidean distance to that centroid is chosen
    as the representative. Singleton patterns select their only member.
    """

    print_header("SELECTING REPRESENTATIVE ADM2 AREAS FROM SCRIPT 03 PATTERNS")

    key_cols = ["ADM2_ID", "ADM2_NAME"]
    data = features.merge(
        membership[["ADM2_ID", "ADM2_NAME", "cluster", "pattern_name"]],
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError("No ADM2 records matched between features and membership tables.")

    # Reproduce Script 03 preprocessing exactly before calculating
    # distances in the clustering feature space. Script 03 converts each
    # clustering feature to numeric, replaces missing values with that
    # feature's median (or 0.0 if the median itself is unavailable), and
    # only then applies StandardScaler. Using the same preprocessing here is
    # essential so that representative selection is consistent with the
    # feature space in which the Script 03 clusters were discovered.
    X = data[CLUSTER_FEATURES].copy()

    imputation_records = []

    for col in CLUSTER_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

        missing_count = int(X[col].isna().sum())
        median_value = X[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        if missing_count > 0:
            X[col] = X[col].fillna(median_value)

        imputation_records.append({
            "feature": col,
            "missing_values_imputed": missing_count,
            "imputation_value_median": float(median_value),
        })

    imputation_qc = pd.DataFrame(imputation_records)
    imputation_qc.to_csv(
        TABLE_DIR / "00_representative_selection_imputation_qc.csv",
        index=False,
    )

    imputed = imputation_qc[
        imputation_qc["missing_values_imputed"] > 0
    ]

    if not imputed.empty:
        print("\nScript 03-consistent median imputation applied:")
        print(
            imputed.to_string(
                index=False,
                formatters={
                    "imputation_value_median": lambda x: f"{x:.6f}"
                },
            )
        )
    else:
        print("\nNo missing clustering-feature values required imputation.")

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    zcols = [f"z_{c}" for c in CLUSTER_FEATURES]
    zdf = pd.DataFrame(Z, columns=zcols, index=data.index)
    data = pd.concat([data, zdf], axis=1)

    records = []

    for cluster, group in data.groupby("cluster", sort=True):
        centroid = group[zcols].mean(axis=0).to_numpy(dtype=float)
        matrix = group[zcols].to_numpy(dtype=float)
        distances = np.linalg.norm(matrix - centroid, axis=1)

        chosen_pos = int(np.argmin(distances))
        chosen = group.iloc[chosen_pos]

        records.append({
            "cluster": int(cluster),
            "pattern_name": chosen["pattern_name"],
            "pattern_size": int(len(group)),
            "representative_ADM2_ID": chosen["ADM2_ID"],
            "representative_ADM2_NAME": chosen["ADM2_NAME"],
            "distance_to_pattern_centroid": float(distances[chosen_pos]),
            "selection_method": (
                "only member of singleton pattern"
                if len(group) == 1
                else "minimum Euclidean distance to pattern centroid in standardized 10-feature space"
            ),
        })

    selected = pd.DataFrame(records).sort_values("cluster").reset_index(drop=True)

    out = TABLE_DIR / "01_representative_pattern_selection.csv"
    selected.to_csv(out, index=False)

    print(selected.to_string(index=False))
    print(f"\nSaved:\n{out}")
    return selected


# =============================================================================
# 7. BUILD REPRESENTATIVE SUMMARY
# =============================================================================

def build_representative_summary(
    annual: pd.DataFrame,
    selected: pd.DataFrame,
    change_points: pd.DataFrame,
    long_term: pd.DataFrame | None,
    change: pd.DataFrame | None,
    trends: pd.DataFrame | None,
) -> pd.DataFrame:

    print_header("BUILDING REPRESENTATIVE PATTERN SUMMARY")
    records = []

    for _, sel in selected.iterrows():
        adm2_id = sel["representative_ADM2_ID"]
        adm2_name = sel["representative_ADM2_NAME"]
        subset = annual[annual["ADM2_ID"] == adm2_id].sort_values("year").copy()

        if subset.empty:
            print(f"WARNING: no annual data found for {adm2_name} ({adm2_id}).")
            continue

        early = subset[subset["year"].between(EARLY_START, EARLY_END)]
        recent = subset[subset["year"].between(RECENT_START, RECENT_END)]
        latest = subset[subset["year"].between(LATEST_START, LATEST_END)]

        annual_pct = pd.to_numeric(subset["annual_flooded_percent"], errors="coerce")
        if annual_pct.notna().any():
            max_idx = annual_pct.idxmax()
            max_year = subset.loc[max_idx, "year"]
        else:
            max_year = np.nan

        cp_match = change_points[change_points["ADM2_ID"] == adm2_id]
        cp_year = np.nan
        cp_shift = np.nan
        cp_direction = np.nan
        cp_method = np.nan
        if not cp_match.empty:
            cp = cp_match.iloc[0]
            cp_year = cp.get("change_point_year", np.nan)
            cp_shift = cp.get("mean_shift_pp", np.nan)
            cp_direction = cp.get("change_direction", np.nan)
            cp_method = cp.get("change_point_method", np.nan)

        early_mean = safe_mean(early["annual_flooded_percent"])
        recent_mean = safe_mean(recent["annual_flooded_percent"])

        record = {
            "cluster": int(sel["cluster"]),
            "pattern_name": sel["pattern_name"],
            "pattern_size": int(sel["pattern_size"]),
            "ADM2_ID": adm2_id,
            "ADM2_NAME": adm2_name,
            "distance_to_pattern_centroid": sel["distance_to_pattern_centroid"],
            "long_term_mean_flooded_percent": safe_mean(subset["annual_flooded_percent"]),
            "early_mean_flooded_percent": early_mean,
            "recent_mean_flooded_percent": recent_mean,
            "latest_5y_mean_flooded_percent": safe_mean(latest["annual_flooded_percent"]),
            "recent_minus_early_pp": recent_mean - early_mean,
            "mean_flood_days_per_year": safe_mean(subset["flood_days"]),
            "max_annual_flooded_percent": safe_max(subset["annual_flooded_percent"]),
            "max_flood_year": max_year,
            "detected_change_point_year": cp_year,
            "change_point_mean_shift_pp": cp_shift,
            "change_point_direction": cp_direction,
            "change_point_method": cp_method,
        }

        # Optional Script 02 values are retained as supporting descriptive context.
        if long_term is not None and "ADM2_NAME" in long_term.columns:
            m = long_term[long_term["ADM2_NAME"] == adm2_name]
            if not m.empty and "hotspot_rank" in m.columns:
                record["long_term_hotspot_rank"] = m.iloc[0]["hotspot_rank"]

        if change is not None and "ADM2_NAME" in change.columns:
            m = change[change["ADM2_NAME"] == adm2_name]
            if not m.empty:
                for col in ["early_rank", "recent_rank"]:
                    if col in m.columns:
                        record[col] = m.iloc[0][col]

        if trends is not None and "ADM2_NAME" in trends.columns:
            m = trends[trends["ADM2_NAME"] == adm2_name]
            if not m.empty:
                if "flooded_percent_trend_pp_per_year" in m.columns:
                    record["trend_pp_per_year"] = m.iloc[0]["flooded_percent_trend_pp_per_year"]
                if "flooded_percent_trend_r2" in m.columns:
                    record["trend_r2"] = m.iloc[0]["flooded_percent_trend_r2"]

        records.append(record)

    summary = pd.DataFrame(records).sort_values("cluster").reset_index(drop=True)
    out = TABLE_DIR / "02_representative_temporal_patterns_summary.csv"
    summary.to_csv(out, index=False)

    display = [
        "cluster", "pattern_name", "ADM2_NAME",
        "early_mean_flooded_percent", "recent_mean_flooded_percent",
        "recent_minus_early_pp", "mean_flood_days_per_year",
        "detected_change_point_year",
    ]
    print(summary[[c for c in display if c in summary.columns]].round(2).to_string(index=False))
    print(f"\nSaved:\n{out}")
    return summary


# =============================================================================
# 8. EXPORT REPRESENTATIVE ANNUAL TRAJECTORIES
# =============================================================================

def export_representative_trajectories(
    annual: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for _, row in summary.iterrows():
        sub = annual[annual["ADM2_ID"] == row["ADM2_ID"]].copy()
        sub["cluster"] = row["cluster"]
        sub["pattern_name"] = row["pattern_name"]
        frames.append(sub)

    trajectories = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = TABLE_DIR / "03_representative_annual_trajectories.csv"
    trajectories.to_csv(out, index=False)
    print(f"Saved:\n{out}")
    return trajectories


# =============================================================================
# 9. FIGURE 1: COMBINED REPRESENTATIVE TRAJECTORIES
# =============================================================================

def plot_combined_trajectories(
    annual: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    print_header("GENERATING COMBINED REPRESENTATIVE TRAJECTORY FIGURE")

    fig, ax = plt.subplots(figsize=(13, 8))

    for _, row in summary.iterrows():
        subset = annual[annual["ADM2_ID"] == row["ADM2_ID"]].sort_values("year")
        label = f"P{int(row['cluster'])}: {row['ADM2_NAME']} - {row['pattern_name']}"
        ax.plot(
            subset["year"],
            subset["annual_flooded_percent"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=label,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual flooded area (%)")
    ax.set_title(
        "Representative Trajectories of the Six ADM2 Temporal Flood Patterns\n"
        "Representatives selected from Script 03 pattern membership"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8.5)
    fig.tight_layout()

    out = FIGURE_DIR / "01_representative_pattern_trajectories_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{out}")


# =============================================================================
# 10. FIGURE 2: ONE PANEL PER TEMPORAL PATTERN
# =============================================================================

def plot_pattern_panels(
    annual: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    print_header("GENERATING REPRESENTATIVE PATTERN PANELS")

    n = len(summary)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, (_, row) in zip(axes, summary.iterrows()):
        subset = annual[annual["ADM2_ID"] == row["ADM2_ID"]].sort_values("year")

        ax.plot(
            subset["year"],
            subset["annual_flooded_percent"],
            marker="o",
            linewidth=1.7,
            markersize=4,
        )

        # Draw only a change point detected by Script 03 for this ADM2.
        cp_year = row.get("detected_change_point_year", np.nan)
        if pd.notna(cp_year):
            ax.axvline(float(cp_year), linestyle="--", linewidth=1.0, alpha=0.65)
            ymax = ax.get_ylim()[1]
            ax.text(
                float(cp_year) + 0.15,
                ymax * 0.94,
                f"Detected change point: {int(cp_year)}",
                fontsize=8,
                va="top",
            )

        ax.set_title(
            f"Pattern {int(row['cluster'])}: {row['pattern_name']}\n"
            f"Representative ADM2: {row['ADM2_NAME']}",
            fontsize=10.5,
        )
        ax.set_ylabel("Flooded area (%)")
        ax.grid(alpha=0.25)

        change_pp = row["recent_minus_early_pp"]
        sign = "+" if pd.notna(change_pp) and change_pp >= 0 else ""
        annotation = (
            f"Early (2000-2007): {row['early_mean_flooded_percent']:.2f}%\n"
            f"Recent (2017-2025): {row['recent_mean_flooded_percent']:.2f}%\n"
            f"Change: {sign}{change_pp:.2f} pp\n"
            f"Mean flood days/year: {row['mean_flood_days_per_year']:.1f}"
        )
        ax.text(
            0.02,
            0.96,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.2,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.78,
                "edgecolor": "none",
            },
        )

    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[max(0, n - ncols):n]:
        ax.set_xlabel("Year")

    fig.suptitle(
        "Representative Annual Flood Trajectories for the Six ADM2 Temporal Patterns (2000-2025)\n"
        "Dashed lines indicate ADM2-specific change points detected in Script 03",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()

    out = FIGURE_DIR / "02_representative_pattern_trajectories_panels.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{out}")


# =============================================================================
# 11. FIGURE 3: EARLY VS RECENT CHANGE BY REPRESENTATIVE PATTERN
# =============================================================================

def plot_change_comparison(summary: pd.DataFrame) -> None:
    print_header("GENERATING REPRESENTATIVE CHANGE COMPARISON")

    plot_df = summary.sort_values("cluster").copy()
    labels = [
        f"P{int(c)}\n{name}"
        for c, name in zip(plot_df["cluster"], plot_df["ADM2_NAME"])
    ]

    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2, plot_df["early_mean_flooded_percent"], width, label="Early (2000-2007)")
    ax.bar(x + width / 2, plot_df["recent_mean_flooded_percent"], width, label="Recent (2017-2025)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean annual flooded area (%)")
    ax.set_title(
        "Early-versus-Recent Flood Extent for Representative ADM2 Temporal Patterns\n"
        "Pattern numbers correspond to Script 03"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = FIGURE_DIR / "03_representative_pattern_change_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{out}")


# =============================================================================
# 12. PRINT INTERPRETATION SUMMARY
# =============================================================================

def print_interpretation_summary(summary: pd.DataFrame) -> None:
    print_header("REPRESENTATIVE TEMPORAL PATTERN INTERPRETATION")

    for _, row in summary.iterrows():
        change = row["recent_minus_early_pp"]
        sign = "+" if pd.notna(change) and change >= 0 else ""

        print(f"\nPattern {int(row['cluster'])}: {row['pattern_name']}")
        print(f"  Representative ADM2: {row['ADM2_NAME']}")
        print(f"  Pattern size: {int(row['pattern_size'])} ADM2 areas")
        print(f"  Early mean: {row['early_mean_flooded_percent']:.2f}%")
        print(f"  Recent mean: {row['recent_mean_flooded_percent']:.2f}%")
        print(f"  Early-to-recent change: {sign}{change:.2f} pp")
        print(f"  Mean flood days/year: {row['mean_flood_days_per_year']:.1f}")

        if pd.notna(row.get("detected_change_point_year", np.nan)):
            print(
                "  Script 03 change point: "
                f"{int(row['detected_change_point_year'])} "
                f"({row.get('change_point_direction', '')}, "
                f"mean shift {row.get('change_point_mean_shift_pp', np.nan):.2f} pp)"
            )
        else:
            print("  Script 03 change point: none available")


# =============================================================================
# 13. MAIN
# =============================================================================

def main() -> None:
    print_header("ADM2 REPRESENTATIVE TEMPORAL PATTERN COMPARISON - SCRIPT 04")

    print(f"Annual indicators:\n{ANNUAL_FILE}")
    print(f"\nScript 03 membership:\n{PATTERN_MEMBERSHIP_FILE}")
    print(f"\nOutput root:\n{OUTPUT_ROOT}")

    annual = load_annual_data()
    features, membership, change_points = load_script03_outputs()
    long_term, change, trends = load_optional_script02_outputs()

    selected = select_pattern_representatives(features, membership)

    summary = build_representative_summary(
        annual=annual,
        selected=selected,
        change_points=change_points,
        long_term=long_term,
        change=change,
        trends=trends,
    )

    export_representative_trajectories(annual, summary)

    plot_combined_trajectories(annual, summary)
    plot_pattern_panels(annual, summary)
    plot_change_comparison(summary)

    print_interpretation_summary(summary)

    print_header("FINISHED")
    print("Generated:")
    print("  tables/00_representative_selection_imputation_qc.csv")
    print("  tables/01_representative_pattern_selection.csv")
    print("  tables/02_representative_temporal_patterns_summary.csv")
    print("  tables/03_representative_annual_trajectories.csv")
    print("  figures/01_representative_pattern_trajectories_combined.png")
    print("  figures/02_representative_pattern_trajectories_panels.png")
    print("  figures/03_representative_pattern_change_comparison.png")
    print(f"\nSaved under:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
