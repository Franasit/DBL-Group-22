#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
23_test_cross_scale_hotspot_robustness_FINAL.py

JBG060 Capstone Data Challenge — South Sudan Flood Analysis

Purpose
-------
Test whether ADM3 flood hotspots and local amplification signals identified
in the ADM2–ADM3 cross-scale analysis are robust to reasonable changes in the
quantile thresholds used to define operational screening thresholds.

Workflow
--------
20 — ADM2 recurring/unusual composition
21 — ADM3 recurring/unusual composition
22 — ADM2–ADM3 cross-scale comparison
23 — Cross-scale hotspot robustness / sensitivity analysis

This script reads the complete Script 20 and Script 21 composition summaries.
It does NOT derive Q70/Q75/Q80 from an already-filtered Script 22 hotspot table.

Interpretation
--------------
Robustness means stability across the tested operational thresholds.
It is NOT:
- a statistical-significance test,
- a causal attribution test,
- a humanitarian priority ranking,
- or an externally validated flood-severity classification.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = None

QUANTILES = [0.70, 0.75, 0.80]
ROBUST_MIN_SCENARIOS = len(QUANTILES)
EPS = 1e-12


# =============================================================================
# PATHS
# =============================================================================

def infer_project_root() -> Path:
    """Infer DBL-Group-22 project root from script location or current folder."""
    if PROJECT_ROOT:
        return Path(PROJECT_ROOT).expanduser().resolve()

    here = Path(__file__).resolve()

    for p in [here.parent, *here.parents]:
        if (p / "processed_data").exists():
            return p

    cwd = Path.cwd().resolve()

    for p in [cwd, *cwd.parents]:
        if (p / "processed_data").exists():
            return p

    return cwd


ROOT = infer_project_root()

COMPOSITION_ROOT = (
    ROOT
    / "processed_data"
    / "cross_dataset_analysis"
    / "03_flood_composition"
)

ADM2_ROOT = COMPOSITION_ROOT / "20_adm2_recurring_unusual_composition"
ADM3_ROOT = COMPOSITION_ROOT / "21_adm3_recurring_unusual_composition"
CROSS_SCALE_ROOT = COMPOSITION_ROOT / "22_adm2_adm3_cross_scale_composition"

OUTPUT_ROOT = COMPOSITION_ROOT / "23_cross_scale_hotspot_robustness"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"

for directory in [TABLE_DIR, FIG_DIR, QC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_first_existing(base_dirs, filenames):
    """Find expected input, with recursive fallback under processed_data."""

    for base in base_dirs:
        for filename in filenames:
            candidates = [
                base / "tables" / filename,
                base / filename,
            ]

            for candidate in candidates:
                if candidate.exists():
                    return candidate

    processed = ROOT / "processed_data"

    if processed.exists():
        for filename in filenames:
            matches = list(processed.rglob(filename))

            if matches:
                matches = sorted(
                    matches,
                    key=lambda p: (
                        0 if "03_flood_composition" in str(p) else 1,
                        len(str(p)),
                        str(p),
                    ),
                )
                return matches[0]

    return None


ADM2_SUMMARY_PATH = find_first_existing(
    [ADM2_ROOT],
    [
        "20_adm2_flood_mask_composition_summary.csv",
        "adm2_flood_mask_composition_summary.csv",
    ],
)

ADM3_SUMMARY_PATH = find_first_existing(
    [ADM3_ROOT],
    [
        "21_adm3_flood_mask_composition_summary.csv",
        "adm3_flood_mask_composition_summary.csv",
    ],
)


LOCAL_SIGNALS_PATH = find_first_existing(
    [CROSS_SCALE_ROOT],
    [
        "22_adm3_local_cross_scale_signals.csv",
        "22_adm3_local_mismatch_hotspots.csv",
    ],
)


# =============================================================================
# LOAD INPUTS
# =============================================================================

print("=" * 100)
print("23 — CROSS-SCALE LOCAL HOTSPOT ROBUSTNESS / SENSITIVITY ANALYSIS")
print("=" * 100)

print(f"Project root : {ROOT}")
print(f"Output dir   : {OUTPUT_ROOT}")

if ADM2_SUMMARY_PATH is None:
    raise FileNotFoundError(
        "Could not find Script 20 ADM2 composition summary.\n"
        f"Expected under: {ADM2_ROOT}"
    )

if ADM3_SUMMARY_PATH is None:
    raise FileNotFoundError(
        "Could not find Script 21 ADM3 composition summary.\n"
        f"Expected under: {ADM3_ROOT}"
    )

print(f"\nADM2 input   : {ADM2_SUMMARY_PATH}")
print(f"ADM3 input   : {ADM3_SUMMARY_PATH}")
print(f"Script 22 local signals: {LOCAL_SIGNALS_PATH}")

adm2 = pd.read_csv(ADM2_SUMMARY_PATH)
adm3 = pd.read_csv(ADM3_SUMMARY_PATH)
local22 = pd.read_csv(LOCAL_SIGNALS_PATH) if LOCAL_SIGNALS_PATH is not None else None

print(f"\nADM2 rows    : {len(adm2):,}")
print(f"ADM3 rows    : {len(adm3):,}")


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

ADM2_REQUIRED = [
    "ADM2_NAME_STD",
    "recent_combined_mean",
    "recent_unusual_only_share",
    "combined_change_recent_minus_early",
    "recent_composition_class",
]

ADM3_REQUIRED = [
    "ADM3_NAME_STD",
    "ADM2_NAME_STD",
    "recent_combined_mean",
    "recent_unusual_only_share",
    "combined_change_recent_minus_early",
    "recent_composition_class",
]


def check_required_columns(df, required, label):
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns:\n"
            f"{missing}\n\nAvailable columns:\n{list(df.columns)}"
        )


check_required_columns(adm2, ADM2_REQUIRED, "ADM2 summary")
check_required_columns(adm3, ADM3_REQUIRED, "ADM3 summary")


# =============================================================================
# NUMERIC CLEANING
# =============================================================================

numeric_columns = [
    "recent_combined_mean",
    "recent_unusual_only_share",
    "combined_change_recent_minus_early",
]

for col in numeric_columns:
    adm2[col] = pd.to_numeric(adm2[col], errors="coerce")
    adm3[col] = pd.to_numeric(adm3[col], errors="coerce")


# =============================================================================
# INPUT QC
# =============================================================================

print("\n" + "=" * 100)
print("INPUT QUALITY CHECK")
print("=" * 100)

duplicate_adm2 = int(adm2["ADM2_NAME_STD"].duplicated().sum())
duplicate_adm3 = int(
    adm3.duplicated(["ADM2_NAME_STD", "ADM3_NAME_STD"]).sum()
)
missing_parent_name = int(adm3["ADM2_NAME_STD"].isna().sum())

print(f"Unique ADM2 names               : {adm2['ADM2_NAME_STD'].nunique():,}")
print(f"Unique ADM3 parent/name pairs   : {adm3[['ADM2_NAME_STD','ADM3_NAME_STD']].drop_duplicates().shape[0]:,}")
print(f"Duplicate ADM2 names            : {duplicate_adm2:,}")
print(f"Duplicate ADM3 parent/name pairs: {duplicate_adm3:,}")
print(f"ADM3 missing parent name        : {missing_parent_name:,}")

if duplicate_adm2:
    raise ValueError("Duplicate ADM2 names found in Script 20 summary.")

if duplicate_adm3:
    raise ValueError(
        "Duplicate ADM3 parent/name pairs found in Script 21 summary."
    )


# =============================================================================
# ATTACH PARENT ADM2 INFORMATION
# =============================================================================

parent = adm2[
    [
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_unusual_only_share",
        "combined_change_recent_minus_early",
        "recent_composition_class",
    ]
].copy()

parent = parent.rename(
    columns={
        "recent_combined_mean": "parent_adm2_recent_combined_mean",
        "recent_unusual_only_share": "parent_adm2_recent_unusual_only_share",
        "combined_change_recent_minus_early": "parent_adm2_combined_change",
        "recent_composition_class": "parent_adm2_composition_class",
    }
)

work = adm3.merge(
    parent,
    on="ADM2_NAME_STD",
    how="left",
    validate="many_to_one",
)

unmatched_parent = int(work["parent_adm2_composition_class"].isna().sum())

print(f"ADM3 unmatched to Script-20 parent: {unmatched_parent:,}")


# Attach Script 22 local-amplification reference flags when available.
if local22 is not None:
    join_cols = ["ADM2_NAME_STD", "ADM3_NAME_STD"]
    keep_cols = [c for c in (
        "ADM2_NAME_STD", "ADM3_NAME_STD",
        "local_amplification_hotspot",
        "local_signal_percentile_score",
        "local_extent_minus_parent_adm2",
        "local_change_minus_parent_adm2",
    ) if c in local22.columns]
    if all(c in local22.columns for c in join_cols):
        work = work.merge(
            local22[keep_cols].drop_duplicates(join_cols),
            on=join_cols,
            how="left",
            validate="one_to_one",
        )


# =============================================================================
# ANALYSIS POPULATION
# =============================================================================

# Derive thresholds only from ADM3 units with a positive recent combined
# flood signal and a defined unusual-only composition share.
valid = work[
    work["recent_combined_mean"].notna()
    & work["recent_unusual_only_share"].notna()
    & (work["recent_combined_mean"] > EPS)
].copy()

positive_change = valid.loc[
    valid["combined_change_recent_minus_early"] > 0,
    "combined_change_recent_minus_early",
].dropna()

print(f"\nADM3 used for extent/composition quantiles: {len(valid):,}")
print(
    "ADM3 excluded from quantile derivation      : "
    f"{len(work) - len(valid):,}"
)
print(f"ADM3 with positive recent change             : {len(positive_change):,}")


# =============================================================================
# CALCULATE THRESHOLDS
# =============================================================================

threshold_rows = []

for q in QUANTILES:
    extent_threshold = float(valid["recent_combined_mean"].quantile(q))
    unusual_threshold = float(
        valid["recent_unusual_only_share"].quantile(q)
    )

    if len(positive_change) > 0:
        change_threshold = float(positive_change.quantile(q))
    else:
        change_threshold = np.nan

    threshold_rows.append(
        {
            "quantile": q,
            "quantile_label": f"Q{int(round(q * 100))}",
            "extent_threshold_pct": extent_threshold,
            "unusual_share_threshold": unusual_threshold,
            "positive_change_threshold_pp": change_threshold,
        }
    )

thresholds = pd.DataFrame(threshold_rows)

thresholds.to_csv(
    TABLE_DIR / "23_threshold_summary.csv",
    index=False,
)

print("\n" + "=" * 100)
print("THRESHOLDS")
print("=" * 100)
print(
    thresholds.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# =============================================================================
# APPLY SENSITIVITY SCENARIOS
# =============================================================================

scenario_rows = []

for _, threshold in thresholds.iterrows():

    q = float(threshold["quantile"])
    q_label = str(threshold["quantile_label"])

    extent_threshold = float(threshold["extent_threshold_pct"])
    unusual_threshold = float(threshold["unusual_share_threshold"])
    change_threshold = float(threshold["positive_change_threshold_pp"])

    basic_col = f"hotspot_basic_{q_label}"
    change_col = f"hotspot_change_{q_label}"
    hidden_col = f"hidden_local_{q_label}"
    hidden_change_col = f"hidden_local_change_{q_label}"
    amplification_col = f"local_amplification_{q_label}"

    # Rule A:
    # high recent combined flood extent + high unusual-only share.
    work[basic_col] = (
        (work["recent_combined_mean"] >= extent_threshold)
        & (work["recent_unusual_only_share"] >= unusual_threshold)
        & (work["recent_combined_mean"] > EPS)
    )

    # Rule B:
    # Rule A + strong positive early-to-recent change.
    if np.isfinite(change_threshold):
        work[change_col] = (
            work[basic_col]
            & (
                work["combined_change_recent_minus_early"]
                >= change_threshold
            )
        )
    else:
        work[change_col] = False

    # Hidden local signal:
    # local ADM3 passes hotspot rule while parent ADM2 is not
    # Unusual-dominated.
    work[hidden_col] = (
        work[basic_col]
        & (
            work["parent_adm2_composition_class"]
            != "Unusual-dominated"
        )
    )

    work[hidden_change_col] = (
        work[change_col]
        & (
            work["parent_adm2_composition_class"]
            != "Unusual-dominated"
        )
    )

    # Local amplification robustness:
    # identify ADM3 units whose recent extent and/or recent increase is strong
    # relative to the parent ADM2 signal. This does NOT require an ADM2/ADM3
    # composition-class mismatch.
    extent_gap = (
        work["recent_combined_mean"]
        - work["parent_adm2_recent_combined_mean"]
    )
    change_gap = (
        work["combined_change_recent_minus_early"]
        - work["parent_adm2_combined_change"]
    )

    extent_gap_positive = extent_gap[extent_gap > 0].dropna()
    change_gap_positive = change_gap[change_gap > 0].dropna()

    extent_gap_thr = (
        float(extent_gap_positive.quantile(q))
        if len(extent_gap_positive) else np.nan
    )
    change_gap_thr = (
        float(change_gap_positive.quantile(q))
        if len(change_gap_positive) else np.nan
    )

    work[amplification_col] = False

    if np.isfinite(extent_gap_thr):
        work[amplification_col] |= extent_gap >= extent_gap_thr

    if np.isfinite(change_gap_thr):
        work[amplification_col] |= change_gap >= change_gap_thr

    scenario_rows.append(
        {
            "scenario": f"{q_label}_basic",
            "quantile": q,
            "rule": "recent extent + unusual-only share",
            "extent_threshold_pct": extent_threshold,
            "unusual_share_threshold": unusual_threshold,
            "change_threshold_pp": np.nan,
            "n_hotspot_adm3": int(work[basic_col].sum()),
            "n_hidden_local_adm3": int(work[hidden_col].sum()),
            "n_parent_adm2_with_hidden_hotspot": int(
                work.loc[work[hidden_col], "ADM2_NAME_STD"].nunique()
            ),
            "n_local_amplification_adm3": int(work[amplification_col].sum()),
            "n_parent_adm2_with_amplification": int(
                work.loc[work[amplification_col], "ADM2_NAME_STD"].nunique()
            ),
        }
    )

    scenario_rows.append(
        {
            "scenario": f"{q_label}_change",
            "quantile": q,
            "rule": (
                "recent extent + unusual-only share "
                "+ positive-change threshold"
            ),
            "extent_threshold_pct": extent_threshold,
            "unusual_share_threshold": unusual_threshold,
            "change_threshold_pp": change_threshold,
            "n_hotspot_adm3": int(work[change_col].sum()),
            "n_hidden_local_adm3": int(
                work[hidden_change_col].sum()
            ),
            "n_parent_adm2_with_hidden_hotspot": int(
                work.loc[
                    work[hidden_change_col],
                    "ADM2_NAME_STD",
                ].nunique()
            ),
            "n_local_amplification_adm3": int(work[amplification_col].sum()),
            "n_parent_adm2_with_amplification": int(
                work.loc[work[amplification_col], "ADM2_NAME_STD"].nunique()
            ),
        }
    )


scenario_summary = pd.DataFrame(scenario_rows)

scenario_summary.to_csv(
    TABLE_DIR / "23_scenario_summary.csv",
    index=False,
)


# =============================================================================
# ROBUSTNESS COUNTS
# =============================================================================

quantile_labels = [f"Q{int(round(q * 100))}" for q in QUANTILES]

basic_cols = [f"hotspot_basic_{q}" for q in quantile_labels]
change_cols = [f"hotspot_change_{q}" for q in quantile_labels]
hidden_cols = [f"hidden_local_{q}" for q in quantile_labels]
hidden_change_cols = [
    f"hidden_local_change_{q}" for q in quantile_labels
]

amplification_cols = [
    f"local_amplification_{q}" for q in quantile_labels
]

work["basic_hotspot_scenario_count"] = (
    work[basic_cols].sum(axis=1).astype(int)
)

work["change_hotspot_scenario_count"] = (
    work[change_cols].sum(axis=1).astype(int)
)

work["hidden_local_scenario_count"] = (
    work[hidden_cols].sum(axis=1).astype(int)
)

work["hidden_local_change_scenario_count"] = (
    work[hidden_change_cols].sum(axis=1).astype(int)
)


work["local_amplification_scenario_count"] = (
    work[amplification_cols].sum(axis=1).astype(int)
)


def classify_robustness(n):
    if n == len(QUANTILES):
        return "Robust across Q70-Q80"

    if n == len(QUANTILES) - 1:
        return "Moderately robust"

    if n == 1:
        return "Threshold-sensitive"

    return "Not identified"


work["basic_hotspot_robustness"] = work[
    "basic_hotspot_scenario_count"
].map(classify_robustness)

work["change_hotspot_robustness"] = work[
    "change_hotspot_scenario_count"
].map(classify_robustness)

work["hidden_local_robustness"] = work[
    "hidden_local_scenario_count"
].map(classify_robustness)

work["hidden_change_robustness"] = work[
    "hidden_local_change_scenario_count"
].map(classify_robustness)


work["local_amplification_robustness"] = work[
    "local_amplification_scenario_count"
].map(classify_robustness)


# =============================================================================
# COMPLETE ADM3 SENSITIVITY TABLE
# =============================================================================

output_columns = [
    col
    for col in [
        "ADM3_ID_STD",
        "ADM3_NAME_STD",
        "ADM2_ID_STD",
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_unusual_only_share",
        "combined_change_recent_minus_early",
        "recent_composition_class",
        "parent_adm2_recent_combined_mean",
        "parent_adm2_recent_unusual_only_share",
        "parent_adm2_combined_change",
        "parent_adm2_composition_class",
        *basic_cols,
        *change_cols,
        *hidden_cols,
        *hidden_change_cols,
        *amplification_cols,
        "basic_hotspot_scenario_count",
        "change_hotspot_scenario_count",
        "hidden_local_scenario_count",
        "hidden_local_change_scenario_count",
        "local_amplification_scenario_count",
        "basic_hotspot_robustness",
        "change_hotspot_robustness",
        "hidden_local_robustness",
        "hidden_change_robustness",
        "local_amplification_robustness",
        "area_km2",
        "small_area_flag",
        "low_support_flag",
        "qc_flag",
    ]
    if col in work.columns
]

sensitivity = work[output_columns].copy()

sensitivity = sensitivity.sort_values(
    [
        "hidden_local_scenario_count",
        "basic_hotspot_scenario_count",
        "recent_combined_mean",
    ],
    ascending=[False, False, False],
)

sensitivity.to_csv(
    TABLE_DIR / "23_adm3_hotspot_sensitivity.csv",
    index=False,
)


# =============================================================================
# ROBUST LOCAL HOTSPOTS
# =============================================================================

robust = work[
    work["basic_hotspot_scenario_count"] >= ROBUST_MIN_SCENARIOS
].copy()

robust = robust.sort_values(
    [
        "hidden_local_scenario_count",
        "recent_combined_mean",
        "recent_unusual_only_share",
    ],
    ascending=[False, False, False],
)

robust_output_columns = [
    col for col in output_columns if col in robust.columns
]

robust[robust_output_columns].to_csv(
    TABLE_DIR / "23_robust_local_hotspots.csv",
    index=False,
)


robust_amplification = work[
    work["local_amplification_scenario_count"] >= ROBUST_MIN_SCENARIOS
].copy()

robust_amplification = robust_amplification.sort_values(
    [
        "local_amplification_scenario_count",
        "recent_combined_mean",
        "combined_change_recent_minus_early",
    ],
    ascending=[False, False, False],
)

robust_amplification[
    [c for c in output_columns if c in robust_amplification.columns]
].to_csv(
    TABLE_DIR / "23_robust_local_amplification_hotspots.csv",
    index=False,
)


# =============================================================================
# ADM2 HIDDEN-HOTSPOT SUMMARY
# =============================================================================

parent_rows = []

for adm2_name, group in work.groupby("ADM2_NAME_STD", dropna=False):

    row = {
        "ADM2_NAME_STD": adm2_name,
        "n_adm3": int(group["ADM3_NAME_STD"].nunique()),
        "parent_adm2_composition_class": (
            group["parent_adm2_composition_class"].iloc[0]
        ),
        "parent_adm2_recent_combined_mean": (
            group["parent_adm2_recent_combined_mean"].iloc[0]
        ),
        "parent_adm2_recent_unusual_only_share": (
            group["parent_adm2_recent_unusual_only_share"].iloc[0]
        ),
    }

    for q_label in quantile_labels:

        hidden_col = f"hidden_local_{q_label}"
        hidden_change_col = f"hidden_local_change_{q_label}"

        row[f"n_hidden_adm3_{q_label}"] = int(
            group[hidden_col].sum()
        )

        row[f"n_hidden_change_adm3_{q_label}"] = int(
            group[hidden_change_col].sum()
        )

    row["max_hidden_local_scenario_count"] = int(
        group["hidden_local_scenario_count"].max()
    )

    row["n_robust_hidden_adm3"] = int(
        (
            group["hidden_local_scenario_count"]
            >= ROBUST_MIN_SCENARIOS
        ).sum()
    )

    parent_rows.append(row)


parent_summary = pd.DataFrame(parent_rows)

parent_summary = parent_summary.sort_values(
    [
        "n_robust_hidden_adm3",
        "max_hidden_local_scenario_count",
    ],
    ascending=False,
)

parent_summary.to_csv(
    TABLE_DIR / "23_adm2_hidden_hotspot_summary.csv",
    index=False,
)


# =============================================================================
# FIGURE 1 — HOTSPOT COUNTS BY THRESHOLD
# =============================================================================

basic_summary = scenario_summary[
    scenario_summary["rule"] == "recent extent + unusual-only share"
].copy()

fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(basic_summary))

ax.bar(
    x,
    basic_summary["n_hotspot_adm3"],
)

ax.set_xticks(x)

ax.set_xticklabels(
    [f"Q{int(q * 100)}" for q in basic_summary["quantile"]]
)

ax.set_ylabel("Number of ADM3 hotspots")
ax.set_xlabel("Quantile threshold")

ax.set_title(
    "Sensitivity of Local ADM3 Hotspot Counts to Threshold Choice"
)

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    FIG_DIR / "23_01_hotspot_counts_by_threshold.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# FIGURE 2 — HOTSPOT STABILITY MATRIX
# =============================================================================

identified = work[
    work["basic_hotspot_scenario_count"] > 0
].copy()

identified = identified.sort_values(
    [
        "basic_hotspot_scenario_count",
        "recent_combined_mean",
    ],
    ascending=[False, False],
)

# Keep figure readable if many units qualify.
plot_df = identified.head(60).copy()

if len(plot_df) > 0:

    matrix = plot_df[basic_cols].astype(int).to_numpy()

    fig_height = max(7, 0.25 * len(plot_df) + 2)

    fig, ax = plt.subplots(
        figsize=(8, fig_height)
    )

    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(len(basic_cols))
    )

    ax.set_xticklabels(quantile_labels)

    y_labels = [
        f"{row['ADM3_NAME_STD']} ({row['ADM2_NAME_STD']})"
        for _, row in plot_df.iterrows()
    ]

    ax.set_yticks(
        np.arange(len(plot_df))
    )

    ax.set_yticklabels(
        y_labels,
        fontsize=7,
    )

    ax.set_xlabel("Hotspot threshold scenario")

    ax.set_title(
        "ADM3 Hotspot Stability Across Q70-Q80 Thresholds"
    )

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "23_02_hotspot_threshold_stability.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# FIGURE 3 — ROBUST HOTSPOTS IN FEATURE SPACE
# =============================================================================

if len(robust) > 0:

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        robust["recent_combined_mean"],
        robust["recent_unusual_only_share"] * 100,
        alpha=0.72,
    )

    label_df = robust[
        robust["hidden_local_scenario_count"]
        >= ROBUST_MIN_SCENARIOS
    ].copy()

    if len(label_df) == 0:
        label_df = robust.nlargest(
            min(15, len(robust)),
            "recent_combined_mean",
        )

    for _, row in label_df.iterrows():

        ax.annotate(
            str(row["ADM3_NAME_STD"]),
            (
                row["recent_combined_mean"],
                row["recent_unusual_only_share"] * 100,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        "Recent mean combined flooded area (%)"
    )

    ax.set_ylabel(
        "Recent unusual-only share (%)"
    )

    ax.set_title(
        "ADM3 Hotspots Robust Across Q70-Q80 Thresholds"
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        FIG_DIR
        / "23_03_robust_hotspot_extent_vs_unusual_share.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# FIGURE 4 — PARENT ADM2 HIDDEN-SIGNAL STABILITY
# =============================================================================

parent_plot = parent_summary[
    parent_summary["max_hidden_local_scenario_count"] > 0
].copy()

parent_plot = parent_plot.head(30)

if len(parent_plot) > 0:

    parent_plot = parent_plot.sort_values(
        "max_hidden_local_scenario_count"
    )

    fig_height = max(
        6,
        0.38 * len(parent_plot) + 2,
    )

    fig, ax = plt.subplots(
        figsize=(10, fig_height)
    )

    y = np.arange(len(parent_plot))

    ax.barh(
        y,
        parent_plot["max_hidden_local_scenario_count"],
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        parent_plot["ADM2_NAME_STD"],
        fontsize=8,
    )

    ax.set_xlabel(
        "Maximum number of Q70-Q80 scenarios supporting a hidden local signal"
    )

    ax.set_title(
        "Parent ADM2 Areas Containing Threshold-Sensitive Local Hotspots"
    )

    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "23_04_parent_adm2_hidden_hotspot_counts.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# QC SUMMARY
# =============================================================================

n_robust = int(
    (
        work["basic_hotspot_scenario_count"]
        >= ROBUST_MIN_SCENARIOS
    ).sum()
)

n_robust_hidden = int(
    (
        work["hidden_local_scenario_count"]
        >= ROBUST_MIN_SCENARIOS
    ).sum()
)

n_robust_change = int(
    (
        work["change_hotspot_scenario_count"]
        >= ROBUST_MIN_SCENARIOS
    ).sum()
)

n_robust_hidden_change = int(
    (
        work["hidden_local_change_scenario_count"]
        >= ROBUST_MIN_SCENARIOS
    ).sum()
)


n_robust_amplification = int(
    (
        work["local_amplification_scenario_count"]
        >= ROBUST_MIN_SCENARIOS
    ).sum()
)

robust_hidden_details = work.loc[
    work["hidden_local_scenario_count"] >= ROBUST_MIN_SCENARIOS,
    [
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_unusual_only_share",
        "combined_change_recent_minus_early",
        "parent_adm2_composition_class",
        "hidden_local_scenario_count",
    ],
].copy()

if len(robust_hidden_details) > 0:
    robust_hidden_text = robust_hidden_details.to_string(index=False)
else:
    robust_hidden_text = "None"


qc_text = f"""
23 — Cross-Scale Local Hotspot Robustness Analysis
==================================================

Project root
------------
{ROOT}

Input ADM2 summary
------------------
{ADM2_SUMMARY_PATH}

Input ADM3 summary
------------------
{ADM3_SUMMARY_PATH}

Purpose
-------
Test whether local ADM3 unusual-flood hotspot identification is stable
across reasonable operational quantile thresholds.

Quantiles tested
----------------
{QUANTILES}

Input coverage
--------------
ADM2 records                         : {len(adm2)}
ADM3 records                         : {len(adm3)}
Duplicate ADM2 names                 : {duplicate_adm2}
Duplicate ADM3 parent/name pairs     : {duplicate_adm3}
ADM3 missing parent name             : {missing_parent_name}
ADM3 unmatched to Script-20 parent   : {unmatched_parent}

Threshold population
--------------------
ADM3 used for extent/composition quantiles : {len(valid)}
ADM3 excluded from threshold derivation    : {len(work) - len(valid)}
ADM3 with positive recent change           : {len(positive_change)}

Thresholds
----------
{thresholds.to_string(index=False)}

Scenario results
----------------
{scenario_summary.to_string(index=False)}

Robustness results
------------------
Robust basic hotspots across Q70/Q75/Q80        : {n_robust}
Robust change-qualified hotspots                : {n_robust_change}
Robust hidden local hotspots                    : {n_robust_hidden}
Robust hidden + change-qualified local hotspots : {n_robust_hidden_change}
Robust local amplification hotspots               : {n_robust_amplification}

Robust hidden local hotspot details
-----------------------------------
{robust_hidden_text}

Interpretation
--------------
A basic local hotspot satisfies both:
1. recent combined flooded extent >= the selected quantile threshold; and
2. recent unusual-only share >= the selected quantile threshold.

A change-qualified hotspot additionally requires positive early-to-recent
combined flood change >= the corresponding quantile among positive changes.

A hidden local hotspot is a qualifying ADM3 whose parent ADM2 is NOT
classified as Unusual-dominated.

"Robust across Q70-Q80" means that the ADM3 satisfies the relevant rule
under all three tested threshold scenarios.

This is a threshold-sensitivity analysis. It is not a statistical
significance test, causal attribution, flood-severity ranking, or
humanitarian priority classification.

Methodological note
-------------------
Thresholds are recalculated from the complete eligible ADM3 population.
They are not calculated from the already-filtered Script 22 hotspot table.
This avoids conditioning the sensitivity test on the original Q75 result.

DONE.
""".strip()

(QC_DIR / "23_hotspot_robustness_qc_summary.txt").write_text(
    qc_text,
    encoding="utf-8",
)


# =============================================================================
# FINAL CONSOLE OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("SCENARIO SUMMARY")
print("=" * 100)

print(
    scenario_summary[
        [
            "scenario",
            "n_hotspot_adm3",
            "n_hidden_local_adm3",
            "n_parent_adm2_with_hidden_hotspot",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 100)
print("ROBUSTNESS SUMMARY")
print("=" * 100)

print(
    f"Robust basic ADM3 hotspots      : {n_robust:,}"
)

print(
    f"Robust change-qualified hotspots: {n_robust_change:,}"
)

print(
    f"Robust hidden local hotspots    : {n_robust_hidden:,}"
)

print(
    "Robust hidden + change hotspots: "
    f"{n_robust_hidden_change:,}"
)

print(
    f"Robust local amplification hotspots: {n_robust_amplification:,}"
)

if len(robust_hidden_details) > 0:

    print("\nRobust hidden local hotspots:")

    print(
        robust_hidden_details.to_string(
            index=False
        )
    )

else:

    print(
        "\nNo hidden local hotspot remained "
        "identified across all Q70/Q75/Q80 scenarios."
    )


print("\nSaved tables:")

for path in sorted(TABLE_DIR.glob("*.csv")):
    print(f"  {path.name}")


print("\nSaved figures:")

for path in sorted(FIG_DIR.glob("*.png")):
    print(f"  {path.name}")


print("\nSaved QC:")

for path in sorted(QC_DIR.glob("*")):
    print(f"  {path.name}")


print("\nInterpretation reminder:")
print(
    "Robustness means stability across the tested "
    "operational thresholds only."
)
print(
    "It is not statistical significance and is not "
    "a humanitarian priority ranking."
)

print("\nDONE.")
