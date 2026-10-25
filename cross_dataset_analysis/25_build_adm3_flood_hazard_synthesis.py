#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25_build_adm3_flood_hazard_synthesis_FINAL_v2.py

JBG060 Capstone Data Challenge — South Sudan Flood Analysis

Purpose
-------
Build a final ADM3 flood-hazard synthesis table by integrating the validated
outputs from Scripts 21, 22, and 23.

This script DOES NOT recompute flood observations or derive new flood masks.
It consolidates already-derived indicators into one transparent ADM3-level
hazard screening layer that can later be combined with exposure and
vulnerability indicators.

Workflow
--------
20 — ADM2 recurring/unusual composition
21 — ADM3 recurring/unusual composition
22 — ADM2–ADM3 cross-scale comparison
23 — Cross-scale hotspot robustness / sensitivity analysis
24 — Final ADM2–ADM3 mapping / visualization
25 — Final ADM3 flood-hazard synthesis

Inputs
------
Script 21:
processed_data/cross_dataset_analysis/03_flood_composition/
21_adm3_recurring_unusual_composition/tables/
    21_adm3_flood_mask_composition_summary.csv

Script 22:
processed_data/cross_dataset_analysis/03_flood_composition/
22_adm2_adm3_cross_scale_composition/tables/
    22_adm3_local_cross_scale_signals.csv
    22_adm2_cross_scale_comparison.csv

Script 23:
processed_data/cross_dataset_analysis/03_flood_composition/
23_cross_scale_hotspot_robustness/tables/
    23_adm3_hotspot_sensitivity.csv
    23_robust_local_amplification_hotspots.csv

Outputs
-------
processed_data/cross_dataset_analysis/03_flood_composition/
25_adm3_flood_hazard_synthesis/

    tables/
        25_adm3_flood_hazard_synthesis.csv
        25_adm3_flood_hazard_class_counts.csv
        25_adm2_flood_hazard_summary.csv
        25_high_priority_hazard_screening_adm3.csv

    figures/
        25_01_hazard_screening_class_counts.png
        25_02_top_adm3_final_hazard_score.png
        25_03_recent_extent_vs_change_by_hazard_class.png

    qc/
        25_hazard_synthesis_qc_summary.txt

Interpretation
--------------
This is a FLOOD-HAZARD screening layer only.

It is NOT:
- a humanitarian priority ranking,
- an exposure index,
- a vulnerability index,
- a causal attribution model,
- or a statistical significance test.

Exposure and vulnerability variables should be integrated only in later steps.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = None

ADM3_SUMMARY_CSV = None
LOCAL_SIGNALS_CSV = None
ADM2_CROSS_SCALE_CSV = None
ROBUSTNESS_CSV = None
ROBUST_AMPLIFICATION_CSV = None

EPS = 1e-12

# Screening thresholds are derived from the empirical ADM3 distributions.
# These are descriptive screening thresholds, not externally validated cut-offs.
HIGH_EXTENT_QUANTILE = 0.75
MODERATE_CHANGE_QUANTILE = 0.50
HIGH_CHANGE_QUANTILE = 0.75

# Transparent score weights.
# Total score is rescaled to 0–100.
WEIGHT_EXTENT = 0.30
WEIGHT_CHANGE = 0.25
WEIGHT_AMPLIFICATION = 0.20
WEIGHT_ROBUSTNESS = 0.15
WEIGHT_COMPOSITION = 0.10

TOP_N = 25


# =============================================================================
# PATHS
# =============================================================================

def infer_project_root() -> Path:
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

ADM3_DIR = (
    COMPOSITION_ROOT
    / "21_adm3_recurring_unusual_composition"
    / "tables"
)

CROSS_SCALE_DIR = (
    COMPOSITION_ROOT
    / "22_adm2_adm3_cross_scale_composition"
    / "tables"
)

ROBUSTNESS_DIR = (
    COMPOSITION_ROOT
    / "23_cross_scale_hotspot_robustness"
    / "tables"
)

OUTPUT_ROOT = (
    COMPOSITION_ROOT
    / "25_adm3_flood_hazard_synthesis"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"

for d in (TABLE_DIR, FIG_DIR, QC_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SAFE INPUT DISCOVERY
# =============================================================================

def resolve_unique_file(manual_path, preferred_path, filename):
    if manual_path:
        p = Path(manual_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manual input does not exist: {p}")
        return p

    if preferred_path.exists():
        return preferred_path

    search_roots = [
        COMPOSITION_ROOT,
        ROOT / "processed_data",
        ROOT,
    ]

    candidates = []
    seen = set()

    for base in search_roots:
        if not base.exists():
            continue

        for p in base.rglob(filename):
            rp = p.resolve()
            key = str(rp)
            if key not in seen:
                candidates.append(rp)
                seen.add(key)

    if not candidates:
        return None

    if len(candidates) > 1:
        msg = "\n".join(f"  - {p}" for p in sorted(candidates))
        raise RuntimeError(
            f"Multiple candidate files found for {filename}.\n"
            f"Please set the corresponding manual CONFIG path.\n{msg}"
        )

    return candidates[0]


ADM3_SUMMARY_PATH = resolve_unique_file(
    ADM3_SUMMARY_CSV,
    ADM3_DIR / "21_adm3_flood_mask_composition_summary.csv",
    "21_adm3_flood_mask_composition_summary.csv",
)

LOCAL_SIGNALS_PATH = resolve_unique_file(
    LOCAL_SIGNALS_CSV,
    CROSS_SCALE_DIR / "22_adm3_local_cross_scale_signals.csv",
    "22_adm3_local_cross_scale_signals.csv",
)

ADM2_CROSS_SCALE_PATH = resolve_unique_file(
    ADM2_CROSS_SCALE_CSV,
    CROSS_SCALE_DIR / "22_adm2_cross_scale_comparison.csv",
    "22_adm2_cross_scale_comparison.csv",
)

ROBUSTNESS_PATH = resolve_unique_file(
    ROBUSTNESS_CSV,
    ROBUSTNESS_DIR / "23_adm3_hotspot_sensitivity.csv",
    "23_adm3_hotspot_sensitivity.csv",
)

ROBUST_AMPLIFICATION_PATH = resolve_unique_file(
    ROBUST_AMPLIFICATION_CSV,
    ROBUSTNESS_DIR / "23_robust_local_amplification_hotspots.csv",
    "23_robust_local_amplification_hotspots.csv",
)


# =============================================================================
# HELPERS
# =============================================================================

def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_name(series):
    return (
        series.astype(str)
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )


def safe_percentile(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True, method="average")


def boolean_series(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


# =============================================================================
# LOAD
# =============================================================================

print("=" * 110)
print("25 — FINAL ADM3 FLOOD-HAZARD SYNTHESIS")
print("=" * 110)

print(f"Project root            : {ROOT}")
print(f"ADM3 summary            : {ADM3_SUMMARY_PATH}")
print(f"Script 22 local signals : {LOCAL_SIGNALS_PATH}")
print(f"Script 22 ADM2 summary  : {ADM2_CROSS_SCALE_PATH}")
print(f"Script 23 sensitivity   : {ROBUSTNESS_PATH}")
print(f"Script 23 robust amp.   : {ROBUST_AMPLIFICATION_PATH}")
print(f"Output directory        : {OUTPUT_ROOT}")

required_paths = {
    "Script 21 ADM3 summary": ADM3_SUMMARY_PATH,
    "Script 22 local signals": LOCAL_SIGNALS_PATH,
    "Script 22 ADM2 cross-scale summary": ADM2_CROSS_SCALE_PATH,
    "Script 23 sensitivity table": ROBUSTNESS_PATH,
    "Script 23 robust amplification table": ROBUST_AMPLIFICATION_PATH,
}

for label, path in required_paths.items():
    if path is None:
        raise FileNotFoundError(f"Could not find {label}.")

adm3 = pd.read_csv(ADM3_SUMMARY_PATH)
local22 = pd.read_csv(LOCAL_SIGNALS_PATH)
adm2_cross = pd.read_csv(ADM2_CROSS_SCALE_PATH)
robustness = pd.read_csv(ROBUSTNESS_PATH)
robust_amp = pd.read_csv(ROBUST_AMPLIFICATION_PATH)


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

require_columns(
    adm3,
    [
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_unusual_only_share",
        "recent_recurring_only_share",
        "recent_overlap_share",
        "combined_change_recent_minus_early",
        "recent_composition_class",
    ],
    "Script 21 ADM3 summary",
)

require_columns(
    local22,
    [
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
        "local_amplification_hotspot",
        "change_cross_scale_class",
    ],
    "Script 22 local signal table",
)

require_columns(
    robustness,
    [
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
        "local_amplification_scenario_count",
        "local_amplification_robustness",
        "basic_hotspot_scenario_count",
        "change_hotspot_scenario_count",
    ],
    "Script 23 robustness table",
)

require_columns(
    robust_amp,
    [
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
    ],
    "Script 23 robust amplification table",
)


# =============================================================================
# STANDARDIZE JOIN KEYS
# =============================================================================

for df in (adm3, local22, robustness, robust_amp):
    df["_adm2_key"] = normalize_name(df["ADM2_NAME_STD"])
    df["_adm3_key"] = normalize_name(df["ADM3_NAME_STD"])

join_key = ["_adm2_key", "_adm3_key"]

# Protect against accidental duplication.
for label, df in (
    ("Script 21", adm3),
    ("Script 22 local signals", local22),
    ("Script 23 robustness", robustness),
    ("Script 23 robust amplification", robust_amp),
):
    duplicated = int(df.duplicated(join_key).sum())
    if duplicated:
        raise ValueError(
            f"{label} contains {duplicated} duplicated ADM2/ADM3 join keys."
        )


# =============================================================================
# BUILD MASTER ADM3 SYNTHESIS
# =============================================================================

master = adm3.copy()

local_keep = [
    c for c in [
        *join_key,
        "local_amplification_hotspot",
        "hidden_composition_mismatch",
        "local_unusual_hotspot",
        "local_strong_increase",
        "local_extent_minus_parent_adm2",
        "local_change_minus_parent_adm2",
        "local_signal_percentile_score",
        "composition_cross_scale_class",
        "change_cross_scale_class",
        "adm2_recent_combined_mean",
        "adm2_recent_unusual_only_share",
        "adm2_combined_change",
        "adm2_recent_composition_class",
    ]
    if c in local22.columns
]

master = master.merge(
    local22[local_keep],
    on=join_key,
    how="left",
    validate="one_to_one",
    suffixes=("", "_s22"),
)

robust_keep = [
    c for c in [
        *join_key,
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
    ]
    if c in robustness.columns
]

master = master.merge(
    robustness[robust_keep],
    on=join_key,
    how="left",
    validate="one_to_one",
)

robust_amp_keys = set(
    zip(
        robust_amp["_adm2_key"].astype(str),
        robust_amp["_adm3_key"].astype(str),
    )
)

master["robust_local_amplification_flag"] = [
    (a, b) in robust_amp_keys
    for a, b in zip(master["_adm2_key"], master["_adm3_key"])
]


# =============================================================================
# NUMERIC CLEANING
# =============================================================================

numeric_cols = [
    "recent_combined_mean",
    "recent_unusual_only_share",
    "recent_recurring_only_share",
    "recent_overlap_share",
    "combined_change_recent_minus_early",
    "local_extent_minus_parent_adm2",
    "local_change_minus_parent_adm2",
    "local_signal_percentile_score",
    "basic_hotspot_scenario_count",
    "change_hotspot_scenario_count",
    "local_amplification_scenario_count",
]

for col in numeric_cols:
    if col in master.columns:
        master[col] = pd.to_numeric(master[col], errors="coerce")


# =============================================================================
# EMPIRICAL SCREENING THRESHOLDS
# =============================================================================

valid_extent = master["recent_combined_mean"].dropna()

positive_change = master.loc[
    master["combined_change_recent_minus_early"] > 0,
    "combined_change_recent_minus_early",
].dropna()

high_extent_threshold = float(
    valid_extent.quantile(HIGH_EXTENT_QUANTILE)
) if len(valid_extent) else np.nan

moderate_change_threshold = float(
    positive_change.quantile(MODERATE_CHANGE_QUANTILE)
) if len(positive_change) else np.nan

high_change_threshold = float(
    positive_change.quantile(HIGH_CHANGE_QUANTILE)
) if len(positive_change) else np.nan

master["high_recent_extent_flag"] = (
    master["recent_combined_mean"] >= high_extent_threshold
)

if np.isfinite(moderate_change_threshold):
    master["moderate_recent_increase_flag"] = (
        master["combined_change_recent_minus_early"]
        >= moderate_change_threshold
    )
else:
    master["moderate_recent_increase_flag"] = False

if np.isfinite(high_change_threshold):
    master["strong_recent_increase_flag"] = (
        master["combined_change_recent_minus_early"] >= high_change_threshold
    )
else:
    master["strong_recent_increase_flag"] = False

master["local_amplification_hotspot"] = boolean_series(
    master,
    "local_amplification_hotspot",
)

master["robust_local_amplification_flag"] = (
    master["robust_local_amplification_flag"].fillna(False).astype(bool)
)

master["local_strong_increase"] = boolean_series(
    master,
    "local_strong_increase",
)


# =============================================================================
# TRANSPARENT COMPONENT SCORES
# =============================================================================

# Extent and change are percentile-normalized to 0–1.
master["extent_percentile"] = safe_percentile(
    master["recent_combined_mean"]
)

master["change_percentile"] = safe_percentile(
    master["combined_change_recent_minus_early"]
)

# Amplification uses either Script 22 percentile score when available,
# or a binary fallback from local amplification.
if "local_signal_percentile_score" in master.columns:
    amplification_component = pd.to_numeric(
        master["local_signal_percentile_score"],
        errors="coerce",
    )
else:
    amplification_component = pd.Series(np.nan, index=master.index)

amplification_component = amplification_component.where(
    amplification_component.notna(),
    master["local_amplification_hotspot"].astype(float),
)

master["amplification_component"] = amplification_component.clip(0, 1)

# Robustness component.
robustness_map = {
    "Robust across Q70-Q80": 1.0,
    "Moderately robust": 2 / 3,
    "Threshold-sensitive": 1 / 3,
    "Not identified": 0.0,
}

if "local_amplification_robustness" in master.columns:
    master["robustness_component"] = (
        master["local_amplification_robustness"]
        .map(robustness_map)
        .fillna(0.0)
    )
else:
    master["robustness_component"] = (
        master["robust_local_amplification_flag"].astype(float)
    )

# Composition component is intentionally low-weight because Script 23 showed
# saturation of unusual-only share at the upper quantiles.
composition_component = pd.to_numeric(
    master["recent_unusual_only_share"],
    errors="coerce",
).fillna(0.0)

master["composition_component"] = composition_component.clip(0, 1)


# =============================================================================
# FINAL HAZARD SCORE
# =============================================================================

weight_sum = (
    WEIGHT_EXTENT
    + WEIGHT_CHANGE
    + WEIGHT_AMPLIFICATION
    + WEIGHT_ROBUSTNESS
    + WEIGHT_COMPOSITION
)

if not np.isclose(weight_sum, 1.0):
    raise ValueError(
        f"Hazard-score weights must sum to 1.0, but sum to {weight_sum:.6f}."
    )

master["final_hazard_score_0_100"] = 100 * (
    WEIGHT_EXTENT * master["extent_percentile"].fillna(0)
    + WEIGHT_CHANGE * master["change_percentile"].fillna(0)
    + WEIGHT_AMPLIFICATION * master["amplification_component"].fillna(0)
    + WEIGHT_ROBUSTNESS * master["robustness_component"].fillna(0)
    + WEIGHT_COMPOSITION * master["composition_component"].fillna(0)
)


# =============================================================================
# FINAL HAZARD SCREENING CLASS
# =============================================================================

def classify_hazard(row):
    robust_amp = bool(row["robust_local_amplification_flag"])
    high_extent = bool(row["high_recent_extent_flag"])
    moderate_increase = bool(row["moderate_recent_increase_flag"])
    strong_increase = bool(row["strong_recent_increase_flag"])
    local_amp = bool(row["local_amplification_hotspot"])

    if robust_amp and (high_extent or strong_increase):
        return "Robust high local hazard"

    if robust_amp:
        return "Robust local amplification"

    if local_amp and strong_increase:
        return "Emerging / worsening local hazard"

    if high_extent and strong_increase:
        return "High recent and worsening hazard"

    if high_extent:
        return "Persistent / high recent hazard"

    if moderate_increase:
        return "Moderate increasing hazard"

    return "Stable / lower local hazard"


master["final_hazard_screening_class"] = master.apply(
    classify_hazard,
    axis=1,
)


# =============================================================================
# OUTPUT TABLE
# =============================================================================

preferred_cols = [
    "ADM3_ID_STD",
    "ADM3_NAME_STD",
    "ADM2_ID_STD",
    "ADM2_NAME_STD",
    "recent_combined_mean",
    "early_combined_mean",
    "combined_change_recent_minus_early",
    "recent_recurring_only_share",
    "recent_unusual_only_share",
    "recent_overlap_share",
    "recent_composition_class",
    "local_amplification_hotspot",
    "robust_local_amplification_flag",
    "local_amplification_scenario_count",
    "local_amplification_robustness",
    "local_strong_increase",
    "local_extent_minus_parent_adm2",
    "local_change_minus_parent_adm2",
    "composition_cross_scale_class",
    "change_cross_scale_class",
    "high_recent_extent_flag",
    "moderate_recent_increase_flag",
    "strong_recent_increase_flag",
    "extent_percentile",
    "change_percentile",
    "amplification_component",
    "robustness_component",
    "composition_component",
    "final_hazard_score_0_100",
    "final_hazard_screening_class",
    "qc_flag",
]

output_cols = [c for c in preferred_cols if c in master.columns]

final_table = master[output_cols].copy()

final_table = final_table.sort_values(
    [
        "final_hazard_score_0_100",
        "recent_combined_mean",
    ],
    ascending=False,
)

final_path = TABLE_DIR / "25_adm3_flood_hazard_synthesis.csv"
final_table.to_csv(final_path, index=False)


# =============================================================================
# CLASS COUNTS
# =============================================================================

class_counts = (
    final_table["final_hazard_screening_class"]
    .value_counts(dropna=False)
    .rename_axis("final_hazard_screening_class")
    .reset_index(name="n_adm3")
)

class_counts["share_adm3"] = (
    class_counts["n_adm3"] / len(final_table)
)

class_counts.to_csv(
    TABLE_DIR / "25_adm3_flood_hazard_class_counts.csv",
    index=False,
)


# =============================================================================
# HIGH-PRIORITY HAZARD SCREENING TABLE
# =============================================================================

high_priority_classes = {
    "Robust high local hazard",
    "Robust local amplification",
    "Emerging / worsening local hazard",
    "High recent and worsening hazard",
    "Persistent / high recent hazard",
}

high_priority = final_table[
    final_table["final_hazard_screening_class"].isin(
        high_priority_classes
    )
].copy()

high_priority.to_csv(
    TABLE_DIR / "25_high_priority_hazard_screening_adm3.csv",
    index=False,
)


# =============================================================================
# ADM2 SUMMARY
# =============================================================================

adm2_summary = (
    final_table.groupby("ADM2_NAME_STD", dropna=False)
    .agg(
        n_adm3=("ADM3_NAME_STD", "nunique"),
        mean_final_hazard_score=("final_hazard_score_0_100", "mean"),
        median_final_hazard_score=("final_hazard_score_0_100", "median"),
        max_final_hazard_score=("final_hazard_score_0_100", "max"),
        mean_recent_combined=("recent_combined_mean", "mean"),
        max_recent_combined=("recent_combined_mean", "max"),
        n_robust_high_local_hazard=(
            "final_hazard_screening_class",
            lambda s: int((s == "Robust high local hazard").sum()),
        ),
        n_robust_local_amplification=(
            "robust_local_amplification_flag",
            lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()),
        ),
    )
    .reset_index()
)

adm2_summary = adm2_summary.sort_values(
    [
        "n_robust_high_local_hazard",
        "max_final_hazard_score",
        "mean_final_hazard_score",
    ],
    ascending=False,
)

adm2_summary.to_csv(
    TABLE_DIR / "25_adm2_flood_hazard_summary.csv",
    index=False,
)


# =============================================================================
# FIGURE 1 — CLASS COUNTS
# =============================================================================

plot_counts = class_counts.sort_values("n_adm3")

fig, ax = plt.subplots(
    figsize=(10, max(6, 0.55 * len(plot_counts) + 2))
)

y = np.arange(len(plot_counts))

ax.barh(
    y,
    plot_counts["n_adm3"],
)

ax.set_yticks(y)
ax.set_yticklabels(
    plot_counts["final_hazard_screening_class"],
    fontsize=9,
)

ax.set_xlabel("Number of ADM3 units")
ax.set_title("Final ADM3 Flood-Hazard Screening Classes")
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "25_01_hazard_screening_class_counts.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# FIGURE 2 — TOP ADM3 FINAL HAZARD SCORE
# =============================================================================

top = final_table.head(min(TOP_N, len(final_table))).copy()
top = top.sort_values("final_hazard_score_0_100")

fig, ax = plt.subplots(
    figsize=(11, max(7, 0.38 * len(top) + 2))
)

labels = [
    f"{r.ADM3_NAME_STD} ({r.ADM2_NAME_STD})"
    for r in top.itertuples()
]

y = np.arange(len(top))

ax.barh(
    y,
    top["final_hazard_score_0_100"],
)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Final flood-hazard screening score (0–100)")
ax.set_title(f"Top {len(top)} ADM3 Units by Final Flood-Hazard Screening Score")
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "25_02_top_adm3_final_hazard_score.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# FIGURE 3 — EXTENT VS CHANGE BY FINAL CLASS
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

classes = list(
    final_table["final_hazard_screening_class"]
    .dropna()
    .unique()
)

for cls in classes:
    subset = final_table[
        final_table["final_hazard_screening_class"] == cls
    ]

    ax.scatter(
        subset["recent_combined_mean"],
        subset["combined_change_recent_minus_early"],
        alpha=0.65,
        label=cls,
    )

ax.axhline(0, linewidth=1)
ax.axvline(
    high_extent_threshold,
    linewidth=1,
    linestyle="--",
)

ax.set_xlabel("Recent mean combined flooded area (%)")
ax.set_ylabel("Recent − early combined flood change (percentage points)")
ax.set_title("ADM3 Flood Extent and Change by Final Hazard Screening Class")
ax.legend(fontsize=7)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "25_03_recent_extent_vs_change_by_hazard_class.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# QC
# =============================================================================

matched_local = int(
    master["local_amplification_hotspot"].notna().sum()
)

matched_robust = int(
    master["local_amplification_robustness"].notna().sum()
) if "local_amplification_robustness" in master.columns else 0

robust_count = int(
    master["robust_local_amplification_flag"].sum()
)

qc_text = f"""25 — Final ADM3 Flood-Hazard Synthesis
=======================================

Project root
------------
{ROOT}

Inputs
------
Script 21 ADM3 summary:
{ADM3_SUMMARY_PATH}

Script 22 local signals:
{LOCAL_SIGNALS_PATH}

Script 22 ADM2 cross-scale summary:
{ADM2_CROSS_SCALE_PATH}

Script 23 sensitivity:
{ROBUSTNESS_PATH}

Script 23 robust amplification:
{ROBUST_AMPLIFICATION_PATH}

Coverage
--------
ADM3 rows in Script 21 master          : {len(adm3)}
ADM3 rows in final synthesis           : {len(final_table)}
ADM3 with Script 22 local information  : {matched_local}
ADM3 with Script 23 robustness info    : {matched_robust}
Robust local amplification ADM3        : {robust_count}

Screening thresholds
--------------------
High recent extent quantile            : Q{HIGH_EXTENT_QUANTILE:.2f}
High recent extent threshold (%)       : {high_extent_threshold:.8f}

Moderate recent increase quantile      : Q{MODERATE_CHANGE_QUANTILE:.2f}
Moderate positive-change threshold (pp): {moderate_change_threshold:.8f}

Strong recent increase quantile        : Q{HIGH_CHANGE_QUANTILE:.2f}
Strong positive-change threshold (pp)  : {high_change_threshold:.8f}

Hazard score weights
--------------------
Recent flood extent                    : {WEIGHT_EXTENT:.2f}
Recent change                          : {WEIGHT_CHANGE:.2f}
Local amplification                    : {WEIGHT_AMPLIFICATION:.2f}
Robustness                             : {WEIGHT_ROBUSTNESS:.2f}
Unusual-flood composition              : {WEIGHT_COMPOSITION:.2f}

Final screening class counts
----------------------------
{class_counts.to_string(index=False)}

Interpretation
--------------
1. This is a flood-hazard screening synthesis, not a humanitarian priority index.
2. The score integrates previously derived flood extent, change, local amplification,
   robustness, and flood-composition information.
3. No exposure or vulnerability variables are included here.
4. The unusual-only composition component is intentionally low-weight because
   Script 23 showed saturation of upper unusual-share quantiles at 1.0.
5. Positive changes smaller than the median positive change (Q50) are not
   classified as "increasing hazard" solely because they are mathematically above zero.
6. Q50 is used as a moderate-increase screening threshold and Q75 as a strong-increase
   threshold. These are transparent analytical assumptions, not externally validated
   flood-risk cut-offs.
7. Exposure and vulnerability should be integrated only in subsequent analysis.

DONE.
"""

(QC_DIR / "25_hazard_synthesis_qc_summary.txt").write_text(
    qc_text,
    encoding="utf-8",
)


# =============================================================================
# FINAL CONSOLE OUTPUT
# =============================================================================

print("\n" + "=" * 110)
print("FINAL HAZARD SYNTHESIS")
print("=" * 110)

print(f"ADM3 synthesized                    : {len(final_table):,}")
print(f"Robust local amplification ADM3    : {robust_count:,}")
print(f"High-priority hazard-screening ADM3: {len(high_priority):,}")

print("\nFinal screening classes:")
print(
    class_counts.to_string(
        index=False
    )
)

print("\nTop 15 ADM3 by final hazard score:")
show_cols = [
    c for c in [
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "combined_change_recent_minus_early",
        "robust_local_amplification_flag",
        "final_hazard_score_0_100",
        "final_hazard_screening_class",
    ]
    if c in final_table.columns
]

print(
    final_table[show_cols]
    .head(15)
    .to_string(index=False)
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
    "This is a validated flood-hazard screening layer only. "
    "Do not interpret it as humanitarian priority until exposure "
    "and vulnerability are integrated."
)

print("\nDONE.")
