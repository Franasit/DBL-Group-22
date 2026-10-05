#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_compare_adm2_adm3_recurring_unusual_composition_FINAL.py

JBG060 Capstone Data Challenge — South Sudan Flood Analysis

Purpose
-------
Compare Script 20 ADM2 flood-composition results with Script 21 ADM3 results to:
1. quantify cross-scale agreement in recent flood composition;
2. identify local ADM3 flood signals that may be masked by ADM2 aggregation;
3. distinguish composition mismatch from intensity amplification;
4. classify cross-scale recent-vs-early change patterns;
5. provide clean tables for the mapping stage (Script 23).

Interpretation rules
--------------------
- Official ADM2 metrics from Script 20 remain the ADM2-scale estimates.
- ADM3 arithmetic means describe the distribution of local administrative units.
- Area-weighted ADM3 summaries are diagnostics only; they are not assumed to
  reconstruct official ADM2 percentages exactly.
- "Composition mismatch" refers to disagreement in recurring/unusual structure.
- "Local amplification" refers to an ADM3 showing substantially higher flood
  intensity or worsening than its parent ADM2, even if both scales share the
  same broad composition class.

Time windows inherited from Scripts 20 and 21
---------------------------------------------
Early  = 2000–2019
Recent = 2020–2025

Expected inputs
---------------
processed_data/cross_dataset_analysis/03_flood_composition/
    20_adm2_recurring_unusual_composition/tables/
        20_adm2_flood_mask_composition_summary.csv

    21_adm3_recurring_unusual_composition/tables/
        21_adm3_flood_mask_composition_summary.csv

Main outputs
------------
22_adm2_adm3_cross_scale_composition/
    tables/
        22_adm2_cross_scale_comparison.csv
        22_adm3_local_cross_scale_signals.csv
        22_adm2_composition_class_counts.csv
        22_adm2_change_class_counts.csv
        22_adm2_hidden_local_signal_summary.csv
        22_adm2_local_amplification_summary.csv
    figures/
        22_01_adm2_vs_area_weighted_adm3_recent_extent.png
        22_02_adm2_vs_adm3_unusual_share.png
        22_03_within_adm2_unusual_share_heterogeneity.png
        22_04_local_amplification_hotspots.png
        22_05_cross_scale_change_comparison.png
        22_06_cross_scale_change_class_counts.png
    qc/
        22_cross_scale_qc_summary.txt
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
ADM2_SUMMARY_CSV = None
ADM3_SUMMARY_CSV = None

EARLY_START = 2000
EARLY_END = 2019
RECENT_START = 2020
RECENT_END = 2025

TOP_N = 25
EPS = 1e-12

# Composition diagnostics.
LOCAL_EXTENT_QUANTILE = 0.75
LOCAL_UNUSUAL_SHARE_QUANTILE = 0.75
LOCAL_CHANGE_QUANTILE = 0.75

# Parent-relative local amplification thresholds.
# These are descriptive screening thresholds, not humanitarian cut-offs.
LOCAL_EXTENT_GAP_QUANTILE = 0.75
LOCAL_CHANGE_GAP_QUANTILE = 0.75

# Cross-scale change classification.
# Changes are in percentage points (recent - early).
CHANGE_NEUTRAL_TOL = 0.50


# =============================================================================
# PATHS
# =============================================================================

def infer_project_root() -> Path:
    if PROJECT_ROOT:
        return Path(PROJECT_ROOT).expanduser().resolve()

    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "processed_data").exists() and (p / "scripts").exists():
            return p

    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "processed_data").exists():
            return p

    return cwd


ROOT = infer_project_root()
COMPOSITION_ROOT = (
    ROOT / "processed_data" / "cross_dataset_analysis" / "03_flood_composition"
)

ADM2_DIR = COMPOSITION_ROOT / "20_adm2_recurring_unusual_composition" / "tables"
ADM3_DIR = COMPOSITION_ROOT / "21_adm3_recurring_unusual_composition" / "tables"

OUTPUT_ROOT = COMPOSITION_ROOT / "22_adm2_adm3_cross_scale_composition"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"

for d in (TABLE_DIR, FIG_DIR, QC_DIR):
    d.mkdir(parents=True, exist_ok=True)


def resolve_input(manual_path, preferred_path: Path, filename: str) -> Path:
    """
    Safe input resolution.

    1. Use manual path if provided.
    2. Use exact preferred path if it exists.
    3. Search composition root, then processed_data.
    4. If multiple candidates exist, stop rather than silently selecting an old file.
    """
    if manual_path:
        p = Path(manual_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manual input does not exist: {p}")
        return p

    if preferred_path.exists():
        return preferred_path

    search_roots = [COMPOSITION_ROOT, ROOT / "processed_data"]
    candidates = []
    seen = set()

    for base in search_roots:
        if not base.exists():
            continue
        for p in base.rglob(filename):
            rp = p.resolve()
            if rp not in seen:
                candidates.append(rp)
                seen.add(rp)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find required input file: {filename}\n"
            f"Expected preferred path: {preferred_path}"
        )

    if len(candidates) > 1:
        joined = "\n".join(f"  - {p}" for p in sorted(candidates))
        raise RuntimeError(
            f"Multiple candidate files found for {filename}.\n"
            "Please set the corresponding manual path in CONFIG.\n"
            f"Candidates:\n{joined}"
        )

    return candidates[0]


ADM2_PATH = resolve_input(
    ADM2_SUMMARY_CSV,
    ADM2_DIR / "20_adm2_flood_mask_composition_summary.csv",
    "20_adm2_flood_mask_composition_summary.csv",
)

ADM3_PATH = resolve_input(
    ADM3_SUMMARY_CSV,
    ADM3_DIR / "21_adm3_flood_mask_composition_summary.csv",
    "21_adm3_flood_mask_composition_summary.csv",
)


# =============================================================================
# HELPERS
# =============================================================================

def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def weighted_mean(values, weights):
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def safe_corr(x, y, method="pearson"):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan
    return float(tmp["x"].corr(tmp["y"], method=method))


def q(series, quantile):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.quantile(quantile)) if len(s) else np.nan


def clean_name(series):
    return series.astype(str).str.strip().str.casefold()


def direction(change, tol=CHANGE_NEUTRAL_TOL):
    if pd.isna(change):
        return "Unknown"
    if change > tol:
        return "Increase"
    if change < -tol:
        return "Decline"
    return "Neutral"


def classify_composition(row):
    adm2_cls = row.get("adm2_recent_composition_class", "")
    share_unusual = row.get("share_adm3_unusual_dominated", np.nan)
    n_hidden = row.get("n_hidden_composition_mismatch", 0)
    diversity = row.get("adm3_class_diversity", 0)

    if adm2_cls == "Unusual-dominated":
        if pd.notna(share_unusual) and share_unusual >= 0.50:
            return "Cross-scale unusual agreement"
        return "ADM2 unusual / locally heterogeneous"

    if n_hidden > 0:
        return "Hidden local unusual composition signal"

    if diversity >= 3:
        return "High within-ADM2 composition heterogeneity"

    return "Broad composition consistency"


def classify_change(row):
    parent_dir = row.get("adm2_change_direction", "Unknown")
    local_worsening = row.get("n_adm3_local_worsening", 0)
    local_improving = row.get("n_adm3_local_improving", 0)
    share_worsening = row.get("share_adm3_local_worsening", np.nan)
    share_improving = row.get("share_adm3_local_improving", np.nan)
    n_amplified = row.get("n_adm3_local_amplification", 0)

    if parent_dir == "Increase":
        if pd.notna(share_worsening) and share_worsening >= 0.50:
            return "Regional increase + Local worsening"
        if n_amplified > 0:
            return "Regional increase + Local hotspot"
        return "Regional increase + Mixed local response"

    if parent_dir == "Neutral":
        if n_amplified > 0 or local_worsening > 0:
            return "Regional neutral + Local hotspot"
        return "Broad cross-scale stability"

    if parent_dir == "Decline":
        if local_worsening > 0:
            return "Regional decline + Local worsening"
        if pd.notna(share_improving) and share_improving >= 0.50:
            return "Regional decline + Local improving"
        return "Regional decline + Mixed local response"

    return "Unclassified change pattern"


# =============================================================================
# LOAD + VALIDATE
# =============================================================================

print("=" * 110)
print("22 — ADM2–ADM3 CROSS-SCALE FLOOD COMPOSITION AND CHANGE COMPARISON")
print("=" * 110)
print(f"Project root : {ROOT}")
print(f"ADM2 input   : {ADM2_PATH}")
print(f"ADM3 input   : {ADM3_PATH}")
print(f"Output dir   : {OUTPUT_ROOT}")
print(f"Early window : {EARLY_START}-{EARLY_END}")
print(f"Recent window: {RECENT_START}-{RECENT_END}\n")

adm2 = pd.read_csv(ADM2_PATH)
adm3 = pd.read_csv(ADM3_PATH)

require_columns(
    adm2,
    [
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_recurring_only_share",
        "recent_unusual_only_share",
        "recent_overlap_share",
        "combined_change_recent_minus_early",
        "unusual_change_recent_minus_early",
        "recent_composition_class",
    ],
    "Script 20 ADM2 summary",
)

require_columns(
    adm3,
    [
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_recurring_only_share",
        "recent_unusual_only_share",
        "recent_overlap_share",
        "combined_change_recent_minus_early",
        "unusual_change_recent_minus_early",
        "recent_composition_class",
    ],
    "Script 21 ADM3 summary",
)

metric_cols = [
    "recent_combined_mean",
    "recent_recurring_mean",
    "recent_unusual_mean",
    "recent_recurring_only_mean",
    "recent_unusual_only_mean",
    "recent_overlap_mean",
    "recent_recurring_only_share",
    "recent_unusual_only_share",
    "recent_overlap_share",
    "combined_change_recent_minus_early",
    "recurring_change_recent_minus_early",
    "unusual_change_recent_minus_early",
    "recent_composition_entropy",
    "area_km2",
]

adm2 = numeric(adm2, metric_cols)
adm3 = numeric(adm3, metric_cols)

# Parent join key.
use_id = "ADM2_ID_STD" in adm2.columns and "ADM2_ID_STD" in adm3.columns
if use_id:
    adm2["_parent_key"] = adm2["ADM2_ID_STD"].astype(str).str.strip()
    adm3["_parent_key"] = adm3["ADM2_ID_STD"].astype(str).str.strip()
    key_label = "ADM2_ID_STD"
else:
    adm2["_parent_key"] = clean_name(adm2["ADM2_NAME_STD"])
    adm3["_parent_key"] = clean_name(adm3["ADM2_NAME_STD"])
    key_label = "normalized ADM2_NAME_STD"

print(f"Cross-scale parent key: {key_label}")
print(f"ADM2 rows: {len(adm2):,} | unique parents: {adm2['_parent_key'].nunique():,}")
print(f"ADM3 rows: {len(adm3):,} | ADM3 units: {adm3['ADM3_NAME_STD'].nunique():,}")

adm2_keys = set(adm2["_parent_key"].dropna())
adm3_keys = set(adm3["_parent_key"].dropna())
missing_parent_keys = sorted(adm3_keys - adm2_keys)
missing_adm3_parents = sorted(adm2_keys - adm3_keys)

if missing_parent_keys:
    print(f"WARNING: {len(missing_parent_keys)} ADM3 parent keys are absent from ADM2 summary.")
if missing_adm3_parents:
    print(f"WARNING: {len(missing_adm3_parents)} ADM2 units have no ADM3 children in summary.")


# =============================================================================
# DATA-DRIVEN ADM3 THRESHOLDS
# =============================================================================

extent_thr = q(adm3["recent_combined_mean"], LOCAL_EXTENT_QUANTILE)
unusual_share_thr = q(adm3["recent_unusual_only_share"], LOCAL_UNUSUAL_SHARE_QUANTILE)
positive_change = adm3.loc[
    adm3["combined_change_recent_minus_early"] > 0,
    "combined_change_recent_minus_early",
]
change_thr = q(positive_change, LOCAL_CHANGE_QUANTILE)

adm3["local_high_extent"] = adm3["recent_combined_mean"] >= extent_thr
adm3["local_high_unusual_share"] = adm3["recent_unusual_only_share"] >= unusual_share_thr
adm3["local_strong_increase"] = adm3["combined_change_recent_minus_early"] >= change_thr
adm3["local_unusual_hotspot"] = (
    adm3["local_high_extent"] & adm3["local_high_unusual_share"]
)

print("\nData-driven ADM3 thresholds:")
print(f"  high recent combined extent >= {extent_thr:.4f}")
print(f"  high unusual-only share     >= {unusual_share_thr:.4f}")
print(f"  strong positive change      >= {change_thr:.4f}")


# =============================================================================
# AGGREGATE ADM3 WITHIN ADM2
# =============================================================================

agg_rows = []
for parent_key, g in adm3.groupby("_parent_key", dropna=False):
    local_dirs = g["combined_change_recent_minus_early"].apply(direction)

    row = {
        "_parent_key": parent_key,
        "n_adm3": int(g["ADM3_NAME_STD"].nunique()),
        "adm3_mean_recent_combined": float(g["recent_combined_mean"].mean()),
        "adm3_median_recent_combined": float(g["recent_combined_mean"].median()),
        "adm3_max_recent_combined": float(g["recent_combined_mean"].max()),
        "adm3_std_recent_combined": float(g["recent_combined_mean"].std(ddof=0)),
        "adm3_range_recent_combined": float(
            g["recent_combined_mean"].max() - g["recent_combined_mean"].min()
        ),
        "adm3_mean_unusual_only_share": float(g["recent_unusual_only_share"].mean()),
        "adm3_median_unusual_only_share": float(g["recent_unusual_only_share"].median()),
        "adm3_max_unusual_only_share": float(g["recent_unusual_only_share"].max()),
        "adm3_std_unusual_only_share": float(g["recent_unusual_only_share"].std(ddof=0)),
        "adm3_mean_recurring_only_share": float(g["recent_recurring_only_share"].mean()),
        "adm3_mean_overlap_share": float(g["recent_overlap_share"].mean()),
        "adm3_mean_combined_change": float(g["combined_change_recent_minus_early"].mean()),
        "adm3_max_combined_change": float(g["combined_change_recent_minus_early"].max()),
        "adm3_min_combined_change": float(g["combined_change_recent_minus_early"].min()),
        "adm3_std_combined_change": float(g["combined_change_recent_minus_early"].std(ddof=0)),
        "adm3_mean_unusual_change": float(g["unusual_change_recent_minus_early"].mean()),
        "adm3_max_unusual_change": float(g["unusual_change_recent_minus_early"].max()),
        "adm3_class_diversity": int(g["recent_composition_class"].nunique()),
        "n_adm3_unusual_dominated": int(
            (g["recent_composition_class"] == "Unusual-dominated").sum()
        ),
        "share_adm3_unusual_dominated": float(
            (g["recent_composition_class"] == "Unusual-dominated").mean()
        ),
        "n_adm3_local_unusual_hotspot_raw": int(g["local_unusual_hotspot"].sum()),
        "share_adm3_local_unusual_hotspot_raw": float(g["local_unusual_hotspot"].mean()),
        "n_adm3_strong_increase": int(g["local_strong_increase"].sum()),
        "n_adm3_local_worsening": int((local_dirs == "Increase").sum()),
        "n_adm3_local_improving": int((local_dirs == "Decline").sum()),
        "n_adm3_local_neutral": int((local_dirs == "Neutral").sum()),
        "share_adm3_local_worsening": float((local_dirs == "Increase").mean()),
        "share_adm3_local_improving": float((local_dirs == "Decline").mean()),
        "share_adm3_local_neutral": float((local_dirs == "Neutral").mean()),
    }

    if "area_km2" in g.columns and g["area_km2"].notna().any():
        row.update({
            "adm3_area_weighted_recent_combined": weighted_mean(
                g["recent_combined_mean"], g["area_km2"]
            ),
            "adm3_area_weighted_unusual_only_share": weighted_mean(
                g["recent_unusual_only_share"], g["area_km2"]
            ),
            "adm3_area_weighted_recurring_only_share": weighted_mean(
                g["recent_recurring_only_share"], g["area_km2"]
            ),
            "adm3_area_weighted_overlap_share": weighted_mean(
                g["recent_overlap_share"], g["area_km2"]
            ),
            "adm3_area_weighted_combined_change": weighted_mean(
                g["combined_change_recent_minus_early"], g["area_km2"]
            ),
            "adm3_area_weighted_unusual_change": weighted_mean(
                g["unusual_change_recent_minus_early"], g["area_km2"]
            ),
        })
    else:
        row.update({
            "adm3_area_weighted_recent_combined": np.nan,
            "adm3_area_weighted_unusual_only_share": np.nan,
            "adm3_area_weighted_recurring_only_share": np.nan,
            "adm3_area_weighted_overlap_share": np.nan,
            "adm3_area_weighted_combined_change": np.nan,
            "adm3_area_weighted_unusual_change": np.nan,
        })

    agg_rows.append(row)

adm3_parent = pd.DataFrame(agg_rows)


# =============================================================================
# MERGE ADM2 + ADM3 PARENT SUMMARIES
# =============================================================================

adm2_keep = [
    c for c in [
        "_parent_key",
        "ADM2_ID_STD",
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_recurring_only_share",
        "recent_unusual_only_share",
        "recent_overlap_share",
        "combined_change_recent_minus_early",
        "recurring_change_recent_minus_early",
        "unusual_change_recent_minus_early",
        "recent_composition_entropy",
        "recent_composition_class",
    ] if c in adm2.columns
]

adm2_base = adm2[adm2_keep].copy().rename(columns={
    "recent_combined_mean": "adm2_recent_combined_mean",
    "recent_recurring_only_share": "adm2_recent_recurring_only_share",
    "recent_unusual_only_share": "adm2_recent_unusual_only_share",
    "recent_overlap_share": "adm2_recent_overlap_share",
    "combined_change_recent_minus_early": "adm2_combined_change",
    "recurring_change_recent_minus_early": "adm2_recurring_change",
    "unusual_change_recent_minus_early": "adm2_unusual_change",
    "recent_composition_entropy": "adm2_composition_entropy",
    "recent_composition_class": "adm2_recent_composition_class",
})

cross = adm2_base.merge(
    adm3_parent,
    on="_parent_key",
    how="left",
    validate="one_to_one",
)

cross["adm2_change_direction"] = cross["adm2_combined_change"].apply(direction)

cross["extent_gap_areaweighted_adm3_minus_adm2"] = (
    cross["adm3_area_weighted_recent_combined"] - cross["adm2_recent_combined_mean"]
)
cross["unusual_share_gap_areaweighted_adm3_minus_adm2"] = (
    cross["adm3_area_weighted_unusual_only_share"] - cross["adm2_recent_unusual_only_share"]
)
cross["change_gap_areaweighted_adm3_minus_adm2"] = (
    cross["adm3_area_weighted_combined_change"] - cross["adm2_combined_change"]
)


# =============================================================================
# LOCAL ADM3 CONTEXT + PARENT-RELATIVE AMPLIFICATION
# =============================================================================

parent_context_cols = [
    "_parent_key",
    "ADM2_NAME_STD",
    "adm2_recent_combined_mean",
    "adm2_recent_unusual_only_share",
    "adm2_combined_change",
    "adm2_change_direction",
    "adm2_recent_composition_class",
]

local = adm3.merge(
    cross[parent_context_cols],
    on="_parent_key",
    how="left",
    validate="many_to_one",
    suffixes=("", "_parent"),
)

local["local_extent_minus_parent_adm2"] = (
    local["recent_combined_mean"] - local["adm2_recent_combined_mean"]
)
local["local_unusual_share_minus_parent_adm2"] = (
    local["recent_unusual_only_share"] - local["adm2_recent_unusual_only_share"]
)
local["local_change_minus_parent_adm2"] = (
    local["combined_change_recent_minus_early"] - local["adm2_combined_change"]
)

extent_gap_thr = q(
    local.loc[local["local_extent_minus_parent_adm2"] > 0, "local_extent_minus_parent_adm2"],
    LOCAL_EXTENT_GAP_QUANTILE,
)
change_gap_thr = q(
    local.loc[local["local_change_minus_parent_adm2"] > 0, "local_change_minus_parent_adm2"],
    LOCAL_CHANGE_GAP_QUANTILE,
)

local["hidden_composition_mismatch"] = (
    local["local_unusual_hotspot"]
    & (local["adm2_recent_composition_class"] != "Unusual-dominated")
)

local["local_extent_amplification"] = (
    local["local_extent_minus_parent_adm2"] >= extent_gap_thr
)
local["local_change_amplification"] = (
    local["local_change_minus_parent_adm2"] >= change_gap_thr
)

# Main local amplification flag: local extent is strongly above the parent and
# either recent flood extent is high or local change is strongly positive.
local["local_amplification_hotspot"] = (
    local["local_extent_amplification"]
    & (local["local_high_extent"] | local["local_change_amplification"])
)

local["local_change_direction"] = local["combined_change_recent_minus_early"].apply(direction)

# Descriptive ranking only; not used for definition.
rank_components = pd.DataFrame(index=local.index)
rank_components["extent_pctile"] = local["recent_combined_mean"].rank(pct=True)
rank_components["unusual_pctile"] = local["recent_unusual_only_share"].rank(pct=True)
rank_components["change_pctile"] = local["combined_change_recent_minus_early"].rank(pct=True)
rank_components["extent_gap_pctile"] = local["local_extent_minus_parent_adm2"].rank(pct=True)
rank_components["change_gap_pctile"] = local["local_change_minus_parent_adm2"].rank(pct=True)
local["local_signal_percentile_score"] = rank_components.mean(axis=1)

# Aggregate local flags back to ADM2.
local_parent_flags = (
    local.groupby("_parent_key", dropna=False)
    .agg(
        n_hidden_composition_mismatch=("hidden_composition_mismatch", "sum"),
        n_adm3_local_amplification=("local_amplification_hotspot", "sum"),
        share_adm3_local_amplification=("local_amplification_hotspot", "mean"),
        max_local_extent_gap=("local_extent_minus_parent_adm2", "max"),
        max_local_change_gap=("local_change_minus_parent_adm2", "max"),
    )
    .reset_index()
)

cross = cross.merge(
    local_parent_flags,
    on="_parent_key",
    how="left",
    validate="one_to_one",
)

for c in [
    "n_hidden_composition_mismatch",
    "n_adm3_local_amplification",
]:
    cross[c] = cross[c].fillna(0).astype(int)

cross["composition_cross_scale_class"] = cross.apply(classify_composition, axis=1)
cross["change_cross_scale_class"] = cross.apply(classify_change, axis=1)

cross = cross.sort_values(
    [
        "n_adm3_local_amplification",
        "n_hidden_composition_mismatch",
        "adm3_max_recent_combined",
        "adm3_max_combined_change",
    ],
    ascending=False,
)

cross_path = TABLE_DIR / "22_adm2_cross_scale_comparison.csv"
cross.drop(columns=["_parent_key"]).to_csv(cross_path, index=False)


# =============================================================================
# LOCAL OUTPUT TABLE
# =============================================================================

local = local.merge(
    cross[[
        "_parent_key",
        "composition_cross_scale_class",
        "change_cross_scale_class",
    ]],
    on="_parent_key",
    how="left",
    validate="many_to_one",
)

local = local.sort_values(
    [
        "local_amplification_hotspot",
        "hidden_composition_mismatch",
        "local_signal_percentile_score",
        "recent_combined_mean",
    ],
    ascending=False,
)

local_cols = [
    c for c in [
        "ADM3_ID_STD",
        "ADM3_NAME_STD",
        "ADM2_ID_STD",
        "ADM2_NAME_STD",
        "area_km2",
        "qc_flag",
        "recent_combined_mean",
        "recent_recurring_only_share",
        "recent_unusual_only_share",
        "recent_overlap_share",
        "combined_change_recent_minus_early",
        "unusual_change_recent_minus_early",
        "recent_composition_class",
        "local_change_direction",
        "adm2_recent_combined_mean",
        "adm2_recent_unusual_only_share",
        "adm2_combined_change",
        "adm2_change_direction",
        "adm2_recent_composition_class",
        "local_extent_minus_parent_adm2",
        "local_unusual_share_minus_parent_adm2",
        "local_change_minus_parent_adm2",
        "local_high_extent",
        "local_high_unusual_share",
        "local_strong_increase",
        "local_unusual_hotspot",
        "hidden_composition_mismatch",
        "local_extent_amplification",
        "local_change_amplification",
        "local_amplification_hotspot",
        "local_signal_percentile_score",
        "composition_cross_scale_class",
        "change_cross_scale_class",
    ] if c in local.columns
]

local_path = TABLE_DIR / "22_adm3_local_cross_scale_signals.csv"
local[local_cols].to_csv(local_path, index=False)


# =============================================================================
# SUMMARY TABLES
# =============================================================================

composition_counts = (
    cross["composition_cross_scale_class"]
    .value_counts(dropna=False)
    .rename_axis("composition_cross_scale_class")
    .reset_index(name="n_adm2")
)
composition_counts["share_adm2"] = composition_counts["n_adm2"] / len(cross)
composition_counts.to_csv(
    TABLE_DIR / "22_adm2_composition_class_counts.csv",
    index=False,
)

change_counts = (
    cross["change_cross_scale_class"]
    .value_counts(dropna=False)
    .rename_axis("change_cross_scale_class")
    .reset_index(name="n_adm2")
)
change_counts["share_adm2"] = change_counts["n_adm2"] / len(cross)
change_counts.to_csv(
    TABLE_DIR / "22_adm2_change_class_counts.csv",
    index=False,
)

hidden_summary = cross.loc[
    cross["n_hidden_composition_mismatch"] > 0,
    [
        c for c in [
            "ADM2_ID_STD",
            "ADM2_NAME_STD",
            "adm2_recent_composition_class",
            "adm2_recent_combined_mean",
            "adm2_recent_unusual_only_share",
            "n_adm3",
            "n_hidden_composition_mismatch",
            "adm3_max_recent_combined",
            "adm3_max_unusual_only_share",
            "adm3_max_combined_change",
            "composition_cross_scale_class",
            "change_cross_scale_class",
        ] if c in cross.columns
    ],
].copy()
hidden_summary.to_csv(
    TABLE_DIR / "22_adm2_hidden_local_signal_summary.csv",
    index=False,
)

amplification_summary = cross.loc[
    cross["n_adm3_local_amplification"] > 0,
    [
        c for c in [
            "ADM2_ID_STD",
            "ADM2_NAME_STD",
            "adm2_recent_combined_mean",
            "adm2_combined_change",
            "adm2_change_direction",
            "n_adm3",
            "n_adm3_local_amplification",
            "share_adm3_local_amplification",
            "max_local_extent_gap",
            "max_local_change_gap",
            "adm3_max_recent_combined",
            "adm3_max_combined_change",
            "composition_cross_scale_class",
            "change_cross_scale_class",
        ] if c in cross.columns
    ],
].copy()
amplification_summary.to_csv(
    TABLE_DIR / "22_adm2_local_amplification_summary.csv",
    index=False,
)


# =============================================================================
# FIGURE 1 — ADM2 VS AREA-WEIGHTED ADM3 EXTENT
# =============================================================================

plot = cross.dropna(
    subset=["adm2_recent_combined_mean", "adm3_area_weighted_recent_combined"]
).copy()

if len(plot):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        plot["adm2_recent_combined_mean"],
        plot["adm3_area_weighted_recent_combined"],
        alpha=0.72,
    )

    lo = min(
        plot["adm2_recent_combined_mean"].min(),
        plot["adm3_area_weighted_recent_combined"].min(),
    )
    hi = max(
        plot["adm2_recent_combined_mean"].max(),
        plot["adm3_area_weighted_recent_combined"].max(),
    )
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="1:1 reference")

    label_df = plot.assign(
        abs_gap=plot["extent_gap_areaweighted_adm3_minus_adm2"].abs()
    ).nlargest(min(10, len(plot)), "abs_gap")

    for _, r in label_df.iterrows():
        ax.annotate(
            str(r["ADM2_NAME_STD"]),
            (
                r["adm2_recent_combined_mean"],
                r["adm3_area_weighted_recent_combined"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Official ADM2 recent mean combined flood extent (%)")
    ax.set_ylabel("Area-weighted mean of ADM3 recent combined extent (%)")
    ax.set_title(
        "Cross-Scale Comparison of Recent Flood Extent\n"
        "ADM3 aggregation is a diagnostic, not an ADM2 reconstruction"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "22_01_adm2_vs_area_weighted_adm3_recent_extent.png",
        dpi=220,
    )
    plt.close(fig)


# =============================================================================
# FIGURE 2 — UNUSUAL-ONLY SHARE ACROSS SCALES
# =============================================================================

plot = cross.dropna(
    subset=["adm2_recent_unusual_only_share", "adm3_area_weighted_unusual_only_share"]
).copy()

if len(plot):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        plot["adm2_recent_unusual_only_share"] * 100,
        plot["adm3_area_weighted_unusual_only_share"] * 100,
        alpha=0.72,
    )
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1, label="1:1 reference")

    label_df = plot.assign(
        abs_gap=plot["unusual_share_gap_areaweighted_adm3_minus_adm2"].abs()
    ).nlargest(min(10, len(plot)), "abs_gap")

    for _, r in label_df.iterrows():
        ax.annotate(
            str(r["ADM2_NAME_STD"]),
            (
                r["adm2_recent_unusual_only_share"] * 100,
                r["adm3_area_weighted_unusual_only_share"] * 100,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("ADM2 unusual-only share of recent combined flood extent (%)")
    ax.set_ylabel("Area-weighted ADM3 unusual-only share (%)")
    ax.set_title("ADM2 vs ADM3 Recent Unusual-Flood Composition")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "22_02_adm2_vs_adm3_unusual_share.png", dpi=220)
    plt.close(fig)


# =============================================================================
# FIGURE 3 — WITHIN-ADM2 UNUSUAL-SHARE HETEROGENEITY
# =============================================================================

plot = cross.nlargest(
    min(TOP_N, len(cross)),
    "adm3_std_unusual_only_share",
).sort_values("adm3_std_unusual_only_share")

if len(plot):
    fig, ax = plt.subplots(figsize=(12, max(7, 0.38 * len(plot) + 2)))
    y = np.arange(len(plot))
    ax.barh(y, plot["adm3_std_unusual_only_share"] * 100)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["ADM2_NAME_STD"], fontsize=8)
    ax.set_xlabel("SD of ADM3 unusual-only share within parent ADM2 (percentage points)")
    ax.set_title("Within-ADM2 Heterogeneity in Recent Unusual-Flood Composition")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "22_03_within_adm2_unusual_share_heterogeneity.png",
        dpi=220,
    )
    plt.close(fig)


# =============================================================================
# FIGURE 4 — LOCAL AMPLIFICATION HOTSPOTS
# =============================================================================

amp = local[local["local_amplification_hotspot"]].copy()
amp = amp.nlargest(
    min(TOP_N, len(amp)),
    "local_signal_percentile_score",
).sort_values("local_extent_minus_parent_adm2")

if len(amp):
    fig, ax = plt.subplots(figsize=(12, max(7, 0.42 * len(amp) + 2)))
    y = np.arange(len(amp))
    ax.barh(y, amp["local_extent_minus_parent_adm2"])

    labels = [
        f"{r.ADM3_NAME_STD} ({r.ADM2_NAME_STD})"
        for r in amp.itertuples()
    ]

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("ADM3 recent flood extent minus parent ADM2 (percentage points)")
    ax.set_title("Local ADM3 Flood-Intensity Amplification Relative to Parent ADM2")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "22_04_local_amplification_hotspots.png", dpi=220)
    plt.close(fig)


# =============================================================================
# FIGURE 5 — CHANGE COMPARISON
# =============================================================================

plot = cross.dropna(
    subset=["adm2_combined_change", "adm3_area_weighted_combined_change"]
).copy()

if len(plot):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        plot["adm2_combined_change"],
        plot["adm3_area_weighted_combined_change"],
        alpha=0.72,
    )
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    lo = min(
        plot["adm2_combined_change"].min(),
        plot["adm3_area_weighted_combined_change"].min(),
    )
    hi = max(
        plot["adm2_combined_change"].max(),
        plot["adm3_area_weighted_combined_change"].max(),
    )
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="1:1 reference")

    label_df = plot.assign(
        abs_gap=plot["change_gap_areaweighted_adm3_minus_adm2"].abs()
    ).nlargest(min(10, len(plot)), "abs_gap")

    for _, r in label_df.iterrows():
        ax.annotate(
            str(r["ADM2_NAME_STD"]),
            (r["adm2_combined_change"], r["adm3_area_weighted_combined_change"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("ADM2 combined change: recent - early (percentage points)")
    ax.set_ylabel("Area-weighted ADM3 combined change (percentage points)")
    ax.set_title("Cross-Scale Comparison of Flood-Extent Change")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "22_05_cross_scale_change_comparison.png", dpi=220)
    plt.close(fig)


# =============================================================================
# FIGURE 6 — CHANGE CLASS COUNTS
# =============================================================================

plot = change_counts.sort_values("n_adm2")
if len(plot):
    fig, ax = plt.subplots(figsize=(12, max(6, 0.55 * len(plot) + 2)))
    y = np.arange(len(plot))
    ax.barh(y, plot["n_adm2"])
    ax.set_yticks(y)
    ax.set_yticklabels(plot["change_cross_scale_class"], fontsize=8)
    ax.set_xlabel("Number of ADM2 areas")
    ax.set_title("Cross-Scale Flood-Change Pattern Classes")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "22_06_cross_scale_change_class_counts.png", dpi=220)
    plt.close(fig)


# =============================================================================
# QC / DIAGNOSTICS
# =============================================================================

pearson_extent = safe_corr(
    cross["adm2_recent_combined_mean"],
    cross["adm3_area_weighted_recent_combined"],
)
spearman_extent = safe_corr(
    cross["adm2_recent_combined_mean"],
    cross["adm3_area_weighted_recent_combined"],
    "spearman",
)
pearson_unusual = safe_corr(
    cross["adm2_recent_unusual_only_share"],
    cross["adm3_area_weighted_unusual_only_share"],
)
spearman_unusual = safe_corr(
    cross["adm2_recent_unusual_only_share"],
    cross["adm3_area_weighted_unusual_only_share"],
    "spearman",
)
pearson_change = safe_corr(
    cross["adm2_combined_change"],
    cross["adm3_area_weighted_combined_change"],
)
spearman_change = safe_corr(
    cross["adm2_combined_change"],
    cross["adm3_area_weighted_combined_change"],
    "spearman",
)

n_hidden_comp_adm3 = int(local["hidden_composition_mismatch"].sum())
n_hidden_comp_adm2 = int((cross["n_hidden_composition_mismatch"] > 0).sum())
n_amp_adm3 = int(local["local_amplification_hotspot"].sum())
n_amp_adm2 = int((cross["n_adm3_local_amplification"] > 0).sum())

qc_text = f"""22 — ADM2–ADM3 Cross-Scale Flood Composition and Change Comparison
===================================================================

Project root
------------
{ROOT}

Inputs
------
ADM2: {ADM2_PATH}
ADM3: {ADM3_PATH}

Output
------
{OUTPUT_ROOT}

Formal time windows inherited from Scripts 20/21
--------------------------------------------------
Early : {EARLY_START}-{EARLY_END}
Recent: {RECENT_START}-{RECENT_END}

Coverage
--------
ADM2 rows                         : {len(adm2)}
ADM3 rows                         : {len(adm3)}
ADM2 parent keys                  : {adm2['_parent_key'].nunique()}
ADM3 parent keys                  : {adm3['_parent_key'].nunique()}
ADM3 parent keys absent from ADM2 : {len(missing_parent_keys)}
ADM2 keys without ADM3 children   : {len(missing_adm3_parents)}
Join key                          : {key_label}

ADM3 global screening thresholds
--------------------------------
Recent combined extent threshold : {extent_thr:.8f} (Q{LOCAL_EXTENT_QUANTILE:.2f})
Unusual-only share threshold     : {unusual_share_thr:.8f} (Q{LOCAL_UNUSUAL_SHARE_QUANTILE:.2f})
Positive change threshold        : {change_thr:.8f} (Q{LOCAL_CHANGE_QUANTILE:.2f} of positive changes)

Parent-relative amplification thresholds
----------------------------------------
Positive extent-gap threshold    : {extent_gap_thr:.8f} (Q{LOCAL_EXTENT_GAP_QUANTILE:.2f})
Positive change-gap threshold    : {change_gap_thr:.8f} (Q{LOCAL_CHANGE_GAP_QUANTILE:.2f})

Composition mismatch
--------------------
ADM3 hidden composition mismatch : {n_hidden_comp_adm3}
ADM2 containing mismatch         : {n_hidden_comp_adm2}

Local amplification
-------------------
ADM3 local amplification hotspots: {n_amp_adm3}
ADM2 containing amplification    : {n_amp_adm2}

Cross-scale diagnostic correlations
-----------------------------------
Recent combined extent:
  Pearson  : {pearson_extent:.6f}
  Spearman : {spearman_extent:.6f}

Recent unusual-only share:
  Pearson  : {pearson_unusual:.6f}
  Spearman : {spearman_unusual:.6f}

Combined change recent - early:
  Pearson  : {pearson_change:.6f}
  Spearman : {spearman_change:.6f}

Composition cross-scale classes
-------------------------------
{composition_counts.to_string(index=False)}

Change cross-scale classes
--------------------------
{change_counts.to_string(index=False)}

Methodological interpretation
-----------------------------
1. Official ADM2 values from Script 20 remain the formal ADM2-scale estimates.
2. ADM3 arithmetic means summarize local-unit distributions and do not reconstruct ADM2 values.
3. Area-weighted ADM3 summaries are diagnostic comparisons only.
4. Hidden composition mismatch identifies an ADM3 with high recent extent and high unusual-only
   share whose parent ADM2 is not itself unusual-dominated.
5. Local amplification is intentionally broader: it identifies ADM3 units that are substantially
   more flood-affected than their parent ADM2 even when the broad composition class agrees.
6. Change classes distinguish regional increase / neutral / decline from local worsening /
   improvement patterns using the same recent-vs-early window as Scripts 20 and 21.
7. Quantile thresholds are analytical screening thresholds and should not be interpreted as
   operational humanitarian cut-offs without sensitivity analysis and external validation.

DONE.
"""

(QC_DIR / "22_cross_scale_qc_summary.txt").write_text(qc_text, encoding="utf-8")


# =============================================================================
# FINAL CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 110)
print("CROSS-SCALE RESULTS")
print("=" * 110)
print(f"ADM2 compared                          : {len(cross):,}")
print(f"ADM3 evaluated                         : {len(local):,}")
print(f"ADM3 hidden composition mismatches     : {n_hidden_comp_adm3:,}")
print(f"ADM2 containing composition mismatch   : {n_hidden_comp_adm2:,}")
print(f"ADM3 local amplification hotspots      : {n_amp_adm3:,}")
print(f"ADM2 containing local amplification    : {n_amp_adm2:,}")
print(f"Extent correlation (Pearson/Spearman)  : {pearson_extent:.3f} / {spearman_extent:.3f}")
print(f"Unusual share corr. (Pearson/Spearman) : {pearson_unusual:.3f} / {spearman_unusual:.3f}")
print(f"Change correlation (Pearson/Spearman)  : {pearson_change:.3f} / {spearman_change:.3f}")

print("\nComposition cross-scale classes:")
print(composition_counts.to_string(index=False))

print("\nChange cross-scale classes:")
print(change_counts.to_string(index=False))

if n_amp_adm3:
    print("\nTop local amplification hotspots:")
    show_cols = [
        c for c in [
            "ADM3_NAME_STD",
            "ADM2_NAME_STD",
            "recent_combined_mean",
            "adm2_recent_combined_mean",
            "local_extent_minus_parent_adm2",
            "combined_change_recent_minus_early",
            "adm2_combined_change",
            "local_change_minus_parent_adm2",
            "local_signal_percentile_score",
        ] if c in local.columns
    ]
    print(
        local.loc[local["local_amplification_hotspot"], show_cols]
        .head(15)
        .to_string(index=False)
    )

print("\nSaved tables:")
for f in sorted(TABLE_DIR.glob("*.csv")):
    print(f"  {f.name}")

print("\nSaved figures:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}")

print("\nSaved QC:")
for f in sorted(QC_DIR.glob("*")):
    print(f"  {f.name}")

print("\nNEXT: Use Script 22 tables as inputs to Script 23 nested ADM2–ADM3 maps.")
print("\nDONE.")
