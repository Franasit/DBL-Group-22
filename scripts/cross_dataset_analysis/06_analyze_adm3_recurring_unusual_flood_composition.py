#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_analyze_adm3_recurring_unusual_flood_composition_v2.py

JBG060 Capstone Data Challenge — South Sudan Flood Analysis

Fixes compared with the previous ADM3 script
----------------------------------------------
1. Column detection no longer misidentifies ADM3_ID as ADM2_ID / area / support.
2. The COMBINED table is used as the master ADM3×year universe.
3. Missing recurring or unusual records are treated as zero after merging.
   This is important because *_by_type.csv may only contain rows where a given
   flood type was actually present.
4. Parent ADM2 name is preserved when available.
5. Small-area / support QC is only applied if a real area/support column exists.

Main outputs
------------
Tables:
- adm3_flood_mask_composition_annual.csv
- adm3_flood_mask_composition_summary.csv
- adm3_recent_change_summary.csv
- adm2_parent_composition_summary.csv
- adm2_parent_composition_heterogeneity.csv
- adm3_qc_summary.csv

Figures:
- 01_adm3_mean_annual_flood_extent_by_mask_type.png
- 02_top30_adm3_recent_composition.png
- 03_adm3_recent_change_recurring_vs_unusual.png
- 04_top30_adm3_unusual_dominated.png
- 05_within_adm2_recent_flood_heterogeneity.png
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

EARLY_START = 2000
EARLY_END = 2019
RECENT_START = 2020
RECENT_END = 2025
TOP_N = 30

PROJECT_ROOT = None
BY_TYPE_CSV = None
COMBINED_CSV = None

SMALL_AREA_KM2_THRESHOLD = 100.0
LOW_SUPPORT_THRESHOLD = 10
DOMINANCE_SHARE = 0.60


# =============================================================================
# PATHS
# =============================================================================

def infer_project_root():
    if PROJECT_ROOT:
        return Path(PROJECT_ROOT).expanduser().resolve()

    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "scripts").exists() and (p / "processed_data").exists():
            return p

    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "processed_data").exists():
            return p

    return cwd


ROOT = infer_project_root()
INPUT_DIR = ROOT / "processed_data" / "flood_analysis" / "adm3"
OUT_DIR = ROOT / "processed_data" / "cross_dataset_analysis" / "adm3_recurring_unusual_composition"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_file(manual_path, filename):
    if manual_path:
        p = Path(manual_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    direct = INPUT_DIR / filename
    if direct.exists():
        return direct

    candidates = list((ROOT / "processed_data").rglob(filename))
    if not candidates:
        candidates = list(ROOT.rglob(filename))

    if not candidates:
        return None

    return sorted(candidates, key=lambda p: len(str(p)))[0]


BY_TYPE_PATH = find_file(BY_TYPE_CSV, "adm3_annual_flood_indicators_by_type.csv")
COMBINED_PATH = find_file(COMBINED_CSV, "adm3_annual_flood_indicators_combined.csv")


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def pick_column(df, exact=(), contains=(), excludes=()):
    """
    Exact match first. Only run fuzzy contains matching if contains is non-empty.
    This avoids the old bug where an empty contains tuple matched the first column.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in exact:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if contains:
        for c in df.columns:
            lc = str(c).lower()
            if all(token.lower() in lc for token in contains):
                if not any(token.lower() in lc for token in excludes):
                    return c

    return None


def detect_columns(df):
    year = pick_column(df, exact=("year", "flood_year"), contains=("year",))

    adm3_name = pick_column(
        df,
        exact=("ADM3_NAME", "adm3_name", "NAME_3", "name_3", "admin3_name"),
        contains=("adm3", "name"),
    )
    adm3_id = pick_column(
        df,
        exact=("ADM3_ID", "adm3_id", "GID_3", "gid_3", "adm3_code"),
    )

    adm2_name = pick_column(
        df,
        exact=("ADM2_NAME", "adm2_name", "NAME_2", "name_2", "admin2_name"),
        contains=("adm2", "name"),
    )
    adm2_id = pick_column(
        df,
        exact=("ADM2_ID", "adm2_id", "GID_2", "gid_2", "adm2_code"),
    )

    metric = None
    for candidate in (
        "annual_flooded_percent",
        "annual_flooded_percentage",
        "flooded_percent",
        "flooded_percentage",
        "annual_unique_flooded_percent",
    ):
        metric = pick_column(df, exact=(candidate,))
        if metric is not None:
            break

    if metric is None:
        for c in df.columns:
            lc = str(c).lower()
            if (
                ("flood" in lc or "flooded" in lc)
                and ("percent" in lc or "percentage" in lc or "pct" in lc)
                and "change" not in lc
            ):
                metric = c
                break

    flood_type = pick_column(
        df,
        exact=("flood_type", "mask_type", "type"),
        contains=("type",),
    )

    area = None
    for candidate in (
        "area_km2",
        "adm3_area_km2",
        "total_area_km2",
        "polygon_area_km2",
        "admin_area_km2",
    ):
        area = pick_column(df, exact=(candidate,))
        if area is not None:
            break

    if area is None:
        for c in df.columns:
            lc = str(c).lower()
            if "area" in lc and "km" in lc and "flood" not in lc:
                area = c
                break

    support = None
    for candidate in (
        "valid_pixel_count",
        "pixel_count",
        "n_pixels",
        "grid_count",
        "observation_count",
        "n_observations",
    ):
        support = pick_column(df, exact=(candidate,))
        if support is not None:
            break

    return {
        "year": year,
        "adm3_name": adm3_name,
        "adm3_id": adm3_id,
        "adm2_name": adm2_name,
        "adm2_id": adm2_id,
        "metric": metric,
        "type": flood_type,
        "area": area,
        "support": support,
    }


def normalise_type(value):
    s = str(value).strip().lower()
    if "recurr" in s:
        return "recurring"
    if "unusual" in s:
        return "unusual"
    if "combin" in s:
        return "combined"
    return s


def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(a, dtype=float)
    np.divide(a, b, out=out, where=np.abs(b) > 1e-12)
    return out


def normalized_entropy(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]

    if len(arr) <= 1:
        return 0.0

    arr = arr / arr.sum()
    h = -np.sum(arr * np.log(arr))
    return float(h / np.log(len(arr)))


# =============================================================================
# LOAD
# =============================================================================

print("=" * 100)
print("ADM3 RECURRING vs UNUSUAL FLOOD-MASK COMPOSITION ANALYSIS — v2")
print("=" * 100)
print(f"Project root : {ROOT}")
print(f"Output dir   : {OUT_DIR}\n")

if BY_TYPE_PATH is None:
    raise FileNotFoundError("Could not find adm3_annual_flood_indicators_by_type.csv")

if COMBINED_PATH is None:
    raise FileNotFoundError("Could not find adm3_annual_flood_indicators_combined.csv")

print(f"By-type file : {BY_TYPE_PATH}")
print(f"Combined file: {COMBINED_PATH}")

by_type = pd.read_csv(BY_TYPE_PATH)
combined_raw = pd.read_csv(COMBINED_PATH)

c1 = detect_columns(by_type)
c2 = detect_columns(combined_raw)

print("\nDetected columns in by-type file:")
for k, v in c1.items():
    print(f"  {k:12s}: {v}")

print("\nDetected columns in combined file:")
for k, v in c2.items():
    print(f"  {k:12s}: {v}")

required_by_type = ("year", "adm3_name", "metric", "type")
missing = [x for x in required_by_type if c1[x] is None]
if missing:
    raise ValueError(
        f"Missing required columns in by-type file: {missing}\n"
        f"Available columns: {list(by_type.columns)}"
    )

required_combined = ("year", "adm3_name", "metric")
missing = [x for x in required_combined if c2[x] is None]
if missing:
    raise ValueError(
        f"Missing required columns in combined file: {missing}\n"
        f"Available columns: {list(combined_raw.columns)}"
    )


# =============================================================================
# CLEAN BY-TYPE
# =============================================================================

by_type = by_type.copy()
by_type["_type"] = by_type[c1["type"]].map(normalise_type)

by_type = by_type[
    by_type["_type"].isin(["recurring", "unusual"])
].copy()

by_type[c1["year"]] = pd.to_numeric(by_type[c1["year"]], errors="coerce")
by_type[c1["metric"]] = pd.to_numeric(by_type[c1["metric"]], errors="coerce")

by_type = by_type.dropna(
    subset=[c1["year"], c1["adm3_name"], c1["metric"]]
)

by_type[c1["year"]] = by_type[c1["year"]].astype(int)

print("\nFlood types in by-type file:")
print(by_type["_type"].value_counts().to_string())


# =============================================================================
# CLEAN COMBINED AND USE IT AS MASTER UNIVERSE
# =============================================================================

combined = combined_raw.copy()

combined[c2["year"]] = pd.to_numeric(combined[c2["year"]], errors="coerce")
combined[c2["metric"]] = pd.to_numeric(combined[c2["metric"]], errors="coerce")

combined = combined.dropna(
    subset=[c2["year"], c2["adm3_name"], c2["metric"]]
)

combined[c2["year"]] = combined[c2["year"]].astype(int)

rename = {
    c2["adm3_name"]: "ADM3_NAME_STD",
    c2["year"]: "year",
    c2["metric"]: "combined",
}

if c2["adm3_id"] is not None:
    rename[c2["adm3_id"]] = "ADM3_ID_STD"

if c2["adm2_name"] is not None:
    rename[c2["adm2_name"]] = "ADM2_NAME_STD"

if c2["adm2_id"] is not None:
    rename[c2["adm2_id"]] = "ADM2_ID_STD"

if c2["area"] is not None:
    rename[c2["area"]] = "area_km2"

if c2["support"] is not None:
    rename[c2["support"]] = "support_count"

combined = combined.rename(columns=rename)

master_cols = ["ADM3_NAME_STD", "year", "combined"]

for optional in (
    "ADM3_ID_STD",
    "ADM2_NAME_STD",
    "ADM2_ID_STD",
    "area_km2",
    "support_count",
):
    if optional in combined.columns:
        master_cols.append(optional)

combined = combined[master_cols].copy()

# Drop exact duplicate master keys safely.
master_key = ["ADM3_NAME_STD", "year"]
if "ADM3_ID_STD" in combined.columns:
    master_key.insert(0, "ADM3_ID_STD")

dup_master = combined.duplicated(master_key, keep=False)

if dup_master.any():
    print(
        f"\nWARNING: {dup_master.sum():,} duplicate combined rows found on "
        f"{master_key}; aggregating them."
    )

    agg = {"combined": "mean"}

    for col in (
        "ADM2_NAME_STD",
        "ADM2_ID_STD",
        "area_km2",
        "support_count",
    ):
        if col in combined.columns:
            agg[col] = "first"

    combined = (
        combined.groupby(master_key, as_index=False)
        .agg(agg)
    )


# =============================================================================
# PIVOT BY-TYPE TO RECURRING / UNUSUAL
# =============================================================================

by_rename = {
    c1["adm3_name"]: "ADM3_NAME_STD",
    c1["year"]: "year",
    c1["metric"]: "flood_pct",
}

if c1["adm3_id"] is not None:
    by_rename[c1["adm3_id"]] = "ADM3_ID_STD"

if c1["adm2_name"] is not None:
    by_rename[c1["adm2_name"]] = "ADM2_NAME_FROM_TYPE"

by_type = by_type.rename(columns=by_rename)

pivot_key = ["ADM3_NAME_STD", "year"]
if "ADM3_ID_STD" in by_type.columns:
    pivot_key.insert(0, "ADM3_ID_STD")

type_wide = (
    by_type.pivot_table(
        index=pivot_key,
        columns="_type",
        values="flood_pct",
        aggfunc="mean",
        dropna=False,
    )
    .reset_index()
)

# A type may be absent for some or all ADM3-years.
for col in ("recurring", "unusual"):
    if col not in type_wide.columns:
        type_wide[col] = 0.0


# =============================================================================
# MERGE ONTO COMBINED MASTER
# =============================================================================

merge_key = ["ADM3_NAME_STD", "year"]

if (
    "ADM3_ID_STD" in combined.columns
    and "ADM3_ID_STD" in type_wide.columns
):
    merge_key.insert(0, "ADM3_ID_STD")

wide = combined.merge(
    type_wide[merge_key + ["recurring", "unusual"]],
    on=merge_key,
    how="left",
    validate="one_to_one",
)

# Missing type rows mean that type was not present for that ADM3-year.
wide["recurring"] = pd.to_numeric(
    wide["recurring"], errors="coerce"
).fillna(0.0)

wide["unusual"] = pd.to_numeric(
    wide["unusual"], errors="coerce"
).fillna(0.0)

wide["combined"] = pd.to_numeric(
    wide["combined"], errors="coerce"
).fillna(0.0)

print("\nMerged ADM3-year universe:")
print(f"  rows             : {len(wide):,}")
print(f"  unique ADM3 names: {wide['ADM3_NAME_STD'].nunique():,}")
print(f"  unique years     : {wide['year'].nunique():,}")
print(f"  missing recurring filled with 0: {(wide['recurring'] == 0).sum():,}")
print(f"  missing unusual filled with 0  : {(wide['unusual'] == 0).sum():,}")


# =============================================================================
# COMPOSITION
# =============================================================================

wide["overlap_raw"] = (
    wide["recurring"]
    + wide["unusual"]
    - wide["combined"]
)

negative_overlap = wide["overlap_raw"] < -1e-6

if negative_overlap.any():
    print("\nWARNING:")
    print(
        f"  {negative_overlap.sum():,} rows have recurring + unusual < combined."
    )
    print(
        "  This suggests the combined product is not always a strict pixel union, "
        "or source rows are incomplete. Overlap-based interpretation should be "
        "treated cautiously until those rows are inspected."
    )

wide["overlap"] = wide["overlap_raw"].clip(lower=0)

wide["recurring_only"] = (
    wide["recurring"] - wide["overlap"]
).clip(lower=0)

wide["unusual_only"] = (
    wide["unusual"] - wide["overlap"]
).clip(lower=0)

wide["component_sum"] = (
    wide["recurring_only"]
    + wide["unusual_only"]
    + wide["overlap"]
)

wide["reconstruction_error"] = (
    wide["component_sum"]
    - wide["combined"]
)

wide["recurring_only_share"] = safe_divide(
    wide["recurring_only"],
    wide["combined"],
)

wide["unusual_only_share"] = safe_divide(
    wide["unusual_only"],
    wide["combined"],
)

wide["overlap_share"] = safe_divide(
    wide["overlap"],
    wide["combined"],
)


# =============================================================================
# QC FLAGS
# =============================================================================

wide["small_area_flag"] = False
wide["low_support_flag"] = False

if "area_km2" in wide.columns:
    wide["area_km2"] = pd.to_numeric(wide["area_km2"], errors="coerce")
    wide["small_area_flag"] = (
        wide["area_km2"] < SMALL_AREA_KM2_THRESHOLD
    ).fillna(False)

if "support_count" in wide.columns:
    wide["support_count"] = pd.to_numeric(
        wide["support_count"], errors="coerce"
    )
    wide["low_support_flag"] = (
        wide["support_count"] < LOW_SUPPORT_THRESHOLD
    ).fillna(False)

wide["qc_flag"] = np.select(
    [
        wide["small_area_flag"] & wide["low_support_flag"],
        wide["small_area_flag"],
        wide["low_support_flag"],
    ],
    [
        "small_area_and_low_support",
        "small_area",
        "low_support",
    ],
    default="ok",
)


# =============================================================================
# SAVE ANNUAL
# =============================================================================

annual_path = OUT_DIR / "adm3_flood_mask_composition_annual.csv"
wide.to_csv(annual_path, index=False)


# =============================================================================
# ADM3 SUMMARY
# =============================================================================

summary_rows = []

group_cols = ["ADM3_NAME_STD"]

if "ADM3_ID_STD" in wide.columns:
    group_cols.insert(0, "ADM3_ID_STD")

if "ADM2_NAME_STD" in wide.columns:
    group_cols.append("ADM2_NAME_STD")

for keys, g in wide.groupby(group_cols, dropna=False):

    if not isinstance(keys, tuple):
        keys = (keys,)

    row = dict(zip(group_cols, keys))

    early = g[
        g["year"].between(EARLY_START, EARLY_END, inclusive="both")
    ]
    recent = g[
        g["year"].between(RECENT_START, RECENT_END, inclusive="both")
    ]

    def m(frame, col):
        return float(frame[col].mean()) if len(frame) else np.nan

    row.update({
        "n_years": int(g["year"].nunique()),

        "longterm_combined_mean": m(g, "combined"),
        "longterm_recurring_mean": m(g, "recurring"),
        "longterm_unusual_mean": m(g, "unusual"),

        "early_combined_mean": m(early, "combined"),
        "early_recurring_mean": m(early, "recurring"),
        "early_unusual_mean": m(early, "unusual"),

        "recent_combined_mean": m(recent, "combined"),
        "recent_recurring_mean": m(recent, "recurring"),
        "recent_unusual_mean": m(recent, "unusual"),

        "recent_recurring_only_mean": m(recent, "recurring_only"),
        "recent_unusual_only_mean": m(recent, "unusual_only"),
        "recent_overlap_mean": m(recent, "overlap"),
    })

    rc = row["recent_combined_mean"]

    if pd.notna(rc) and rc > 0:
        row["recent_recurring_only_share"] = (
            row["recent_recurring_only_mean"] / rc
        )
        row["recent_unusual_only_share"] = (
            row["recent_unusual_only_mean"] / rc
        )
        row["recent_overlap_share"] = (
            row["recent_overlap_mean"] / rc
        )
    else:
        row["recent_recurring_only_share"] = 0.0
        row["recent_unusual_only_share"] = 0.0
        row["recent_overlap_share"] = 0.0

    row["combined_change_recent_minus_early"] = (
        row["recent_combined_mean"] - row["early_combined_mean"]
    )
    row["recurring_change_recent_minus_early"] = (
        row["recent_recurring_mean"] - row["early_recurring_mean"]
    )
    row["unusual_change_recent_minus_early"] = (
        row["recent_unusual_mean"] - row["early_unusual_mean"]
    )

    row["recent_composition_entropy"] = normalized_entropy(
        [
            row["recent_recurring_only_share"],
            row["recent_unusual_only_share"],
            row["recent_overlap_share"],
        ]
    )

    row["small_area_flag"] = bool(g["small_area_flag"].any())
    row["low_support_flag"] = bool(g["low_support_flag"].any())

    if "area_km2" in g.columns:
        vals = pd.to_numeric(g["area_km2"], errors="coerce")
        row["area_km2"] = float(vals.median()) if vals.notna().any() else np.nan

    if "support_count" in g.columns:
        vals = pd.to_numeric(g["support_count"], errors="coerce")
        row["support_count_median"] = (
            float(vals.median()) if vals.notna().any() else np.nan
        )

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)


def classify(row):
    if pd.isna(row["recent_combined_mean"]) or row["recent_combined_mean"] <= 0:
        return "No / negligible flood signal"

    components = {
        "Recurring-dominated": row["recent_recurring_only_share"],
        "Unusual-dominated": row["recent_unusual_only_share"],
        "High overlap": row["recent_overlap_share"],
    }

    label, value = max(components.items(), key=lambda x: x[1])

    if value >= DOMINANCE_SHARE:
        return label

    return "Mixed composition"


summary["recent_composition_class"] = summary.apply(classify, axis=1)

summary["qc_flag"] = np.select(
    [
        summary["small_area_flag"] & summary["low_support_flag"],
        summary["small_area_flag"],
        summary["low_support_flag"],
    ],
    [
        "small_area_and_low_support",
        "small_area",
        "low_support",
    ],
    default="ok",
)

summary = summary.sort_values(
    ["recent_combined_mean", "longterm_combined_mean"],
    ascending=False,
)

summary_path = OUT_DIR / "adm3_flood_mask_composition_summary.csv"
summary.to_csv(summary_path, index=False)


# =============================================================================
# CHANGE + QC TABLES
# =============================================================================

change_cols = [
    c for c in (
        "ADM3_ID_STD",
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
    )
    if c in summary.columns
]

change_cols += [
    "early_combined_mean",
    "recent_combined_mean",
    "combined_change_recent_minus_early",
    "recurring_change_recent_minus_early",
    "unusual_change_recent_minus_early",
    "recent_recurring_only_share",
    "recent_unusual_only_share",
    "recent_overlap_share",
    "recent_composition_entropy",
    "recent_composition_class",
    "qc_flag",
]

change_path = OUT_DIR / "adm3_recent_change_summary.csv"
summary[change_cols].to_csv(change_path, index=False)

qc_cols = [
    c for c in (
        "ADM3_ID_STD",
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
        "area_km2",
        "support_count_median",
        "small_area_flag",
        "low_support_flag",
        "qc_flag",
    )
    if c in summary.columns
]

qc_path = OUT_DIR / "adm3_qc_summary.csv"
summary[qc_cols].to_csv(qc_path, index=False)


# =============================================================================
# PARENT ADM2 SUMMARIES
# =============================================================================

parent_summary = None
parent_heterogeneity = None

if "ADM2_NAME_STD" in summary.columns:

    parent_summary = (
        summary.groupby("ADM2_NAME_STD", dropna=False)
        .agg(
            n_adm3=("ADM3_NAME_STD", "nunique"),
            mean_recent_combined=("recent_combined_mean", "mean"),
            median_recent_combined=("recent_combined_mean", "median"),
            max_recent_combined=("recent_combined_mean", "max"),
            mean_recent_recurring_only_share=("recent_recurring_only_share", "mean"),
            mean_recent_unusual_only_share=("recent_unusual_only_share", "mean"),
            mean_recent_overlap_share=("recent_overlap_share", "mean"),
            mean_composition_entropy=("recent_composition_entropy", "mean"),
            mean_combined_change=("combined_change_recent_minus_early", "mean"),
            max_combined_change=("combined_change_recent_minus_early", "max"),
        )
        .reset_index()
    )

    parent_path = OUT_DIR / "adm2_parent_composition_summary.csv"
    parent_summary.to_csv(parent_path, index=False)

    rows = []

    for adm2_name, g in summary.groupby("ADM2_NAME_STD", dropna=False):

        class_freq = (
            g["recent_composition_class"]
            .value_counts(normalize=True)
            .values
        )

        rows.append({
            "ADM2_NAME_STD": adm2_name,
            "n_adm3": int(g["ADM3_NAME_STD"].nunique()),
            "composition_class_entropy": normalized_entropy(class_freq),
            "std_unusual_only_share": float(
                g["recent_unusual_only_share"].std(ddof=0)
            ),
            "std_recurring_only_share": float(
                g["recent_recurring_only_share"].std(ddof=0)
            ),
            "std_recent_combined_mean": float(
                g["recent_combined_mean"].std(ddof=0)
            ),
            "range_recent_combined_mean": float(
                g["recent_combined_mean"].max()
                - g["recent_combined_mean"].min()
            ),
            "max_recent_combined_mean": float(
                g["recent_combined_mean"].max()
            ),
            "mean_recent_combined_mean": float(
                g["recent_combined_mean"].mean()
            ),
        })

    parent_heterogeneity = pd.DataFrame(rows).sort_values(
        [
            "range_recent_combined_mean",
            "std_recent_combined_mean",
            "std_unusual_only_share",
        ],
        ascending=False,
    )

    hetero_path = OUT_DIR / "adm2_parent_composition_heterogeneity.csv"
    parent_heterogeneity.to_csv(hetero_path, index=False)


# =============================================================================
# FIGURE 1 — MEAN TRAJECTORY
# =============================================================================

national = (
    wide.groupby("year")[["combined", "recurring", "unusual"]]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(13, 7))

ax.plot(
    national["year"],
    national["combined"],
    marker="o",
    linewidth=2.5,
    label="Combined",
)

ax.plot(
    national["year"],
    national["recurring"],
    marker="o",
    label="Recurring",
)

ax.plot(
    national["year"],
    national["unusual"],
    marker="o",
    label="Unusual",
)

ax.axvline(RECENT_START, linestyle="--", alpha=0.7)

ax.set_title("ADM3 Mean Annual Flood Extent by Flood-Mask Type")
ax.set_xlabel("Year")
ax.set_ylabel("Mean annual flooded area (%)")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "01_adm3_mean_annual_flood_extent_by_mask_type.png",
    dpi=220,
)
plt.close(fig)


# =============================================================================
# FIGURE 2 — TOP ADM3 COMPOSITION
# =============================================================================

plot_df = (
    summary.head(TOP_N)
    .copy()
    .sort_values("recent_combined_mean")
)

fig, ax = plt.subplots(
    figsize=(12, max(8, 0.38 * len(plot_df) + 2))
)

y = np.arange(len(plot_df))

v1 = plot_df["recent_recurring_only_mean"].to_numpy()
v2 = plot_df["recent_overlap_mean"].to_numpy()
v3 = plot_df["recent_unusual_only_mean"].to_numpy()

ax.barh(y, v1, label="Recurring only")
ax.barh(y, v2, left=v1, label="Recurring ∩ unusual")
ax.barh(y, v3, left=v1 + v2, label="Unusual only")

labels = []
for _, r in plot_df.iterrows():
    label = str(r["ADM3_NAME_STD"])
    if "ADM2_NAME_STD" in plot_df.columns:
        label += f" ({r['ADM2_NAME_STD']})"
    labels.append(label)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Recent mean annual flooded area (%)")
ax.set_title(
    f"Recent Flood-Mask Composition of Top {len(plot_df)} ADM3 Areas "
    f"({RECENT_START}-{RECENT_END})"
)
ax.legend()
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "02_top30_adm3_recent_composition.png",
    dpi=220,
)
plt.close(fig)


# =============================================================================
# FIGURE 3 — CHANGE SCATTER
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

x = summary["recurring_change_recent_minus_early"]
y = summary["unusual_change_recent_minus_early"]

sizes = (
    20
    + 15
    * np.sqrt(
        summary["recent_combined_mean"]
        .clip(lower=0)
        .fillna(0)
    )
)

ax.scatter(x, y, s=sizes, alpha=0.65)
ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)

label_df = summary.nlargest(
    min(15, len(summary)),
    "combined_change_recent_minus_early",
)

for _, r in label_df.iterrows():
    ax.annotate(
        str(r["ADM3_NAME_STD"]),
        (
            r["recurring_change_recent_minus_early"],
            r["unusual_change_recent_minus_early"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel(
    "Change in recurring flooded area: recent - early (percentage points)"
)
ax.set_ylabel(
    "Change in unusual flooded area: recent - early (percentage points)"
)
ax.set_title(
    "ADM3 Recurring vs Unusual Contribution to Recent Flood Change"
)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "03_adm3_recent_change_recurring_vs_unusual.png",
    dpi=220,
)
plt.close(fig)


# =============================================================================
# FIGURE 4 — TOP UNUSUAL-DOMINATED ADM3
# =============================================================================

unusual_df = summary[
    summary["recent_composition_class"] == "Unusual-dominated"
].copy()

unusual_df = (
    unusual_df.nlargest(min(TOP_N, len(unusual_df)), "recent_combined_mean")
    .sort_values("recent_combined_mean")
)

if len(unusual_df):

    fig, ax = plt.subplots(
        figsize=(12, max(8, 0.38 * len(unusual_df) + 2))
    )

    y = np.arange(len(unusual_df))

    ax.barh(
        y,
        unusual_df["recent_combined_mean"],
    )

    labels = []

    for _, r in unusual_df.iterrows():
        label = str(r["ADM3_NAME_STD"])

        if "ADM2_NAME_STD" in unusual_df.columns:
            label += f" ({r['ADM2_NAME_STD']})"

        labels.append(label)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Recent mean combined flooded area (%)")
    ax.set_title(
        f"Top {len(unusual_df)} Unusual-Dominated ADM3 Flood Hotspots"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "04_top30_adm3_unusual_dominated.png",
        dpi=220,
    )
    plt.close(fig)


# =============================================================================
# FIGURE 5 — WITHIN-ADM2 FLOOD HETEROGENEITY
# =============================================================================

if parent_heterogeneity is not None and len(parent_heterogeneity):

    plot_het = (
        parent_heterogeneity.head(30)
        .copy()
        .sort_values("range_recent_combined_mean")
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    y = np.arange(len(plot_het))

    ax.barh(
        y,
        plot_het["range_recent_combined_mean"],
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        plot_het["ADM2_NAME_STD"],
        fontsize=8,
    )
    ax.set_xlabel(
        "Range of recent mean combined flooded area among ADM3 units "
        "(percentage points)"
    )
    ax.set_title(
        "Within-ADM2 Heterogeneity of Recent ADM3 Flood Extent"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "05_within_adm2_recent_flood_heterogeneity.png",
        dpi=220,
    )
    plt.close(fig)


# =============================================================================
# FINAL QA
# =============================================================================

print("\n" + "=" * 100)
print("QUALITY CHECK")
print("=" * 100)

print(f"Unique ADM3                  : {summary['ADM3_NAME_STD'].nunique():,}")
print(f"Unique years                 : {wide['year'].nunique():,}")
print(f"ADM3-year rows               : {len(wide):,}")
print(f"Negative raw-overlap rows    : {(wide['overlap_raw'] < -1e-6).sum():,}")
print(
    "Max abs reconstruction error: "
    f"{wide['reconstruction_error'].abs().max():.8f}"
)

if "area_km2" in summary.columns:
    print(f"Small-area flagged ADM3      : {summary['small_area_flag'].sum():,}")
else:
    print("Small-area QC                : no real area column found; not applied")

if "support_count_median" in summary.columns:
    print(f"Low-support flagged ADM3     : {summary['low_support_flag'].sum():,}")
else:
    print("Low-support QC               : no support-count column found; not applied")

print("\nRecent composition classes:")
print(summary["recent_composition_class"].value_counts().to_string())

print("\nTop 15 ADM3 by recent combined flood extent:")

display_cols = [
    c for c in (
        "ADM3_NAME_STD",
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_recurring_only_share",
        "recent_unusual_only_share",
        "recent_overlap_share",
        "combined_change_recent_minus_early",
        "recent_composition_class",
        "qc_flag",
    )
    if c in summary.columns
]

print(
    summary[display_cols]
    .head(15)
    .to_string(index=False)
)

if parent_heterogeneity is not None:
    print("\nTop 10 parent ADM2 by within-ADM3 flood-extent heterogeneity:")
    print(
        parent_heterogeneity[
            [
                "ADM2_NAME_STD",
                "n_adm3",
                "range_recent_combined_mean",
                "std_recent_combined_mean",
                "std_unusual_only_share",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

print("\nSaved outputs:")
for f in sorted(OUT_DIR.glob("*.csv")):
    print(f"  TABLE  {f.name}")

for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  FIGURE {f.name}")

print("\nDONE.")
