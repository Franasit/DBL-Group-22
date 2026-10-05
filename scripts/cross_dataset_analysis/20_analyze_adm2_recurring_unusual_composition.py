#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20_analyze_adm2_recurring_unusual_composition_FINAL.py

JBG060 Capstone Data Challenge — South Sudan Flood Analysis

Purpose
-------
Analyse the composition of ADM2 annual flood extent into:
1) recurring flood,
2) unusual flood,
3) recurring-only,
4) unusual-only,
5) recurring ∩ unusual overlap,
6) combined flood extent.

This script is the final reference ADM2 component of:

    processed_data/cross_dataset_analysis/03_flood_composition/

Formal sequence
---------------
20 — ADM2 recurring/unusual composition
21 — ADM3 recurring/unusual composition
22 — ADM2–ADM3 cross-scale composition comparison

Important methodological rule
-----------------------------
If recurring (R), unusual (U), and combined (C) are derived from compatible
pixel sets, inclusion-exclusion gives:

    overlap = R + U - C
    recurring_only = R - overlap
    unusual_only = U - overlap

and therefore:

    C = recurring_only + overlap + unusual_only

The script explicitly audits negative raw overlap and reconstruction error
before the decomposition is interpreted.

Outputs
-------
processed_data/cross_dataset_analysis/03_flood_composition/20_adm2_recurring_unusual_composition/

    tables/
        20_adm2_flood_mask_composition_annual.csv
        20_adm2_flood_mask_composition_summary.csv
        20_adm2_recent_change_summary.csv
        20_adm2_qc_summary.csv
        20_adm2_composition_class_counts.csv

    figures/
        20_01_adm2_mean_annual_flood_extent_by_mask_type.png
        20_02_top_adm2_recent_composition.png
        20_03_adm2_recent_change_recurring_vs_unusual.png
        20_04_adm2_recent_composition_scatter.png
        20_05_top_adm2_unusual_dominated.png

    qc/
        20_adm2_composition_qc_summary.txt
        20_negative_overlap_rows.csv              [only if present]
        20_reconstruction_error_rows.csv          [only if present]

Notes on time windows
---------------------
For this composition module, the formal comparison is retained as:

    Early  = 2000–2019
    Recent = 2020–2025

This is intentionally a composition-specific comparison window and is NOT
silently substituted for the temporal-pattern windows used in Scripts 12–15.
The report should state this distinction explicitly.
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

ADM_LEVEL = "adm2"

EARLY_START = 2000
EARLY_END = 2019
RECENT_START = 2020
RECENT_END = 2025

TOP_N = 20
DOMINANCE_SHARE = 0.60

PROJECT_ROOT = None

# Optional manual input paths. Leave as None to use deterministic automatic search.
# If more than one valid copy of an input file exists at the same priority level,
# the script will stop and ask you to set the path manually rather than silently
# selecting an older duplicate.
BY_TYPE_CSV = None
COMBINED_CSV = None

# Numerical QA tolerance.
OVERLAP_TOL = 1e-6
RECONSTRUCTION_TOL = 1e-6


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

# Deterministic input search priority.
# Priority 1 is the formal ADM2 flood-analysis area used by the current pipeline.
# Lower-priority roots are retained only for backward compatibility.
PREFERRED_INPUT_ROOTS = [
    ROOT / "processed_data" / "cross_dataset_analysis" / "01_adm2_flood",
    ROOT / "processed_data" / "cross_dataset_analysis",
    ROOT / "processed_data",
]

OUTPUT_ROOT = (
    ROOT
    / "processed_data"
    / "cross_dataset_analysis"
    / "03_flood_composition"
    / "20_adm2_recurring_unusual_composition"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"

for d in (TABLE_DIR, FIG_DIR, QC_DIR):
    d.mkdir(parents=True, exist_ok=True)


def find_file(manual_path, filename):
    """
    Locate an input file deterministically.

    Rules
    -----
    1. A manually supplied path always wins.
    2. Search preferred roots in priority order.
    3. Within the first priority root containing matches, accept exactly one match.
    4. If multiple matches exist at that same priority level, stop rather than
       silently choosing a potentially outdated copy.
    """
    if manual_path:
        p = Path(manual_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manual path does not exist: {p}")
        if not p.is_file():
            raise FileNotFoundError(f"Manual path is not a file: {p}")
        return p

    searched = []

    for base in PREFERRED_INPUT_ROOTS:
        base = base.resolve()
        searched.append(base)

        if not base.exists():
            continue

        matches = []

        direct = base / filename
        if direct.exists() and direct.is_file():
            matches.append(direct.resolve())

        matches.extend(
            p.resolve()
            for p in base.rglob(filename)
            if p.is_file() and p.resolve() not in matches
        )

        matches = sorted(set(matches), key=lambda p: (len(str(p)), str(p)))

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            listed = "\n".join(f"  - {p}" for p in matches)
            raise RuntimeError(
                f"Multiple copies of {filename!r} were found under the same "
                f"priority root:\n{base}\n\n{listed}\n\n"
                "To avoid reading an outdated CSV, set the corresponding "
                "manual path in the CONFIG section."
            )

    searched_text = "\n".join(f"  - {p}" for p in searched)
    raise FileNotFoundError(
        f"Could not find {filename!r}. Searched roots:\n{searched_text}"
    )


BY_TYPE_PATH = find_file(
    BY_TYPE_CSV,
    "adm2_annual_flood_indicators_by_type.csv",
)

COMBINED_PATH = find_file(
    COMBINED_CSV,
    "adm2_annual_flood_indicators_combined.csv",
)


# =============================================================================
# COLUMN HELPERS
# =============================================================================

def pick_column(df, exact=(), contains=(), excludes=()):
    """
    Exact match first. Fuzzy matching only runs when `contains` is non-empty.
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
    year = pick_column(
        df,
        exact=("year", "flood_year"),
        contains=("year",),
    )

    adm2_name = pick_column(
        df,
        exact=("ADM2_NAME", "adm2_name", "NAME_2", "name_2", "admin2_name"),
        contains=("adm2", "name"),
    )

    adm2_id = pick_column(
        df,
        exact=("ADM2_ID", "adm2_id", "GID_2", "gid_2", "adm2_code", "ADM2_PCODE"),
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

    return {
        "year": year,
        "adm2_name": adm2_name,
        "adm2_id": adm2_id,
        "metric": metric,
        "type": flood_type,
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
    """
    Normalized Shannon entropy of positive composition shares.
    0 = one component dominates completely.
    1 = components are maximally even.
    """
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
print("20 — ADM2 RECURRING vs UNUSUAL FLOOD-MASK COMPOSITION ANALYSIS")
print("=" * 100)
print("Version      : FINAL / current reference")
print(f"Project root : {ROOT}")
print(f"Output dir   : {OUTPUT_ROOT}")
print(f"Early window : {EARLY_START}-{EARLY_END}")
print(f"Recent window: {RECENT_START}-{RECENT_END}")
print()

print(f"By-type file : {BY_TYPE_PATH}")
print(f"Combined file: {COMBINED_PATH}")

by_type_raw = pd.read_csv(BY_TYPE_PATH)
combined_raw = pd.read_csv(COMBINED_PATH)

c1 = detect_columns(by_type_raw)
c2 = detect_columns(combined_raw)

print("\nDetected columns in by-type file:")
for k, v in c1.items():
    print(f"  {k:12s}: {v}")

print("\nDetected columns in combined file:")
for k, v in c2.items():
    print(f"  {k:12s}: {v}")

required_by_type = ("year", "adm2_name", "metric", "type")
missing = [x for x in required_by_type if c1[x] is None]

if missing:
    raise ValueError(
        f"Missing required columns in by-type file: {missing}\n"
        f"Available columns: {list(by_type_raw.columns)}"
    )

required_combined = ("year", "adm2_name", "metric")
missing = [x for x in required_combined if c2[x] is None]

if missing:
    raise ValueError(
        f"Missing required columns in combined file: {missing}\n"
        f"Available columns: {list(combined_raw.columns)}"
    )


# =============================================================================
# CLEAN BY-TYPE
# =============================================================================

by_type = by_type_raw.copy()

by_type["_type"] = by_type[c1["type"]].map(normalise_type)
by_type = by_type[
    by_type["_type"].isin(["recurring", "unusual"])
].copy()

by_type[c1["year"]] = pd.to_numeric(
    by_type[c1["year"]],
    errors="coerce",
)

by_type[c1["metric"]] = pd.to_numeric(
    by_type[c1["metric"]],
    errors="coerce",
)

by_type = by_type.dropna(
    subset=[c1["year"], c1["adm2_name"], c1["metric"]]
)

by_type[c1["year"]] = by_type[c1["year"]].astype(int)

print("\nFlood types in by-type file:")
print(by_type["_type"].value_counts().to_string())


# =============================================================================
# CLEAN COMBINED — MASTER ADM2 × YEAR UNIVERSE
# =============================================================================

combined = combined_raw.copy()

combined[c2["year"]] = pd.to_numeric(
    combined[c2["year"]],
    errors="coerce",
)

combined[c2["metric"]] = pd.to_numeric(
    combined[c2["metric"]],
    errors="coerce",
)

combined = combined.dropna(
    subset=[c2["year"], c2["adm2_name"], c2["metric"]]
)

combined[c2["year"]] = combined[c2["year"]].astype(int)

rename = {
    c2["adm2_name"]: "ADM2_NAME_STD",
    c2["year"]: "year",
    c2["metric"]: "combined",
}

if c2["adm2_id"] is not None:
    rename[c2["adm2_id"]] = "ADM2_ID_STD"

combined = combined.rename(columns=rename)

master_cols = ["ADM2_NAME_STD", "year", "combined"]

if "ADM2_ID_STD" in combined.columns:
    master_cols.insert(0, "ADM2_ID_STD")

combined = combined[master_cols].copy()

master_key = ["ADM2_NAME_STD", "year"]

if "ADM2_ID_STD" in combined.columns:
    master_key.insert(0, "ADM2_ID_STD")

dup_master = combined.duplicated(master_key, keep=False)

if dup_master.any():
    print(
        f"\nWARNING: {dup_master.sum():,} duplicate combined rows found on "
        f"{master_key}; aggregating duplicate metric values by mean."
    )

    combined = (
        combined.groupby(master_key, as_index=False)
        .agg(combined=("combined", "mean"))
    )


# =============================================================================
# PIVOT RECURRING / UNUSUAL
# =============================================================================

by_rename = {
    c1["adm2_name"]: "ADM2_NAME_STD",
    c1["year"]: "year",
    c1["metric"]: "flood_pct",
}

if c1["adm2_id"] is not None:
    by_rename[c1["adm2_id"]] = "ADM2_ID_STD"

by_type = by_type.rename(columns=by_rename)

pivot_key = ["ADM2_NAME_STD", "year"]

if "ADM2_ID_STD" in by_type.columns:
    pivot_key.insert(0, "ADM2_ID_STD")

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

for col in ("recurring", "unusual"):
    if col not in type_wide.columns:
        type_wide[col] = 0.0


# =============================================================================
# MERGE ONTO COMBINED MASTER
# =============================================================================

merge_key = ["ADM2_NAME_STD", "year"]

if (
    "ADM2_ID_STD" in combined.columns
    and "ADM2_ID_STD" in type_wide.columns
):
    merge_key.insert(0, "ADM2_ID_STD")

wide = combined.merge(
    type_wide[merge_key + ["recurring", "unusual"]],
    on=merge_key,
    how="left",
    validate="one_to_one",
)

# A missing type row means that flood type was absent for that ADM2-year.
missing_recurring_before_fill = int(wide["recurring"].isna().sum())
missing_unusual_before_fill = int(wide["unusual"].isna().sum())

wide["recurring"] = pd.to_numeric(
    wide["recurring"],
    errors="coerce",
).fillna(0.0)

wide["unusual"] = pd.to_numeric(
    wide["unusual"],
    errors="coerce",
).fillna(0.0)

wide["combined"] = pd.to_numeric(
    wide["combined"],
    errors="coerce",
).fillna(0.0)

for col in ("recurring", "unusual", "combined"):
    wide[col] = wide[col].clip(lower=0)

print("\nMerged ADM2-year universe:")
print(f"  rows                        : {len(wide):,}")
print(f"  unique ADM2 names           : {wide['ADM2_NAME_STD'].nunique():,}")
print(f"  unique years                : {wide['year'].nunique():,}")
print(f"  missing recurring -> zero   : {missing_recurring_before_fill:,}")
print(f"  missing unusual -> zero     : {missing_unusual_before_fill:,}")


# =============================================================================
# COMPOSITION
# =============================================================================

wide["overlap_raw"] = (
    wide["recurring"]
    + wide["unusual"]
    - wide["combined"]
)

negative_overlap = wide["overlap_raw"] < -OVERLAP_TOL

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
    wide["component_sum"] - wide["combined"]
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

wide["composition_share_sum"] = (
    wide["recurring_only_share"]
    + wide["unusual_only_share"]
    + wide["overlap_share"]
)

if negative_overlap.any():
    print("\nWARNING:")
    print(
        f"  {negative_overlap.sum():,} rows have recurring + unusual < combined."
    )
    print(
        "  These rows violate strict inclusion-exclusion expectations. "
        "They are exported to QC and should be inspected before literal "
        "overlap interpretation."
    )

    wide.loc[negative_overlap].to_csv(
        QC_DIR / "20_negative_overlap_rows.csv",
        index=False,
    )

material_reconstruction = (
    wide["reconstruction_error"].abs() > RECONSTRUCTION_TOL
)

if material_reconstruction.any():
    wide.loc[material_reconstruction].to_csv(
        QC_DIR / "20_reconstruction_error_rows.csv",
        index=False,
    )


# =============================================================================
# SAVE ANNUAL TABLE
# =============================================================================

annual_path = TABLE_DIR / "20_adm2_flood_mask_composition_annual.csv"
wide.to_csv(annual_path, index=False)


# =============================================================================
# ADM2 SUMMARY
# =============================================================================

summary_rows = []

group_cols = ["ADM2_NAME_STD"]

if "ADM2_ID_STD" in wide.columns:
    group_cols.insert(0, "ADM2_ID_STD")

for keys, g in wide.groupby(group_cols, dropna=False):

    if not isinstance(keys, tuple):
        keys = (keys,)

    row = dict(zip(group_cols, keys))

    g = g.sort_values("year")

    early = g[
        g["year"].between(
            EARLY_START,
            EARLY_END,
            inclusive="both",
        )
    ]

    recent = g[
        g["year"].between(
            RECENT_START,
            RECENT_END,
            inclusive="both",
        )
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
        row["recent_combined_mean"]
        - row["early_combined_mean"]
    )

    row["recurring_change_recent_minus_early"] = (
        row["recent_recurring_mean"]
        - row["early_recurring_mean"]
    )

    row["unusual_change_recent_minus_early"] = (
        row["recent_unusual_mean"]
        - row["early_unusual_mean"]
    )

    row["recent_composition_entropy"] = normalized_entropy([
        row["recent_recurring_only_share"],
        row["recent_unusual_only_share"],
        row["recent_overlap_share"],
    ])

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)


def classify(row):
    if (
        pd.isna(row["recent_combined_mean"])
        or row["recent_combined_mean"] <= 0
    ):
        return "No / negligible flood signal"

    components = {
        "Recurring-dominated": row["recent_recurring_only_share"],
        "Unusual-dominated": row["recent_unusual_only_share"],
        "High overlap": row["recent_overlap_share"],
    }

    label, value = max(
        components.items(),
        key=lambda x: x[1],
    )

    if value >= DOMINANCE_SHARE:
        return label

    return "Mixed composition"


summary["recent_composition_class"] = summary.apply(
    classify,
    axis=1,
)

summary = summary.sort_values(
    ["recent_combined_mean", "longterm_combined_mean"],
    ascending=False,
)

summary_path = (
    TABLE_DIR
    / "20_adm2_flood_mask_composition_summary.csv"
)
summary.to_csv(summary_path, index=False)


# =============================================================================
# CHANGE SUMMARY
# =============================================================================

change_cols = [
    c for c in (
        "ADM2_ID_STD",
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
]

change_path = (
    TABLE_DIR
    / "20_adm2_recent_change_summary.csv"
)

summary[change_cols].to_csv(
    change_path,
    index=False,
)


# =============================================================================
# QC SUMMARY TABLE
# =============================================================================

qc_rows = []

for keys, g in wide.groupby(group_cols, dropna=False):

    if not isinstance(keys, tuple):
        keys = (keys,)

    row = dict(zip(group_cols, keys))

    row.update({
        "n_years": int(g["year"].nunique()),
        "negative_overlap_rows": int(
            (g["overlap_raw"] < -OVERLAP_TOL).sum()
        ),
        "max_abs_reconstruction_error": float(
            g["reconstruction_error"].abs().max()
        ),
        "max_abs_share_sum_error_for_positive_combined": float(
            (
                g.loc[
                    g["combined"] > 0,
                    "composition_share_sum",
                ]
                - 1.0
            ).abs().max()
            if (g["combined"] > 0).any()
            else 0.0
        ),
    })

    qc_rows.append(row)

qc_summary = pd.DataFrame(qc_rows)

qc_path = TABLE_DIR / "20_adm2_qc_summary.csv"
qc_summary.to_csv(qc_path, index=False)


# =============================================================================
# COMPOSITION CLASS COUNTS
# =============================================================================

class_counts = (
    summary["recent_composition_class"]
    .value_counts(dropna=False)
    .rename_axis("recent_composition_class")
    .reset_index(name="n_adm2")
)

class_counts["share_adm2"] = (
    class_counts["n_adm2"] / len(summary)
)

class_counts_path = (
    TABLE_DIR
    / "20_adm2_composition_class_counts.csv"
)

class_counts.to_csv(
    class_counts_path,
    index=False,
)


# =============================================================================
# FIGURE 1 — MEAN TRAJECTORY
# =============================================================================

national = (
    wide.groupby("year")[
        ["combined", "recurring", "unusual"]
    ]
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

ax.axvline(
    RECENT_START,
    linestyle="--",
    alpha=0.7,
)

ax.set_title(
    "ADM2 Mean Annual Flood Extent by Flood-Mask Type"
)
ax.set_xlabel("Year")
ax.set_ylabel("Mean annual flooded area (%)")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR
    / "20_01_adm2_mean_annual_flood_extent_by_mask_type.png",
    dpi=220,
)
plt.close(fig)


# =============================================================================
# FIGURE 2 — TOP ADM2 RECENT COMPOSITION
# =============================================================================

plot_df = (
    summary.head(TOP_N)
    .copy()
    .sort_values("recent_combined_mean")
)

fig, ax = plt.subplots(
    figsize=(12, max(7, 0.42 * len(plot_df) + 2))
)

y = np.arange(len(plot_df))

v1 = plot_df["recent_recurring_only_mean"].to_numpy()
v2 = plot_df["recent_overlap_mean"].to_numpy()
v3 = plot_df["recent_unusual_only_mean"].to_numpy()

ax.barh(
    y,
    v1,
    label="Recurring only",
)

ax.barh(
    y,
    v2,
    left=v1,
    label="Recurring ∩ unusual",
)

ax.barh(
    y,
    v3,
    left=v1 + v2,
    label="Unusual only",
)

ax.set_yticks(y)
ax.set_yticklabels(
    plot_df["ADM2_NAME_STD"],
    fontsize=8,
)

ax.set_xlabel(
    "Recent mean annual flooded area (%)"
)

ax.set_title(
    f"Recent Flood-Mask Composition of Top "
    f"{len(plot_df)} ADM2 Areas "
    f"({RECENT_START}-{RECENT_END})"
)

ax.legend()
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR
    / "20_02_top_adm2_recent_composition.png",
    dpi=220,
)
plt.close(fig)


# =============================================================================
# FIGURE 3 — CHANGE SCATTER
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

x = summary[
    "recurring_change_recent_minus_early"
]

y = summary[
    "unusual_change_recent_minus_early"
]

sizes = (
    30
    + 20
    * np.sqrt(
        summary["recent_combined_mean"]
        .clip(lower=0)
        .fillna(0)
    )
)

ax.scatter(
    x,
    y,
    s=sizes,
    alpha=0.72,
)

ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)

label_df = summary.nlargest(
    min(10, len(summary)),
    "combined_change_recent_minus_early",
)

for _, r in label_df.iterrows():
    ax.annotate(
        str(r["ADM2_NAME_STD"]),
        (
            r["recurring_change_recent_minus_early"],
            r["unusual_change_recent_minus_early"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel(
    "Change in recurring flooded area: "
    "recent - early (percentage points)"
)

ax.set_ylabel(
    "Change in unusual flooded area: "
    "recent - early (percentage points)"
)

ax.set_title(
    "ADM2 Recurring vs Unusual Contribution "
    "to Recent Flood Change"
)

ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR
    / "20_03_adm2_recent_change_recurring_vs_unusual.png",
    dpi=220,
)
plt.close(fig)


# =============================================================================
# FIGURE 4 — RECENT COMPOSITION SCATTER
# =============================================================================

valid = summary[
    summary["recent_combined_mean"] > 0
].copy()

fig, ax = plt.subplots(figsize=(10, 8))

bubble = (
    30
    + 700
    * valid[
        "recent_overlap_share"
    ].clip(0, 1)
)

ax.scatter(
    valid["recent_recurring_only_share"] * 100,
    valid["recent_unusual_only_share"] * 100,
    s=bubble,
    alpha=0.68,
)

for _, r in valid.nlargest(
    min(8, len(valid)),
    "recent_combined_mean",
).iterrows():

    ax.annotate(
        str(r["ADM2_NAME_STD"]),
        (
            r["recent_recurring_only_share"] * 100,
            r["recent_unusual_only_share"] * 100,
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel(
    "Recurring-only share of recent combined "
    "flood extent (%)"
)

ax.set_ylabel(
    "Unusual-only share of recent combined "
    "flood extent (%)"
)

ax.set_title(
    "Recent Flood-Mask Composition — ADM2\n"
    "Bubble size = recurring-unusual overlap share"
)

ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR
    / "20_04_adm2_recent_composition_scatter.png",
    dpi=220,
)
plt.close(fig)


# =============================================================================
# FIGURE 5 — TOP UNUSUAL-DOMINATED ADM2
# =============================================================================

unusual_df = summary[
    summary["recent_composition_class"]
    == "Unusual-dominated"
].copy()

unusual_df = (
    unusual_df
    .nlargest(
        min(TOP_N, len(unusual_df)),
        "recent_combined_mean",
    )
    .sort_values("recent_combined_mean")
)

if len(unusual_df):

    fig, ax = plt.subplots(
        figsize=(
            12,
            max(7, 0.42 * len(unusual_df) + 2),
        )
    )

    y = np.arange(len(unusual_df))

    ax.barh(
        y,
        unusual_df["recent_combined_mean"],
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        unusual_df["ADM2_NAME_STD"],
        fontsize=8,
    )

    ax.set_xlabel(
        "Recent mean combined flooded area (%)"
    )

    ax.set_title(
        f"Top {len(unusual_df)} "
        "Unusual-Dominated ADM2 Flood Areas"
    )

    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIG_DIR
        / "20_05_top_adm2_unusual_dominated.png",
        dpi=220,
    )
    plt.close(fig)


# =============================================================================
# FINAL QC
# =============================================================================

n_adm2 = int(summary["ADM2_NAME_STD"].nunique())
n_years = int(wide["year"].nunique())
negative_n = int(
    (wide["overlap_raw"] < -OVERLAP_TOL).sum()
)

max_reconstruction_error = float(
    wide["reconstruction_error"].abs().max()
)

positive_combined = wide["combined"] > 0

if positive_combined.any():
    max_share_error = float(
        (
            wide.loc[
                positive_combined,
                "composition_share_sum",
            ]
            - 1.0
        )
        .abs()
        .max()
    )
else:
    max_share_error = 0.0

qc_text = f"""20 — ADM2 Recurring vs Unusual Flood Composition
====================================================

Project root:
{ROOT}

Input by-type:
{BY_TYPE_PATH}

Input combined:
{COMBINED_PATH}

Input selection rule:
Deterministic priority search; ambiguous duplicate inputs are rejected.

Output root:
{OUTPUT_ROOT}

Time windows
------------
Early  : {EARLY_START}-{EARLY_END}
Recent : {RECENT_START}-{RECENT_END}

Coverage
--------
Unique ADM2             : {n_adm2}
Unique years            : {n_years}
ADM2-year rows          : {len(wide)}
Missing recurring -> 0  : {missing_recurring_before_fill}
Missing unusual -> 0    : {missing_unusual_before_fill}

Inclusion-exclusion QA
----------------------
Negative raw-overlap rows      : {negative_n}
Max abs reconstruction error   : {max_reconstruction_error:.10f}
Max abs composition-share error: {max_share_error:.10f}

Interpretation rule
-------------------
Dominance threshold: {DOMINANCE_SHARE:.2f}

Composition class counts
------------------------
{class_counts.to_string(index=False)}

Methodological note
-------------------
Early/recent windows in this composition module are 2000-2019 and 2020-2025.
These are composition-specific windows and should not be confused with the
temporal-pattern windows used in Scripts 12-15.

DONE.
"""

qc_text_path = (
    QC_DIR
    / "20_adm2_composition_qc_summary.txt"
)

qc_text_path.write_text(
    qc_text,
    encoding="utf-8",
)

print("\n" + "=" * 100)
print("QUALITY CHECK")
print("=" * 100)

print(f"Unique ADM2                  : {n_adm2:,}")
print(f"Unique years                 : {n_years:,}")
print(f"ADM2-year rows               : {len(wide):,}")
print(f"Negative raw-overlap rows    : {negative_n:,}")
print(
    "Max abs reconstruction error: "
    f"{max_reconstruction_error:.10f}"
)
print(
    "Max abs composition-share error: "
    f"{max_share_error:.10f}"
)

print("\nRecent composition classes:")
print(
    summary[
        "recent_composition_class"
    ]
    .value_counts()
    .to_string()
)

print("\nTop ADM2 by recent combined flood extent:")

display_cols = [
    c for c in (
        "ADM2_ID_STD",
        "ADM2_NAME_STD",
        "recent_combined_mean",
        "recent_recurring_only_share",
        "recent_unusual_only_share",
        "recent_overlap_share",
        "combined_change_recent_minus_early",
        "recent_composition_class",
    )
    if c in summary.columns
]

print(
    summary[display_cols]
    .head(10)
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

print("\nNEXT:")
print(
    "Run Script 21 for ADM3 composition, then use "
    "Scripts 20 + 21 as inputs to Script 22 "
    "ADM2-ADM3 cross-scale composition comparison."
)

print("\nDONE.")
