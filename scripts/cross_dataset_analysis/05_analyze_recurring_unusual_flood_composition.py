#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_analyze_recurring_unusual_flood_composition_v2.py

Fix for the previous version:
- adm2_annual_flood_indicators_by_type.csv contains ONLY recurring + unusual.
- combined is stored separately (typically adm2_annual_flood_indicators_combined.csv),
  or may already exist in adm2_annual_flood_indicators_all.csv.

This version:
1) reads recurring + unusual from *_by_type.csv
2) reads combined from *_combined.csv
3) merges by ADM_ID + ADM_NAME + year
4) computes overlap with inclusion-exclusion
5) creates summary tables + figures

Works for ADM2 or ADM3 by changing ADM_LEVEL.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

ADM_LEVEL = "adm2"     # change to "adm3" later if needed
RECENT_START = 2020
EARLY_END = 2019
TOP_N = 20

PROJECT_ROOT = None

# Optional manual paths. Leave as None for automatic search.
BY_TYPE_CSV = None
COMBINED_CSV = None


# ============================================================================
# PATHS
# ============================================================================

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

OUT_DIR = (
    ROOT
    / "processed_data"
    / "cross_dataset_analysis"
    / f"{ADM_LEVEL}_recurring_unusual_composition"
)
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FILE FINDING
# ============================================================================

def find_file(manual_path, filename):
    if manual_path:
        p = Path(manual_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manual path does not exist: {p}")
        return p

    candidates = list((ROOT / "processed_data").rglob(filename))

    if not candidates:
        candidates = list(ROOT.rglob(filename))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: len(str(p)))
    return candidates[0]


by_type_path = find_file(
    BY_TYPE_CSV,
    f"{ADM_LEVEL}_annual_flood_indicators_by_type.csv",
)

combined_path = find_file(
    COMBINED_CSV,
    f"{ADM_LEVEL}_annual_flood_indicators_combined.csv",
)

all_path = find_file(
    None,
    f"{ADM_LEVEL}_annual_flood_indicators_all.csv",
)


# ============================================================================
# COLUMN HELPERS
# ============================================================================

def pick_column(df, exact=(), contains=()):
    lower_map = {str(c).lower(): c for c in df.columns}

    for x in exact:
        if x.lower() in lower_map:
            return lower_map[x.lower()]

    for c in df.columns:
        lc = str(c).lower()
        if all(x.lower() in lc for x in contains):
            return c

    return None


def detect_common_columns(df):
    year = pick_column(df, exact=("year", "flood_year"), contains=("year",))

    if ADM_LEVEL == "adm2":
        name = pick_column(
            df,
            exact=("ADM2_NAME", "adm2_name", "name_2", "admin2_name"),
            contains=("adm2", "name"),
        )
        ident = pick_column(
            df,
            exact=("ADM2_ID", "adm2_id", "gid_2", "adm2_code"),
        )
    else:
        name = pick_column(
            df,
            exact=("ADM3_NAME", "adm3_name", "name_3", "admin3_name"),
            contains=("adm3", "name"),
        )
        ident = pick_column(
            df,
            exact=("ADM3_ID", "adm3_id", "gid_3", "adm3_code"),
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
                "flood" in lc
                and ("percent" in lc or "percentage" in lc or "pct" in lc)
                and "change" not in lc
            ):
                metric = c
                break

    return {
        "year": year,
        "name": name,
        "id": ident,
        "metric": metric,
    }


def normalise_type(x):
    s = str(x).strip().lower()
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


# ============================================================================
# LOAD recurring + unusual
# ============================================================================

print("=" * 92)
print("RECURRING vs UNUSUAL FLOOD-MASK COMPOSITION ANALYSIS — v2")
print("=" * 92)
print(f"Project root : {ROOT}")
print(f"ADM level    : {ADM_LEVEL.upper()}")
print()

if by_type_path is None:
    raise FileNotFoundError(
        f"Could not find {ADM_LEVEL}_annual_flood_indicators_by_type.csv"
    )

print(f"By-type file : {by_type_path}")

by_type = pd.read_csv(by_type_path)
c1 = detect_common_columns(by_type)

type_col = pick_column(
    by_type,
    exact=("flood_type", "mask_type", "type"),
    contains=("type",),
)

required = {
    "year": c1["year"],
    "name": c1["name"],
    "metric": c1["metric"],
    "type": type_col,
}

missing = [k for k, v in required.items() if v is None]
if missing:
    raise ValueError(
        f"Missing required columns in by-type file: {missing}\n"
        f"Columns: {list(by_type.columns)}"
    )

by_type["_type"] = by_type[type_col].map(normalise_type)
by_type = by_type[by_type["_type"].isin(["recurring", "unusual"])].copy()

by_type[c1["year"]] = pd.to_numeric(by_type[c1["year"]], errors="coerce")
by_type[c1["metric"]] = pd.to_numeric(by_type[c1["metric"]], errors="coerce")
by_type = by_type.dropna(subset=[c1["year"], c1["name"], c1["metric"]])
by_type[c1["year"]] = by_type[c1["year"]].astype(int)

print("\nFlood types in by-type file:")
print(by_type["_type"].value_counts().to_string())

index_cols = [c1["name"], c1["year"]]
if c1["id"] is not None:
    index_cols.insert(0, c1["id"])

wide = (
    by_type.pivot_table(
        index=index_cols,
        columns="_type",
        values=c1["metric"],
        aggfunc="mean",
    )
    .reset_index()
)

for col in ("recurring", "unusual"):
    if col not in wide.columns:
        raise ValueError(f"Missing '{col}' after pivot.")


# ============================================================================
# LOAD combined
# ============================================================================

combined = None

if combined_path is not None:
    print(f"Combined file: {combined_path}")
    combined_raw = pd.read_csv(combined_path)
    c2 = detect_common_columns(combined_raw)

    missing2 = [k for k in ("year", "name", "metric") if c2[k] is None]
    if missing2:
        raise ValueError(
            f"Missing required columns in combined file: {missing2}\n"
            f"Columns: {list(combined_raw.columns)}"
        )

    combined_raw[c2["year"]] = pd.to_numeric(
        combined_raw[c2["year"]], errors="coerce"
    )
    combined_raw[c2["metric"]] = pd.to_numeric(
        combined_raw[c2["metric"]], errors="coerce"
    )
    combined_raw = combined_raw.dropna(
        subset=[c2["year"], c2["name"], c2["metric"]]
    )
    combined_raw[c2["year"]] = combined_raw[c2["year"]].astype(int)

    # standardize names for merge
    rename_map = {
        c2["name"]: c1["name"],
        c2["year"]: c1["year"],
        c2["metric"]: "combined",
    }

    if c2["id"] is not None and c1["id"] is not None:
        rename_map[c2["id"]] = c1["id"]

    combined = combined_raw.rename(columns=rename_map).copy()

    keep_cols = [c1["name"], c1["year"], "combined"]
    if c1["id"] is not None and c1["id"] in combined.columns:
        keep_cols.insert(0, c1["id"])

    combined = combined[keep_cols].copy()

elif all_path is not None:
    print(f"No separate combined file found.")
    print(f"Using all-types file: {all_path}")

    all_raw = pd.read_csv(all_path)
    c2 = detect_common_columns(all_raw)
    all_type_col = pick_column(
        all_raw,
        exact=("flood_type", "mask_type", "type"),
        contains=("type",),
    )

    if all_type_col is None:
        raise ValueError(
            "all-types file exists, but no flood_type column was found."
        )

    all_raw["_type"] = all_raw[all_type_col].map(normalise_type)
    combined_raw = all_raw[all_raw["_type"] == "combined"].copy()

    combined_raw[c2["year"]] = pd.to_numeric(
        combined_raw[c2["year"]], errors="coerce"
    )
    combined_raw[c2["metric"]] = pd.to_numeric(
        combined_raw[c2["metric"]], errors="coerce"
    )

    rename_map = {
        c2["name"]: c1["name"],
        c2["year"]: c1["year"],
        c2["metric"]: "combined",
    }

    if c2["id"] is not None and c1["id"] is not None:
        rename_map[c2["id"]] = c1["id"]

    combined = combined_raw.rename(columns=rename_map)

    keep_cols = [c1["name"], c1["year"], "combined"]
    if c1["id"] is not None and c1["id"] in combined.columns:
        keep_cols.insert(0, c1["id"])

    combined = combined[keep_cols].copy()

else:
    raise FileNotFoundError(
        "\nCould not find a source for combined flood indicators.\n"
        f"Expected one of:\n"
        f"  {ADM_LEVEL}_annual_flood_indicators_combined.csv\n"
        f"  {ADM_LEVEL}_annual_flood_indicators_all.csv\n"
    )


# ============================================================================
# MERGE
# ============================================================================

merge_cols = [c1["name"], c1["year"]]

if (
    c1["id"] is not None
    and c1["id"] in wide.columns
    and c1["id"] in combined.columns
):
    merge_cols.insert(0, c1["id"])

wide = wide.merge(
    combined,
    on=merge_cols,
    how="left",
    validate="one_to_one",
)

missing_combined = int(wide["combined"].isna().sum())

print(f"\nRows after merge         : {len(wide):,}")
print(f"Rows missing combined    : {missing_combined:,}")

if missing_combined:
    sample = wide.loc[wide["combined"].isna(), merge_cols].head(10)
    print("\nSample unmatched keys:")
    print(sample.to_string(index=False))
    raise ValueError(
        "\nSome recurring/unusual rows could not be matched to combined rows. "
        "Check ADM IDs/names and years."
    )

for col in ("recurring", "unusual", "combined"):
    wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0).clip(lower=0)


# ============================================================================
# COMPOSITION
# ============================================================================

wide["overlap_raw"] = wide["recurring"] + wide["unusual"] - wide["combined"]

# IMPORTANT QA:
# If recurring/unusual/combined were computed as exact set unions,
# overlap_raw should normally be >= 0, allowing only tiny numerical negatives.
material_negative = wide["overlap_raw"] < -1e-6

if material_negative.any():
    print("\nWARNING:")
    print(
        f"{material_negative.sum():,} rows have a negative raw overlap estimate."
    )
    print(
        "This means recurring + unusual < combined for those rows, which should "
        "not happen for a strict pixel-union combined layer. Inspect the source "
        "definitions before interpreting overlap literally."
    )

wide["overlap"] = wide["overlap_raw"].clip(lower=0)
wide["recurring_only"] = (wide["recurring"] - wide["overlap"]).clip(lower=0)
wide["unusual_only"] = (wide["unusual"] - wide["overlap"]).clip(lower=0)

wide["component_sum"] = (
    wide["recurring_only"]
    + wide["unusual_only"]
    + wide["overlap"]
)

wide["reconstruction_error"] = wide["component_sum"] - wide["combined"]

wide["recurring_only_share"] = safe_divide(
    wide["recurring_only"], wide["combined"]
)
wide["unusual_only_share"] = safe_divide(
    wide["unusual_only"], wide["combined"]
)
wide["overlap_share"] = safe_divide(
    wide["overlap"], wide["combined"]
)

annual_out = OUT_DIR / "flood_mask_composition_annual.csv"
wide.to_csv(annual_out, index=False)


# ============================================================================
# SUMMARY
# ============================================================================

name_col = c1["name"]
year_col = c1["year"]

rows = []

for region, g in wide.groupby(name_col):
    g = g.sort_values(year_col)
    early = g[g[year_col] <= EARLY_END]
    recent = g[g[year_col] >= RECENT_START]

    def m(frame, col):
        return float(frame[col].mean()) if len(frame) else np.nan

    row = {
        name_col: region,
        "n_years": int(g[year_col].nunique()),

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
    }

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

    rows.append(row)

summary = pd.DataFrame(rows)

def classify(row):
    if row["recent_combined_mean"] <= 0:
        return "No / negligible flood signal"

    vals = {
        "Recurring-dominated": row["recent_recurring_only_share"],
        "Unusual-dominated": row["recent_unusual_only_share"],
        "High overlap": row["recent_overlap_share"],
    }

    label, value = max(vals.items(), key=lambda x: x[1])

    if value >= 0.60:
        return label

    return "Mixed composition"


summary["recent_composition_class"] = summary.apply(classify, axis=1)

summary = summary.sort_values(
    ["recent_combined_mean", "longterm_combined_mean"],
    ascending=False,
)

summary_out = OUT_DIR / "flood_mask_composition_summary.csv"
summary.to_csv(summary_out, index=False)

change_cols = [
    name_col,
    "early_combined_mean",
    "recent_combined_mean",
    "combined_change_recent_minus_early",
    "recurring_change_recent_minus_early",
    "unusual_change_recent_minus_early",
    "recent_recurring_only_share",
    "recent_unusual_only_share",
    "recent_overlap_share",
    "recent_composition_class",
]

change_out = OUT_DIR / "flood_mask_recent_change_summary.csv"
summary[change_cols].to_csv(change_out, index=False)


# ============================================================================
# FIGURE 1
# ============================================================================

national = (
    wide.groupby(year_col)[["recurring", "unusual", "combined"]]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(13, 7))

ax.plot(
    national[year_col],
    national["combined"],
    marker="o",
    linewidth=2.5,
    label="Combined",
)
ax.plot(
    national[year_col],
    national["recurring"],
    marker="o",
    label="Recurring",
)
ax.plot(
    national[year_col],
    national["unusual"],
    marker="o",
    label="Unusual",
)

ax.axvline(RECENT_START, linestyle="--", alpha=0.7)
ax.set_title(
    f"{ADM_LEVEL.upper()} Mean Annual Flood Extent by Flood-Mask Type"
)
ax.set_xlabel("Year")
ax.set_ylabel("Mean annual flooded area (%)")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "01_national_recurring_unusual_mean_trajectory.png",
    dpi=220,
)
plt.close(fig)


# ============================================================================
# FIGURE 2
# ============================================================================

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

ax.barh(y, v1, label="Recurring only")
ax.barh(y, v2, left=v1, label="Recurring ∩ unusual")
ax.barh(y, v3, left=v1 + v2, label="Unusual only")

ax.set_yticks(y)
ax.set_yticklabels(plot_df[name_col])
ax.set_xlabel("Recent mean annual flooded area (%)")
ax.set_title(
    f"Recent Flood-Mask Composition of Top {len(plot_df)} "
    f"{ADM_LEVEL.upper()} Areas ({RECENT_START}-{int(wide[year_col].max())})"
)
ax.legend()
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "02_top_regions_mask_composition_recent.png",
    dpi=220,
)
plt.close(fig)


# ============================================================================
# FIGURE 3
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

x = summary["recurring_change_recent_minus_early"]
y = summary["unusual_change_recent_minus_early"]

size = (
    30
    + 20
    * np.sqrt(
        summary["recent_combined_mean"].clip(lower=0).fillna(0)
    )
)

ax.scatter(x, y, s=size, alpha=0.72)
ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)

label_df = summary.nlargest(
    min(10, len(summary)),
    "combined_change_recent_minus_early",
)

for _, r in label_df.iterrows():
    ax.annotate(
        str(r[name_col]),
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
    f"Recurring vs Unusual Contribution to Recent Flood Change — "
    f"{ADM_LEVEL.upper()}"
)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "03_recent_change_recurring_vs_unusual.png",
    dpi=220,
)
plt.close(fig)


# ============================================================================
# FIGURE 4
# ============================================================================

valid = summary[summary["recent_combined_mean"] > 0].copy()

fig, ax = plt.subplots(figsize=(10, 8))

bubble = (
    30
    + 700
    * valid["recent_overlap_share"].clip(0, 1)
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
        str(r[name_col]),
        (
            r["recent_recurring_only_share"] * 100,
            r["recent_unusual_only_share"] * 100,
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel(
    "Recurring-only share of recent combined flood extent (%)"
)
ax.set_ylabel(
    "Unusual-only share of recent combined flood extent (%)"
)
ax.set_title(
    f"Recent Flood-Mask Composition — {ADM_LEVEL.upper()}\n"
    "Bubble size = recurring-unusual overlap share"
)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "04_recent_composition_scatter.png",
    dpi=220,
)
plt.close(fig)


# ============================================================================
# FIGURE 5 — RUBKONA
# ============================================================================

rub = wide[
    wide[name_col]
    .astype(str)
    .str.contains("rubkona", case=False, na=False)
].copy()

if len(rub):
    rub = rub.sort_values(year_col)

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(
        rub[year_col],
        rub["combined"],
        marker="o",
        linewidth=2.8,
        label="Combined",
    )
    ax.plot(
        rub[year_col],
        rub["recurring"],
        marker="o",
        linewidth=1.8,
        label="Recurring",
    )
    ax.plot(
        rub[year_col],
        rub["unusual"],
        marker="o",
        linewidth=1.8,
        label="Unusual",
    )

    ax.axvline(RECENT_START, linestyle="--", alpha=0.7)

    ax.set_title(
        "Rubkona: Recurring vs Unusual Flood-Mask Decomposition"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual flooded area (%)")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "05_rubkona_recurring_unusual_decomposition.png",
        dpi=220,
    )
    plt.close(fig)


# ============================================================================
# QA
# ============================================================================

print("\n" + "=" * 92)
print("QUALITY CHECK")
print("=" * 92)

print(f"Annual rows                : {len(wide):,}")
print(f"Expected ADM x year count  : {wide[name_col].nunique()} x {wide[year_col].nunique()}")
print(f"Missing combined rows      : {wide['combined'].isna().sum():,}")
print(
    f"Negative raw-overlap rows  : "
    f"{(wide['overlap_raw'] < -1e-6).sum():,}"
)
print(
    "Max absolute reconstruction error: "
    f"{wide['reconstruction_error'].abs().max():.8f}"
)

print("\nRecent composition classes:")
print(summary["recent_composition_class"].value_counts().to_string())

print("\nTop 10 recent combined-flood areas:")
print(
    summary[
        [
            name_col,
            "recent_combined_mean",
            "recent_recurring_only_share",
            "recent_unusual_only_share",
            "recent_overlap_share",
            "recent_composition_class",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nSaved tables:")
print(f"  {annual_out}")
print(f"  {summary_out}")
print(f"  {change_out}")

print("\nSaved figures:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}")

print("\nDONE.")
