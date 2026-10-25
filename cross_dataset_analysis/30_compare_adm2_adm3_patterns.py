"""
30_compare_adm2_adm3_patterns.py

JBG060 South Sudan Flood Analysis
FINAL ADM2 × ADM3 cross-scale temporal-pattern comparison

Purpose
-------
Compare the finalized ADM2 temporal regimes with the finalized, coverage-aware
ADM3 temporal typology. This script does NOT rerun clustering and does NOT use
the obsolete ADM2 multi-label / old ADM3 hotspot taxonomy.

Expected finalized classes
--------------------------
ADM2:
- Persistent low / stable
- Gradual expansion / increasing
- Recent acceleration / expansion
- Recurrent / persistent flooding
- Episodic / high-volatility flooding
- Extreme recent expansion

ADM3:
- Recurrent / persistent
- Episodic / high-volatility
- Moderate recent expansion
- Strong recent expansion
- Insufficient source coverage
- No detected flood record

Input strategy
--------------
Because earlier project versions used different output filenames, this script
discovers the finalized ADM2 and ADM3 CSVs inside:
  processed_data/cross_dataset_analysis/01_adm2_flood/
  processed_data/cross_dataset_analysis/02_adm3_flood/

It scores candidate CSVs by:
1) expected class labels,
2) expected geography columns,
3) expected temporal metrics,
4) preference for final/finalized/classification/summary files,
5) preference for Script 14 outputs for ADM3.

The selected files and detected columns are written to QC output.

Outputs
-------
processed_data/cross_dataset_analysis/04_cross_scale/
└── 30_compare_adm2_adm3_patterns/
    ├── tables/
    │   ├── 30_01_adm3_cross_scale_comparison.csv
    │   ├── 30_02_adm2_cross_scale_summary.csv
    │   ├── 30_03_relationship_counts.csv
    │   ├── 30_04_adm2_adm3_pattern_matrix.csv
    │   ├── 30_05_local_stronger_recent_expansion.csv
    │   ├── 30_06_regional_expansion_local_persistence.csv
    │   ├── 30_07_regional_persistent_local_expansion.csv
    │   ├── 30_08_regional_low_local_nonlow_signal.csv
    │   ├── 30_09_high_heterogeneity_adm2.csv
    │   ├── 30_10_coverage_status_by_adm2.csv
    │   └── 30_11_qc_summary.csv
    └── qc/
        └── 30_cross_scale_qc.txt

Important interpretation rule
-----------------------------
The ADM2 and ADM3 taxonomies were derived independently and are not forced into
one-to-one equivalence. "Cross-scale relationship" is therefore descriptive:
it identifies broad alignment, local strengthening, persistence/volatility
differences, and within-parent heterogeneity. Coverage-status ADM3 units are
reported but excluded from hydrological relationship and entropy calculations.
"""

from __future__ import annotations

from pathlib import Path
import math
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path("/Users/shibai/Documents/GitHub/DBL-Group-22")

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "cross_dataset_analysis"
)

ADM2_ROOT = PROCESSED_ROOT / "01_adm2_flood"
ADM3_ROOT = PROCESSED_ROOT / "02_adm3_flood"

OUTPUT_ROOT = (
    PROCESSED_ROOT
    / "04_cross_scale"
    / "30_compare_adm2_adm3_patterns"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
QC_DIR = OUTPUT_ROOT / "qc"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. FINAL TAXONOMIES
# =============================================================================

ADM2_CLASSES = [
    "Persistent low / stable",
    "Gradual expansion / increasing",
    "Recent acceleration / expansion",
    "Recurrent / persistent flooding",
    "Episodic / high-volatility flooding",
    "Extreme recent expansion",
]

ADM3_HYDRO_CLASSES = [
    "Recurrent / persistent",
    "Episodic / high-volatility",
    "Moderate recent expansion",
    "Strong recent expansion",
]

ADM3_STATUS_CLASSES = [
    "Insufficient source coverage",
    "No detected flood record",
]

ADM3_ALL_CLASSES = ADM3_HYDRO_CLASSES + ADM3_STATUS_CLASSES

# =============================================================================
# 3. HELPERS
# =============================================================================

def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[_\-\/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_name(value) -> str:
    return norm_text(value)


def detect_column(columns, candidates):
    exact = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]

    normalized = {norm_text(c): c for c in columns}
    for candidate in candidates:
        key = norm_text(candidate)
        if key in normalized:
            return normalized[key]
    return None


def normalized_entropy(series: pd.Series) -> float:
    counts = series.dropna().value_counts().astype(float)
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    h = -float(np.sum(p * np.log(p)))
    hmax = math.log(len(counts))
    return h / hmax if hmax > 0 else 0.0


def dominant_value(series: pd.Series):
    counts = series.dropna().value_counts()
    return counts.index[0] if not counts.empty else np.nan


def dominant_share(series: pd.Series) -> float:
    counts = series.dropna().value_counts()
    if counts.empty:
        return np.nan
    return float(counts.iloc[0] / counts.sum())


def canonicalize(value, allowed):
    """Map harmless punctuation/case variants to the exact final label."""
    key = norm_text(value)
    mapping = {norm_text(x): x for x in allowed}
    return mapping.get(key, str(value).strip() if not pd.isna(value) else np.nan)


def safe_read_header(path: Path):
    try:
        return pd.read_csv(path, nrows=5)
    except Exception:
        return None


def discover_final_csv(root: Path, level: str):
    """Use verified final outputs, never heuristic CSV discovery."""
    if level == "ADM2":
        path = root / "temporal_patterns" / "tables" / "03_pattern_membership.csv"
        pattern_col = "pattern_name"
        name_col, id_col = "ADM2_NAME", "ADM2_ID"
        expected_rows = 79
    elif level == "ADM3":
        path = (root / "14_finalize_adm3_temporal_pattern_classification" /
                "tables" / "14_adm3_final_temporal_pattern_classification_all512.csv")
        pattern_col = "final_temporal_pattern"
        name_col, id_col = "ADM3_NAME", "ADM3_ID"
        expected_rows = 512
    else:
        raise ValueError(level)
    if not path.is_file():
        raise FileNotFoundError(f"Required FINAL {level} file missing: {path}")
    header = pd.read_csv(path, nrows=0).columns
    required = [pattern_col, name_col, id_col]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"FINAL {level} file has missing columns {missing}: {path}")
    df = pd.read_csv(path)
    if len(df) != expected_rows:
        raise ValueError(f"FINAL {level} must have {expected_rows} rows; got {len(df)}: {path}")
    if df[id_col].isna().any() or df[id_col].astype(str).duplicated().any():
        raise ValueError(f"FINAL {level} contains blank/duplicate IDs: {path}")
    expected_classes = ADM2_CLASSES if level == "ADM2" else ADM3_ALL_CLASSES
    values = set(df[pattern_col].dropna().map(lambda v: canonicalize(v, expected_classes)))
    unknown = values - set(expected_classes)
    if unknown or df[pattern_col].isna().any():
        raise ValueError(f"Unknown/missing final {level} labels: {unknown} in {path}")
    print(f"Verified {level}: {len(df)} rows, {len(values)} classes, {path}")
    return dict(path=path, pattern_col=pattern_col, name_col=name_col,
                id_col=id_col, rows=len(df))


def find_parent_adm2_column(df: pd.DataFrame):
    return detect_column(
        df.columns,
        [
            "ADM2_NAME",
            "adm2_name",
            "PARENT_ADM2_NAME",
            "parent_adm2_name",
            "parent_adm2",
            "county",
            "county_name",
            "NAME_2",
        ],
    )


def find_metric(df: pd.DataFrame, candidates):
    col = detect_column(df.columns, candidates)
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


# =============================================================================
# 4. DISCOVER + LOAD FINAL ADM2
# =============================================================================

print("=" * 88)
print("SCRIPT 30 — FINAL ADM2 × ADM3 CROSS-SCALE TEMPORAL COMPARISON")
print("=" * 88)

adm2_info = discover_final_csv(ADM2_ROOT, "ADM2")
ADM2_FILE = adm2_info["path"]

adm2_raw = pd.read_csv(ADM2_FILE)
ADM2_FEATURES_FILE = ADM2_ROOT / "temporal_patterns" / "tables" / "01_adm2_temporal_features.csv"
if not ADM2_FEATURES_FILE.is_file():
    raise FileNotFoundError(f"Missing ADM2 temporal features: {ADM2_FEATURES_FILE}")
adm2_features = pd.read_csv(ADM2_FEATURES_FILE)
if adm2_features["ADM2_ID"].duplicated().any():
    raise ValueError("Duplicate ADM2_ID in temporal features")
if len(adm2_features) != 79:
    raise ValueError(f"Expected 79 ADM2 feature rows, got {len(adm2_features)}")
adm2_raw = adm2_raw.merge(
    adm2_features.drop(columns=["ADM2_NAME"], errors="ignore"),
    on="ADM2_ID", how="left", validate="one_to_one", indicator=True
)
if not adm2_raw["_merge"].eq("both").all():
    raise ValueError("ADM2 features and final membership IDs do not match")
adm2_raw = adm2_raw.drop(columns="_merge")

adm2_name_col = adm2_info["name_col"]
adm2_id_col = adm2_info["id_col"]
adm2_pattern_col = adm2_info["pattern_col"]

adm2 = pd.DataFrame(index=adm2_raw.index)

if adm2_name_col is not None:
    adm2["ADM2_NAME"] = adm2_raw[adm2_name_col].astype(str).str.strip()
else:
    adm2["ADM2_NAME"] = adm2_raw[adm2_id_col].astype(str).str.strip()

if adm2_id_col is not None:
    adm2["ADM2_ID"] = adm2_raw[adm2_id_col].astype(str).str.strip()
else:
    adm2["ADM2_ID"] = np.nan

adm2["ADM2_PATTERN"] = adm2_raw[adm2_pattern_col].map(
    lambda x: canonicalize(x, ADM2_CLASSES)
)
adm2["ADM2_KEY"] = adm2["ADM2_NAME"].map(norm_name)

# Preserve useful ADM2 metrics when present.
adm2["ADM2_long_term_mean"] = find_metric(
    adm2_raw, ["long_term_mean", "mean_flood", "mean_flooded_pct", "mean_annual_flooded_pct", "mean"]
)
adm2["ADM2_recent_minus_early"] = find_metric(
    adm2_raw, ["recent_minus_early", "change_pp", "early_recent_change", "recent_early_change"]
)
adm2["ADM2_trend_slope"] = find_metric(
    adm2_raw, ["trend_slope", "long_term_slope", "trend_pp_per_year"]
)
adm2["ADM2_recent_5y_change"] = find_metric(
    adm2_raw, ["recent_5y_change", "recent_5y_change_pp", "recent_five_year_change", "recent5_change"]
)

adm2_unknown = sorted(
    set(adm2["ADM2_PATTERN"].dropna()) - set(ADM2_CLASSES)
)
if adm2_unknown:
    raise ValueError(
        "\nSelected ADM2 file contains labels outside the finalized six-class typology:\n"
        + "\n".join(f"  - {x}" for x in adm2_unknown)
        + f"\n\nSelected file:\n{ADM2_FILE}"
    )

if adm2["ADM2_KEY"].duplicated().any():
    dup = adm2.loc[adm2["ADM2_KEY"].duplicated(False), ["ADM2_NAME", "ADM2_PATTERN"]]
    raise ValueError(
        "\nDuplicate ADM2 units found in selected final classification:\n"
        + dup.to_string(index=False)
    )

print(f"\nSelected FINAL ADM2 file:\n{ADM2_FILE}")
print(f"ADM2 rows: {len(adm2):,}")
print("\nADM2 final class counts:")
print(adm2["ADM2_PATTERN"].value_counts(dropna=False).to_string())

# =============================================================================
# 5. DISCOVER + LOAD FINAL ADM3
# =============================================================================

adm3_info = discover_final_csv(ADM3_ROOT, "ADM3")
ADM3_FILE = adm3_info["path"]

adm3_raw = pd.read_csv(ADM3_FILE)

adm3_name_col = adm3_info["name_col"]
adm3_id_col = adm3_info["id_col"]
adm3_pattern_col = adm3_info["pattern_col"]
parent_col = find_parent_adm2_column(adm3_raw)

if parent_col is None:
    raise ValueError(
        "\nThe selected final ADM3 table does not contain a parent ADM2 name column.\n"
        f"Selected file:\n{ADM3_FILE}\n"
        f"Columns:\n{list(adm3_raw.columns)}"
    )

adm3 = pd.DataFrame(index=adm3_raw.index)

if adm3_name_col is not None:
    adm3["ADM3_NAME"] = adm3_raw[adm3_name_col].astype(str).str.strip()
else:
    adm3["ADM3_NAME"] = adm3_raw[adm3_id_col].astype(str).str.strip()

if adm3_id_col is not None:
    adm3["ADM3_ID"] = adm3_raw[adm3_id_col].astype(str).str.strip()
else:
    adm3["ADM3_ID"] = np.arange(1, len(adm3_raw) + 1).astype(str)

adm3["ADM2_NAME"] = adm3_raw[parent_col].astype(str).str.strip()
adm3["ADM2_KEY"] = adm3["ADM2_NAME"].map(norm_name)
adm3["ADM3_PATTERN"] = adm3_raw[adm3_pattern_col].map(
    lambda x: canonicalize(x, ADM3_ALL_CLASSES)
)

# Preserve useful ADM3 metrics when present.
adm3["ADM3_long_term_mean"] = find_metric(
    adm3_raw, ["long_term_mean", "mean_flood", "mean_flooded_pct", "mean_annual_flooded_pct", "mean"]
)
adm3["ADM3_recent_minus_early"] = find_metric(
    adm3_raw, ["recent_minus_early", "change_pp", "early_recent_change", "recent_early_change"]
)
adm3["ADM3_trend_slope"] = find_metric(
    adm3_raw, ["trend_slope", "long_term_slope", "trend_pp_per_year"]
)
adm3["ADM3_recent_5y_change"] = find_metric(
    adm3_raw, ["recent_5y_change", "recent_5y_change_pp", "recent_five_year_change", "recent5_change"]
)

adm3_unknown = sorted(
    set(adm3["ADM3_PATTERN"].dropna()) - set(ADM3_ALL_CLASSES)
)
if adm3_unknown:
    raise ValueError(
        "\nSelected ADM3 file contains labels outside the finalized typology:\n"
        + "\n".join(f"  - {x}" for x in adm3_unknown)
        + f"\n\nSelected file:\n{ADM3_FILE}"
    )

if adm3["ADM3_ID"].duplicated().any():
    raise ValueError("Duplicate ADM3 IDs")

print(f"\nSelected FINAL ADM3 file:\n{ADM3_FILE}")
print(f"ADM3 rows: {len(adm3):,}")
print("\nADM3 final class counts:")
print(adm3["ADM3_PATTERN"].value_counts(dropna=False).to_string())

# =============================================================================
# 6. MERGE ADM2 -> ADM3
# =============================================================================

# Match by official administrative PCODE hierarchy, not spelling of county names.
# Example: ADM3 SS060101 belongs to ADM2 SS0601.
# Refuse to continue if the expected hierarchy does not validate.
adm3["PARENT_ADM2_ID"] = adm3["ADM3_ID"].str.upper().str.strip().str[:6]
adm2["ADM2_ID"] = adm2["ADM2_ID"].str.upper().str.strip()

if not adm3["ADM3_ID"].str.fullmatch(r"SS[0-9]{6}").all():
    raise ValueError("ADM3_ID does not consistently use SS + 6 digit PCODE format")
if not adm2["ADM2_ID"].str.fullmatch(r"SS[0-9]{4}").all():
    raise ValueError("ADM2_ID does not consistently use SS + 4 digit PCODE format")

cross = adm3.merge(
    adm2,
    left_on="PARENT_ADM2_ID",
    right_on="ADM2_ID",
    how="left",
    suffixes=("", "_ADM2"),
    validate="many_to_one",
    indicator=True,
)

missing_parent = int((cross["_merge"] != "both").sum())

# Always output an audit of administrative names, even when PCODEs all match.
cross["parent_name_matches"] = (
    cross["ADM2_NAME"].map(norm_name)
    == cross["ADM2_NAME_ADM2"].map(norm_name)
)
name_disagreements = cross.loc[
    cross["_merge"].eq("both") & ~cross["parent_name_matches"],
    ["ADM3_ID", "ADM3_NAME", "PARENT_ADM2_ID", "ADM2_NAME", "ADM2_NAME_ADM2"]
].copy()
name_disagreements.to_csv(TABLE_DIR / "30_12_adm2_parent_name_disagreements.csv", index=False)

print("\n" + "=" * 88)
print("PARENT–CHILD MERGE QC — PCODE")
print("=" * 88)
print(f"ADM3 rows: {len(cross):,}")
print(f"Unique ADM3 IDs: {cross['ADM3_ID'].nunique():,}")
print(f"Unique parent ADM2 IDs in ADM3 table: {cross['PARENT_ADM2_ID'].nunique():,}")
print(f"ADM3 rows without matched ADM2 PCODE: {missing_parent:,}")
print(f"ADM3 rows with differing parent ADM2 names: {len(name_disagreements):,}")

if missing_parent:
    cross.loc[cross["_merge"] != "both", [
        "ADM3_ID", "ADM3_NAME", "PARENT_ADM2_ID", "ADM2_NAME"
    ]].to_csv(TABLE_DIR / "30_13_unmatched_adm2_ids.csv", index=False)
    raise ValueError("Some ADM3 parent PCODEs cannot be matched. See 30_13_unmatched_adm2_ids.csv")

# Aggregation key must use the matched ADM2 ID, not the possibly inconsistent name.
cross["ADM2_KEY"] = cross["PARENT_ADM2_ID"]
cross = cross.drop(columns="_merge")

# =============================================================================
# 7. COVERAGE-AWARE ANALYTICAL STATUS
# =============================================================================

cross["ADM3_IS_STATUS"] = cross["ADM3_PATTERN"].isin(ADM3_STATUS_CLASSES)
cross["ADM3_IS_HYDRO"] = cross["ADM3_PATTERN"].isin(ADM3_HYDRO_CLASSES)

cross["ADM3_IS_EXPANSION"] = cross["ADM3_PATTERN"].isin(
    ["Moderate recent expansion", "Strong recent expansion"]
)
cross["ADM3_IS_STRONG_EXPANSION"] = cross["ADM3_PATTERN"].eq(
    "Strong recent expansion"
)
cross["ADM3_IS_PERSISTENT"] = cross["ADM3_PATTERN"].eq(
    "Recurrent / persistent"
)
cross["ADM3_IS_VOLATILE"] = cross["ADM3_PATTERN"].eq(
    "Episodic / high-volatility"
)

cross["ADM2_IS_LOW"] = cross["ADM2_PATTERN"].eq(
    "Persistent low / stable"
)
cross["ADM2_IS_EXPANSION"] = cross["ADM2_PATTERN"].isin(
    [
        "Gradual expansion / increasing",
        "Recent acceleration / expansion",
        "Extreme recent expansion",
    ]
)
cross["ADM2_IS_STRONG_EXPANSION"] = cross["ADM2_PATTERN"].isin(
    ["Recent acceleration / expansion", "Extreme recent expansion"]
)
cross["ADM2_IS_PERSISTENT"] = cross["ADM2_PATTERN"].eq(
    "Recurrent / persistent flooding"
)
cross["ADM2_IS_VOLATILE"] = cross["ADM2_PATTERN"].eq(
    "Episodic / high-volatility flooding"
)

# =============================================================================
# 8. DESCRIPTIVE CROSS-SCALE RELATIONSHIP
# =============================================================================

def classify_relationship(row):
    # Data-status rows are never interpreted as hydrological relationships.
    if row["ADM3_PATTERN"] == "Insufficient source coverage":
        return "Coverage status: insufficient source coverage"
    if row["ADM3_PATTERN"] == "No detected flood record":
        return "Coverage status: no detected flood record"

    a2 = row["ADM2_PATTERN"]
    a3 = row["ADM3_PATTERN"]

    # Strong/local recent expansion.
    if a3 == "Strong recent expansion":
        if a2 in ["Recent acceleration / expansion", "Extreme recent expansion"]:
            return "Cross-scale strong recent expansion alignment"
        return "Local strong recent expansion beyond parent regime"

    if a3 == "Moderate recent expansion":
        if a2 in [
            "Gradual expansion / increasing",
            "Recent acceleration / expansion",
            "Extreme recent expansion",
        ]:
            return "Cross-scale recent expansion alignment"
        return "Local moderate recent expansion beyond parent regime"

    # Persistence.
    if a3 == "Recurrent / persistent":
        if a2 == "Recurrent / persistent flooding":
            return "Cross-scale persistence alignment"
        if a2 in [
            "Gradual expansion / increasing",
            "Recent acceleration / expansion",
            "Extreme recent expansion",
        ]:
            return "Regional expansion + local persistence"
        if a2 == "Persistent low / stable":
            return "Regional low/stable + local persistence"
        if a2 == "Episodic / high-volatility flooding":
            return "Regional volatility + local persistence"

    # Volatility.
    if a3 == "Episodic / high-volatility":
        if a2 == "Episodic / high-volatility flooding":
            return "Cross-scale volatility alignment"
        if a2 == "Recurrent / persistent flooding":
            return "Regional persistence + local volatility"
        if a2 in [
            "Gradual expansion / increasing",
            "Recent acceleration / expansion",
            "Extreme recent expansion",
        ]:
            return "Regional expansion + local volatility"
        if a2 == "Persistent low / stable":
            return "Regional low/stable + local volatility"

    return "Other cross-scale combination"


cross["cross_scale_relationship"] = cross.apply(
    classify_relationship, axis=1
)

# Explicit analytical flags.
cross["flag_local_expansion_nonexpanding_parent"] = (
    cross["ADM3_IS_HYDRO"]
    & cross["ADM3_IS_EXPANSION"]
    & ~cross["ADM2_IS_EXPANSION"]
).astype(int)

cross["flag_regional_expansion_local_persistence"] = (
    cross["ADM3_IS_HYDRO"]
    & cross["ADM2_IS_EXPANSION"]
    & cross["ADM3_IS_PERSISTENT"]
).astype(int)

cross["flag_regional_persistent_local_expansion"] = (
    cross["ADM3_IS_HYDRO"]
    & cross["ADM2_IS_PERSISTENT"]
    & cross["ADM3_IS_EXPANSION"]
).astype(int)

cross["flag_regional_low_local_nonlow_signal"] = (
    cross["ADM3_IS_HYDRO"]
    & cross["ADM2_IS_LOW"]
    & (
        cross["ADM3_IS_EXPANSION"]
        | cross["ADM3_IS_PERSISTENT"]
        | cross["ADM3_IS_VOLATILE"]
    )
).astype(int)

# Continuous direction disagreement is supplementary only.
def direction(v):
    if pd.isna(v):
        return "unknown"
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "flat"

cross["ADM2_trend_direction"] = cross["ADM2_trend_slope"].map(direction)
cross["ADM3_trend_direction"] = cross["ADM3_trend_slope"].map(direction)

cross["continuous_trend_opposite"] = (
    cross["ADM3_IS_HYDRO"]
    & (
        (
            cross["ADM2_trend_direction"].eq("positive")
            & cross["ADM3_trend_direction"].eq("negative")
        )
        |
        (
            cross["ADM2_trend_direction"].eq("negative")
            & cross["ADM3_trend_direction"].eq("positive")
        )
    )
).astype(int)

# =============================================================================
# 9. SAVE FULL ADM3-LEVEL CROSS-SCALE TABLE
# =============================================================================

cross = cross.sort_values(["ADM2_NAME", "ADM3_NAME"])

cross.to_csv(
    TABLE_DIR / "30_01_adm3_cross_scale_comparison.csv",
    index=False,
)

# =============================================================================
# 10. ADM2-LEVEL SUMMARY OF INTERNAL ADM3 HETEROGENEITY
# =============================================================================

summary_rows = []

for adm2_key, group in cross.groupby("ADM2_KEY", sort=True):
    hydro = group[group["ADM3_IS_HYDRO"]].copy()
    status = group[group["ADM3_IS_STATUS"]].copy()

    adm2_name = (
        group["ADM2_NAME_ADM2"].dropna().iloc[0]
        if "ADM2_NAME_ADM2" in group.columns
        and group["ADM2_NAME_ADM2"].notna().any()
        else group["ADM2_NAME"].iloc[0]
    )

    row = {
        "ADM2_NAME": adm2_name,
        "ADM2_PATTERN": group["ADM2_PATTERN"].iloc[0],
        "n_adm3_total": len(group),
        "n_adm3_hydrological": len(hydro),
        "n_adm3_status": len(status),
        "n_insufficient_source_coverage": int(
            group["ADM3_PATTERN"].eq("Insufficient source coverage").sum()
        ),
        "n_no_detected_flood_record": int(
            group["ADM3_PATTERN"].eq("No detected flood record").sum()
        ),
        "adm3_dominant_hydrological_pattern": dominant_value(
            hydro["ADM3_PATTERN"]
        ) if len(hydro) else np.nan,
        "adm3_dominant_hydrological_share": dominant_share(
            hydro["ADM3_PATTERN"]
        ) if len(hydro) else np.nan,
        "adm3_hydrological_pattern_entropy": normalized_entropy(
            hydro["ADM3_PATTERN"]
        ) if len(hydro) else np.nan,
        "adm3_unique_hydrological_pattern_count": int(
            hydro["ADM3_PATTERN"].nunique()
        ) if len(hydro) else 0,
        "adm3_persistent_count": int(hydro["ADM3_IS_PERSISTENT"].sum()),
        "adm3_persistent_share": float(hydro["ADM3_IS_PERSISTENT"].mean())
            if len(hydro) else np.nan,
        "adm3_volatile_count": int(hydro["ADM3_IS_VOLATILE"].sum()),
        "adm3_volatile_share": float(hydro["ADM3_IS_VOLATILE"].mean())
            if len(hydro) else np.nan,
        "adm3_moderate_expansion_count": int(
            hydro["ADM3_PATTERN"].eq("Moderate recent expansion").sum()
        ),
        "adm3_strong_expansion_count": int(
            hydro["ADM3_PATTERN"].eq("Strong recent expansion").sum()
        ),
        "adm3_any_expansion_share": float(hydro["ADM3_IS_EXPANSION"].mean())
            if len(hydro) else np.nan,
        "local_expansion_nonexpanding_parent_count": int(
            hydro["flag_local_expansion_nonexpanding_parent"].sum()
        ),
        "continuous_trend_opposite_count": int(
            hydro["continuous_trend_opposite"].sum()
        ),
        "mean_adm3_long_term_mean": hydro["ADM3_long_term_mean"].mean(),
        "mean_adm3_recent_minus_early": hydro["ADM3_recent_minus_early"].mean(),
        "mean_adm3_trend_slope": hydro["ADM3_trend_slope"].mean(),
        "max_adm3_recent_minus_early": hydro["ADM3_recent_minus_early"].max(),
        "max_adm3_recent_5y_change": hydro["ADM3_recent_5y_change"].max(),
    }

    summary_rows.append(row)

adm2_summary = pd.DataFrame(summary_rows)

adm2_summary = adm2_summary.sort_values(
    ["adm3_hydrological_pattern_entropy", "adm3_any_expansion_share"],
    ascending=[False, False],
)

adm2_summary.to_csv(
    TABLE_DIR / "30_02_adm2_cross_scale_summary.csv",
    index=False,
)

# =============================================================================
# 11. RELATIONSHIP COUNTS + PATTERN MATRIX
# =============================================================================

relationship_counts = (
    cross["cross_scale_relationship"]
    .value_counts(dropna=False)
    .rename_axis("cross_scale_relationship")
    .reset_index(name="adm3_count")
)

relationship_counts["share_of_all_512_percent"] = (
    relationship_counts["adm3_count"] / len(cross) * 100
)

hydro_n = int(cross["ADM3_IS_HYDRO"].sum())
relationship_counts["share_of_hydrological_adm3_percent"] = np.where(
    relationship_counts["cross_scale_relationship"].str.startswith("Coverage status:"),
    np.nan,
    relationship_counts["adm3_count"] / hydro_n * 100 if hydro_n else np.nan,
)

relationship_counts.to_csv(
    TABLE_DIR / "30_03_relationship_counts.csv",
    index=False,
)

hydro_cross = cross[cross["ADM3_IS_HYDRO"]].copy()

pattern_matrix = pd.crosstab(
    hydro_cross["ADM2_PATTERN"],
    hydro_cross["ADM3_PATTERN"],
    margins=True,
)

pattern_matrix.to_csv(
    TABLE_DIR / "30_04_adm2_adm3_pattern_matrix.csv"
)

# =============================================================================
# 12. IMPORTANT CASE TABLES
# =============================================================================

cross.loc[
    cross["flag_local_expansion_nonexpanding_parent"].eq(1)
].to_csv(
    TABLE_DIR / "30_05_local_expansion_nonexpanding_parent.csv",
    index=False,
)

cross.loc[
    cross["flag_regional_expansion_local_persistence"].eq(1)
].to_csv(
    TABLE_DIR / "30_06_regional_expansion_local_persistence.csv",
    index=False,
)

cross.loc[
    cross["flag_regional_persistent_local_expansion"].eq(1)
].to_csv(
    TABLE_DIR / "30_07_regional_persistent_local_expansion.csv",
    index=False,
)

cross.loc[
    cross["flag_regional_low_local_nonlow_signal"].eq(1)
].to_csv(
    TABLE_DIR / "30_08_regional_low_local_nonlow_signal.csv",
    index=False,
)

# High heterogeneity = upper quartile of entropy among ADM2 with >=2 hydro ADM3.
eligible_entropy = adm2_summary.loc[
    adm2_summary["n_adm3_hydrological"].ge(2),
    "adm3_hydrological_pattern_entropy",
].dropna()

entropy_q75 = (
    float(eligible_entropy.quantile(0.75))
    if len(eligible_entropy)
    else np.nan
)

high_heterogeneity = adm2_summary[
    adm2_summary["adm3_hydrological_pattern_entropy"].ge(entropy_q75)
].copy()

high_heterogeneity.to_csv(
    TABLE_DIR / "30_09_high_heterogeneity_adm2.csv",
    index=False,
)

coverage_by_adm2 = (
    cross.groupby(["PARENT_ADM2_ID", "ADM2_NAME_ADM2", "ADM2_PATTERN"], dropna=False)
    .agg(
        n_adm3_total=("ADM3_ID", "size"),
        n_hydrological=("ADM3_IS_HYDRO", "sum"),
        n_status=("ADM3_IS_STATUS", "sum"),
        n_insufficient_source_coverage=(
            "ADM3_PATTERN",
            lambda s: int((s == "Insufficient source coverage").sum()),
        ),
        n_no_detected_flood_record=(
            "ADM3_PATTERN",
            lambda s: int((s == "No detected flood record").sum()),
        ),
    )
    .reset_index()
)

if len(coverage_by_adm2) != 79 or coverage_by_adm2["PARENT_ADM2_ID"].nunique() != 79:
    raise ValueError("ADM2 coverage must have exactly 79 unique parent PCODEs")
if int(coverage_by_adm2["n_adm3_total"].sum()) != 512:
    raise ValueError("Coverage summary must account for all 512 ADM3 units")
coverage_by_adm2 = coverage_by_adm2.rename(columns={"ADM2_NAME_ADM2": "ADM2_NAME"})
coverage_by_adm2.to_csv(
    TABLE_DIR / "30_10_coverage_status_by_adm2.csv",
    index=False,
)

# =============================================================================
# 13. QC
# =============================================================================

adm2_counts = adm2["ADM2_PATTERN"].value_counts()
adm3_counts = adm3["ADM3_PATTERN"].value_counts()

qc_rows = [
    ("selected_adm2_file", str(ADM2_FILE)),
    ("selected_adm3_file", str(ADM3_FILE)),
    ("adm2_detected_pattern_column", str(adm2_pattern_col)),
    ("adm3_detected_pattern_column", str(adm3_pattern_col)),
    ("adm3_detected_parent_adm2_column", str(parent_col)),
    ("adm2_rows", len(adm2)),
    ("adm2_unique", adm2["ADM2_KEY"].nunique()),
    ("adm3_rows", len(adm3)),
    ("adm3_unique_ids", adm3["ADM3_ID"].nunique()),
    ("adm3_parent_adm2_unique", adm3["ADM2_KEY"].nunique()),
    ("adm3_parent_missing_after_merge", missing_parent),
    ("adm3_hydrological_units", hydro_n),
    ("adm3_status_units", int(cross["ADM3_IS_STATUS"].sum())),
    (
        "adm3_insufficient_source_coverage",
        int(cross["ADM3_PATTERN"].eq("Insufficient source coverage").sum()),
    ),
    (
        "adm3_no_detected_flood_record",
        int(cross["ADM3_PATTERN"].eq("No detected flood record").sum()),
    ),
    ("entropy_q75", entropy_q75),
    (
        "local_expansion_nonexpanding_parent_count",
        int(cross["flag_local_expansion_nonexpanding_parent"].sum()),
    ),
    (
        "regional_expansion_local_persistence_count",
        int(cross["flag_regional_expansion_local_persistence"].sum()),
    ),
    (
        "regional_persistent_local_expansion_count",
        int(cross["flag_regional_persistent_local_expansion"].sum()),
    ),
    (
        "regional_low_local_nonlow_signal_count",
        int(cross["flag_regional_low_local_nonlow_signal"].sum()),
    ),
    (
        "continuous_trend_opposite_count",
        int(cross["continuous_trend_opposite"].sum()),
    ),
]

for label in ADM2_CLASSES:
    qc_rows.append(
        (
            f"ADM2_count__{label}",
            int(adm2_counts.get(label, 0)),
        )
    )

for label in ADM3_ALL_CLASSES:
    qc_rows.append(
        (
            f"ADM3_count__{label}",
            int(adm3_counts.get(label, 0)),
        )
    )

qc = pd.DataFrame(qc_rows, columns=["metric", "value"])

qc.to_csv(
    TABLE_DIR / "30_11_qc_summary.csv",
    index=False,
)

# Hard expected-count checks based on finalized project outputs.
expected_adm2_counts = {
    "Persistent low / stable": 43,
    "Gradual expansion / increasing": 21,
    "Recent acceleration / expansion": 6,
    "Recurrent / persistent flooding": 5,
    "Episodic / high-volatility flooding": 3,
    "Extreme recent expansion": 1,
}

expected_adm3_counts = {
    "Recurrent / persistent": 223,
    "Episodic / high-volatility": 218,
    "Moderate recent expansion": 39,
    "Strong recent expansion": 10,
    "Insufficient source coverage": 13,
    "No detected flood record": 9,
}

adm2_count_check = all(
    int(adm2_counts.get(k, 0)) == v
    for k, v in expected_adm2_counts.items()
)

adm3_count_check = all(
    int(adm3_counts.get(k, 0)) == v
    for k, v in expected_adm3_counts.items()
)

qc_text = []
qc_text.append("SCRIPT 30 — FINAL ADM2 × ADM3 CROSS-SCALE TEMPORAL COMPARISON")
qc_text.append("=" * 88)
qc_text.append("")
qc_text.append(f"Selected ADM2 file: {ADM2_FILE}")
qc_text.append(f"Selected ADM3 file: {ADM3_FILE}")
qc_text.append("")
qc_text.append(f"ADM2 rows / unique: {len(adm2)} / {adm2['ADM2_KEY'].nunique()}")
qc_text.append(f"ADM3 rows / unique IDs: {len(adm3)} / {adm3['ADM3_ID'].nunique()}")
qc_text.append(f"Unmatched ADM3 parent rows: {missing_parent}")
qc_text.append("")
qc_text.append(f"ADM2 finalized class-count check: {'PASS' if adm2_count_check else 'WARNING'}")
qc_text.append(f"ADM3 finalized class-count check: {'PASS' if adm3_count_check else 'WARNING'}")
qc_text.append("")
qc_text.append("ADM2 class counts:")
qc_text.append(adm2_counts.to_string())
qc_text.append("")
qc_text.append("ADM3 class counts:")
qc_text.append(adm3_counts.to_string())
qc_text.append("")
qc_text.append(f"Hydrological ADM3 units used for relationship analysis: {hydro_n}")
qc_text.append(
    "Coverage/data-status ADM3 units excluded from hydrological relationship and entropy analysis: "
    f"{int(cross['ADM3_IS_STATUS'].sum())}"
)
qc_text.append("")
qc_text.append(f"High-heterogeneity entropy Q75 threshold: {entropy_q75:.6f}")
qc_text.append("")
qc_text.append("Interpretation:")
qc_text.append(
    "- ADM2 and ADM3 classifications were derived independently; no one-to-one taxonomy is imposed."
)
qc_text.append(
    "- Coverage-status classes are preserved in outputs but are not interpreted as flood regimes."
)
qc_text.append(
    "- Cross-scale flags are descriptive screening relationships, not causal or significance tests."
)
qc_text.append(
    "- Entropy describes within-ADM2 diversity of hydrological ADM3 classes only."
)

(QC_DIR / "30_cross_scale_qc.txt").write_text(
    "\n".join(qc_text),
    encoding="utf-8",
)

# =============================================================================
# 14. TERMINAL REPORT
# =============================================================================

print("\n" + "=" * 88)
print("FINAL TAXONOMY QC")
print("=" * 88)
print(f"ADM2 final-count check: {'PASS' if adm2_count_check else 'WARNING'}")
print(f"ADM3 final-count check: {'PASS' if adm3_count_check else 'WARNING'}")
print(f"Parent ADM2 match: {'PASS' if missing_parent == 0 else 'FAIL'}")

print("\n" + "=" * 88)
print("CROSS-SCALE RELATIONSHIP COUNTS")
print("=" * 88)
print(relationship_counts.to_string(index=False))

print("\n" + "=" * 88)
print("TOP ADM2 BY INTERNAL ADM3 HYDROLOGICAL HETEROGENEITY")
print("=" * 88)

display_cols = [
    "ADM2_NAME",
    "ADM2_PATTERN",
    "n_adm3_hydrological",
    "adm3_dominant_hydrological_pattern",
    "adm3_dominant_hydrological_share",
    "adm3_hydrological_pattern_entropy",
    "adm3_any_expansion_share",
]

print(
    adm2_summary[display_cols]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 88)
print("KEY CASE COUNTS")
print("=" * 88)
print(
    "Local expansion with non-expanding parent:",
    int(cross["flag_local_expansion_nonexpanding_parent"].sum()),
)
print(
    "Regional expansion + local persistence:",
    int(cross["flag_regional_expansion_local_persistence"].sum()),
)
print(
    "Regional persistence + local expansion:",
    int(cross["flag_regional_persistent_local_expansion"].sum()),
)
print(
    "Regional low/stable + local non-low hydrological signal:",
    int(cross["flag_regional_low_local_nonlow_signal"].sum()),
)
print(
    "Continuous opposite trend direction:",
    int(cross["continuous_trend_opposite"].sum()),
)
print(
    "High-heterogeneity ADM2 (entropy >= Q75):",
    len(high_heterogeneity),
)

print("\n" + "=" * 88)
print("OUTPUT")
print("=" * 88)
print(f"Output root:\n{OUTPUT_ROOT}")
print(f"\nTables:\n{TABLE_DIR}")
print(f"\nQC:\n{QC_DIR}")
print("\nScript 30 completed.")
