#!/usr/bin/env python3
"""
12_analyze_adm3_temporal_patterns.py

JBG060 Capstone Data Challenge
South Sudan Flood Analysis
ADM3 Temporal Pattern Discovery, 2000-2025

PURPOSE
-------
Discover temporal flood-pattern structure at ADM3 level while explicitly
incorporating the source-coverage audit from Script 11.

This script follows the ADM2 temporal-pattern methodology as closely as
possible:

    annual flooded percentage
        -> temporal features
        -> median imputation (feature space only)
        -> StandardScaler
        -> Ward hierarchical clustering
        -> silhouette diagnostics
        -> PCA for interpretation/visualization only
        -> centroid-nearest representative ADM3
        -> candidate change point

IMPORTANT COVERAGE RULE
-----------------------
Script 10 constructed a complete ADM3 x year panel and filled absent indicator
cells with zero. Script 11 showed that all-zero series do not represent one
homogeneous data condition.

Therefore:
1. ADM3 units with incomplete source coverage are NOT used in the main
   temporal clustering.
2. ADM3 units with complete expected source files but no detected positive
   flood record are retained in the final classification as a transparent
   descriptive category:
       "No detected flood record"
   but are NOT forced into the temporal clustering.
3. Main clustering is performed only on ADM3 units with detected flooding and
   adequate source-file coverage.
4. No parent ADM2 value is copied into an ADM3 unit.

PCA is NOT used as clustering input. It is used only to summarize and interpret
the standardized temporal-feature space.

INPUTS
------
Script 10:
processed_data/cross_dataset_analysis/02_adm3_flood/
    10_build_adm3_flood_indicators/tables/
        adm3_annual_flood_indicators_combined.csv

Script 11:
processed_data/cross_dataset_analysis/02_adm3_flood/
    11_audit_adm3_flood_coverage/tables/
        11_adm3_coverage_summary.csv
        11_adm3_year_coverage_audit.csv

OUTPUTS
-------
processed_data/cross_dataset_analysis/02_adm3_flood/
    12_analyze_adm3_temporal_patterns/
        tables/
        figures/
        qc/

The script writes:
- temporal features
- coverage-aware analysis eligibility
- silhouette diagnostics
- cluster assignments
- descriptive cluster profiles
- PCA scores and loadings
- centroid-nearest representatives
- candidate change points
- final 512-ADM3 classification table
- QC summary
- diagnostic figures

TIME WINDOWS
------------
Study period : 2000-2025
Early period : 2000-2007
Recent period: 2017-2025
Latest 5 yr  : 2021-2025
Previous 5 yr: 2016-2020

These early/recent windows intentionally match the ADM2 analysis so that
subsequent ADM2-ADM3 cross-scale comparisons use comparable definitions.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path("/Users/shibai/Documents/GitHub/DBL-Group-22")

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "cross_dataset_analysis"
)

ADM3_ROOT = PROCESSED_ROOT / "02_adm3_flood"

SCRIPT10_ROOT = ADM3_ROOT / "10_build_adm3_flood_indicators"
SCRIPT11_ROOT = ADM3_ROOT / "11_audit_adm3_flood_coverage"

ANNUAL_FILE = (
    SCRIPT10_ROOT
    / "tables"
    / "adm3_annual_flood_indicators_combined.csv"
)

COVERAGE_SUMMARY_FILE = (
    SCRIPT11_ROOT
    / "tables"
    / "11_adm3_coverage_summary.csv"
)

COVERAGE_YEAR_FILE = (
    SCRIPT11_ROOT
    / "tables"
    / "11_adm3_year_coverage_audit.csv"
)

OUTPUT_ROOT = ADM3_ROOT / "12_analyze_adm3_temporal_patterns"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"

for directory in [TABLE_DIR, FIGURE_DIR, QC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

START_YEAR = 2000
END_YEAR = 2025
EXPECTED_YEARS = END_YEAR - START_YEAR + 1

EARLY_START = 2000
EARLY_END = 2007
RECENT_START = 2017
RECENT_END = 2025

PREVIOUS_5Y_START = 2016
PREVIOUS_5Y_END = 2020
RECENT_5Y_START = 2021
RECENT_5Y_END = 2025

K_MIN = 3
K_MAX = 8

# Keep k=6 as the preferred interpretability target when its silhouette is
# close to the best solution, mirroring the ADM2 reasoning. If k=6 is clearly
# inferior, the script selects the best silhouette solution instead.
PREFERRED_K = 6
SILHOUETTE_TOLERANCE = 0.02

# =============================================================================
# 2. OPTIONAL / REQUIRED ANALYTICAL PACKAGES
# =============================================================================

try:
    from scipy.stats import linregress
    from scipy.cluster.hierarchy import linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError as exc:
    raise ImportError(
        "Script 12 requires scipy. Install it with: pip install scipy"
    ) from exc

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
except ImportError as exc:
    raise ImportError(
        "Script 12 requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False

# =============================================================================
# 3. HELPERS
# =============================================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_mean(series: pd.Series) -> float:
    values = safe_numeric(series).dropna()
    return float(values.mean()) if len(values) else np.nan


def safe_std(series: pd.Series) -> float:
    values = safe_numeric(series).dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else np.nan


def calculate_linear_trend(years: pd.Series, values: pd.Series):
    x = safe_numeric(years)
    y = safe_numeric(values)
    valid = x.notna() & y.notna()
    x = x[valid].astype(float)
    y = y[valid].astype(float)

    if len(x) < 3:
        return np.nan, np.nan, np.nan

    if y.nunique() <= 1:
        return 0.0, 1.0, 0.0

    result = linregress(x, y)
    return (
        float(result.slope),
        float(result.pvalue),
        float(result.rvalue ** 2),
    )


def longest_consecutive_true(years: pd.Series, mask: pd.Series) -> int:
    selected = sorted(
        safe_numeric(years[mask]).dropna().astype(int).unique().tolist()
    )
    if not selected:
        return 0

    best = 1
    current = 1
    for previous, current_year in zip(selected[:-1], selected[1:]):
        if current_year == previous + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return int(best)


def candidate_change_point(years: np.ndarray, values: np.ndarray):
    """
    Return one candidate change-point year.

    If ruptures is installed, use Binary Segmentation with L2 cost, consistent
    with the ADM2 methodology. Otherwise return NaN rather than silently
    substituting a different method.
    """
    if not RUPTURES_AVAILABLE:
        return np.nan

    years = np.asarray(years, dtype=int)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)
    years = years[valid]
    values = values[valid]

    if len(values) < 8 or np.nanstd(values) <= 1e-12:
        return np.nan

    signal = values.reshape(-1, 1)

    try:
        algo = rpt.Binseg(model="l2").fit(signal)
        bkps = algo.predict(n_bkps=1)
        # ruptures returns segment end indices. The final endpoint equals len.
        candidates = [b for b in bkps if b < len(years)]
        if not candidates:
            return np.nan
        idx = candidates[0]
        return int(years[idx])
    except Exception:
        return np.nan


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan, "none": np.nan})
    )


# =============================================================================
# 4. LOAD SCRIPT 10 AND SCRIPT 11 OUTPUTS
# =============================================================================

print("=" * 88)
print("SCRIPT 12 — ADM3 TEMPORAL PATTERN ANALYSIS")
print("=" * 88)

require_file(ANNUAL_FILE, "Script 10 annual combined ADM3 indicators")
require_file(COVERAGE_SUMMARY_FILE, "Script 11 ADM3 coverage summary")

annual = pd.read_csv(ANNUAL_FILE)
coverage = pd.read_csv(COVERAGE_SUMMARY_FILE)

print(f"\nScript 10 annual file:\n{ANNUAL_FILE}")
print(f"Rows: {len(annual):,} | Columns: {len(annual.columns):,}")

print(f"\nScript 11 coverage file:\n{COVERAGE_SUMMARY_FILE}")
print(f"Rows: {len(coverage):,} | Columns: {len(coverage.columns):,}")

# =============================================================================
# 5. IDENTIFY KEY COLUMNS ROBUSTLY
# =============================================================================

annual_id_col = first_existing_column(
    annual,
    ["ADM3_ID", "adm3_id", "ADM3_PCODE", "adm3_pcode"],
)
annual_name_col = first_existing_column(
    annual,
    ["ADM3_NAME", "adm3_name", "ADM3_EN", "adm3_en"],
)
annual_year_col = first_existing_column(annual, ["year", "YEAR", "Year"])
annual_value_col = first_existing_column(
    annual,
    ["annual_flooded_percent", "flooded_percent"],
)

if annual_id_col is None or annual_name_col is None:
    raise ValueError(
        "Could not identify ADM3 ID/name columns in Script 10 annual file."
    )
if annual_year_col is None or annual_value_col is None:
    raise ValueError(
        "Could not identify year/annual_flooded_percent columns in Script 10 file."
    )

coverage_id_col = first_existing_column(
    coverage,
    ["ADM3_PCODE", "adm3_pcode", "ADM3_ID", "adm3_id"],
)
coverage_name_col = first_existing_column(
    coverage,
    ["ADM3_NAME", "adm3_name", "ADM3_EN", "adm3_en"],
)

# =============================================================================
# 6. CLEAN SCRIPT 10 ANNUAL PANEL
# =============================================================================

annual = annual.copy()
annual["year"] = safe_numeric(annual[annual_year_col])
annual["annual_flooded_percent"] = safe_numeric(annual[annual_value_col])

annual = annual[
    annual["year"].between(START_YEAR, END_YEAR)
].copy()

annual["year"] = annual["year"].astype(int)

# Canonical names used downstream.
annual["ADM3_ID_CANON"] = annual[annual_id_col].astype(str)
annual["ADM3_NAME_CANON"] = annual[annual_name_col].astype(str)

identity_candidates = [
    "ADM0_NAME",
    "ADM1_NAME",
    "ADM2_NAME",
    annual_name_col,
]
identity_candidates = [
    c for c in identity_candidates
    if c in annual.columns
]

# =============================================================================
# 7. BUILD COVERAGE-AWARE ELIGIBILITY TABLE
# =============================================================================

print("\n" + "=" * 88)
print("BUILD COVERAGE-AWARE ELIGIBILITY")
print("=" * 88)

coverage = coverage.copy()

# We derive eligibility from Script 11's actual summary fields rather than
# assuming one exact column name from an earlier code version.
complete_year_col = first_existing_column(
    coverage,
    [
        "years_complete_tile_files",
        "complete_tile_file_years",
        "n_complete_tile_years",
        "years_complete",
    ],
)

flood_year_col = first_existing_column(
    coverage,
    [
        "years_with_flood_records",
        "years_with_flood_record",
        "flood_record_years",
        "years_flood_detected",
    ],
)

total_record_col = first_existing_column(
    coverage,
    [
        "total_flood_records",
        "flood_records_total",
        "n_flood_records",
        "total_event_records",
    ],
)

interpretation_col = first_existing_column(
    coverage,
    [
        "coverage_interpretation",
        "coverage_status",
        "summary_status",
        "overall_qc_status",
        "audit_interpretation",
    ],
)

# Create a robust join. Prefer exact ID/PCODE when compatible; otherwise use
# normalized ADM3 name.
annual_identity = (
    annual[
        ["ADM3_ID_CANON", "ADM3_NAME_CANON"]
    ]
    .drop_duplicates()
    .copy()
)
annual_identity["name_key"] = normalize_text(annual_identity["ADM3_NAME_CANON"])

coverage["coverage_row"] = np.arange(len(coverage))

if coverage_id_col is not None:
    coverage["coverage_id_key"] = coverage[coverage_id_col].astype(str)
else:
    coverage["coverage_id_key"] = np.nan

if coverage_name_col is not None:
    coverage["name_key"] = normalize_text(coverage[coverage_name_col])
else:
    coverage["name_key"] = np.nan

# First try ID join.
eligibility = annual_identity.merge(
    coverage,
    left_on="ADM3_ID_CANON",
    right_on="coverage_id_key",
    how="left",
    suffixes=("", "_cov"),
)

id_matches = eligibility["coverage_row"].notna().sum()

# If ID matching is weak, fall back to name join.
if id_matches < max(1, int(0.80 * len(annual_identity))):
    print(
        f"ID-based Script10/11 match found only {id_matches} ADM3; "
        "using normalized ADM3-name matching instead."
    )
    eligibility = annual_identity.merge(
        coverage,
        on="name_key",
        how="left",
        suffixes=("", "_cov"),
    )

matched = int(eligibility["coverage_row"].notna().sum())
print(f"Coverage rows matched to Script 10 ADM3: {matched}/{len(annual_identity)}")

if matched != len(annual_identity):
    unmatched = eligibility[
        eligibility["coverage_row"].isna()
    ][["ADM3_ID_CANON", "ADM3_NAME_CANON"]]
    unmatched.to_csv(
        QC_DIR / "12_unmatched_script10_script11_adm3.csv",
        index=False,
    )
    raise ValueError(
        "Not all Script 10 ADM3 units could be matched to Script 11 coverage "
        "summary. See qc/12_unmatched_script10_script11_adm3.csv before clustering."
    )

# Derive a conservative coverage group.
def derive_coverage_group(row) -> str:
    # Best case: explicit complete-year count from Script 11.
    complete_years = np.nan
    if complete_year_col is not None:
        complete_years = pd.to_numeric(
            pd.Series([row.get(complete_year_col)]),
            errors="coerce",
        ).iloc[0]

    flood_years = np.nan
    if flood_year_col is not None:
        flood_years = pd.to_numeric(
            pd.Series([row.get(flood_year_col)]),
            errors="coerce",
        ).iloc[0]

    total_records = np.nan
    if total_record_col is not None:
        total_records = pd.to_numeric(
            pd.Series([row.get(total_record_col)]),
            errors="coerce",
        ).iloc[0]

    text = ""
    if interpretation_col is not None:
        text = str(row.get(interpretation_col, "")).lower()

    incomplete_words = [
        "incomplete", "partial", "no_tile", "no tile",
        "insufficient", "missing source", "missing tile",
    ]

    if any(word in text for word in incomplete_words):
        return "Incomplete source coverage"

    if pd.notna(complete_years) and int(complete_years) < EXPECTED_YEARS:
        return "Incomplete source coverage"

    # Complete expected source files but no positive flood records.
    no_detection = False
    if pd.notna(total_records):
        no_detection = float(total_records) == 0
    elif pd.notna(flood_years):
        no_detection = float(flood_years) == 0
    elif "zero" in text or "no detected" in text or "no flood record" in text:
        no_detection = True

    if (
        pd.notna(complete_years)
        and int(complete_years) == EXPECTED_YEARS
        and no_detection
    ):
        return "No detected flood record"

    if pd.notna(complete_years) and int(complete_years) == EXPECTED_YEARS:
        return "Eligible detected-flood series"

    # If the summary contains an explicit complete/processed interpretation but
    # no numeric complete-year field, use that text cautiously.
    if (
        ("complete" in text or "26y_processed" in text or "processed" in text)
        and not any(word in text for word in incomplete_words)
    ):
        if no_detection:
            return "No detected flood record"
        return "Eligible detected-flood series"

    return "Coverage status unresolved"


eligibility["analysis_coverage_group"] = eligibility.apply(
    derive_coverage_group,
    axis=1,
)

# Cross-check against Script 10 actual flood detections so eligibility is not
# dependent on Script 11 column naming alone.
detection_check = (
    annual.groupby(["ADM3_ID_CANON", "ADM3_NAME_CANON"], as_index=False)
    .agg(
        script10_flood_year_count=(
            "annual_flooded_percent",
            lambda s: int((safe_numeric(s).fillna(0) > 0).sum()),
        ),
        script10_max_flooded_percent=(
            "annual_flooded_percent",
            lambda s: float(safe_numeric(s).max()),
        ),
    )
)

eligibility = eligibility.merge(
    detection_check,
    on=["ADM3_ID_CANON", "ADM3_NAME_CANON"],
    how="left",
)

# A coverage-complete unit with no Script 10 detection is a descriptive
# zero-detection category, not a clustering case.
mask_complete_but_zero = (
    eligibility["analysis_coverage_group"]
    == "Eligible detected-flood series"
) & (
    eligibility["script10_flood_year_count"] == 0
)
eligibility.loc[
    mask_complete_but_zero,
    "analysis_coverage_group",
] = "No detected flood record"

# A purported no-detection unit with positive Script 10 flood values is a
# mismatch requiring review.
mask_zero_but_positive = (
    eligibility["analysis_coverage_group"]
    == "No detected flood record"
) & (
    eligibility["script10_flood_year_count"] > 0
)

if mask_zero_but_positive.any():
    eligibility.loc[
        mask_zero_but_positive,
        "analysis_coverage_group",
    ] = "Coverage status unresolved"

eligibility["eligible_for_main_clustering"] = (
    eligibility["analysis_coverage_group"]
    == "Eligible detected-flood series"
)

eligibility.to_csv(
    TABLE_DIR / "12_adm3_analysis_eligibility.csv",
    index=False,
)

print(
    eligibility["analysis_coverage_group"]
    .value_counts(dropna=False)
    .to_string()
)

if (eligibility["analysis_coverage_group"] == "Coverage status unresolved").any():
    unresolved = eligibility[
        eligibility["analysis_coverage_group"]
        == "Coverage status unresolved"
    ]
    unresolved.to_csv(
        QC_DIR / "12_unresolved_coverage_status.csv",
        index=False,
    )
    raise ValueError(
        "Some ADM3 coverage statuses remain unresolved. "
        "Inspect qc/12_unresolved_coverage_status.csv before clustering."
    )

eligible_ids = set(
    eligibility.loc[
        eligibility["eligible_for_main_clustering"],
        "ADM3_ID_CANON",
    ].astype(str)
)

# =============================================================================
# 8. BUILD TEMPORAL FEATURES FOR ELIGIBLE ADM3
# =============================================================================

print("\n" + "=" * 88)
print("BUILD ADM3 TEMPORAL FEATURES")
print("=" * 88)

feature_rows = []

for adm3_id, group in annual.groupby("ADM3_ID_CANON"):
    if str(adm3_id) not in eligible_ids:
        continue

    group = group.sort_values("year").copy()
    years = group["year"].astype(int)
    values = group["annual_flooded_percent"].astype(float)

    if len(group) != EXPECTED_YEARS:
        raise ValueError(
            f"Eligible ADM3 {adm3_id} has {len(group)} annual rows; "
            f"expected {EXPECTED_YEARS}."
        )

    identity = {
        "ADM3_ID": adm3_id,
        "ADM3_NAME": str(group["ADM3_NAME_CANON"].iloc[0]),
    }

    for column in ["ADM0_NAME", "ADM1_NAME", "ADM2_NAME"]:
        if column in group.columns:
            valid = group[column].dropna()
            identity[column] = (
                str(valid.iloc[0]) if len(valid) else np.nan
            )

    early = values[years.between(EARLY_START, EARLY_END)]
    recent = values[years.between(RECENT_START, RECENT_END)]
    previous_5y = values[
        years.between(PREVIOUS_5Y_START, PREVIOUS_5Y_END)
    ]
    recent_5y = values[
        years.between(RECENT_5Y_START, RECENT_5Y_END)
    ]

    long_slope, trend_pvalue, trend_r2 = calculate_linear_trend(
        years, values
    )

    recent_years = years[years.between(RECENT_START, RECENT_END)]
    recent_values = values[years.between(RECENT_START, RECENT_END)]
    recent_slope, _, _ = calculate_linear_trend(
        recent_years, recent_values
    )

    early_slope, _, _ = calculate_linear_trend(
        years[years.between(EARLY_START, EARLY_END)],
        early,
    )

    acceleration = (
        recent_slope - early_slope
        if np.isfinite(recent_slope) and np.isfinite(early_slope)
        else np.nan
    )

    early_mean = safe_mean(early)
    recent_mean = safe_mean(recent)
    previous_5y_mean = safe_mean(previous_5y)
    recent_5y_mean = safe_mean(recent_5y)

    change_pp = recent_mean - early_mean
    recent_5y_change_pp = recent_5y_mean - previous_5y_mean

    flood_mask = values > 0
    flood_year_count = int(flood_mask.sum())
    flood_year_fraction = flood_year_count / EXPECTED_YEARS
    longest_run = longest_consecutive_true(years, flood_mask)

    median_value = float(values.median())
    max_value = float(values.max())
    max_minus_median = max_value - median_value

    peak_idx = values.idxmax()
    peak_year = int(group.loc[peak_idx, "year"])
    peak_value = float(group.loc[peak_idx, "annual_flooded_percent"])

    post_peak = group[group["year"] > peak_year]["annual_flooded_percent"]
    if len(post_peak):
        post_peak_decline = peak_value - float(post_peak.iloc[-1])
    else:
        post_peak_decline = np.nan

    long_mean = float(values.mean())
    std_value = float(values.std(ddof=1))
    cv_flood = (
        std_value / long_mean
        if abs(long_mean) > 1e-12
        else np.nan
    )

    cp_year = candidate_change_point(
        years.to_numpy(),
        values.to_numpy(),
    )

    feature_rows.append(
        {
            **identity,
            "n_years": len(group),
            "long_term_mean": long_mean,
            "median_flooded_percent": median_value,
            "early_mean": early_mean,
            "recent_mean": recent_mean,
            "recent_5y_mean": recent_5y_mean,
            "previous_5y_mean": previous_5y_mean,
            "max_flooded_percent": max_value,
            "trend_slope": long_slope,
            "trend_pvalue": trend_pvalue,
            "trend_r2": trend_r2,
            "recent_slope": recent_slope,
            "early_slope": early_slope,
            "acceleration": acceleration,
            "change_pp": change_pp,
            "recent_5y_change_pp": recent_5y_change_pp,
            "flood_year_count": flood_year_count,
            "flood_year_fraction": flood_year_fraction,
            "longest_consecutive_run": longest_run,
            "max_minus_median": max_minus_median,
            "post_peak_decline": post_peak_decline,
            "cv_flood": cv_flood,
            "peak_year": peak_year,
            "peak_value": peak_value,
            "candidate_change_point": cp_year,
        }
    )

features = pd.DataFrame(feature_rows)

if features.empty:
    raise ValueError("No ADM3 units are eligible for temporal clustering.")

features.to_csv(
    TABLE_DIR / "12_adm3_temporal_features.csv",
    index=False,
)

print(f"Eligible ADM3 temporal feature rows: {len(features):,}")

# =============================================================================
# 9. CLUSTERING FEATURE MATRIX — SAME 10 CONCEPTS AS ADM2
# =============================================================================

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

X_raw = features[CLUSTER_FEATURES].copy()

imputation_rows = []
for column in CLUSTER_FEATURES:
    missing_count = int(X_raw[column].isna().sum())
    median_value = float(X_raw[column].median())
    if not np.isfinite(median_value):
        median_value = 0.0
    imputation_rows.append(
        {
            "feature": column,
            "missing_count": missing_count,
            "imputation_median": median_value,
        }
    )
    X_raw[column] = X_raw[column].fillna(median_value)

pd.DataFrame(imputation_rows).to_csv(
    QC_DIR / "12_feature_imputation_summary.csv",
    index=False,
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=[f"z_{c}" for c in CLUSTER_FEATURES],
)
scaled_df.insert(0, "ADM3_ID", features["ADM3_ID"].values)
scaled_df.insert(1, "ADM3_NAME", features["ADM3_NAME"].values)
scaled_df.to_csv(
    TABLE_DIR / "12_adm3_standardized_temporal_features.csv",
    index=False,
)

# =============================================================================
# 10. WARD HIERARCHICAL CLUSTERING + SILHOUETTE
# =============================================================================

print("\n" + "=" * 88)
print("WARD HIERARCHICAL CLUSTERING")
print("=" * 88)

Z = linkage(X_scaled, method="ward")

silhouette_rows = []
cluster_solutions = {}

max_k_allowed = min(K_MAX, len(features) - 1)

for k in range(K_MIN, max_k_allowed + 1):
    labels = fcluster(Z, t=k, criterion="maxclust")
    n_actual = len(np.unique(labels))

    if n_actual < 2 or n_actual >= len(features):
        score = np.nan
    else:
        score = float(silhouette_score(X_scaled, labels))

    cluster_solutions[k] = labels
    silhouette_rows.append(
        {
            "k_requested": k,
            "k_actual": n_actual,
            "silhouette_score": score,
        }
    )

silhouette_df = pd.DataFrame(silhouette_rows)
silhouette_df.to_csv(
    TABLE_DIR / "12_adm3_silhouette_diagnostics.csv",
    index=False,
)

valid_sil = silhouette_df.dropna(subset=["silhouette_score"]).copy()
if valid_sil.empty:
    raise ValueError("No valid silhouette solution could be calculated.")

best_row = valid_sil.loc[valid_sil["silhouette_score"].idxmax()]
best_k = int(best_row["k_requested"])
best_score = float(best_row["silhouette_score"])

preferred_row = valid_sil[
    valid_sil["k_requested"] == PREFERRED_K
]

if len(preferred_row):
    preferred_score = float(preferred_row.iloc[0]["silhouette_score"])
    if best_score - preferred_score <= SILHOUETTE_TOLERANCE:
        selected_k = PREFERRED_K
        selection_reason = (
            f"k={PREFERRED_K} retained for cross-scale interpretability; "
            f"its silhouette ({preferred_score:.4f}) is within "
            f"{SILHOUETTE_TOLERANCE:.3f} of the best score "
            f"({best_score:.4f}, k={best_k})."
        )
    else:
        selected_k = best_k
        selection_reason = (
            f"k={best_k} selected because its silhouette ({best_score:.4f}) "
            f"is more than {SILHOUETTE_TOLERANCE:.3f} above the preferred "
            f"k={PREFERRED_K} solution."
        )
else:
    selected_k = best_k
    selection_reason = (
        f"k={best_k} selected as the best valid silhouette solution."
    )

labels = cluster_solutions[selected_k]
features["cluster_id"] = labels.astype(int)

print(silhouette_df.to_string(index=False))
print(f"\nSelected k: {selected_k}")
print(selection_reason)

with open(
    QC_DIR / "12_cluster_selection_summary.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(f"Selected k: {selected_k}\n")
    f.write(f"Best silhouette k: {best_k}\n")
    f.write(f"Best silhouette score: {best_score:.6f}\n")
    f.write(selection_reason + "\n")

# =============================================================================
# 11. PCA — INTERPRETATION ONLY
# =============================================================================

print("\n" + "=" * 88)
print("PCA FOR INTERPRETATION ONLY")
print("=" * 88)

pca = PCA(n_components=min(len(CLUSTER_FEATURES), len(features)))
X_pca = pca.fit_transform(X_scaled)

pca_scores = pd.DataFrame(
    {
        "ADM3_ID": features["ADM3_ID"].values,
        "ADM3_NAME": features["ADM3_NAME"].values,
        "cluster_id": features["cluster_id"].values,
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1] if X_pca.shape[1] > 1 else np.nan,
    }
)

pca_scores.to_csv(
    TABLE_DIR / "12_adm3_pca_scores.csv",
    index=False,
)

loadings = pd.DataFrame(
    pca.components_.T,
    index=CLUSTER_FEATURES,
    columns=[
        f"PC{i+1}"
        for i in range(pca.components_.shape[0])
    ],
).reset_index().rename(columns={"index": "feature"})

loadings.to_csv(
    TABLE_DIR / "12_adm3_pca_loadings.csv",
    index=False,
)

explained = pd.DataFrame(
    {
        "component": [
            f"PC{i+1}"
            for i in range(len(pca.explained_variance_ratio_))
        ],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_percent": (
            pca.explained_variance_ratio_ * 100
        ),
        "cumulative_explained_percent": (
            np.cumsum(pca.explained_variance_ratio_) * 100
        ),
    }
)

explained.to_csv(
    TABLE_DIR / "12_adm3_pca_explained_variance.csv",
    index=False,
)

print(
    f"PC1 explained variance: "
    f"{explained.loc[0, 'explained_variance_percent']:.2f}%"
)
if len(explained) > 1:
    print(
        f"PC2 explained variance: "
        f"{explained.loc[1, 'explained_variance_percent']:.2f}%"
    )
    print(
        f"PC1+PC2 cumulative: "
        f"{explained.loc[1, 'cumulative_explained_percent']:.2f}%"
    )

# =============================================================================
# 12. CLUSTER PROFILES + DATA-DRIVEN DESCRIPTIVE LABELS
# =============================================================================

profile_metrics = [
    "long_term_mean",
    "early_mean",
    "recent_mean",
    "change_pp",
    "recent_5y_change_pp",
    "trend_slope",
    "recent_slope",
    "acceleration",
    "flood_year_fraction",
    "longest_consecutive_run",
    "max_minus_median",
    "cv_flood",
]

profiles = (
    features.groupby("cluster_id")[profile_metrics]
    .mean()
    .reset_index()
)

counts = (
    features["cluster_id"]
    .value_counts()
    .rename_axis("cluster_id")
    .reset_index(name="adm3_count")
)

profiles = profiles.merge(counts, on="cluster_id", how="left")

# Transparent descriptive naming based on cluster profiles.
# Labels are assigned after clustering; they do not affect membership.
profiles["descriptive_label"] = ""

remaining = set(profiles["cluster_id"].astype(int).tolist())

def assign_extreme(metric, mode, label):
    global remaining
    subset = profiles[profiles["cluster_id"].isin(remaining)]
    if subset.empty:
        return
    idx = (
        subset[metric].idxmax()
        if mode == "max"
        else subset[metric].idxmin()
    )
    cid = int(profiles.loc[idx, "cluster_id"])
    profiles.loc[idx, "descriptive_label"] = label
    remaining.discard(cid)

# Prioritize substantively distinct temporal dimensions.
assign_extreme(
    "change_pp",
    "max",
    "Strong recent expansion",
)
assign_extreme(
    "flood_year_fraction",
    "max",
    "Recurrent / persistent",
)
assign_extreme(
    "cv_flood",
    "max",
    "Episodic / high-volatility",
)

# Of the remaining clusters, identify the most stable/low and most declining.
if remaining:
    subset = profiles[profiles["cluster_id"].isin(remaining)].copy()
    low_score = (
        subset["long_term_mean"].rank(pct=True)
        + subset["change_pp"].abs().rank(pct=True)
    )
    idx = low_score.idxmin()
    cid = int(profiles.loc[idx, "cluster_id"])
    profiles.loc[idx, "descriptive_label"] = "Relatively low / stable"
    remaining.discard(cid)

if remaining:
    subset = profiles[profiles["cluster_id"].isin(remaining)]
    idx = subset["change_pp"].idxmin()
    cid = int(profiles.loc[idx, "cluster_id"])
    profiles.loc[idx, "descriptive_label"] = "Declining / weakening"
    remaining.discard(cid)

# Any remaining cluster(s) are labelled by moderate positive change.
for cid in sorted(remaining):
    profiles.loc[
        profiles["cluster_id"] == cid,
        "descriptive_label",
    ] = "Gradual / intermediate change"

label_map = dict(
    zip(
        profiles["cluster_id"].astype(int),
        profiles["descriptive_label"],
    )
)

features["temporal_pattern"] = (
    features["cluster_id"].map(label_map)
)

profiles.to_csv(
    TABLE_DIR / "12_adm3_cluster_profiles.csv",
    index=False,
)

# =============================================================================
# 13. CENTROID-NEAREST REPRESENTATIVE ADM3
# =============================================================================

representative_rows = []

for cid in sorted(features["cluster_id"].unique()):
    member_idx = np.where(features["cluster_id"].to_numpy() == cid)[0]
    member_matrix = X_scaled[member_idx, :]
    centroid = member_matrix.mean(axis=0)
    distances = np.linalg.norm(member_matrix - centroid, axis=1)
    local_best = int(np.argmin(distances))
    global_idx = int(member_idx[local_best])

    representative_rows.append(
        {
            "cluster_id": int(cid),
            "temporal_pattern": label_map[int(cid)],
            "ADM3_ID": features.iloc[global_idx]["ADM3_ID"],
            "ADM3_NAME": features.iloc[global_idx]["ADM3_NAME"],
            "ADM2_NAME": (
                features.iloc[global_idx]["ADM2_NAME"]
                if "ADM2_NAME" in features.columns
                else np.nan
            ),
            "centroid_distance": float(distances[local_best]),
        }
    )

representatives = pd.DataFrame(representative_rows)
representatives.to_csv(
    TABLE_DIR / "12_adm3_centroid_nearest_representatives.csv",
    index=False,
)

# =============================================================================
# 14. FINAL 512-ADM3 CLASSIFICATION
# =============================================================================

cluster_output_columns = [
    "ADM3_ID",
    "ADM3_NAME",
    "ADM1_NAME",
    "ADM2_NAME",
    "cluster_id",
    "temporal_pattern",
    "long_term_mean",
    "early_mean",
    "recent_mean",
    "change_pp",
    "recent_5y_mean",
    "recent_5y_change_pp",
    "trend_slope",
    "trend_pvalue",
    "trend_r2",
    "flood_year_count",
    "flood_year_fraction",
    "longest_consecutive_run",
    "max_flooded_percent",
    "candidate_change_point",
]
cluster_output_columns = [
    c for c in cluster_output_columns
    if c in features.columns
]

clustered = features[cluster_output_columns].copy()
clustered["analysis_status"] = "Included in main clustering"

# Build rows for non-clustered ADM3.
noncluster = eligibility[
    ~eligibility["eligible_for_main_clustering"]
].copy()

noncluster_rows = []
for _, row in noncluster.iterrows():
    group_name = row["analysis_coverage_group"]

    if group_name == "No detected flood record":
        pattern = "No detected flood record"
        status = (
            "Not clustered: expected source-file coverage complete, "
            "but no positive flood record detected"
        )
    else:
        pattern = "Insufficient source coverage"
        status = (
            "Not clustered: incomplete source coverage; zero values "
            "must not be interpreted as observed no-flood years"
        )

    noncluster_rows.append(
        {
            "ADM3_ID": row["ADM3_ID_CANON"],
            "ADM3_NAME": row["ADM3_NAME_CANON"],
            "cluster_id": np.nan,
            "temporal_pattern": pattern,
            "analysis_status": status,
        }
    )

final_classification = pd.concat(
    [
        clustered,
        pd.DataFrame(noncluster_rows),
    ],
    ignore_index=True,
    sort=False,
)

final_classification = final_classification.sort_values(
    ["temporal_pattern", "ADM3_NAME"]
)

final_classification.to_csv(
    TABLE_DIR / "12_adm3_temporal_pattern_classification_all512.csv",
    index=False,
)

features.to_csv(
    TABLE_DIR / "12_adm3_cluster_assignments_main_analysis.csv",
    index=False,
)

# =============================================================================
# 15. CHANGE-POINT SUMMARY
# =============================================================================

change_points = features[
    [
        c for c in [
            "ADM3_ID",
            "ADM3_NAME",
            "ADM2_NAME",
            "cluster_id",
            "temporal_pattern",
            "candidate_change_point",
        ]
        if c in features.columns
    ]
].copy()

change_points.to_csv(
    TABLE_DIR / "12_adm3_candidate_change_points.csv",
    index=False,
)

# =============================================================================
# 16. OPTIONAL DIAGNOSTIC FIGURES
# =============================================================================

if MATPLOTLIB_AVAILABLE:
    # Silhouette diagnostic
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        silhouette_df["k_requested"],
        silhouette_df["silhouette_score"],
        marker="o",
    )
    ax.axvline(selected_k, linestyle="--")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("ADM3 Ward clustering — silhouette diagnostics")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "12_silhouette_diagnostics.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # PCA scatter. PCA is visualization only.
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_scores["PC1"],
        pca_scores["PC2"],
        c=pca_scores["cluster_id"],
        s=28,
        alpha=0.8,
    )
    ax.set_xlabel(
        f"PC1 ({explained.loc[0, 'explained_variance_percent']:.1f}%)"
    )
    pc2_pct = (
        explained.loc[1, "explained_variance_percent"]
        if len(explained) > 1
        else np.nan
    )
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)")
    ax.set_title(
        "ADM3 temporal-feature PCA (visualization only)"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "12_pca_temporal_patterns.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

# =============================================================================
# 17. QC SUMMARY
# =============================================================================

coverage_counts = (
    eligibility["analysis_coverage_group"]
    .value_counts()
    .to_dict()
)

final_counts = (
    final_classification["temporal_pattern"]
    .value_counts()
    .rename_axis("temporal_pattern")
    .reset_index(name="adm3_count")
)
final_counts["share_percent_of_512"] = (
    final_counts["adm3_count"]
    / len(final_classification)
    * 100
)
final_counts.to_csv(
    TABLE_DIR / "12_adm3_temporal_pattern_counts_all512.csv",
    index=False,
)

qc_lines = [
    "SCRIPT 12 — ADM3 TEMPORAL PATTERN ANALYSIS QC",
    "=" * 60,
    f"Study years: {START_YEAR}-{END_YEAR}",
    f"Early period: {EARLY_START}-{EARLY_END}",
    f"Recent period: {RECENT_START}-{RECENT_END}",
    f"Script 10 ADM3 count: {annual['ADM3_ID_CANON'].nunique()}",
    f"Script 11 matched ADM3 count: {matched}",
    f"Eligible main-clustering ADM3: {len(features)}",
    f"Final classification rows: {len(final_classification)}",
    f"Selected k: {selected_k}",
    f"Best silhouette k: {best_k}",
    f"Best silhouette score: {best_score:.6f}",
    f"Ruptures available: {RUPTURES_AVAILABLE}",
    "",
    "Coverage groups:",
]
for key, value in coverage_counts.items():
    qc_lines.append(f"  {key}: {value}")

qc_lines += [
    "",
    "Cluster selection:",
    f"  {selection_reason}",
    "",
    "Important interpretation:",
    "  PCA is visualization/interpretation only; clustering uses all 10 standardized temporal features.",
    "  Incomplete-coverage ADM3 are not used in the main clustering.",
    "  Complete-source/no-detection ADM3 are retained as a separate descriptive category.",
    "  No ADM2 value is copied into ADM3.",
    "  Candidate change points are not causal evidence.",
]

with open(
    QC_DIR / "12_analysis_qc_summary.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(qc_lines))

# =============================================================================
# 18. CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 88)
print("SCRIPT 12 COMPLETED")
print("=" * 88)

print(f"\nMain clustering ADM3: {len(features)}")
print(f"Selected k: {selected_k}")
print("\nCluster profiles:")
print(
    profiles[
        [
            "cluster_id",
            "descriptive_label",
            "adm3_count",
            "long_term_mean",
            "change_pp",
            "flood_year_fraction",
            "cv_flood",
        ]
    ].to_string(index=False)
)

print("\nFinal 512-ADM3 pattern/status counts:")
print(final_counts.to_string(index=False))

print(f"\nOutput root:\n{OUTPUT_ROOT}")
print("\nNext step: Script 13 — visualize ADM3 temporal patterns.")
