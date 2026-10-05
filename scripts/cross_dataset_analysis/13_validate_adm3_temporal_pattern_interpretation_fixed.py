#!/usr/bin/env python3
"""
13_validate_adm3_temporal_pattern_interpretation.py

JBG060 Capstone Data Challenge
South Sudan Flood Analysis
ADM3 Temporal Pattern Validation and Interpretation

This script validates and interprets the ADM3 temporal clusters produced by
Script 12. It DOES NOT re-run clustering and DOES NOT change cluster membership.
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
PROCESSED_ROOT = PROJECT_ROOT / "processed_data" / "cross_dataset_analysis"
ADM3_ROOT = PROCESSED_ROOT / "02_adm3_flood"

SCRIPT10_ROOT = ADM3_ROOT / "10_build_adm3_flood_indicators"
SCRIPT12_ROOT = ADM3_ROOT / "12_analyze_adm3_temporal_patterns"
SCRIPT12_TABLE_DIR = SCRIPT12_ROOT / "tables"

OUTPUT_ROOT = ADM3_ROOT / "13_validate_adm3_temporal_pattern_interpretation"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"

for directory in [TABLE_DIR, FIGURE_DIR, QC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. INPUT FILES
# =============================================================================

ANNUAL_FILE = SCRIPT10_ROOT / "tables" / "adm3_annual_flood_indicators_combined.csv"
ASSIGNMENT_FILE = SCRIPT12_TABLE_DIR / "12_adm3_cluster_assignments_main_analysis.csv"
PROFILE_FILE = SCRIPT12_TABLE_DIR / "12_adm3_cluster_profiles.csv"
PCA_LOADING_FILE = SCRIPT12_TABLE_DIR / "12_adm3_pca_loadings.csv"
PCA_VARIANCE_FILE = SCRIPT12_TABLE_DIR / "12_adm3_pca_explained_variance.csv"
REPRESENTATIVE_FILE = SCRIPT12_TABLE_DIR / "12_adm3_centroid_nearest_representatives.csv"

START_YEAR, END_YEAR = 2000, 2025
EARLY_START, EARLY_END = 2000, 2007
RECENT_START, RECENT_END = 2017, 2025
PREVIOUS_5Y_START, PREVIOUS_5Y_END = 2016, 2020
RECENT_5Y_START, RECENT_5Y_END = 2021, 2025

CLUSTER_FEATURES = [
    "trend_slope", "recent_slope", "acceleration", "change_pp",
    "recent_5y_change_pp", "flood_year_fraction",
    "longest_consecutive_run", "max_minus_median",
    "post_peak_decline", "cv_flood",
]

# Descriptive validation threshold only; does not affect clustering.
NEAR_ZERO_CHANGE_PP = 2.0

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"\n{description} not found:\n{path}")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_percent(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100.0 if denominator else np.nan


print("=" * 96)
print("SCRIPT 13 — ADM3 TEMPORAL PATTERN VALIDATION AND INTERPRETATION")
print("=" * 96)

required_files = {
    "Script 10 annual ADM3 indicators": ANNUAL_FILE,
    "Script 12 cluster assignments": ASSIGNMENT_FILE,
    "Script 12 cluster profiles": PROFILE_FILE,
    "Script 12 PCA loadings": PCA_LOADING_FILE,
    "Script 12 PCA explained variance": PCA_VARIANCE_FILE,
    "Script 12 representatives": REPRESENTATIVE_FILE,
}
for description, path in required_files.items():
    require_file(path, description)

annual = pd.read_csv(ANNUAL_FILE)
assignments = pd.read_csv(ASSIGNMENT_FILE)
profiles12 = pd.read_csv(PROFILE_FILE)
pca_loadings = pd.read_csv(PCA_LOADING_FILE)
pca_variance = pd.read_csv(PCA_VARIANCE_FILE)
representatives = pd.read_csv(REPRESENTATIVE_FILE)

annual_id_col = first_existing_column(annual, ["ADM3_ID", "adm3_id", "ADM3_PCODE", "adm3_pcode"])
annual_name_col = first_existing_column(annual, ["ADM3_NAME", "adm3_name", "ADM3_EN", "adm3_en"])
annual_year_col = first_existing_column(annual, ["year", "YEAR", "Year"])
annual_value_col = first_existing_column(annual, ["annual_flooded_percent", "flooded_percent"])

if any(x is None for x in [annual_id_col, annual_name_col, annual_year_col, annual_value_col]):
    raise ValueError("Could not identify required columns in Script 10 annual file.")

annual = annual.copy()
annual["ADM3_ID_CANON"] = annual[annual_id_col].astype(str)
annual["ADM3_NAME_CANON"] = annual[annual_name_col].astype(str)
annual["year"] = safe_numeric(annual[annual_year_col])
annual["annual_flooded_percent"] = safe_numeric(annual[annual_value_col])
annual = annual[annual["year"].between(START_YEAR, END_YEAR)].copy()
annual["year"] = annual["year"].astype(int)

assignments = assignments.copy()
assignments["cluster_id"] = pd.to_numeric(assignments["cluster_id"], errors="coerce")
if assignments["cluster_id"].isna().any():
    raise ValueError("Missing cluster_id detected in Script 12 assignments.")
assignments["cluster_id"] = assignments["cluster_id"].astype(int)
assignments["ADM3_ID"] = assignments["ADM3_ID"].astype(str)
cluster_ids = sorted(assignments["cluster_id"].unique())

missing_features = [f for f in CLUSTER_FEATURES if f not in assignments.columns]
if missing_features:
    raise ValueError("Missing clustering features: " + ", ".join(missing_features))
for feature in CLUSTER_FEATURES:
    assignments[feature] = safe_numeric(assignments[feature])

# =============================================================================
# 3. FULL CLUSTER FEATURE DISTRIBUTIONS
# =============================================================================

distribution_rows = []
for cid in cluster_ids:
    subset = assignments[assignments["cluster_id"] == cid]
    label = subset["temporal_pattern"].iloc[0]
    for feature in CLUSTER_FEATURES:
        values = safe_numeric(subset[feature]).dropna()
        if values.empty:
            continue
        distribution_rows.append({
            "cluster_id": cid,
            "script12_label": label,
            "feature": feature,
            "n": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(ddof=1),
            "q25": values.quantile(.25),
            "q75": values.quantile(.75),
            "iqr": values.quantile(.75) - values.quantile(.25),
            "min": values.min(),
            "max": values.max(),
        })

distribution_df = pd.DataFrame(distribution_rows)
distribution_df.to_csv(TABLE_DIR / "01_cluster_feature_distribution_summary.csv", index=False)

# =============================================================================
# 4. MEMBER-LEVEL CHANGE DIAGNOSTICS
# =============================================================================

diag_cols = [c for c in [
    "ADM3_ID", "ADM3_NAME", "ADM1_NAME", "ADM2_NAME", "cluster_id",
    "temporal_pattern", "early_mean", "recent_mean", "change_pp",
    "previous_5y_mean", "recent_5y_mean", "recent_5y_change_pp",
    "trend_slope", "recent_slope", "acceleration", "flood_year_fraction",
    "longest_consecutive_run", "cv_flood", "max_minus_median",
    "post_peak_decline",
] if c in assignments.columns]

member_diag = assignments[diag_cols].copy()
member_diag["change_direction"] = np.select(
    [member_diag["change_pp"] > NEAR_ZERO_CHANGE_PP,
     member_diag["change_pp"] < -NEAR_ZERO_CHANGE_PP],
    ["Increase", "Decrease"],
    default="Approximately stable",
)
member_diag["recent_5y_direction"] = np.select(
    [member_diag["recent_5y_change_pp"] > NEAR_ZERO_CHANGE_PP,
     member_diag["recent_5y_change_pp"] < -NEAR_ZERO_CHANGE_PP],
    ["Recent 5-year increase", "Recent 5-year decrease"],
    default="Recent 5-year approximately stable",
)
member_diag.to_csv(TABLE_DIR / "06_cluster_member_change_diagnostics.csv", index=False)

change_rows = []
for cid in cluster_ids:
    subset = member_diag[member_diag["cluster_id"] == cid]
    n = len(subset)
    label = subset["temporal_pattern"].iloc[0]

    inc = int((subset["change_direction"] == "Increase").sum())
    stable = int((subset["change_direction"] == "Approximately stable").sum())
    dec = int((subset["change_direction"] == "Decrease").sum())
    r_inc = int((subset["recent_5y_direction"] == "Recent 5-year increase").sum())
    r_stable = int((subset["recent_5y_direction"] == "Recent 5-year approximately stable").sum())
    r_dec = int((subset["recent_5y_direction"] == "Recent 5-year decrease").sum())

    change_rows.append({
        "cluster_id": cid,
        "script12_label": label,
        "adm3_count": n,
        "increase_n": inc,
        "increase_percent": safe_percent(inc, n),
        "approximately_stable_n": stable,
        "approximately_stable_percent": safe_percent(stable, n),
        "decrease_n": dec,
        "decrease_percent": safe_percent(dec, n),
        "recent_5y_increase_n": r_inc,
        "recent_5y_increase_percent": safe_percent(r_inc, n),
        "recent_5y_stable_n": r_stable,
        "recent_5y_stable_percent": safe_percent(r_stable, n),
        "recent_5y_decrease_n": r_dec,
        "recent_5y_decrease_percent": safe_percent(r_dec, n),
        "mean_early": safe_numeric(subset["early_mean"]).mean(),
        "median_early": safe_numeric(subset["early_mean"]).median(),
        "mean_recent": safe_numeric(subset["recent_mean"]).mean(),
        "median_recent": safe_numeric(subset["recent_mean"]).median(),
        "mean_change_pp": safe_numeric(subset["change_pp"]).mean(),
        "median_change_pp": safe_numeric(subset["change_pp"]).median(),
        "mean_recent_5y_change_pp": safe_numeric(subset["recent_5y_change_pp"]).mean(),
        "median_recent_5y_change_pp": safe_numeric(subset["recent_5y_change_pp"]).median(),
        "mean_flood_year_fraction": safe_numeric(subset["flood_year_fraction"]).mean(),
        "median_flood_year_fraction": safe_numeric(subset["flood_year_fraction"]).median(),
        "mean_cv_flood": safe_numeric(subset["cv_flood"]).mean(),
        "median_cv_flood": safe_numeric(subset["cv_flood"]).median(),
    })

change_summary = pd.DataFrame(change_rows)
change_summary.to_csv(TABLE_DIR / "02_cluster_change_direction_summary.csv", index=False)

# =============================================================================
# 5. PCA INTERPRETATION
# =============================================================================

pca_out = pca_loadings.copy()
for pc in [c for c in pca_out.columns if c.startswith("PC")]:
    pca_out[f"{pc}_absolute_loading"] = safe_numeric(pca_out[pc]).abs()
    pca_out[f"{pc}_rank_by_absolute_loading"] = (
        pca_out[f"{pc}_absolute_loading"].rank(ascending=False, method="min")
    )
pca_out.to_csv(TABLE_DIR / "04_pca_loading_interpretation.csv", index=False)

# =============================================================================
# 6. JOIN ANNUAL TRAJECTORIES TO CLUSTERS
# =============================================================================

# Use an explicit canonical join key to avoid collisions with any ADM3_ID
# column already present in the Script 10 annual file.
cluster_lookup = assignments[
    ["ADM3_ID", "ADM3_NAME", "cluster_id", "temporal_pattern"]
].copy()

cluster_lookup = cluster_lookup.rename(
    columns={
        "ADM3_ID": "CLUSTER_ADM3_ID",
        "ADM3_NAME": "CLUSTER_ADM3_NAME",
    }
)

cluster_lookup["CLUSTER_ADM3_ID"] = (
    cluster_lookup["CLUSTER_ADM3_ID"].astype(str)
)

annual_clustered = annual.merge(
    cluster_lookup,
    left_on="ADM3_ID_CANON",
    right_on="CLUSTER_ADM3_ID",
    how="inner",
    validate="many_to_one",
)

# Keep one explicit canonical identifier for all downstream trajectory lookups.
annual_clustered["ADM3_CLUSTER_KEY"] = annual_clustered["ADM3_ID_CANON"].astype(str)

# =============================================================================
# 7. REPRESENTATIVE ADM3 SUMMARY
# =============================================================================

representatives["cluster_id"] = pd.to_numeric(representatives["cluster_id"], errors="coerce").astype(int)
representatives["ADM3_ID"] = representatives["ADM3_ID"].astype(str)

rep_rows = []
for _, rep in representatives.iterrows():
    cid = int(rep["cluster_id"])
    adm3_id = str(rep["ADM3_ID"])
    subset = annual_clustered[annual_clustered["ADM3_CLUSTER_KEY"] == adm3_id].sort_values("year")
    if subset.empty:
        continue

    early = subset[subset["year"].between(EARLY_START, EARLY_END)]["annual_flooded_percent"]
    recent = subset[subset["year"].between(RECENT_START, RECENT_END)]["annual_flooded_percent"]
    prev5 = subset[subset["year"].between(PREVIOUS_5Y_START, PREVIOUS_5Y_END)]["annual_flooded_percent"]
    recent5 = subset[subset["year"].between(RECENT_5Y_START, RECENT_5Y_END)]["annual_flooded_percent"]

    rep_rows.append({
        "cluster_id": cid,
        "temporal_pattern": rep.get("temporal_pattern", np.nan),
        "ADM3_ID": adm3_id,
        "ADM3_NAME": rep.get("ADM3_NAME", np.nan),
        "ADM2_NAME": rep.get("ADM2_NAME", np.nan),
        "centroid_distance": rep.get("centroid_distance", np.nan),
        "early_mean": safe_numeric(early).mean(),
        "recent_mean": safe_numeric(recent).mean(),
        "change_pp": safe_numeric(recent).mean() - safe_numeric(early).mean(),
        "previous_5y_mean": safe_numeric(prev5).mean(),
        "recent_5y_mean": safe_numeric(recent5).mean(),
        "recent_5y_change_pp": safe_numeric(recent5).mean() - safe_numeric(prev5).mean(),
        "flood_year_count": int((safe_numeric(subset["annual_flooded_percent"]).fillna(0) > 0).sum()),
        "maximum_flooded_percent": safe_numeric(subset["annual_flooded_percent"]).max(),
    })

representative_summary = pd.DataFrame(rep_rows)
representative_summary.to_csv(TABLE_DIR / "05_representative_adm3_summary.csv", index=False)

# =============================================================================
# 8. DIAGNOSTIC LABEL INTERPRETATION
# =============================================================================

validation_rows = []
for _, row in change_summary.iterrows():
    cid = int(row["cluster_id"])
    label = row["script12_label"]
    inc = row["increase_percent"]
    stable = row["approximately_stable_percent"]
    dec = row["decrease_percent"]
    mean_change = row["mean_change_pp"]
    median_change = row["median_change_pp"]
    flood_fraction = row["mean_flood_year_fraction"]
    cv = row["mean_cv_flood"]

    if inc >= 70 and mean_change >= 15:
        interpretation = "Strong recent expansion"
        evidence = f"{inc:.1f}% increase; mean change={mean_change:.2f} pp."
    elif inc >= 60 and mean_change > NEAR_ZERO_CHANGE_PP:
        interpretation = "Moderate / widespread expansion"
        evidence = f"{inc:.1f}% increase; mean={mean_change:.2f} pp; median={median_change:.2f} pp."
    elif flood_fraction >= 0.80 and abs(mean_change) < 10:
        interpretation = "Recurrent / persistent"
        evidence = f"Flood-year fraction={flood_fraction:.3f}; mean change={mean_change:.2f} pp."
    elif cv >= 1.0 and flood_fraction < 0.80:
        interpretation = "Episodic / high-volatility"
        evidence = f"Mean CV={cv:.2f}; flood-year fraction={flood_fraction:.3f}."
    elif stable >= 60 and abs(mean_change) <= NEAR_ZERO_CHANGE_PP:
        interpretation = "Relatively stable"
        evidence = f"{stable:.1f}% within ±{NEAR_ZERO_CHANGE_PP:.1f} pp."
    elif dec >= 60 and mean_change < -NEAR_ZERO_CHANGE_PP:
        interpretation = "Declining / weakening"
        evidence = f"{dec:.1f}% decrease; mean change={mean_change:.2f} pp."
    else:
        interpretation = "Mixed / intermediate temporal behaviour"
        evidence = f"Increase={inc:.1f}%, stable={stable:.1f}%, decrease={dec:.1f}%."

    validation_rows.append({
        "cluster_id": cid,
        "adm3_count": int(row["adm3_count"]),
        "script12_automatic_label": label,
        "validation_interpretation": interpretation,
        "label_agreement": "Supported" if str(label).lower() == interpretation.lower() else "Review recommended",
        "evidence_summary": evidence,
        "increase_percent": inc,
        "stable_percent": stable,
        "decrease_percent": dec,
        "recent_5y_increase_percent": row["recent_5y_increase_percent"],
        "recent_5y_decrease_percent": row["recent_5y_decrease_percent"],
        "mean_change_pp": mean_change,
        "median_change_pp": median_change,
        "mean_recent_5y_change_pp": row["mean_recent_5y_change_pp"],
        "mean_flood_year_fraction": flood_fraction,
        "mean_cv_flood": cv,
    })

validation_df = pd.DataFrame(validation_rows)
validation_df.to_csv(TABLE_DIR / "03_cluster_label_validation_summary.csv", index=False)

# =============================================================================
# 9. ANNUAL CLUSTER TRAJECTORY SUMMARY
# =============================================================================

cluster_year_summary = (
    annual_clustered.groupby(["cluster_id", "year"])["annual_flooded_percent"]
    .agg(
        mean="mean",
        median="median",
        q25=lambda x: x.quantile(.25),
        q75=lambda x: x.quantile(.75),
    )
    .reset_index()
)
cluster_year_summary.to_csv(TABLE_DIR / "07_cluster_annual_trajectory_summary.csv", index=False)

# =============================================================================
# 10. FIGURES
# =============================================================================

if MATPLOTLIB_AVAILABLE:
    plot_specs = [
        ("change_pp", "01_cluster_change_pp_boxplot.png",
         "Recent mean − early mean (percentage points)",
         "ADM3 early-to-recent flood change by temporal cluster"),
        ("recent_5y_change_pp", "02_cluster_recent_5y_change_boxplot.png",
         "2021–2025 mean − 2016–2020 mean (percentage points)",
         "Recent five-year flood change by ADM3 temporal cluster"),
        ("flood_year_fraction", "03_cluster_flood_year_fraction_boxplot.png",
         "Flood-year fraction",
         "Flood recurrence by ADM3 temporal cluster"),
        ("cv_flood", "04_cluster_cv_boxplot.png",
         "Coefficient of variation",
         "Flood variability by ADM3 temporal cluster"),
    ]

    for feature, filename, ylabel, title in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 6))
        data = [
            assignments.loc[assignments["cluster_id"] == cid, feature].dropna().values
            for cid in cluster_ids
        ]
        ax.boxplot(data, tick_labels=[f"C{cid}" for cid in cluster_ids], showfliers=True)
        if feature in ["change_pp", "recent_5y_change_pp"]:
            ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=240, bbox_inches="tight")
        plt.close(fig)

    # Representative trajectories
    fig, ax = plt.subplots(figsize=(12, 7))
    for _, rep in representatives.iterrows():
        cid = int(rep["cluster_id"])
        adm3_id = str(rep["ADM3_ID"])
        subset = annual_clustered[annual_clustered["ADM3_CLUSTER_KEY"] == adm3_id].sort_values("year")
        if subset.empty:
            continue
        ax.plot(
            subset["year"], subset["annual_flooded_percent"],
            marker="o", markersize=3, linewidth=1.5,
            label=f"C{cid}: {rep.get('ADM3_NAME', adm3_id)} — {rep.get('temporal_pattern', '')}"
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual flooded area (%)")
    ax.set_title("Centroid-nearest representative ADM3 temporal trajectories")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_representative_temporal_trajectories.png",
                dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Cluster median trajectories
    fig, ax = plt.subplots(figsize=(12, 7))
    for cid in cluster_ids:
        subset = cluster_year_summary[cluster_year_summary["cluster_id"] == cid]
        label = assignments.loc[assignments["cluster_id"] == cid, "temporal_pattern"].iloc[0]
        ax.plot(subset["year"], subset["median"], marker="o", markersize=3,
                linewidth=2, label=f"C{cid}: {label}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median annual flooded area (%)")
    ax.set_title("Median temporal trajectory of each ADM3 cluster")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_cluster_median_temporal_trajectories.png",
                dpi=240, bbox_inches="tight")
    plt.close(fig)

# Standardized cluster profile
cluster_feature_means = assignments.groupby("cluster_id")[CLUSTER_FEATURES].mean()

if SKLEARN_AVAILABLE:
    scaler = StandardScaler()
    standardized = pd.DataFrame(
        scaler.fit_transform(cluster_feature_means),
        index=cluster_feature_means.index,
        columns=CLUSTER_FEATURES,
    )
    standardized.to_csv(TABLE_DIR / "08_standardized_cluster_feature_profiles.csv")

    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(13, 6))
        im = ax.imshow(standardized.values, aspect="auto")
        ax.set_xticks(np.arange(len(CLUSTER_FEATURES)))
        ax.set_xticklabels(CLUSTER_FEATURES, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(standardized)))
        ax.set_yticklabels([f"Cluster {cid}" for cid in standardized.index])
        ax.set_title("Standardized ADM3 temporal-feature profiles by cluster")
        fig.colorbar(im, ax=ax, label="Standardized cluster mean")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "07_cluster_feature_heatmap.png",
                    dpi=240, bbox_inches="tight")
        plt.close(fig)

if MATPLOTLIB_AVAILABLE:
    pcs = [c for c in ["PC1", "PC2"] if c in pca_loadings.columns]
    if pcs:
        matrix = pca_loadings.set_index("feature")[pcs]
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(matrix.values, aspect="auto")
        ax.set_xticks(np.arange(len(pcs)))
        ax.set_xticklabels(pcs)
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_title("PCA loadings for ADM3 temporal features")
        fig.colorbar(im, ax=ax, label="PCA loading")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "08_pca_loading_heatmap.png",
                    dpi=240, bbox_inches="tight")
        plt.close(fig)

# =============================================================================
# 11. QC SUMMARY
# =============================================================================

qc_lines = [
    "SCRIPT 13 — ADM3 TEMPORAL PATTERN VALIDATION QC",
    "=" * 72,
    f"Study period: {START_YEAR}-{END_YEAR}",
    f"Clustered ADM3 received from Script 12: {len(assignments)}",
    f"Number of clusters: {len(cluster_ids)}",
    f"Cluster IDs: {cluster_ids}",
    "",
    "IMPORTANT:",
    "Script 13 does not re-run clustering or change ADM3 cluster membership.",
    "It validates interpretation and descriptive naming of Script 12 clusters.",
    f"Approximately stable diagnostic threshold: ±{NEAR_ZERO_CHANGE_PP:.1f} percentage points.",
    "",
    "Validation results:",
]

for _, row in validation_df.iterrows():
    qc_lines += [
        "",
        f"Cluster {int(row['cluster_id'])}",
        f"  Script 12 label: {row['script12_automatic_label']}",
        f"  Diagnostic interpretation: {row['validation_interpretation']}",
        f"  Increase: {row['increase_percent']:.1f}%",
        f"  Stable: {row['stable_percent']:.1f}%",
        f"  Decrease: {row['decrease_percent']:.1f}%",
        f"  Mean change: {row['mean_change_pp']:.2f} pp",
        f"  Median change: {row['median_change_pp']:.2f} pp",
        f"  Flood-year fraction: {row['mean_flood_year_fraction']:.3f}",
        f"  Mean CV: {row['mean_cv_flood']:.3f}",
    ]

qc_lines += [
    "",
    "CAUTION:",
    "Diagnostic labels are not automatically accepted as final scientific labels.",
    "Review distributions, representative trajectories, cluster median trajectories,",
    "and PCA loadings together before finalizing temporal-pattern names.",
]

(QC_DIR / "13_validation_qc_summary.txt").write_text(
    "\n".join(qc_lines), encoding="utf-8"
)

print("\n" + "=" * 96)
print("SCRIPT 13 COMPLETED")
print("=" * 96)
print(validation_df[
    ["cluster_id", "adm3_count", "script12_automatic_label",
     "validation_interpretation", "increase_percent", "stable_percent",
     "decrease_percent", "mean_change_pp", "median_change_pp",
     "mean_flood_year_fraction", "mean_cv_flood"]
].to_string(index=False))
print(f"\nOutput root:\n{OUTPUT_ROOT}")
print("\nNext step: review Script 13 evidence, finalize cluster names, then map them in Script 14.")
