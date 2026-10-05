#!/usr/bin/env python3
"""
14_finalize_adm3_temporal_pattern_classification.py

JBG060 Capstone Data Challenge
South Sudan Flood Analysis
Final Validated ADM3 Temporal Pattern Classification, 2000-2025

PURPOSE
-------
Finalize the ADM3 temporal-pattern typology after the discovery analysis in
Script 12 and the interpretation-validation analysis in Script 13.

Analytical sequence:
    Script 12 -> temporal pattern discovery / preliminary labels
    Script 13 -> interpretation validation
    Script 14 -> final validated temporal typology

IMPORTANT
---------
Script 14 DOES NOT:
- re-run hierarchical clustering;
- change k;
- change ADM3 cluster membership;
- re-run PCA;
- alter Script 10/11 coverage classifications.

Instead, it preserves the Script 12 clustering solution and uses Script 13
validation evidence to finalize the descriptive interpretation of each cluster.

The key validated revision is:
    Script 12 preliminary:
        "Relatively low / stable"
    Script 14 final:
        "Moderate recent expansion"

The other three preliminary cluster labels are retained because Script 13
supported their interpretation:
    "Strong recent expansion"
    "Episodic / high-volatility"
    "Recurrent / persistent"

Non-clustered categories remain unchanged:
    "No detected flood record"
    "Insufficient source coverage"

INPUTS
------
Script 12:
processed_data/cross_dataset_analysis/02_adm3_flood/
    12_analyze_adm3_temporal_patterns/tables/

Script 13:
processed_data/cross_dataset_analysis/02_adm3_flood/
    13_validate_adm3_temporal_pattern_interpretation/tables/

OUTPUTS
-------
processed_data/cross_dataset_analysis/02_adm3_flood/
    14_finalize_adm3_temporal_pattern_classification/
        tables/
        figures/
        qc/

Main outputs:
- 14_cluster_label_revision_audit.csv
- 14_adm3_final_cluster_assignments.csv
- 14_adm3_final_temporal_pattern_classification_all512.csv
- 14_adm3_final_temporal_pattern_counts_all512.csv
- 14_final_cluster_profiles.csv
- 14_final_representative_adm3.csv
- 14_final_candidate_change_points.csv
- 14_final_pattern_methodological_summary.csv
- 14_final_typology_summary.png
- 14_final_validation_qc_summary.txt
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

SCRIPT12_ROOT = (
    ADM3_ROOT
    / "12_analyze_adm3_temporal_patterns"
)

SCRIPT13_ROOT = (
    ADM3_ROOT
    / "13_validate_adm3_temporal_pattern_interpretation"
)

SCRIPT12_TABLE_DIR = SCRIPT12_ROOT / "tables"
SCRIPT12_QC_DIR = SCRIPT12_ROOT / "qc"

SCRIPT13_TABLE_DIR = SCRIPT13_ROOT / "tables"
SCRIPT13_QC_DIR = SCRIPT13_ROOT / "qc"

OUTPUT_ROOT = (
    ADM3_ROOT
    / "14_finalize_adm3_temporal_pattern_classification"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"

for directory in [TABLE_DIR, FIGURE_DIR, QC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. INPUT FILES
# =============================================================================

SCRIPT12_ASSIGNMENTS_FILE = (
    SCRIPT12_TABLE_DIR
    / "12_adm3_cluster_assignments_main_analysis.csv"
)

SCRIPT12_CLASSIFICATION_FILE = (
    SCRIPT12_TABLE_DIR
    / "12_adm3_temporal_pattern_classification_all512.csv"
)

SCRIPT12_COUNTS_FILE = (
    SCRIPT12_TABLE_DIR
    / "12_adm3_temporal_pattern_counts_all512.csv"
)

SCRIPT12_PROFILES_FILE = (
    SCRIPT12_TABLE_DIR
    / "12_adm3_cluster_profiles.csv"
)

SCRIPT12_REPRESENTATIVES_FILE = (
    SCRIPT12_TABLE_DIR
    / "12_adm3_centroid_nearest_representatives.csv"
)

SCRIPT12_CHANGE_POINTS_FILE = (
    SCRIPT12_TABLE_DIR
    / "12_adm3_candidate_change_points.csv"
)

SCRIPT12_SILHOUETTE_FILE = (
    SCRIPT12_TABLE_DIR
    / "12_adm3_silhouette_diagnostics.csv"
)

SCRIPT12_PCA_VARIANCE_FILE = (
    SCRIPT12_TABLE_DIR
    / "12_adm3_pca_explained_variance.csv"
)

SCRIPT12_CLUSTER_SELECTION_FILE = (
    SCRIPT12_QC_DIR
    / "12_cluster_selection_summary.txt"
)

SCRIPT13_VALIDATION_FILE = (
    SCRIPT13_TABLE_DIR
    / "03_cluster_label_validation_summary.csv"
)

SCRIPT13_CHANGE_SUMMARY_FILE = (
    SCRIPT13_TABLE_DIR
    / "02_cluster_change_direction_summary.csv"
)

SCRIPT13_REPRESENTATIVE_SUMMARY_FILE = (
    SCRIPT13_TABLE_DIR
    / "05_representative_adm3_summary.csv"
)


# =============================================================================
# 3. FINAL VALIDATED TYPOLOGY
# =============================================================================

# IMPORTANT:
# These are post-clustering descriptive interpretations.
# They do not affect cluster membership.

FINAL_LABEL_MAP = {
    1: "Strong recent expansion",
    2: "Episodic / high-volatility",
    3: "Moderate recent expansion",
    4: "Recurrent / persistent",
}

EXPECTED_PRELIMINARY_LABELS = {
    1: "Strong recent expansion",
    2: "Episodic / high-volatility",
    3: "Relatively low / stable",
    4: "Recurrent / persistent",
}

NONCLUSTER_PATTERNS = {
    "No detected flood record",
    "Insufficient source coverage",
}


# =============================================================================
# 4. HELPERS
# =============================================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def labels_equal(a, b) -> bool:
    return normalize_label(a).lower() == normalize_label(b).lower()


# =============================================================================
# 5. LOAD INPUTS
# =============================================================================

print("=" * 96)
print("SCRIPT 14 — FINAL VALIDATED ADM3 TEMPORAL PATTERN CLASSIFICATION")
print("=" * 96)

required_files = {
    "Script 12 cluster assignments": SCRIPT12_ASSIGNMENTS_FILE,
    "Script 12 all-512 classification": SCRIPT12_CLASSIFICATION_FILE,
    "Script 12 cluster profiles": SCRIPT12_PROFILES_FILE,
    "Script 12 representatives": SCRIPT12_REPRESENTATIVES_FILE,
    "Script 12 change points": SCRIPT12_CHANGE_POINTS_FILE,
    "Script 12 silhouette diagnostics": SCRIPT12_SILHOUETTE_FILE,
    "Script 12 PCA variance": SCRIPT12_PCA_VARIANCE_FILE,
    "Script 13 validation summary": SCRIPT13_VALIDATION_FILE,
    "Script 13 change-direction summary": SCRIPT13_CHANGE_SUMMARY_FILE,
}

for description, path in required_files.items():
    require_file(path, description)

assignments12 = pd.read_csv(SCRIPT12_ASSIGNMENTS_FILE)
classification12 = pd.read_csv(SCRIPT12_CLASSIFICATION_FILE)
profiles12 = pd.read_csv(SCRIPT12_PROFILES_FILE)
representatives12 = pd.read_csv(SCRIPT12_REPRESENTATIVES_FILE)
change_points12 = pd.read_csv(SCRIPT12_CHANGE_POINTS_FILE)
silhouette12 = pd.read_csv(SCRIPT12_SILHOUETTE_FILE)
pca_variance12 = pd.read_csv(SCRIPT12_PCA_VARIANCE_FILE)
validation13 = pd.read_csv(SCRIPT13_VALIDATION_FILE)
change_summary13 = pd.read_csv(SCRIPT13_CHANGE_SUMMARY_FILE)

representative_summary13 = None
if SCRIPT13_REPRESENTATIVE_SUMMARY_FILE.exists():
    representative_summary13 = pd.read_csv(
        SCRIPT13_REPRESENTATIVE_SUMMARY_FILE
    )

print(f"\nScript 12 clustered ADM3: {len(assignments12):,}")
print(f"Script 12 final classification rows: {len(classification12):,}")
print(f"Script 13 validation rows: {len(validation13):,}")


# =============================================================================
# 6. BASIC INTEGRITY CHECKS
# =============================================================================

for df_name, df in [
    ("Script 12 assignments", assignments12),
    ("Script 12 profiles", profiles12),
    ("Script 12 representatives", representatives12),
    ("Script 13 validation", validation13),
]:
    if "cluster_id" not in df.columns:
        raise ValueError(
            f"{df_name} does not contain cluster_id."
        )

assignments12["cluster_id"] = pd.to_numeric(
    assignments12["cluster_id"],
    errors="coerce",
)

if assignments12["cluster_id"].isna().any():
    raise ValueError(
        "Script 12 main-clustering assignments contain missing cluster_id."
    )

assignments12["cluster_id"] = assignments12["cluster_id"].astype(int)

profiles12["cluster_id"] = pd.to_numeric(
    profiles12["cluster_id"],
    errors="coerce",
).astype(int)

representatives12["cluster_id"] = pd.to_numeric(
    representatives12["cluster_id"],
    errors="coerce",
).astype(int)

validation13["cluster_id"] = pd.to_numeric(
    validation13["cluster_id"],
    errors="coerce",
).astype(int)

change_summary13["cluster_id"] = pd.to_numeric(
    change_summary13["cluster_id"],
    errors="coerce",
).astype(int)

cluster_ids = sorted(
    assignments12["cluster_id"].unique().tolist()
)

expected_cluster_ids = sorted(FINAL_LABEL_MAP.keys())

if cluster_ids != expected_cluster_ids:
    raise ValueError(
        "\nUnexpected Script 12 cluster IDs.\n"
        f"Observed: {cluster_ids}\n"
        f"Expected: {expected_cluster_ids}\n"
        "Script 14 is intentionally tied to the validated four-cluster "
        "solution. Review before proceeding."
    )

if len(assignments12) != 490:
    print(
        "\nWARNING:"
        f" Expected 490 clustered ADM3 from the validated Script 12 run, "
        f"but found {len(assignments12)}."
    )

if len(classification12) != 512:
    print(
        "\nWARNING:"
        f" Expected 512 total ADM3 classification rows, "
        f"but found {len(classification12)}."
    )


# =============================================================================
# 7. VERIFY SCRIPT 12 PRELIMINARY LABELS
# =============================================================================

print("\n" + "=" * 96)
print("VERIFY SCRIPT 12 PRELIMINARY LABELS")
print("=" * 96)

script12_label_col = first_existing_column(
    assignments12,
    ["temporal_pattern", "descriptive_label"],
)

if script12_label_col is None:
    raise ValueError(
        "Could not identify Script 12 preliminary temporal-pattern label."
    )

observed_preliminary = (
    assignments12
    .groupby("cluster_id")[script12_label_col]
    .agg(lambda s: sorted(set(s.dropna().astype(str))))
    .to_dict()
)

for cid in cluster_ids:
    observed = observed_preliminary.get(cid, [])
    expected = EXPECTED_PRELIMINARY_LABELS[cid]

    print(
        f"Cluster {cid}: "
        f"observed={observed} | expected={expected}"
    )

    if len(observed) != 1:
        raise ValueError(
            f"Cluster {cid} has inconsistent preliminary labels: {observed}"
        )

    if not labels_equal(observed[0], expected):
        raise ValueError(
            f"\nCluster {cid} preliminary label differs from the "
            f"validated Script 12 run.\n"
            f"Observed: {observed[0]}\n"
            f"Expected: {expected}\n"
            "Do not silently finalize a different upstream clustering run."
        )


# =============================================================================
# 8. VERIFY SCRIPT 13 VALIDATION EVIDENCE
# =============================================================================

print("\n" + "=" * 96)
print("VERIFY SCRIPT 13 VALIDATION EVIDENCE")
print("=" * 96)

required_validation_columns = [
    "cluster_id",
    "adm3_count",
    "script12_automatic_label",
    "validation_interpretation",
    "increase_percent",
    "stable_percent",
    "decrease_percent",
    "mean_change_pp",
    "median_change_pp",
    "mean_flood_year_fraction",
    "mean_cv_flood",
]

missing_validation_columns = [
    c for c in required_validation_columns
    if c not in validation13.columns
]

if missing_validation_columns:
    raise ValueError(
        "Script 13 validation summary is missing columns:\n"
        + "\n".join(missing_validation_columns)
    )

if sorted(validation13["cluster_id"].unique().tolist()) != cluster_ids:
    raise ValueError(
        "Script 13 validation does not contain the same four clusters "
        "as Script 12."
    )

# Cluster 3 is the specifically validated revision.
cluster3 = validation13[
    validation13["cluster_id"] == 3
].iloc[0]

if cluster3["increase_percent"] < 90:
    raise ValueError(
        "Cluster 3 no longer shows the strong within-cluster increase "
        "consistency used to justify the validated label revision."
    )

if cluster3["mean_change_pp"] <= 2:
    raise ValueError(
        "Cluster 3 mean change is no longer sufficiently positive to "
        "support the validated expansion interpretation."
    )

if cluster3["median_change_pp"] <= 2:
    raise ValueError(
        "Cluster 3 median change is no longer sufficiently positive to "
        "support the validated expansion interpretation."
    )

print(
    "Cluster 3 validation evidence:"
    f"\n  increase_percent = {cluster3['increase_percent']:.1f}%"
    f"\n  mean_change_pp = {cluster3['mean_change_pp']:.2f}"
    f"\n  median_change_pp = {cluster3['median_change_pp']:.2f}"
)


# =============================================================================
# 9. BUILD LABEL REVISION AUDIT TABLE
# =============================================================================

audit_rows = []

for cid in cluster_ids:

    val = validation13[
        validation13["cluster_id"] == cid
    ].iloc[0]

    preliminary = EXPECTED_PRELIMINARY_LABELS[cid]
    final_label = FINAL_LABEL_MAP[cid]

    changed = not labels_equal(
        preliminary,
        final_label,
    )

    if cid == 3:
        decision = (
            "Revised after Script 13 validation. The preliminary "
            "'stable' interpretation was not supported because the "
            "cluster showed consistent positive early-to-recent change "
            "across its members, with positive mean and median change."
        )
    else:
        decision = (
            "Retained after Script 13 validation; the preliminary "
            "interpretation was supported by the validation diagnostics."
        )

    audit_rows.append(
        {
            "cluster_id": cid,
            "adm3_count": int(val["adm3_count"]),
            "script12_preliminary_label": preliminary,
            "script13_validation_interpretation":
                val["validation_interpretation"],
            "script14_final_label": final_label,
            "label_changed": changed,
            "cluster_membership_changed": False,
            "increase_percent":
                val["increase_percent"],
            "stable_percent":
                val["stable_percent"],
            "decrease_percent":
                val["decrease_percent"],
            "mean_change_pp":
                val["mean_change_pp"],
            "median_change_pp":
                val["median_change_pp"],
            "mean_recent_5y_change_pp":
                val.get(
                    "mean_recent_5y_change_pp",
                    np.nan,
                ),
            "mean_flood_year_fraction":
                val["mean_flood_year_fraction"],
            "mean_cv_flood":
                val["mean_cv_flood"],
            "finalization_decision":
                decision,
        }
    )

label_audit = pd.DataFrame(audit_rows)

label_audit.to_csv(
    TABLE_DIR / "14_cluster_label_revision_audit.csv",
    index=False,
)


# =============================================================================
# 10. FINALIZE MAIN-CLUSTER ASSIGNMENTS
# =============================================================================

final_assignments = assignments12.copy()

final_assignments = final_assignments.rename(
    columns={
        script12_label_col:
            "script12_preliminary_temporal_pattern"
    }
)

final_assignments["final_temporal_pattern"] = (
    final_assignments["cluster_id"]
    .map(FINAL_LABEL_MAP)
)

if final_assignments["final_temporal_pattern"].isna().any():
    raise ValueError(
        "At least one clustered ADM3 could not be mapped to a final label."
    )

final_assignments["label_validation_status"] = np.where(
    final_assignments["cluster_id"] == 3,
    "Revised after Script 13 validation",
    "Preliminary label retained after Script 13 validation",
)

final_assignments["cluster_membership_changed"] = False

# Put the final label close to the identifying fields.
preferred_order = [
    "ADM3_ID",
    "ADM3_NAME",
    "ADM1_NAME",
    "ADM2_NAME",
    "cluster_id",
    "script12_preliminary_temporal_pattern",
    "final_temporal_pattern",
    "label_validation_status",
    "cluster_membership_changed",
]

remaining_columns = [
    c for c in final_assignments.columns
    if c not in preferred_order
]

final_assignments = final_assignments[
    [c for c in preferred_order if c in final_assignments.columns]
    + remaining_columns
]

final_assignments.to_csv(
    TABLE_DIR / "14_adm3_final_cluster_assignments.csv",
    index=False,
)


# =============================================================================
# 11. FINALIZE ALL-512 CLASSIFICATION
# =============================================================================

classification14 = classification12.copy()

if "temporal_pattern" not in classification14.columns:
    raise ValueError(
        "Script 12 all-512 classification lacks temporal_pattern."
    )

classification14 = classification14.rename(
    columns={
        "temporal_pattern":
            "script12_preliminary_temporal_pattern"
    }
)

classification14["final_temporal_pattern"] = (
    classification14[
        "script12_preliminary_temporal_pattern"
    ]
)

cluster_mask = classification14["cluster_id"].notna()

classification14.loc[
    cluster_mask,
    "cluster_id",
] = pd.to_numeric(
    classification14.loc[
        cluster_mask,
        "cluster_id",
    ],
    errors="coerce",
)

for cid, final_label in FINAL_LABEL_MAP.items():
    mask = (
        classification14["cluster_id"] == cid
    )
    classification14.loc[
        mask,
        "final_temporal_pattern",
    ] = final_label

# Explicitly preserve non-clustered coverage categories.
noncluster_values = set(
    classification14.loc[
        ~cluster_mask,
        "script12_preliminary_temporal_pattern",
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

unexpected_noncluster = (
    noncluster_values
    - NONCLUSTER_PATTERNS
)

if unexpected_noncluster:
    raise ValueError(
        "Unexpected non-clustered Script 12 categories found:\n"
        + "\n".join(sorted(unexpected_noncluster))
    )

classification14["label_validation_status"] = np.where(
    ~cluster_mask,
    "Coverage-status category retained from Script 12",
    np.where(
        classification14["cluster_id"] == 3,
        "Revised after Script 13 validation",
        "Preliminary label retained after Script 13 validation",
    ),
)

classification14["cluster_membership_changed"] = False

classification14 = classification14.sort_values(
    [
        "final_temporal_pattern",
        "ADM3_NAME",
    ]
)

classification14.to_csv(
    TABLE_DIR
    / "14_adm3_final_temporal_pattern_classification_all512.csv",
    index=False,
)


# =============================================================================
# 12. FINAL COUNTS
# =============================================================================

final_counts = (
    classification14[
        "final_temporal_pattern"
    ]
    .value_counts()
    .rename_axis(
        "final_temporal_pattern"
    )
    .reset_index(
        name="adm3_count"
    )
)

final_counts["share_percent_of_512"] = (
    final_counts["adm3_count"]
    / len(classification14)
    * 100
)

final_counts.to_csv(
    TABLE_DIR
    / "14_adm3_final_temporal_pattern_counts_all512.csv",
    index=False,
)


# =============================================================================
# 13. FINAL CLUSTER PROFILES
# =============================================================================

profiles14 = profiles12.copy()

profile_label_col = first_existing_column(
    profiles14,
    ["descriptive_label", "temporal_pattern"],
)

if profile_label_col is not None:
    profiles14 = profiles14.rename(
        columns={
            profile_label_col:
                "script12_preliminary_label"
        }
    )
else:
    profiles14[
        "script12_preliminary_label"
    ] = profiles14[
        "cluster_id"
    ].map(EXPECTED_PRELIMINARY_LABELS)

profiles14["final_temporal_pattern"] = (
    profiles14["cluster_id"]
    .map(FINAL_LABEL_MAP)
)

# Add Script 13 member-level validation statistics.
profile_validation_columns = [
    "cluster_id",
    "increase_percent",
    "approximately_stable_percent",
    "decrease_percent",
    "recent_5y_increase_percent",
    "recent_5y_decrease_percent",
    "median_change_pp",
    "median_recent_5y_change_pp",
]

available_profile_validation_columns = [
    c for c in profile_validation_columns
    if c in change_summary13.columns
]

profiles14 = profiles14.merge(
    change_summary13[
        available_profile_validation_columns
    ],
    on="cluster_id",
    how="left",
)

profiles14.to_csv(
    TABLE_DIR
    / "14_final_cluster_profiles.csv",
    index=False,
)


# =============================================================================
# 14. FINAL REPRESENTATIVE ADM3 TABLE
# =============================================================================

representatives14 = representatives12.copy()

if "temporal_pattern" in representatives14.columns:
    representatives14 = representatives14.rename(
        columns={
            "temporal_pattern":
                "script12_preliminary_temporal_pattern"
        }
    )

representatives14["final_temporal_pattern"] = (
    representatives14["cluster_id"]
    .map(FINAL_LABEL_MAP)
)

if representative_summary13 is not None:
    representative_summary13[
        "cluster_id"
    ] = pd.to_numeric(
        representative_summary13[
            "cluster_id"
        ],
        errors="coerce",
    ).astype(int)

    rep_extra_cols = [
        c for c in [
            "cluster_id",
            "early_mean",
            "recent_mean",
            "change_pp",
            "previous_5y_mean",
            "recent_5y_mean",
            "recent_5y_change_pp",
            "flood_year_count",
            "maximum_flooded_percent",
        ]
        if c in representative_summary13.columns
    ]

    representatives14 = (
        representatives14
        .merge(
            representative_summary13[
                rep_extra_cols
            ],
            on="cluster_id",
            how="left",
            suffixes=("", "_script13"),
        )
    )

representatives14.to_csv(
    TABLE_DIR
    / "14_final_representative_adm3.csv",
    index=False,
)


# =============================================================================
# 15. FINAL CHANGE-POINT TABLE
# =============================================================================

change_points14 = change_points12.copy()

if "cluster_id" in change_points14.columns:
    change_points14["cluster_id"] = pd.to_numeric(
        change_points14["cluster_id"],
        errors="coerce",
    )

if "temporal_pattern" in change_points14.columns:
    change_points14 = change_points14.rename(
        columns={
            "temporal_pattern":
                "script12_preliminary_temporal_pattern"
        }
    )

change_points14["final_temporal_pattern"] = (
    change_points14["cluster_id"]
    .map(FINAL_LABEL_MAP)
)

change_points14.to_csv(
    TABLE_DIR
    / "14_final_candidate_change_points.csv",
    index=False,
)


# =============================================================================
# 16. METHODOLOGICAL SUMMARY TABLE
# =============================================================================

method_rows = [
    {
        "step": "12",
        "analytical_role": "Temporal pattern discovery",
        "operation": (
            "Coverage-aware temporal feature construction; median imputation "
            "in feature space; StandardScaler; Ward hierarchical clustering; "
            "silhouette diagnostics; PCA for interpretation only; preliminary "
            "post-hoc descriptive labels."
        ),
        "effect_on_cluster_membership": (
            "Creates the four-cluster solution for 490 eligible ADM3."
        ),
    },
    {
        "step": "13",
        "analytical_role": "Interpretation validation",
        "operation": (
            "Checks cluster feature distributions, individual ADM3 change "
            "directions, mean versus median behaviour, PCA loadings, "
            "centroid-nearest representatives and cluster temporal trajectories."
        ),
        "effect_on_cluster_membership": (
            "None. Validates interpretation only."
        ),
    },
    {
        "step": "14",
        "analytical_role": "Final temporal typology",
        "operation": (
            "Preserves Script 12 cluster membership and finalizes descriptive "
            "labels using Script 13 validation evidence."
        ),
        "effect_on_cluster_membership": (
            "None. Cluster 3 is relabelled from 'Relatively low / stable' "
            "to 'Moderate recent expansion'; membership is unchanged."
        ),
    },
]

method_summary = pd.DataFrame(method_rows)

method_summary.to_csv(
    TABLE_DIR
    / "14_final_pattern_methodological_summary.csv",
    index=False,
)


# =============================================================================
# 17. OPTIONAL FINAL TYPOLOGY FIGURE
# =============================================================================

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

if MATPLOTLIB_AVAILABLE:

    clustered_counts = (
        final_assignments[
            "final_temporal_pattern"
        ]
        .value_counts()
        .reindex(
            [
                FINAL_LABEL_MAP[1],
                FINAL_LABEL_MAP[2],
                FINAL_LABEL_MAP[3],
                FINAL_LABEL_MAP[4],
            ]
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        clustered_counts.index,
        clustered_counts.values,
    )

    ax.set_ylabel(
        "Number of ADM3 units"
    )

    ax.set_xlabel(
        "Final validated temporal pattern"
    )

    ax.set_title(
        "Final validated ADM3 flood temporal typology"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    for i, value in enumerate(
        clustered_counts.values
    ):
        ax.text(
            i,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "14_final_typology_summary.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 18. QC / VALIDATION SUMMARY
# =============================================================================

best_silhouette_row = (
    silhouette12
    .dropna(
        subset=["silhouette_score"]
    )
    .sort_values(
        "silhouette_score",
        ascending=False,
    )
    .iloc[0]
)

best_k = int(
    best_silhouette_row[
        "k_requested"
    ]
)

best_silhouette = float(
    best_silhouette_row[
        "silhouette_score"
    ]
)

pc1_pct = np.nan
pc2_pct = np.nan
pc12_pct = np.nan

if (
    "component" in pca_variance12.columns
    and "explained_variance_percent"
    in pca_variance12.columns
):
    pc1_rows = pca_variance12[
        pca_variance12["component"] == "PC1"
    ]
    pc2_rows = pca_variance12[
        pca_variance12["component"] == "PC2"
    ]

    if len(pc1_rows):
        pc1_pct = float(
            pc1_rows.iloc[0][
                "explained_variance_percent"
            ]
        )

    if len(pc2_rows):
        pc2_pct = float(
            pc2_rows.iloc[0][
                "explained_variance_percent"
            ]
        )

    if np.isfinite(pc1_pct) and np.isfinite(pc2_pct):
        pc12_pct = pc1_pct + pc2_pct


qc_lines = [
    "SCRIPT 14 — FINAL VALIDATED ADM3 TEMPORAL TYPOLOGY",
    "=" * 72,
    "",
    "Analytical lineage:",
    "  Script 12 = temporal pattern discovery / preliminary interpretation",
    "  Script 13 = interpretation validation",
    "  Script 14 = final validated temporal typology",
    "",
    f"Script 12 clustered ADM3: {len(assignments12)}",
    f"Final all-ADM3 classification rows: {len(classification14)}",
    f"Observed cluster IDs: {cluster_ids}",
    f"Best silhouette k from Script 12: {best_k}",
    f"Best silhouette score from Script 12: {best_silhouette:.6f}",
    f"PC1 explained variance: {pc1_pct:.2f}%",
    f"PC2 explained variance: {pc2_pct:.2f}%",
    f"PC1 + PC2: {pc12_pct:.2f}%",
    "",
    "FINAL VALIDATED LABELS:",
]

for _, row in label_audit.iterrows():
    qc_lines += [
        "",
        f"Cluster {int(row['cluster_id'])}",
        f"  ADM3 count: {int(row['adm3_count'])}",
        (
            "  Script 12 preliminary: "
            f"{row['script12_preliminary_label']}"
        ),
        (
            "  Script 13 diagnostic: "
            f"{row['script13_validation_interpretation']}"
        ),
        (
            "  Script 14 final: "
            f"{row['script14_final_label']}"
        ),
        (
            "  Label changed: "
            f"{row['label_changed']}"
        ),
        "  Cluster membership changed: False",
        (
            "  Increase share: "
            f"{row['increase_percent']:.1f}%"
        ),
        (
            "  Mean change: "
            f"{row['mean_change_pp']:.2f} pp"
        ),
        (
            "  Median change: "
            f"{row['median_change_pp']:.2f} pp"
        ),
    ]

qc_lines += [
    "",
    "INTERPRETATION RULES:",
    (
        "  1. Script 14 does not claim that validation created new clusters."
    ),
    (
        "  2. Cluster membership remains exactly as produced by Script 12."
    ),
    (
        "  3. Script 13 validation is used only to finalize semantic interpretation."
    ),
    (
        "  4. Cluster 3's preliminary 'Relatively low / stable' label is revised "
        "because the within-cluster evidence shows consistent positive temporal change."
    ),
    (
        "  5. 'Moderate recent expansion' describes temporal behaviour; it does "
        "not imply that the cluster is spatially widespread across South Sudan."
    ),
    (
        "  6. Non-clustered categories remain coverage-aware descriptive categories."
    ),
    (
        "  7. PCA remains interpretation/visualization only and does not determine "
        "cluster membership."
    ),
    "",
    "NEXT STEP:",
    (
        "  Script 15 can spatially map the final validated ADM3 temporal patterns."
    ),
]

with open(
    QC_DIR
    / "14_final_validation_qc_summary.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "\n".join(qc_lines)
    )


# =============================================================================
# 19. CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 96)
print("FINAL LABEL REVISION AUDIT")
print("=" * 96)

print(
    label_audit[
        [
            "cluster_id",
            "adm3_count",
            "script12_preliminary_label",
            "script13_validation_interpretation",
            "script14_final_label",
            "label_changed",
            "cluster_membership_changed",
        ]
    ].to_string(
        index=False
    )
)

print("\n" + "=" * 96)
print("FINAL 512-ADM3 CLASSIFICATION COUNTS")
print("=" * 96)

print(
    final_counts.to_string(
        index=False
    )
)

print("\n" + "=" * 96)
print("SCRIPT 14 COMPLETED")
print("=" * 96)

print(
    f"\nOutput root:\n{OUTPUT_ROOT}"
)

print(
    "\nIMPORTANT:"
    "\nScript 14 preserves Script 12 cluster membership."
    "\nOnly the validated descriptive interpretation is finalized."
)

print(
    "\nNext step:"
    "\nScript 15 — map the final validated ADM3 temporal patterns."
)
