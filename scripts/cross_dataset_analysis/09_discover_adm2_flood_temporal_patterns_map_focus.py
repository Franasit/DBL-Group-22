#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_discover_adm2_flood_temporal_patterns_finegrained.py

South Sudan Flood Analysis
JBG060 Capstone Data Challenge

PURPOSE
-------
Discover interpretable ADM2 flood TEMPORAL PATTERNS for 2000–2025.

This version differs from the earlier script in four important ways:

1. All ADM2 areas remain in the clustering.
   Rubkona is NOT removed. If it forms a singleton cluster, that is retained as
   a legitimate rare / extreme temporal pattern.

2. Clustering focuses mainly on TEMPORAL SHAPE and CHANGE rather than absolute
   flood severity. This is important because the research question is:
       "How does flooding change over time?"
   rather than only:
       "Which ADM2 has more flooded area?"

3. Fine-grained solutions are explored (k = 3..8 by default).
   The default selected solution is k = 6 because, for the current dataset,
   it gives a useful balance between detail and interpretability. You can
   change SELECTED_K after inspecting the validation outputs.

4. Cluster numbers are translated into descriptive pattern names AFTER
   clustering by inspecting each cluster's observed feature profile.
   The names are descriptive summaries, not externally validated classes.

MAIN OUTPUT IDEA
----------------
Example pattern types that may emerge:
    - Persistent low / stable
    - Gradual expansion
    - Recent acceleration
    - Recurrent high flooding
    - Episodic / high-volatility flooding
    - Extreme recent expansion

IMPORTANT
---------
This is temporal-pattern discovery, NOT a flood-risk ranking.
Risk should later combine:
    Hazard + Exposure + Vulnerability

INPUT
-----
processed_data/flood_analysis/adm2/
    adm2_annual_flood_indicators_combined.csv

Required columns:
    ADM2_ID
    ADM2_NAME
    year
    annual_flooded_percent

Optional:
    flood_type
If present, only flood_type == "combined" is used.

OUTPUT
------
processed_data/flood_analysis/adm2/pattern_discovery_finegrained/

tables/
    01_adm2_temporal_features.csv
    02_k_validation.csv
    03_pattern_membership.csv
    04_pattern_profiles.csv
    05_pattern_name_guide.csv
    06_change_points.csv
    07_pattern_change_point_summary.csv

figures/
    01_k_validation.png
    02_dendrogram_selected_k.png
    03_pca_patterns.png
    04_pattern_mean_trajectories.png
    05_pattern_small_multiples.png
    06_pattern_profile_heatmap.png
    07_change_point_distribution.png
    08_pattern_map.png
    09_selected_pattern_examples.png
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False


# =============================================================================
# 0. PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/shibai/Documents/GitHub/DBL-Group-22"
)

ANNUAL_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm2"
    / "adm2_annual_flood_indicators_combined.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm2"
    / "pattern_discovery_finegrained"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# If automatic boundary discovery still cannot locate the correct polygon file,
# set this manually to the ADM2 polygon layer.
BOUNDARY_FILE = None

STUDY_START_YEAR = 2000
STUDY_END_YEAR = 2025

EARLY_START = 2000
EARLY_END = 2007

RECENT_START = 2017
RECENT_END = 2025

RECENT_5Y_START = 2021
RECENT_5Y_END = 2025

PREVIOUS_5Y_START = 2016
PREVIOUS_5Y_END = 2020

FLOOD_YEAR_THRESHOLD = 1.0

# Explore fine-grained solutions.
K_MIN = 3
K_MAX = 8

# -------------------------------------------------------------------------
# IMPORTANT:
# For the current dataset, k=6 is a useful starting point:
# it is detailed enough to separate several temporal behaviours while still
# remaining interpretable. Rubkona may remain as a singleton pattern.
#
# This is intentionally editable. Inspect 02_k_validation.csv and the
# trajectory panels before deciding whether k=5, 6, or 7 is best for reporting.
# -------------------------------------------------------------------------
SELECTED_K = 6

# A small cluster is NOT rejected.
# It is simply flagged as a "rare pattern" for interpretation.
RARE_PATTERN_MAX_SIZE = 2

REFERENCE_ADM2 = [
    "Rubkona",
    "Mayom",
    "Twic East",
    "Luakpiny/Nasir",
]


# =============================================================================
# 1. HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


def safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def safe_std(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.std(ddof=0)) if len(values) else np.nan


def linear_slope(
    years: pd.Series,
    values: pd.Series,
) -> float:
    x = pd.to_numeric(years, errors="coerce")
    y = pd.to_numeric(values, errors="coerce")

    valid = x.notna() & y.notna()
    x = x[valid].to_numpy(dtype=float)
    y = y[valid].to_numpy(dtype=float)

    if len(x) < 2:
        return np.nan

    if np.allclose(y, y[0]):
        return 0.0

    return float(np.polyfit(x, y, 1)[0])


def longest_true_run(values: list[bool]) -> int:
    best = 0
    current = 0

    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def normalise_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
    )


def canonical_match_key(name: str) -> str:
    key = " ".join(normalise_name(name).split())

    aliases = {
        "luakpiny nasir": "luakpiny nasir",
        "nasir": "luakpiny nasir",
    }

    return aliases.get(key, key)


# =============================================================================
# 2. LOAD ANNUAL DATA
# =============================================================================

def load_annual_data() -> pd.DataFrame:
    print_header("LOAD ANNUAL ADM2 FLOOD DATA")

    require_file(ANNUAL_FILE)

    df = pd.read_csv(ANNUAL_FILE)

    required = {
        "ADM2_ID",
        "ADM2_NAME",
        "year",
        "annual_flooded_percent",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    df = df.copy()

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["annual_flooded_percent"] = pd.to_numeric(
        df["annual_flooded_percent"],
        errors="coerce",
    )

    if "flood_type" in df.columns:
        df = df[
            df["flood_type"]
            .astype(str)
            .str.lower()
            .eq("combined")
        ].copy()

    df = df[
        df["year"].between(
            STUDY_START_YEAR,
            STUDY_END_YEAR,
        )
    ].copy()

    df = df.dropna(
        subset=[
            "ADM2_NAME",
            "year",
            "annual_flooded_percent",
        ]
    )

    # If duplicate ADM2-year rows exist, average them.
    df = (
        df.groupby(
            ["ADM2_ID", "ADM2_NAME", "year"],
            as_index=False,
        )["annual_flooded_percent"]
        .mean()
    )

    df = df.sort_values(
        ["ADM2_NAME", "year"]
    ).reset_index(drop=True)

    print(f"Rows loaded: {len(df):,}")
    print(f"ADM2 areas: {df['ADM2_NAME'].nunique()}")
    print(
        f"Years: {int(df['year'].min())}"
        f"–{int(df['year'].max())}"
    )

    return df


# =============================================================================
# 3. TEMPORAL FEATURE ENGINEERING
# =============================================================================

def build_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build both:
      A. temporal-shape/change features used for clustering
      B. severity features used AFTER clustering for interpretation
    """

    print_header("TEMPORAL FEATURE ENGINEERING")

    records = []

    for adm2_name, subset in df.groupby("ADM2_NAME"):
        subset = subset.sort_values("year").copy()

        adm2_id = subset["ADM2_ID"].iloc[0]

        years = subset["year"]
        values = subset["annual_flooded_percent"]

        early = subset[
            subset["year"].between(
                EARLY_START,
                EARLY_END,
            )
        ]

        recent = subset[
            subset["year"].between(
                RECENT_START,
                RECENT_END,
            )
        ]

        recent_5y = subset[
            subset["year"].between(
                RECENT_5Y_START,
                RECENT_5Y_END,
            )
        ]

        previous_5y = subset[
            subset["year"].between(
                PREVIOUS_5Y_START,
                PREVIOUS_5Y_END,
            )
        ]

        mean_flood = safe_mean(values)
        median_flood = float(values.median())
        min_flood = float(values.min())
        max_flood = float(values.max())
        std_flood = safe_std(values)

        if (
            pd.notna(mean_flood)
            and not np.isclose(mean_flood, 0)
        ):
            cv_flood = std_flood / mean_flood
        else:
            cv_flood = np.nan

        peak_idx = values.idxmax()
        peak_year = int(
            df.loc[peak_idx, "year"]
        )

        early_mean = safe_mean(
            early["annual_flooded_percent"]
        )

        recent_mean = safe_mean(
            recent["annual_flooded_percent"]
        )

        change_pp = (
            recent_mean - early_mean
            if pd.notna(recent_mean)
            and pd.notna(early_mean)
            else np.nan
        )

        trend_slope = linear_slope(
            years,
            values,
        )

        recent_slope = linear_slope(
            recent["year"],
            recent["annual_flooded_percent"],
        )

        previous_slope = linear_slope(
            previous_5y["year"],
            previous_5y["annual_flooded_percent"],
        )

        acceleration = (
            recent_slope - previous_slope
            if pd.notna(recent_slope)
            and pd.notna(previous_slope)
            else np.nan
        )

        recent_5y_mean = safe_mean(
            recent_5y["annual_flooded_percent"]
        )

        previous_5y_mean = safe_mean(
            previous_5y["annual_flooded_percent"]
        )

        recent_5y_change_pp = (
            recent_5y_mean
            - previous_5y_mean
            if pd.notna(recent_5y_mean)
            and pd.notna(previous_5y_mean)
            else np.nan
        )

        max_minus_median = (
            max_flood - median_flood
        )

        flood_flags = (
            values >= FLOOD_YEAR_THRESHOLD
        ).tolist()

        flood_year_count = int(
            np.sum(flood_flags)
        )

        flood_year_fraction = (
            flood_year_count / len(subset)
            if len(subset)
            else np.nan
        )

        longest_consecutive_run = longest_true_run(
            flood_flags
        )

        post_peak = subset[
            subset["year"] > peak_year
        ]

        if len(post_peak):
            post_peak_mean = safe_mean(
                post_peak[
                    "annual_flooded_percent"
                ]
            )

            post_peak_decline = (
                max_flood - post_peak_mean
            )
        else:
            post_peak_mean = np.nan
            post_peak_decline = np.nan

        records.append(
            {
                "ADM2_ID": adm2_id,
                "ADM2_NAME": adm2_name,
                "n_years": len(subset),

                # Severity / level
                "mean_flood": mean_flood,
                "median_flood": median_flood,
                "min_flood": min_flood,
                "max_flood": max_flood,
                "recent_mean": recent_mean,
                "recent_5y_mean": recent_5y_mean,
                "previous_5y_mean": previous_5y_mean,

                # Variability
                "std_flood": std_flood,
                "cv_flood": cv_flood,
                "max_minus_median": max_minus_median,

                # Change / temporal shape
                "early_mean": early_mean,
                "change_pp": change_pp,
                "trend_slope": trend_slope,
                "recent_slope": recent_slope,
                "previous_slope": previous_slope,
                "acceleration": acceleration,
                "recent_5y_change_pp": recent_5y_change_pp,

                # Persistence
                "flood_year_count": flood_year_count,
                "flood_year_fraction": flood_year_fraction,
                "longest_consecutive_run": longest_consecutive_run,

                # Peak / recovery
                "peak_year": peak_year,
                "post_peak_mean": post_peak_mean,
                "post_peak_decline": post_peak_decline,
            }
        )

    features = pd.DataFrame(records)

    features.to_csv(
        TABLE_DIR
        / "01_adm2_temporal_features.csv",
        index=False,
    )

    print(
        f"Feature rows: {len(features)}"
    )

    return features


# =============================================================================
# 4. SHAPE-ORIENTED CLUSTERING FEATURES
# =============================================================================

def get_shape_feature_columns() -> list[str]:
    """
    These are deliberately focused on HOW flooding changes over time.

    Absolute level variables such as mean_flood and max_flood are NOT used
    directly to drive clustering. They are retained for interpretation later.

    Why:
    Two ADM2 areas can have different absolute flood levels but still share
    the same temporal pattern, e.g. both may be rapidly expanding.
    """

    return [
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


def prepare_shape_matrix(
    features: pd.DataFrame,
):
    shape_cols = get_shape_feature_columns()

    X = features[
        shape_cols
    ].copy()

    for col in shape_cols:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

        median_value = X[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        X[col] = X[col].fillna(
            median_value
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, shape_cols, scaler


# =============================================================================
# 5. TEST MULTIPLE k VALUES
# =============================================================================

def evaluate_k_values(
    X_scaled: np.ndarray,
) -> pd.DataFrame:
    print_header(
        "FINE-GRAINED CLUSTER VALIDATION"
    )

    records = []

    max_k = min(
        K_MAX,
        len(X_scaled) - 1,
    )

    for k in range(
        K_MIN,
        max_k + 1,
    ):
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels,
        )

        counts = (
            pd.Series(labels)
            .value_counts()
            .sort_values(
                ascending=False
            )
        )

        rare_count = int(
            (counts <= RARE_PATTERN_MAX_SIZE)
            .sum()
        )

        records.append(
            {
                "k": k,
                "silhouette_score": score,
                "largest_cluster": int(
                    counts.max()
                ),
                "smallest_cluster": int(
                    counts.min()
                ),
                "rare_pattern_clusters": rare_count,
                "cluster_sizes": " | ".join(
                    str(int(x))
                    for x in counts.tolist()
                ),
            }
        )

        print(
            f"k={k}: "
            f"silhouette={score:.4f}; "
            f"sizes={counts.tolist()}; "
            f"rare clusters={rare_count}"
        )

    validation = pd.DataFrame(
        records
    )

    validation.to_csv(
        TABLE_DIR
        / "02_k_validation.csv",
        index=False,
    )

    return validation


def plot_k_validation(
    validation: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.plot(
        validation["k"],
        validation["silhouette_score"],
        marker="o",
        linewidth=1.8,
    )

    selected = validation[
        validation["k"] == SELECTED_K
    ]

    if not selected.empty:
        row = selected.iloc[0]

        ax.scatter(
            [SELECTED_K],
            [row["silhouette_score"]],
            marker="*",
            s=180,
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            label=(
                f"Selected k={SELECTED_K}"
            ),
        )

    for _, row in validation.iterrows():
        ax.annotate(
            row["cluster_sizes"],
            (
                row["k"],
                row["silhouette_score"],
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    ax.set_title(
        "Fine-Grained Temporal Pattern Validation"
    )

    ax.set_xlabel(
        "Number of temporal patterns (k)"
    )

    ax.set_ylabel(
        "Silhouette score"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "01_k_validation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 6. FIT SELECTED FINE-GRAINED CLUSTERING
# =============================================================================

def fit_selected_clustering(
    features: pd.DataFrame,
    X_scaled: np.ndarray,
) -> pd.DataFrame:
    if not (
        K_MIN <= SELECTED_K <= K_MAX
    ):
        raise ValueError(
            f"SELECTED_K={SELECTED_K} is outside "
            f"the tested range {K_MIN}–{K_MAX}."
        )

    model = AgglomerativeClustering(
        n_clusters=SELECTED_K,
        linkage="ward",
    )

    raw_labels = model.fit_predict(
        X_scaled
    )

    # Renumber by cluster size for stable, human-friendly output.
    counts = (
        pd.Series(raw_labels)
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    relabel = {
        raw_label: i + 1
        for i, raw_label in enumerate(
            counts.index
        )
    }

    membership = features[
        [
            "ADM2_ID",
            "ADM2_NAME",
        ]
    ].copy()

    membership["cluster"] = [
        relabel[x]
        for x in raw_labels
    ]

    sizes = (
        membership["cluster"]
        .value_counts()
        .to_dict()
    )

    membership["cluster_size"] = (
        membership["cluster"]
        .map(sizes)
    )

    membership["is_rare_pattern"] = (
        membership["cluster_size"]
        <= RARE_PATTERN_MAX_SIZE
    ).astype(int)

    return membership


# =============================================================================
# 7. BUILD CLUSTER / PATTERN PROFILES
# =============================================================================

def build_pattern_profiles(
    features: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    merged = features.merge(
        membership[
            [
                "ADM2_NAME",
                "cluster",
                "cluster_size",
                "is_rare_pattern",
            ]
        ],
        on="ADM2_NAME",
        how="inner",
    )

    profile_cols = [
        "mean_flood",
        "max_flood",
        "std_flood",
        "cv_flood",
        "early_mean",
        "recent_mean",
        "change_pp",
        "trend_slope",
        "recent_slope",
        "acceleration",
        "recent_5y_mean",
        "previous_5y_mean",
        "recent_5y_change_pp",
        "max_minus_median",
        "flood_year_fraction",
        "longest_consecutive_run",
        "post_peak_decline",
    ]

    profiles = (
        merged.groupby(
            "cluster"
        )
        .agg(
            n_areas=(
                "ADM2_NAME",
                "count",
            ),
            **{
                col: (
                    col,
                    "mean",
                )
                for col in profile_cols
            },
        )
        .reset_index()
    )

    profiles[
        "is_rare_pattern"
    ] = (
        profiles["n_areas"]
        <= RARE_PATTERN_MAX_SIZE
    ).astype(int)

    return profiles


# =============================================================================
# 8. ASSIGN DESCRIPTIVE PATTERN NAMES AFTER CLUSTERING
# =============================================================================

def assign_descriptive_pattern_names(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Give each discovered cluster a readable descriptive name.

    Important:
    These names are assigned AFTER clustering.
    They do not determine cluster membership.

    The rules describe the observed cluster profile and are intentionally
    transparent. Review them together with the trajectory figures before using
    the names in a final report.
    """

    p = profiles.copy()

    # Rank-based helper values across the discovered clusters.
    def rank_pct(
        col: str,
    ) -> pd.Series:
        return p[col].rank(
            pct=True,
            method="average",
        )

    p["_mean_rank"] = rank_pct(
        "mean_flood"
    )

    p["_trend_rank"] = rank_pct(
        "trend_slope"
    )

    p["_recent_change_rank"] = rank_pct(
        "recent_5y_change_pp"
    )

    p["_acceleration_rank"] = rank_pct(
        "acceleration"
    )

    p["_recurrence_rank"] = rank_pct(
        "flood_year_fraction"
    )

    p["_run_rank"] = rank_pct(
        "longest_consecutive_run"
    )

    p["_volatility_rank"] = rank_pct(
        "cv_flood"
    )

    p["_spike_rank"] = rank_pct(
        "max_minus_median"
    )

    names = {}

    remaining = set(
        p["cluster"].astype(int)
    )

    # ---------------------------------------------------------------------
    # A. Extreme recent expansion
    # Rare/tiny cluster + strongest recent change.
    # This should naturally capture Rubkona if it remains unique.
    # ---------------------------------------------------------------------
    rare = p[
        p["is_rare_pattern"] == 1
    ]

    if not rare.empty:
        candidate = rare.sort_values(
            [
                "recent_5y_change_pp",
                "recent_slope",
            ],
            ascending=False,
        ).iloc[0]

        c = int(
            candidate["cluster"]
        )

        names[c] = (
            "Extreme recent expansion"
        )

        remaining.discard(c)

    # ---------------------------------------------------------------------
    # B. Persistent low / stable
    # Find the remaining cluster with the lowest long-run level and very weak
    # change.
    # ---------------------------------------------------------------------
    if remaining:
        candidate_pool = p[
            p["cluster"].isin(
                remaining
            )
        ].copy()

        candidate_pool[
            "_low_stable_score"
        ] = (
            candidate_pool[
                "_mean_rank"
            ]
            + candidate_pool[
                "_trend_rank"
            ]
            + candidate_pool[
                "_recent_change_rank"
            ]
        )

        candidate = candidate_pool.sort_values(
            "_low_stable_score"
        ).iloc[0]

        c = int(
            candidate["cluster"]
        )

        names[c] = (
            "Persistent low / stable"
        )

        remaining.discard(c)

    # ---------------------------------------------------------------------
    # C. Recent acceleration
    # Strong recent change + strong positive acceleration.
    # ---------------------------------------------------------------------
    if remaining:
        candidate_pool = p[
            p["cluster"].isin(
                remaining
            )
        ].copy()

        candidate_pool[
            "_recent_accel_score"
        ] = (
            candidate_pool[
                "_recent_change_rank"
            ]
            + candidate_pool[
                "_acceleration_rank"
            ]
            + candidate_pool[
                "_trend_rank"
            ]
        )

        candidate = candidate_pool.sort_values(
            "_recent_accel_score",
            ascending=False,
        ).iloc[0]

        c = int(
            candidate["cluster"]
        )

        names[c] = (
            "Recent acceleration / expansion"
        )

        remaining.discard(c)

    # ---------------------------------------------------------------------
    # D. Recurrent / persistent flooding
    # High flood-year frequency + long consecutive runs + higher mean level.
    # ---------------------------------------------------------------------
    if remaining:
        candidate_pool = p[
            p["cluster"].isin(
                remaining
            )
        ].copy()

        candidate_pool[
            "_recurrence_score"
        ] = (
            candidate_pool[
                "_recurrence_rank"
            ]
            + candidate_pool[
                "_run_rank"
            ]
            + candidate_pool[
                "_mean_rank"
            ]
        )

        candidate = candidate_pool.sort_values(
            "_recurrence_score",
            ascending=False,
        ).iloc[0]

        c = int(
            candidate["cluster"]
        )

        names[c] = (
            "Recurrent / persistent flooding"
        )

        remaining.discard(c)

    # ---------------------------------------------------------------------
    # E. Episodic / high-volatility flooding
    # Strong relative volatility and large spikes.
    # ---------------------------------------------------------------------
    if remaining:
        candidate_pool = p[
            p["cluster"].isin(
                remaining
            )
        ].copy()

        candidate_pool[
            "_volatility_score"
        ] = (
            candidate_pool[
                "_volatility_rank"
            ]
            + candidate_pool[
                "_spike_rank"
            ]
        )

        candidate = candidate_pool.sort_values(
            "_volatility_score",
            ascending=False,
        ).iloc[0]

        c = int(
            candidate["cluster"]
        )

        names[c] = (
            "Episodic / high-volatility flooding"
        )

        remaining.discard(c)

    # ---------------------------------------------------------------------
    # F. Remaining positive-change cluster(s)
    # Usually a more gradual expansion pattern.
    # ---------------------------------------------------------------------
    for c in sorted(
        remaining
    ):
        row = p[
            p["cluster"] == c
        ].iloc[0]

        if (
            row["trend_slope"] > 0
            and row[
                "recent_5y_change_pp"
            ] > 0
        ):
            names[c] = (
                "Gradual expansion / increasing"
            )
        elif (
            row["trend_slope"] < 0
            and row["change_pp"] < 0
        ):
            names[c] = (
                "Declining / recovery"
            )
        else:
            names[c] = (
                "Mixed / moderate temporal pattern"
            )

    p["pattern_name"] = (
        p["cluster"]
        .astype(int)
        .map(names)
    )

    # Make duplicate generic names distinct only if necessary.
    duplicate_counts = (
        p["pattern_name"]
        .value_counts()
    )

    for pattern_name, count in (
        duplicate_counts.items()
    ):
        if count <= 1:
            continue

        idxs = p.index[
            p["pattern_name"]
            == pattern_name
        ].tolist()

        for j, idx in enumerate(
            idxs,
            start=1,
        ):
            p.loc[
                idx,
                "pattern_name",
            ] = (
                f"{pattern_name} {j}"
            )

    drop_cols = [
        c
        for c in p.columns
        if c.startswith("_")
    ]

    p = p.drop(
        columns=drop_cols
    )

    return p


def attach_pattern_names(
    membership: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    names = profiles[
        [
            "cluster",
            "pattern_name",
            "n_areas",
            "is_rare_pattern",
        ]
    ].copy()

    result = membership.drop(
        columns=[
            "is_rare_pattern",
        ],
        errors="ignore",
    ).merge(
        names,
        on="cluster",
        how="left",
    )

    result = result.rename(
        columns={
            "n_areas":
            "pattern_size",
        }
    )

    return result


def save_pattern_name_guide(
    profiles: pd.DataFrame,
) -> None:
    guide = profiles[
        [
            "cluster",
            "pattern_name",
            "n_areas",
            "is_rare_pattern",
            "mean_flood",
            "max_flood",
            "flood_year_fraction",
            "trend_slope",
            "recent_slope",
            "acceleration",
            "recent_5y_change_pp",
            "cv_flood",
            "max_minus_median",
        ]
    ].copy()

    guide[
        "interpretation_note"
    ] = (
        "Name assigned after clustering from observed cluster profile; "
        "review trajectory before final reporting."
    )

    guide.to_csv(
        TABLE_DIR
        / "05_pattern_name_guide.csv",
        index=False,
    )


# =============================================================================
# 9. PCA
# =============================================================================

def plot_pca_patterns(
    X_scaled: np.ndarray,
    membership: pd.DataFrame,
) -> None:
    pca = PCA(
        n_components=2
    )

    pcs = pca.fit_transform(
        X_scaled
    )

    pca_df = membership.copy()

    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]

    explained = (
        pca.explained_variance_ratio_
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for cluster_id in sorted(
        pca_df["cluster"].unique()
    ):
        sub = pca_df[
            pca_df["cluster"]
            == cluster_id
        ]

        label = (
            sub["pattern_name"]
            .iloc[0]
        )

        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=55,
            alpha=0.8,
            label=(
                f"{label} (n={len(sub)})"
            ),
        )

    for _, row in pca_df.iterrows():
        if (
            row["ADM2_NAME"]
            in REFERENCE_ADM2
            or row[
                "is_rare_pattern"
            ] == 1
        ):
            ax.annotate(
                row["ADM2_NAME"],
                (
                    row["PC1"],
                    row["PC2"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(
        "Fine-Grained ADM2 Flood Temporal Patterns"
    )

    ax.set_xlabel(
        f"PC1 ({explained[0]:.1f}% variance)"
    )

    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}% variance)"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        fontsize=7,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "03_pca_patterns.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 10. DENDROGRAM
# =============================================================================

def plot_selected_dendrogram(
    X_scaled: np.ndarray,
    features: pd.DataFrame,
) -> None:
    Z = linkage(
        X_scaled,
        method="ward",
    )

    fig, ax = plt.subplots(
        figsize=(16, 7.5)
    )

    dendrogram(
        Z,
        labels=features[
            "ADM2_NAME"
        ].tolist(),
        leaf_rotation=90,
        leaf_font_size=6,
        ax=ax,
    )

    ax.set_title(
        "Hierarchical Structure of ADM2 Temporal Flood Patterns"
    )

    ax.set_ylabel(
        "Ward distance"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "02_dendrogram_selected_k.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 11. PATTERN TRAJECTORIES
# =============================================================================

def build_pattern_year_means(
    df: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    merged = df.merge(
        membership[
            [
                "ADM2_NAME",
                "cluster",
                "pattern_name",
            ]
        ],
        on="ADM2_NAME",
        how="inner",
    )

    pattern_year = (
        merged.groupby(
            [
                "cluster",
                "pattern_name",
                "year",
            ],
            as_index=False,
        )["annual_flooded_percent"]
        .mean()
    )

    return pattern_year


def plot_pattern_mean_trajectories(
    df: pd.DataFrame,
    membership: pd.DataFrame,
) -> None:
    pattern_year = build_pattern_year_means(
        df,
        membership,
    )

    sizes = (
        membership.groupby(
            [
                "cluster",
                "pattern_name",
            ]
        )
        .size()
        .to_dict()
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for (
        cluster_id,
        pattern_name,
    ), sub in pattern_year.groupby(
        [
            "cluster",
            "pattern_name",
        ]
    ):
        n = sizes[
            (
                cluster_id,
                pattern_name,
            )
        ]

        ax.plot(
            sub["year"],
            sub["annual_flooded_percent"],
            marker="o",
            markersize=3,
            linewidth=1.9,
            label=(
                f"{pattern_name} (n={n})"
            ),
        )

    ax.set_title(
        "Mean Annual Flooded Area by Temporal Pattern"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Mean annual flooded area (%)"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        fontsize=7,
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "04_pattern_mean_trajectories.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_pattern_small_multiples(
    df: pd.DataFrame,
    membership: pd.DataFrame,
) -> None:
    """
    One panel per discovered pattern.

    Thin lines = individual ADM2 trajectories.
    Thick line = cluster/pattern mean.

    This is the most important interpretation figure because it shows whether
    a cluster truly has a coherent temporal shape.
    """

    merged = df.merge(
        membership[
            [
                "ADM2_NAME",
                "cluster",
                "pattern_name",
                "pattern_size",
            ]
        ],
        on="ADM2_NAME",
        how="inner",
    )

    pattern_info = (
        membership[
            [
                "cluster",
                "pattern_name",
                "pattern_size",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "cluster"
        )
    )

    n_patterns = len(
        pattern_info
    )

    ncols = 2
    nrows = math.ceil(
        n_patterns / ncols
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            14,
            4.4 * nrows,
        ),
        sharex=True,
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for ax, (_, info) in zip(
        axes,
        pattern_info.iterrows(),
    ):
        cluster_id = int(
            info["cluster"]
        )

        pattern_name = (
            info["pattern_name"]
        )

        sub = merged[
            merged["cluster"]
            == cluster_id
        ].copy()

        # Individual ADM2 lines.
        for _, area in sub.groupby(
            "ADM2_NAME"
        ):
            area = area.sort_values(
                "year"
            )

            ax.plot(
                area["year"],
                area[
                    "annual_flooded_percent"
                ],
                linewidth=0.8,
                alpha=0.20,
            )

        # Pattern mean.
        mean_line = (
            sub.groupby(
                "year",
                as_index=False,
            )[
                "annual_flooded_percent"
            ]
            .mean()
        )

        ax.plot(
            mean_line["year"],
            mean_line[
                "annual_flooded_percent"
            ],
            linewidth=2.5,
            marker="o",
            markersize=3,
            label="Pattern mean",
        )

        # Label singleton / rare areas.
        if (
            int(
                info["pattern_size"]
            )
            <= RARE_PATTERN_MAX_SIZE
        ):
            names = ", ".join(
                sorted(
                    sub[
                        "ADM2_NAME"
                    ].unique()
                )
            )

            ax.text(
                0.03,
                0.91,
                f"Rare pattern: {names}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )

        ax.set_title(
            (
                f"{pattern_name}\n"
                f"n={int(info['pattern_size'])}"
            ),
            fontsize=11,
        )

        ax.set_ylabel(
            "Flooded area (%)"
        )

        ax.grid(
            alpha=0.2
        )

    for ax in axes[
        n_patterns:
    ]:
        ax.axis("off")

    for ax in axes:
        if ax.has_data():
            ax.set_xlabel(
                "Year"
            )

    fig.suptitle(
        "Temporal Shape Within Each Discovered Flood Pattern",
        fontsize=15,
        y=1.01,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "05_pattern_small_multiples.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 12. PROFILE HEATMAP
# =============================================================================

def plot_pattern_profile_heatmap(
    profiles: pd.DataFrame,
) -> None:
    cols = [
        "mean_flood",
        "flood_year_fraction",
        "trend_slope",
        "recent_slope",
        "acceleration",
        "recent_5y_change_pp",
        "cv_flood",
        "max_minus_median",
    ]

    heat = profiles.set_index(
        "pattern_name"
    )[cols].copy()

    for col in cols:
        std = heat[col].std(
            ddof=0
        )

        if (
            pd.isna(std)
            or np.isclose(
                std,
                0,
            )
        ):
            heat[col] = 0.0
        else:
            heat[col] = (
                heat[col]
                - heat[col].mean()
            ) / std

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    image = ax.imshow(
        heat.values,
        aspect="auto",
        cmap="coolwarm",
    )

    ax.set_xticks(
        np.arange(
            len(cols)
        )
    )

    ax.set_xticklabels(
        cols,
        rotation=35,
        ha="right",
        fontsize=8,
    )

    ax.set_yticks(
        np.arange(
            len(heat.index)
        )
    )

    ax.set_yticklabels(
        heat.index,
        fontsize=8,
    )

    ax.set_title(
        "How the Discovered Temporal Patterns Differ"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Standardised cluster mean",
        shrink=0.8,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "06_pattern_profile_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 13. CHANGE-POINT DETECTION
# =============================================================================

def detect_change_point_fallback(
    subset: pd.DataFrame,
) -> tuple[float, float]:
    subset = subset.sort_values(
        "year"
    )

    years = subset[
        "year"
    ].to_numpy()

    values = subset[
        "annual_flooded_percent"
    ].to_numpy(
        dtype=float
    )

    if len(values) < 8:
        return np.nan, np.nan

    best_year = np.nan
    best_shift = np.nan
    best_abs_shift = -np.inf

    for split in range(
        4,
        len(values) - 4 + 1,
    ):
        left = values[
            :split
        ]

        right = values[
            split:
        ]

        shift = (
            np.nanmean(right)
            - np.nanmean(left)
        )

        if abs(
            shift
        ) > best_abs_shift:
            best_abs_shift = abs(
                shift
            )

            best_shift = shift

            if split < len(
                years
            ):
                best_year = int(
                    years[split]
                )

    return (
        best_year,
        best_shift,
    )


def detect_change_point_ruptures(
    subset: pd.DataFrame,
) -> tuple[float, float]:
    subset = subset.sort_values(
        "year"
    )

    years = subset[
        "year"
    ].to_numpy()

    values = subset[
        "annual_flooded_percent"
    ].to_numpy(
        dtype=float
    )

    if len(values) < 8:
        return np.nan, np.nan

    signal = values.reshape(
        -1,
        1,
    )

    model = rpt.Binseg(
        model="l2"
    ).fit(
        signal
    )

    bkps = model.predict(
        n_bkps=1
    )

    if not bkps:
        return np.nan, np.nan

    split = bkps[0]

    if split >= len(
        values
    ):
        return np.nan, np.nan

    change_year = int(
        years[split]
    )

    before = values[
        :split
    ]

    after = values[
        split:
    ]

    shift = (
        np.nanmean(after)
        - np.nanmean(before)
    )

    return (
        change_year,
        shift,
    )


def detect_change_points(
    df: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    print_header(
        "CHANGE-POINT DETECTION"
    )

    method = (
        "ruptures_binseg"
        if HAS_RUPTURES
        else "fallback_mean_shift"
    )

    records = []

    for adm2_name, subset in df.groupby(
        "ADM2_NAME"
    ):
        adm2_id = subset[
            "ADM2_ID"
        ].iloc[0]

        if HAS_RUPTURES:
            year, shift = (
                detect_change_point_ruptures(
                    subset
                )
            )
        else:
            year, shift = (
                detect_change_point_fallback(
                    subset
                )
            )

        if pd.isna(
            shift
        ):
            direction = "none"
        elif shift > 0:
            direction = "increase"
        elif shift < 0:
            direction = "decrease"
        else:
            direction = "none"

        records.append(
            {
                "ADM2_ID": adm2_id,
                "ADM2_NAME": adm2_name,
                "change_point_year": year,
                "mean_shift_pp": shift,
                "change_direction": direction,
                "change_point_method": method,
            }
        )

    cp = pd.DataFrame(
        records
    )

    cp = cp.merge(
        membership[
            [
                "ADM2_NAME",
                "cluster",
                "pattern_name",
            ]
        ],
        on="ADM2_NAME",
        how="left",
    )

    cp.to_csv(
        TABLE_DIR
        / "06_change_points.csv",
        index=False,
    )

    return cp


def build_pattern_change_point_summary(
    change_points: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        change_points.dropna(
            subset=[
                "change_point_year",
            ]
        )
        .groupby(
            [
                "pattern_name",
                "change_point_year",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                "n_adm2",
            }
        )
    )

    summary.to_csv(
        TABLE_DIR
        / "07_pattern_change_point_summary.csv",
        index=False,
    )

    return summary


def plot_change_point_distribution(
    change_points: pd.DataFrame,
) -> None:
    counts = (
        change_points[
            "change_point_year"
        ]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    if counts.empty:
        return

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.bar(
        counts.index.astype(
            str
        ),
        counts.values,
    )

    for bar, value in zip(
        bars,
        counts.values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.3,
            str(
                int(value)
            ),
            ha="center",
            fontsize=9,
        )

    ax.set_title(
        "When Did ADM2 Flood Trajectories Change Most?"
    )

    ax.set_xlabel(
        "Detected change-point year"
    )

    ax.set_ylabel(
        "Number of ADM2 areas"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "07_change_point_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 14. SELECTED EXAMPLES
# =============================================================================

def plot_selected_pattern_examples(
    df: pd.DataFrame,
    membership: pd.DataFrame,
    change_points: pd.DataFrame,
) -> None:
    """
    Select one representative ADM2 per pattern:
    the ADM2 closest to that pattern's mean trajectory.
    """

    merged = df.merge(
        membership[
            [
                "ADM2_NAME",
                "cluster",
                "pattern_name",
            ]
        ],
        on="ADM2_NAME",
        how="inner",
    )

    examples = []

    for (
        cluster_id,
        pattern_name,
    ), sub in merged.groupby(
        [
            "cluster",
            "pattern_name",
        ]
    ):
        pivot = sub.pivot_table(
            index="year",
            columns="ADM2_NAME",
            values="annual_flooded_percent",
        )

        mean_series = pivot.mean(
            axis=1
        )

        distances = {}

        for name in pivot.columns:
            diff = (
                pivot[name]
                - mean_series
            )

            distances[name] = (
                np.nanmean(
                    diff ** 2
                )
            )

        representative = min(
            distances,
            key=distances.get,
        )

        examples.append(
            {
                "cluster": cluster_id,
                "pattern_name": pattern_name,
                "ADM2_NAME": representative,
            }
        )

    examples = pd.DataFrame(
        examples
    ).sort_values(
        "cluster"
    )

    n = len(
        examples
    )

    ncols = 2
    nrows = math.ceil(
        n / ncols
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            14,
            4.3 * nrows,
        ),
        sharex=True,
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for ax, (_, row) in zip(
        axes,
        examples.iterrows(),
    ):
        name = row[
            "ADM2_NAME"
        ]

        sub = (
            df[
                df["ADM2_NAME"]
                == name
            ]
            .sort_values(
                "year"
            )
        )

        ax.plot(
            sub["year"],
            sub[
                "annual_flooded_percent"
            ],
            marker="o",
            linewidth=1.8,
        )

        cp_row = change_points[
            change_points[
                "ADM2_NAME"
            ] == name
        ]

        if not cp_row.empty:
            cp_year = cp_row[
                "change_point_year"
            ].iloc[0]

            shift = cp_row[
                "mean_shift_pp"
            ].iloc[0]

            if pd.notna(
                cp_year
            ):
                ax.axvline(
                    cp_year,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.7,
                )

                ax.text(
                    0.03,
                    0.92,
                    (
                        f"Change point: "
                        f"{int(cp_year)}\n"
                        f"Mean shift: "
                        f"{shift:+.2f} pp"
                    ),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )

        ax.set_title(
            (
                f"{row['pattern_name']}\n"
                f"Representative: {name}"
            ),
            fontsize=10,
        )

        ax.set_ylabel(
            "Flooded area (%)"
        )

        ax.grid(
            alpha=0.2
        )

    for ax in axes[
        n:
    ]:
        ax.axis(
            "off"
        )

    for ax in axes:
        if ax.has_data():
            ax.set_xlabel(
                "Year"
            )

    fig.suptitle(
        "Representative ADM2 Example for Each Temporal Pattern",
        fontsize=15,
        y=1.01,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "09_selected_pattern_examples.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 15. ADM2 POLYGON BOUNDARY
# =============================================================================

def guess_adm2_name_column(
    gdf,
) -> str:
    candidates = [
        "ADM2_NAME",
        "adm2_name",
        "ADM2_EN",
        "admin2Name",
        "ADMIN2NAME",
        "County_EN",
        "COUNTYNAME",
        "county",
        "NAME_2",
        "shapeName",
        "name",
    ]

    for col in candidates:
        if col in gdf.columns:
            return col

    raise ValueError(
        "Could not identify ADM2 name column.\n"
        f"Available columns:\n{list(gdf.columns)}"
    )


def is_polygon_layer(
    gdf,
) -> bool:
    if gdf.empty:
        return False

    geom_types = set(
        gdf.geometry.geom_type.dropna()
    )

    allowed = {
        "Polygon",
        "MultiPolygon",
    }

    return bool(
        geom_types
    ) and geom_types.issubset(
        allowed
    )


def find_adm2_boundary() -> Path:
    if gpd is None:
        raise ImportError(
            "geopandas is not installed."
        )

    if BOUNDARY_FILE is not None:
        path = Path(
            BOUNDARY_FILE
        )

        if not path.exists():
            raise FileNotFoundError(
                f"BOUNDARY_FILE does not exist:\n{path}"
            )

        test = gpd.read_file(
            path
        )

        if not is_polygon_layer(
            test
        ):
            raise ValueError(
                "BOUNDARY_FILE is not a Polygon/MultiPolygon layer."
            )

        return path

    roots = [
        PROJECT_ROOT
        / "raw_data",
        PROJECT_ROOT
        / "processed_data",
    ]

    candidates = []

    for root in roots:
        if not root.exists():
            continue

        for pattern in [
            "*.shp",
            "*.gpkg",
            "*.geojson",
        ]:
            candidates.extend(
                root.rglob(
                    pattern
                )
            )

    if not candidates:
        raise FileNotFoundError(
            "No spatial boundary files found."
        )

    def score(
        path: Path,
    ) -> int:
        name = str(
            path
        ).lower()

        value = 0

        if "adm2" in name:
            value += 10

        if "county" in name:
            value += 7

        if "boundary" in name:
            value += 5

        if "ssd" in name:
            value += 2

        if "south" in name:
            value += 2

        if "sudan" in name:
            value += 2

        if "point" in name:
            value -= 30

        return value

    candidates = sorted(
        candidates,
        key=score,
        reverse=True,
    )

    checked = []

    for path in candidates:
        try:
            test = gpd.read_file(
                path
            )

            if test.empty:
                continue

            if not is_polygon_layer(
                test
            ):
                checked.append(
                    (
                        str(path),
                        "not polygon geometry",
                    )
                )
                continue

            name_col = guess_adm2_name_column(
                test
            )

            n_names = (
                test[name_col]
                .astype(str)
                .nunique()
            )

            if (
                60
                <= n_names
                <= 120
            ):
                print(
                    f"Selected ADM2 polygon boundary:\n{path}"
                )

                return path

            checked.append(
                (
                    str(path),
                    (
                        f"{n_names} unique names; "
                        "not likely ADM2"
                    ),
                )
            )

        except Exception as exc:
            checked.append(
                (
                    str(path),
                    str(exc),
                )
            )

    msg = (
        "Could not identify a suitable ADM2 polygon layer.\n"
        "Please set BOUNDARY_FILE manually."
    )

    if checked:
        msg += "\n\nSome checked files:\n"

        for path, reason in checked[
            :10
        ]:
            msg += (
                f"- {path}\n"
                f"  {reason}\n"
            )

    raise FileNotFoundError(
        msg
    )


def load_adm2_boundary():
    path = find_adm2_boundary()

    gdf = gpd.read_file(
        path
    ).copy()

    name_col = guess_adm2_name_column(
        gdf
    )

    if not is_polygon_layer(
        gdf
    ):
        raise ValueError(
            "Selected boundary is not polygon geometry."
        )

    gdf[
        "_match_key"
    ] = gdf[
        name_col
    ].map(
        canonical_match_key
    )

    # Merge multipart pieces belonging to the same ADM2 name.
    gdf = gdf.dissolve(
        by="_match_key",
        as_index=False,
        aggfunc="first",
    )

    print(
        f"ADM2 polygon count after dissolve: {len(gdf)}"
    )

    return (
        gdf,
        name_col,
    )


# =============================================================================
# 16. PATTERN MAP
# =============================================================================

def plot_pattern_map(
    adm2_gdf,
    name_col: str,
    membership: pd.DataFrame,
) -> None:
    gdf = adm2_gdf.copy()

    m = membership.copy()

    m[
        "_match_key"
    ] = m[
        "ADM2_NAME"
    ].map(
        canonical_match_key
    )

    mapped = gdf.merge(
        m[
            [
                "_match_key",
                "ADM2_NAME",
                "cluster",
                "pattern_name",
                "pattern_size",
                "is_rare_pattern",
            ]
        ],
        on="_match_key",
        how="left",
    )

    matched = (
        mapped["cluster"]
        .notna()
        .sum()
    )

    print(
        f"Pattern map matched polygons: "
        f"{matched}/{len(mapped)}"
    )

    fig, ax = plt.subplots(
        figsize=(11, 12)
    )

    mapped[
        mapped["cluster"]
        .isna()
    ].plot(
        ax=ax,
        color="lightgray",
        edgecolor="white",
        linewidth=0.45,
    )

    main = mapped[
        mapped["cluster"]
        .notna()
    ].copy()

    main[
        "cluster"
    ] = main[
        "cluster"
    ].astype(
        int
    )

    main.plot(
        ax=ax,
        column="cluster",
        categorical=True,
        cmap="tab10",
        legend=True,
        edgecolor="white",
        linewidth=0.6,
        legend_kwds={
            "title":
            "Temporal pattern cluster",
            "loc":
            "lower left",
        },
    )

    # Outline rare/singleton patterns rather than excluding them.
    rare = main[
        main[
            "is_rare_pattern"
        ] == 1
    ]

    if not rare.empty:
        rare.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=2.4,
            hatch="///",
        )

        for _, row in rare.iterrows():
            p = row.geometry.representative_point()

            ax.text(
                p.x,
                p.y,
                (
                    f"{row['ADM2_NAME']}\n"
                    f"{row['pattern_name']}"
                ),
                fontsize=7,
                ha="center",
                va="center",
                bbox={
                    "facecolor":
                    "white",
                    "edgecolor":
                    "black",
                    "linewidth":
                    0.4,
                    "alpha":
                    0.9,
                    "pad":
                    1.1,
                },
            )

    ax.set_title(
        "South Sudan ADM2 Fine-Grained Flood Temporal Patterns\n"
        f"{STUDY_START_YEAR}–{STUDY_END_YEAR}"
    )

    ax.text(
        0.01,
        0.01,
        (
            "Colours represent temporal pattern classes, not flood-risk ranks. "
            "Hatching marks rare/singleton patterns."
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "08_pattern_map.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 17. SAVE FINAL TABLES
# =============================================================================

def save_final_tables(
    membership: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    membership = membership.sort_values(
        [
            "cluster",
            "ADM2_NAME",
        ]
    )

    membership.to_csv(
        TABLE_DIR
        / "03_pattern_membership.csv",
        index=False,
    )

    profiles = profiles.sort_values(
        "cluster"
    )

    profiles.to_csv(
        TABLE_DIR
        / "04_pattern_profiles.csv",
        index=False,
    )

    save_pattern_name_guide(
        profiles
    )


# =============================================================================
# 18. PRINT HUMAN-READABLE SUMMARY
# =============================================================================

def print_pattern_summary(
    membership: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    print_header(
        "DISCOVERED TEMPORAL PATTERNS"
    )

    for _, row in profiles.sort_values(
        "cluster"
    ).iterrows():
        cluster_id = int(
            row["cluster"]
        )

        members = (
            membership[
                membership["cluster"]
                == cluster_id
            ]["ADM2_NAME"]
            .sort_values()
            .tolist()
        )

        print(
            f"\nPattern {cluster_id}: "
            f"{row['pattern_name']}"
        )

        print(
            f"  ADM2 areas: {int(row['n_areas'])}"
        )

        if int(
            row["is_rare_pattern"]
        ) == 1:
            print(
                "  Rare pattern: YES"
            )

        print(
            f"  Mean flood: "
            f"{row['mean_flood']:.3f}%"
        )

        print(
            f"  Long-term slope: "
            f"{row['trend_slope']:+.3f} pp/year"
        )

        print(
            f"  Recent slope: "
            f"{row['recent_slope']:+.3f} pp/year"
        )

        print(
            f"  Recent 5y change: "
            f"{row['recent_5y_change_pp']:+.3f} pp"
        )

        print(
            f"  Flood-year fraction: "
            f"{row['flood_year_fraction']:.3f}"
        )

        print(
            "  Members: "
            + ", ".join(
                members
            )
        )



# =============================================================================
# 18A. PRESENTATION-READY TRAJECTORY FIGURES
# =============================================================================

def plot_common_pattern_trajectories(
    df: pd.DataFrame,
    membership: pd.DataFrame,
) -> None:
    """
    Plot only non-rare/common temporal patterns.

    This avoids allowing the Rubkona singleton/extreme trajectory to stretch
    the y-axis and visually compress all other patterns.
    """
    common_membership = membership[
        membership["is_rare_pattern"] == 0
    ].copy()

    if common_membership.empty:
        return

    pattern_year = build_pattern_year_means(
        df,
        common_membership,
    )

    sizes = (
        common_membership.groupby(
            ["cluster", "pattern_name"]
        )
        .size()
        .to_dict()
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for (
        cluster_id,
        pattern_name,
    ), sub in pattern_year.groupby(
        ["cluster", "pattern_name"]
    ):
        n = sizes[
            (
                cluster_id,
                pattern_name,
            )
        ]

        ax.plot(
            sub["year"],
            sub["annual_flooded_percent"],
            marker="o",
            markersize=3,
            linewidth=2.0,
            label=f"{pattern_name} (n={n})",
        )

    ax.set_title(
        "Common ADM2 Flood Temporal Patterns"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean annual flooded area (%)")
    ax.grid(alpha=0.2)
    ax.legend(
        fontsize=8,
        loc="upper left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "04a_common_pattern_trajectories.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_rare_pattern_trajectories(
    df: pd.DataFrame,
    membership: pd.DataFrame,
) -> None:
    """
    Plot rare/singleton temporal patterns separately.

    For the current dataset this should make Rubkona easy to interpret without
    distorting the common-pattern comparison.
    """
    rare_membership = membership[
        membership["is_rare_pattern"] == 1
    ].copy()

    if rare_membership.empty:
        return

    merged = df.merge(
        rare_membership[
            [
                "ADM2_NAME",
                "cluster",
                "pattern_name",
            ]
        ],
        on="ADM2_NAME",
        how="inner",
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 6)
    )

    for (
        pattern_name,
        adm2_name,
    ), sub in merged.groupby(
        [
            "pattern_name",
            "ADM2_NAME",
        ]
    ):
        sub = sub.sort_values("year")

        ax.plot(
            sub["year"],
            sub["annual_flooded_percent"],
            marker="o",
            markersize=4,
            linewidth=2.3,
            label=f"{adm2_name} — {pattern_name}",
        )

    ax.set_title(
        "Rare / Extreme ADM2 Temporal Pattern"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual flooded area (%)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "04b_rare_extreme_pattern_trajectory.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 18B. ADM2 PATTERN POINT MAP
# =============================================================================

def plot_pattern_point_map(
    adm2_gdf,
    name_col: str,
    membership: pd.DataFrame,
) -> None:
    """
    Presentation-ready map requested for interpretation.

    Design:
      - South Sudan ADM2 polygon boundaries in the background.
      - One coloured point per ADM2, placed at a representative point.
      - Point colour identifies the discovered temporal pattern.
      - Small / high-interest pattern groups are labelled directly.
      - A side panel lists pattern counts and the ADM2 names for smaller groups.

    The map is intended to answer:
      "WHERE are the different temporal patterns located?"
    """

    gdf = adm2_gdf.copy()

    m = membership.copy()
    m["_match_key"] = (
        m["ADM2_NAME"]
        .map(canonical_match_key)
    )

    mapped = gdf.merge(
        m[
            [
                "_match_key",
                "ADM2_NAME",
                "cluster",
                "pattern_name",
                "pattern_size",
                "is_rare_pattern",
            ]
        ],
        on="_match_key",
        how="left",
    )

    matched = int(
        mapped["cluster"]
        .notna()
        .sum()
    )

    print(
        f"Pattern point map matched polygons: "
        f"{matched}/{len(mapped)}"
    )

    if matched == 0:
        raise ValueError(
            "No ADM2 polygon names matched the pattern-membership table."
        )

    # Representative points stay inside polygons, unlike ordinary centroids.
    mapped = mapped.copy()
    mapped["map_point"] = (
        mapped.geometry
        .representative_point()
    )

    point_gdf = gpd.GeoDataFrame(
        mapped.drop(
            columns="geometry"
        ),
        geometry="map_point",
        crs=mapped.crs,
    )

    # A stable colour per cluster. We use matplotlib's categorical palette
    # rather than manually hard-coding scientific meaning into colours.
    cluster_ids = sorted(
        membership[
            "cluster"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    cmap = plt.get_cmap(
        "tab10"
    )

    cluster_colours = {
        cluster_id:
        cmap(
            (i % 10)
            / 10
        )
        for i, cluster_id in enumerate(
            cluster_ids
        )
    }

    # Layout: map left, text summary right.
    fig = plt.figure(
        figsize=(16, 11)
    )

    ax = fig.add_axes(
        [
            0.04,
            0.06,
            0.66,
            0.88,
        ]
    )

    info_ax = fig.add_axes(
        [
            0.72,
            0.08,
            0.26,
            0.84,
        ]
    )

    # Background ADM2 polygons.
    mapped.plot(
        ax=ax,
        color="whitesmoke",
        edgecolor="lightgray",
        linewidth=0.6,
    )

    # Draw one point group at a time.
    pattern_rows = (
        membership[
            [
                "cluster",
                "pattern_name",
                "pattern_size",
                "is_rare_pattern",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "cluster"
        )
    )

    for _, pinfo in pattern_rows.iterrows():
        cluster_id = int(
            pinfo["cluster"]
        )
        pattern_name = str(
            pinfo["pattern_name"]
        )

        sub = point_gdf[
            point_gdf[
                "cluster"
            ] == cluster_id
        ]

        if sub.empty:
            continue

        is_rare = int(
            pinfo[
                "is_rare_pattern"
            ]
        ) == 1

        point_size = (
            150
            if is_rare
            else 65
        )

        marker = (
            "*"
            if is_rare
            else "o"
        )

        ax.scatter(
            sub.geometry.x,
            sub.geometry.y,
            s=point_size,
            marker=marker,
            color=cluster_colours[
                cluster_id
            ],
            edgecolor="black",
            linewidth=(
                1.0
                if is_rare
                else 0.45
            ),
            alpha=0.92,
            label=(
                f"{pattern_name} "
                f"(n={int(pinfo['pattern_size'])})"
            ),
            zorder=5,
        )

    # ------------------------------------------------------------------
    # Labels:
    # Label all ADM2 names for small/high-interest groups.
    # Do NOT label the 43 stable-low areas because that would make the map
    # unreadable. The full names remain in the CSV.
    # ------------------------------------------------------------------
    label_threshold = 6

    label_df = point_gdf[
        (
            point_gdf[
                "pattern_size"
            ]
            <= label_threshold
        )
        | (
            point_gdf[
                "is_rare_pattern"
            ]
            == 1
        )
    ].copy()

    # Small deterministic offsets reduce exact text overlap.
    offsets = [
        (5, 5),
        (5, -10),
        (-5, 5),
        (-5, -10),
        (8, 0),
        (-8, 0),
    ]

    for idx, (_, row) in enumerate(
        label_df.sort_values(
            [
                "cluster",
                "ADM2_NAME",
            ]
        ).iterrows()
    ):
        dx, dy = offsets[
            idx
            % len(offsets)
        ]

        ax.annotate(
            row["ADM2_NAME"],
            (
                row.geometry.x,
                row.geometry.y,
            ),
            xytext=(
                dx,
                dy,
            ),
            textcoords="offset points",
            fontsize=7.2,
            fontweight=(
                "bold"
                if row[
                    "is_rare_pattern"
                ] == 1
                else "normal"
            ),
            bbox={
                "facecolor":
                "white",
                "edgecolor":
                "none",
                "alpha":
                0.72,
                "pad":
                0.6,
            },
            zorder=6,
        )

    ax.set_title(
        "South Sudan ADM2 Flood Temporal Patterns\n"
        "Coloured points show the pattern assigned to each ADM2",
        fontsize=15,
        pad=12,
    )

    ax.set_axis_off()

    ax.legend(
        fontsize=7.5,
        loc="lower left",
        frameon=True,
        title=(
            "Temporal pattern"
        ),
        title_fontsize=8.5,
    )

    # ------------------------------------------------------------------
    # Right-side summary panel.
    # For small groups list every ADM2; for large groups show count and
    # direct the reader to the membership CSV.
    # ------------------------------------------------------------------
    info_ax.axis("off")

    info_ax.text(
        0.0,
        1.0,
        "Pattern membership",
        fontsize=14,
        fontweight="bold",
        va="top",
    )

    y = 0.955

    for _, pinfo in pattern_rows.iterrows():
        cluster_id = int(
            pinfo["cluster"]
        )

        pattern_name = str(
            pinfo["pattern_name"]
        )

        n = int(
            pinfo[
                "pattern_size"
            ]
        )

        members = (
            membership[
                membership[
                    "cluster"
                ] == cluster_id
            ]["ADM2_NAME"]
            .sort_values()
            .tolist()
        )

        info_ax.scatter(
            [0.02],
            [y],
            s=70,
            color=cluster_colours[
                cluster_id
            ],
            edgecolor="black",
            linewidth=0.4,
            transform=info_ax.transAxes,
            clip_on=False,
        )

        info_ax.text(
            0.07,
            y,
            f"{pattern_name} (n={n})",
            fontsize=9.3,
            fontweight="bold",
            va="center",
            transform=info_ax.transAxes,
        )

        y -= 0.034

        if n <= 6:
            member_text = ", ".join(
                members
            )
        elif n <= 21:
            member_text = (
                ", ".join(
                    members
                )
            )
        else:
            member_text = (
                f"{n} ADM2 areas "
                "(full list in 03_pattern_membership.csv)"
            )

        # Wrap manually to keep the panel readable.
        words = member_text.split()
        lines = []
        current = ""

        for word in words:
            proposed = (
                word
                if not current
                else current
                + " "
                + word
            )

            if len(
                proposed
            ) > 38:
                lines.append(
                    current
                )
                current = word
            else:
                current = proposed

        if current:
            lines.append(
                current
            )

        for line in lines:
            info_ax.text(
                0.07,
                y,
                line,
                fontsize=7.8,
                va="top",
                transform=info_ax.transAxes,
            )
            y -= 0.026

        y -= 0.020

    info_ax.text(
        0.0,
        0.01,
        (
            "Note: colours represent temporal-pattern classes, "
            "not flood-risk rankings."
        ),
        fontsize=7.7,
        va="bottom",
        transform=info_ax.transAxes,
    )

    fig.savefig(
        FIGURE_DIR
        / "10_pattern_point_map_with_names.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 18C. COMPACT PATTERN MEMBERSHIP SUMMARY
# =============================================================================

def save_pattern_membership_summary(
    membership: pd.DataFrame,
) -> None:
    """
    Save one row per pattern with the full ADM2 membership as a readable list.
    """
    rows = []

    for (
        cluster_id,
        pattern_name,
    ), sub in membership.groupby(
        [
            "cluster",
            "pattern_name",
        ]
    ):
        names = (
            sub["ADM2_NAME"]
            .sort_values()
            .tolist()
        )

        rows.append(
            {
                "cluster":
                int(cluster_id),
                "pattern_name":
                pattern_name,
                "n_adm2":
                len(names),
                "adm2_names":
                " | ".join(
                    names
                ),
                "is_rare_pattern":
                int(
                    sub[
                        "is_rare_pattern"
                    ].max()
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    ).sort_values(
        "cluster"
    )

    summary.to_csv(
        TABLE_DIR
        / "08_pattern_membership_summary.csv",
        index=False,
    )


# =============================================================================
# 19. MAIN
# =============================================================================

def main() -> None:
    print_header(
        "ADM2 FINE-GRAINED FLOOD TEMPORAL PATTERN DISCOVERY"
    )

    print(
        f"Annual input:\n{ANNUAL_FILE}"
    )

    print(
        f"\nOutput root:\n{OUTPUT_ROOT}"
    )

    print(
        f"\nSelected fine-grained solution: k={SELECTED_K}"
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    df = load_annual_data()

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------
    features = build_temporal_features(
        df
    )

    (
        X,
        X_scaled,
        shape_cols,
        scaler,
    ) = prepare_shape_matrix(
        features
    )

    print(
        "\nShape-oriented clustering variables:"
    )

    for col in shape_cols:
        print(
            f"  - {col}"
        )

    # -------------------------------------------------------------------------
    # Compare k
    # -------------------------------------------------------------------------
    validation = evaluate_k_values(
        X_scaled
    )

    plot_k_validation(
        validation
    )

    # -------------------------------------------------------------------------
    # Fit selected k
    # -------------------------------------------------------------------------
    membership = fit_selected_clustering(
        features,
        X_scaled,
    )

    profiles = build_pattern_profiles(
        features,
        membership,
    )

    profiles = assign_descriptive_pattern_names(
        profiles
    )

    membership = attach_pattern_names(
        membership,
        profiles,
    )

    save_final_tables(
        membership,
        profiles,
    )

    save_pattern_membership_summary(
        membership
    )

    # -------------------------------------------------------------------------
    # Clustering figures
    # -------------------------------------------------------------------------
    plot_selected_dendrogram(
        X_scaled,
        features,
    )

    plot_pca_patterns(
        X_scaled,
        membership,
    )

    plot_pattern_mean_trajectories(
        df,
        membership,
    )

    # Presentation-ready versions:
    # common patterns and rare/extreme patterns are separated so the extreme
    # case does not compress the y-axis for the other patterns.
    plot_common_pattern_trajectories(
        df,
        membership,
    )

    plot_rare_pattern_trajectories(
        df,
        membership,
    )

    plot_pattern_small_multiples(
        df,
        membership,
    )

    plot_pattern_profile_heatmap(
        profiles
    )

    # -------------------------------------------------------------------------
    # Change points
    # -------------------------------------------------------------------------
    change_points = detect_change_points(
        df,
        membership,
    )

    build_pattern_change_point_summary(
        change_points
    )

    plot_change_point_distribution(
        change_points
    )

    plot_selected_pattern_examples(
        df,
        membership,
        change_points,
    )

    # -------------------------------------------------------------------------
    # Map
    # -------------------------------------------------------------------------
    if gpd is not None:
        try:
            (
                adm2_gdf,
                adm2_name_col,
            ) = load_adm2_boundary()

            plot_pattern_map(
                adm2_gdf,
                adm2_name_col,
                membership,
            )

            # Main presentation map:
            # light ADM2 boundary background + one coloured point per ADM2
            # + labels for smaller / high-interest pattern groups.
            plot_pattern_point_map(
                adm2_gdf,
                adm2_name_col,
                membership,
            )

        except Exception as exc:
            print(
                "\nPattern map could not be generated:"
            )

            print(
                exc
            )
    else:
        print(
            "\nGeoPandas not installed; map skipped."
        )

    # -------------------------------------------------------------------------
    # Human-readable output
    # -------------------------------------------------------------------------
    print_pattern_summary(
        membership,
        profiles,
    )

    print_header(
        "FINISHED"
    )

    print(
        f"""
The analysis kept ALL ADM2 areas in clustering.

Rare/singleton clusters are retained as legitimate rare temporal patterns.
They are NOT automatically discarded as outliers.

Selected k:
  {SELECTED_K}

Primary interpretation figures:
  04a_common_pattern_trajectories.png
  04b_rare_extreme_pattern_trajectory.png
  05_pattern_small_multiples.png
  08_pattern_map.png
  09_selected_pattern_examples.png
  10_pattern_point_map_with_names.png

Most useful membership table:
  08_pattern_membership_summary.csv

Method-validation figures:
  01_k_validation.png
  02_dendrogram_selected_k.png
  03_pca_patterns.png
  06_pattern_profile_heatmap.png

Change-point figure:
  07_change_point_distribution.png

Most useful tables:
  03_pattern_membership.csv
  04_pattern_profiles.csv
  05_pattern_name_guide.csv
  06_change_points.csv

IMPORTANT:
Pattern names are assigned AFTER clustering from the observed cluster profiles.
Please inspect 05_pattern_small_multiples.png before treating the descriptive
pattern names as final scientific labels.

Saved under:
  {OUTPUT_ROOT}
"""
    )


if __name__ == "__main__":
    main()
