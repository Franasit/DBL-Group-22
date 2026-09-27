"""
04_compare_adm2_adm3_patterns_v5.py

South Sudan Flood Analysis
Multi-label ADM2 × ADM3 cross-scale temporal comparison

========================================================================
WHY V5?
========================================================================

ADM2 pattern discovery is multi-label and data-driven, e.g.

    Chronic hotspot | Episodic extreme | High volatility |
    Accelerating | Recent resurgence

ADM3 temporal classification is a single categorical class, e.g.

    Persistent hotspot
    Emerging hotspot
    Increasing
    Declining
    Episodic / volatile
    Stable moderate
    Stable low

V5 therefore DOES NOT compress ADM2 into one coarse semantic class.

Instead it keeps the independent ADM2 dimensions:

    extreme_exceptional_case
    emerging_hotspot
    chronic_hotspot
    episodic_extreme
    high_volatility
    accelerating
    declining
    historical_peak_recovery
    recent_resurgence
    persistent_low
    no_dominant_pattern

and compares those dimensions with the ADM3 temporal pattern.

This gives a more faithful multi-scale interpretation.

========================================================================
INPUTS
========================================================================

ADM2 feature table:
processed_data/flood_analysis/adm2/pattern_discovery_outlier_aware/tables/
    01_adm2_temporal_features.csv

ADM2 final multi-label table:
processed_data/flood_analysis/adm2/pattern_discovery_outlier_aware/tables/
    07_adm2_pattern_labels.csv

ADM3 temporal summary:
processed_data/flood_analysis/adm3/temporal_patterns/
    adm3_temporal_pattern_summary.csv

========================================================================
OUTPUTS
========================================================================

processed_data/flood_analysis/cross_scale_adm2_adm3_v5/

01_adm2_multilabel_dimensions_v5.csv
02_adm3_cross_scale_comparison_v5.csv
03_primary_relationship_counts_v5.csv
04_adm2_cross_scale_summary_v5.csv
05_regional_chronic_local_hotspot_v5.csv
06_regional_neutral_local_hotspot_v5.csv
07_regional_accelerating_local_increase_v5.csv
08_regional_decline_local_worsening_v5.csv
09_regional_volatile_local_hotspot_v5.csv
10_regional_low_local_emerging_v5.csv
11_cross_scale_disagreement_v5.csv
12_high_heterogeneity_adm2_v5.csv
13_dimension_alignment_counts_v5.csv
14_cross_scale_qc_v5.csv
15_unmatched_adm2_names_v5.csv
"""

from __future__ import annotations

from pathlib import Path
import math
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# 1. PATHS
# ============================================================================

PROJECT_ROOT = Path(
    "/Users/shibai/Documents/GitHub/DBL-Group-22"
)

ADM2_TABLE_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm2"
    / "pattern_discovery_outlier_aware"
    / "tables"
)

ADM2_FEATURE_FILE = (
    ADM2_TABLE_DIR
    / "01_adm2_temporal_features.csv"
)

ADM2_LABEL_FILE = (
    ADM2_TABLE_DIR
    / "07_adm2_pattern_labels.csv"
)

ADM3_SUMMARY_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm3"
    / "temporal_patterns"
    / "adm3_temporal_pattern_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "cross_scale_adm2_adm3_v5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# 2. ADM2 MULTI-LABEL DIMENSIONS
# ============================================================================

ADM2_DIMENSIONS = [
    "extreme_exceptional_case",
    "emerging_hotspot",
    "chronic_hotspot",
    "episodic_extreme",
    "high_volatility",
    "accelerating",
    "declining",
    "historical_peak_recovery",
    "recent_resurgence",
    "persistent_low",
]

ALL_ADM2_DIMENSIONS = (
    ADM2_DIMENSIONS
    + [
        "no_dominant_pattern"
    ]
)


# ============================================================================
# 3. ADM3 TEMPORAL GROUPS
# ============================================================================

ADM3_HOTSPOT = {
    "Persistent hotspot",
    "Emerging hotspot",
}

ADM3_WORSENING = {
    "Persistent hotspot",
    "Emerging hotspot",
    "Increasing",
}

ADM3_INCREASING = {
    "Emerging hotspot",
    "Increasing",
}

ADM3_DECLINING = {
    "Declining",
}

ADM3_VOLATILE = {
    "Episodic / volatile",
}

ADM3_STABLE = {
    "Stable moderate",
    "Stable low",
}


# ============================================================================
# 4. HELPERS
# ============================================================================

def require_file(
    path: Path,
    description: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


def normalize_text(
    value,
) -> str:

    if pd.isna(value):
        return ""

    text = (
        str(value)
        .strip()
        .lower()
    )

    text = re.sub(
        r"[/+_\-]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_name(
    value,
) -> str:

    return normalize_text(
        value
    )


def detect_column(
    columns,
    candidates,
):

    lower_map = {
        str(col).lower():
            col
        for col in columns
    }

    for candidate in candidates:

        if (
            candidate.lower()
            in lower_map
        ):

            return lower_map[
                candidate.lower()
            ]

    normalized_map = {
        normalize_text(
            col
        ):
            col
        for col in columns
    }

    for candidate in candidates:

        key = normalize_text(
            candidate
        )

        if key in normalized_map:

            return normalized_map[
                key
            ]

    return None


def to_binary(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(
        series
    ):

        return (
            series
            .fillna(False)
            .astype(int)
        )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().any():

        return (
            numeric
            .fillna(0)
            .ne(0)
            .astype(int)
        )

    text = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return (
        text.isin(
            {
                "1",
                "true",
                "yes",
                "y",
                "t",
            }
        )
        .astype(int)
    )


def safe_numeric(
    series,
):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def normalized_entropy(
    series: pd.Series,
) -> float:

    counts = (
        series
        .dropna()
        .value_counts()
        .astype(float)
    )

    if len(counts) <= 1:
        return 0.0

    p = (
        counts
        /
        counts.sum()
    )

    entropy = -float(
        np.sum(
            p
            *
            np.log(p)
        )
    )

    max_entropy = (
        math.log(
            len(counts)
        )
    )

    if max_entropy == 0:
        return 0.0

    return (
        entropy
        /
        max_entropy
    )


def dominant_value(
    series: pd.Series,
):

    counts = (
        series
        .dropna()
        .value_counts()
    )

    if counts.empty:
        return np.nan

    return counts.index[0]


def dominant_share(
    series: pd.Series,
) -> float:

    counts = (
        series
        .dropna()
        .value_counts()
    )

    if counts.empty:
        return np.nan

    return float(
        counts.iloc[0]
        /
        counts.sum()
    )


# ============================================================================
# 5. LOAD ADM2 LABEL TABLE
# ============================================================================

def load_adm2_labels():

    print("=" * 80)
    print("LOAD ADM2 MULTI-LABEL PATTERN TABLE")
    print("=" * 80)

    require_file(
        ADM2_LABEL_FILE,
        "ADM2 pattern-label table",
    )

    df = pd.read_csv(
        ADM2_LABEL_FILE
    )

    adm2_name_col = detect_column(
        df.columns,
        [
            "ADM2_NAME",
            "adm2_name",
            "ADM2",
            "county",
            "county_name",
            "NAME_2",
        ],
    )

    pattern_col = detect_column(
        df.columns,
        [
            "pattern_label",
            "pattern_labels",
            "pattern_type",
            "final_pattern",
            "final_pattern_label",
            "temporal_pattern",
            "label",
        ],
    )

    if adm2_name_col is None:

        raise ValueError(
            "\nCould not detect ADM2 name column in "
            "07_adm2_pattern_labels.csv.\n"
            f"Columns:\n{list(df.columns)}"
        )

    if pattern_col is None:

        # Fallback: look for the single pattern/label text column.
        text_candidates = [
            col
            for col in df.columns
            if (
                "pattern"
                in str(col).lower()
                or
                "label"
                in str(col).lower()
            )
        ]

        # Exclude the binary dimension columns.
        text_candidates = [
            col
            for col in text_candidates
            if col
            not in ADM2_DIMENSIONS
        ]

        if len(text_candidates) == 1:

            pattern_col = (
                text_candidates[0]
            )

        else:

            raise ValueError(
                "\nCould not detect final ADM2 pattern label column.\n"
                f"Columns:\n{list(df.columns)}"
            )

    df = df.rename(
        columns={
            adm2_name_col:
                "ADM2_NAME",

            pattern_col:
                "ADM2_FINAL_PATTERN",
        }
    )

    df[
        "ADM2_KEY"
    ] = (
        df[
            "ADM2_NAME"
        ]
        .map(
            normalize_name
        )
    )

    # --------------------------------------------------------
    # Ensure all multi-label dimensions exist.
    # If missing, derive them from final label text.
    # --------------------------------------------------------

    label_text = (
        df[
            "ADM2_FINAL_PATTERN"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    keyword_map = {
        "extreme_exceptional_case":
            "extreme exceptional case",

        "emerging_hotspot":
            "emerging hotspot",

        "chronic_hotspot":
            "chronic hotspot",

        "episodic_extreme":
            "episodic extreme",

        "high_volatility":
            "high volatility",

        "accelerating":
            "accelerating",

        "declining":
            "declining",

        "historical_peak_recovery":
            "historical peak",

        "recent_resurgence":
            "recent resurgence",

        "persistent_low":
            "persistent low",
    }

    for (
        dimension,
        keyword,
    ) in keyword_map.items():

        if dimension in df.columns:

            df[
                dimension
            ] = to_binary(
                df[
                    dimension
                ]
            )

        else:

            df[
                dimension
            ] = (
                label_text
                .str.contains(
                    keyword,
                    regex=False,
                )
                .astype(int)
            )

    # --------------------------------------------------------
    # No dominant special pattern
    # --------------------------------------------------------

    explicit_no_dominant = (
        label_text
        .str.contains(
            "no dominant special pattern",
            regex=False,
        )
    )

    no_flag = (
        df[
            ADM2_DIMENSIONS
        ]
        .sum(
            axis=1
        )
        ==
        0
    )

    df[
        "no_dominant_pattern"
    ] = (
        explicit_no_dominant
        |
        no_flag
    ).astype(int)

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    duplicate_count = int(
        df[
            "ADM2_KEY"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        print(
            f"WARNING: {duplicate_count} duplicate ADM2 labels found."
        )

        df = (
            df
            .drop_duplicates(
                subset=[
                    "ADM2_KEY"
                ],
                keep="first",
            )
        )

    print(
        f"ADM2 label rows: {len(df):,}"
    )

    print(
        f"Unique ADM2: {df['ADM2_KEY'].nunique():,}"
    )

    print(
        "\nADM2 dimension counts:"
    )

    counts = (
        df[
            ALL_ADM2_DIMENSIONS
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        counts.to_string()
    )

    return df


# ============================================================================
# 6. LOAD ADM2 FEATURE TABLE
# ============================================================================

def load_adm2_features():

    print("\n" + "=" * 80)
    print("LOAD ADM2 TEMPORAL FEATURES")
    print("=" * 80)

    require_file(
        ADM2_FEATURE_FILE,
        "ADM2 temporal-feature table",
    )

    df = pd.read_csv(
        ADM2_FEATURE_FILE
    )

    name_col = detect_column(
        df.columns,
        [
            "ADM2_NAME",
            "adm2_name",
            "ADM2",
            "county",
            "county_name",
            "NAME_2",
        ],
    )

    if name_col is None:

        raise ValueError(
            "\nCould not detect ADM2 name column in "
            "01_adm2_temporal_features.csv.\n"
            f"Columns:\n{list(df.columns)}"
        )

    df = df.rename(
        columns={
            name_col:
                "ADM2_NAME_FEATURE"
        }
    )

    df[
        "ADM2_KEY"
    ] = (
        df[
            "ADM2_NAME_FEATURE"
        ]
        .map(
            normalize_name
        )
    )

    duplicate_count = int(
        df[
            "ADM2_KEY"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        print(
            f"WARNING: {duplicate_count} duplicate ADM2 feature rows found."
        )

        df = (
            df
            .drop_duplicates(
                subset=[
                    "ADM2_KEY"
                ],
                keep="first",
            )
        )

    print(
        f"ADM2 feature rows: {len(df):,}"
    )

    return df


# ============================================================================
# 7. MERGE ADM2 LABELS + FEATURES
# ============================================================================

def build_adm2_table(
    labels,
    features,
):

    print("\n" + "=" * 80)
    print("MERGE ADM2 MULTI-LABELS + FEATURES")
    print("=" * 80)

    merged = labels.merge(
        features,
        on="ADM2_KEY",
        how="outer",
        suffixes=(
            "_label",
            "_feature",
        ),
        indicator=True,
    )

    print(
        merged[
            "_merge"
        ]
        .value_counts()
        .to_string()
    )

    merged[
        "ADM2_NAME_FINAL"
    ] = (
        merged[
            "ADM2_NAME"
        ]
        .combine_first(
            merged[
                "ADM2_NAME_FEATURE"
            ]
            if (
                "ADM2_NAME_FEATURE"
                in merged.columns
            )
            else pd.Series(
                index=merged.index,
                dtype=object,
            )
        )
    )

    merged = merged.drop(
        columns=[
            "_merge"
        ]
    )

    return merged


# ============================================================================
# 8. LOAD ADM3 TEMPORAL SUMMARY
# ============================================================================

def load_adm3():

    print("\n" + "=" * 80)
    print("LOAD ADM3 TEMPORAL PATTERNS")
    print("=" * 80)

    require_file(
        ADM3_SUMMARY_FILE,
        "ADM3 temporal-pattern summary",
    )

    df = pd.read_csv(
        ADM3_SUMMARY_FILE
    )

    required = [
        "ADM3_ID",
        "ADM2_NAME",
        "ADM3_NAME",
        "temporal_pattern",
        "long_term_mean",
        "recent_minus_early",
        "trend_slope",
    ]

    missing = [
        col
        for col in required
        if col
        not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nADM3 summary missing required columns:\n"
            + "\n".join(
                missing
            )
        )

    # Rename ADM3 continuous metrics BEFORE merging with ADM2 features.
    # This prevents pandas from creating ambiguous *_x / *_y columns.
    df = df.rename(
        columns={
            "long_term_mean":
                "ADM3_long_term_mean",

            "recent_minus_early":
                "ADM3_recent_minus_early",

            "trend_slope":
                "ADM3_trend_slope",
        }
    )

    df[
        "ADM2_KEY"
    ] = (
        df[
            "ADM2_NAME"
        ]
        .map(
            normalize_name
        )
    )

    df[
        "ADM3_PATTERN"
    ] = (
        df[
            "temporal_pattern"
        ]
        .astype(str)
        .str.strip()
    )

    print(
        f"ADM3 rows: {len(df):,}"
    )

    print(
        f"Unique ADM3: {df['ADM3_ID'].nunique():,}"
    )

    print(
        f"Unique parent ADM2: {df['ADM2_KEY'].nunique():,}"
    )

    return df


# ============================================================================
# 9. ADM3 DERIVED BINARY DIMENSIONS
# ============================================================================

def add_adm3_dimensions(
    df,
):

    df = df.copy()

    pattern = (
        df[
            "ADM3_PATTERN"
        ]
    )

    df[
        "ADM3_is_hotspot"
    ] = (
        pattern
        .isin(
            ADM3_HOTSPOT
        )
        .astype(int)
    )

    df[
        "ADM3_is_persistent_hotspot"
    ] = (
        pattern
        .eq(
            "Persistent hotspot"
        )
        .astype(int)
    )

    df[
        "ADM3_is_emerging_hotspot"
    ] = (
        pattern
        .eq(
            "Emerging hotspot"
        )
        .astype(int)
    )

    df[
        "ADM3_is_increasing"
    ] = (
        pattern
        .isin(
            ADM3_INCREASING
        )
        .astype(int)
    )

    df[
        "ADM3_is_worsening"
    ] = (
        pattern
        .isin(
            ADM3_WORSENING
        )
        .astype(int)
    )

    df[
        "ADM3_is_declining"
    ] = (
        pattern
        .isin(
            ADM3_DECLINING
        )
        .astype(int)
    )

    df[
        "ADM3_is_volatile"
    ] = (
        pattern
        .isin(
            ADM3_VOLATILE
        )
        .astype(int)
    )

    df[
        "ADM3_is_stable"
    ] = (
        pattern
        .isin(
            ADM3_STABLE
        )
        .astype(int)
    )

    return df


# ============================================================================
# 10. DIMENSION-BY-DIMENSION FLAGS
# ============================================================================

def add_cross_scale_flags(
    df,
):

    df = df.copy()

    # --------------------------------------------------------
    # Regional chronic hotspot relationships
    # --------------------------------------------------------

    df[
        "flag_regional_chronic_local_hotspot"
    ] = (
        (
            df[
                "chronic_hotspot"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_hotspot"
            ]
            ==
            1
        )
    ).astype(int)

    df[
        "flag_regional_chronic_local_persistent"
    ] = (
        (
            df[
                "chronic_hotspot"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_persistent_hotspot"
            ]
            ==
            1
        )
    ).astype(int)

    df[
        "flag_regional_chronic_local_decline"
    ] = (
        (
            df[
                "chronic_hotspot"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_declining"
            ]
            ==
            1
        )
    ).astype(int)

    # --------------------------------------------------------
    # Regional neutral background relationships
    # --------------------------------------------------------

    df[
        "flag_regional_neutral_local_hotspot"
    ] = (
        (
            df[
                "no_dominant_pattern"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_hotspot"
            ]
            ==
            1
        )
    ).astype(int)

    df[
        "flag_regional_neutral_local_increase"
    ] = (
        (
            df[
                "no_dominant_pattern"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_increasing"
            ]
            ==
            1
        )
    ).astype(int)

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    df[
        "flag_regional_accelerating_local_increase"
    ] = (
        (
            df[
                "accelerating"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_increasing"
            ]
            ==
            1
        )
    ).astype(int)

    df[
        "flag_regional_accelerating_local_decline"
    ] = (
        (
            df[
                "accelerating"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_declining"
            ]
            ==
            1
        )
    ).astype(int)

    # --------------------------------------------------------
    # Regional decline
    # --------------------------------------------------------

    df[
        "flag_regional_decline_local_decline"
    ] = (
        (
            df[
                "declining"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_declining"
            ]
            ==
            1
        )
    ).astype(int)

    df[
        "flag_regional_decline_local_worsening"
    ] = (
        (
            df[
                "declining"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_worsening"
            ]
            ==
            1
        )
    ).astype(int)

    # --------------------------------------------------------
    # Regional volatility / episodic behaviour
    # --------------------------------------------------------

    df[
        "flag_regional_volatile_local_hotspot"
    ] = (
        (
            (
                df[
                    "high_volatility"
                ]
                ==
                1
            )
            |
            (
                df[
                    "episodic_extreme"
                ]
                ==
                1
            )
        )
        &
        (
            df[
                "ADM3_is_hotspot"
            ]
            ==
            1
        )
    ).astype(int)

    df[
        "flag_regional_volatile_local_volatile"
    ] = (
        (
            (
                df[
                    "high_volatility"
                ]
                ==
                1
            )
            |
            (
                df[
                    "episodic_extreme"
                ]
                ==
                1
            )
        )
        &
        (
            df[
                "ADM3_is_volatile"
            ]
            ==
            1
        )
    ).astype(int)

    # --------------------------------------------------------
    # Persistent-low regional background
    # --------------------------------------------------------

    df[
        "flag_regional_low_local_emerging"
    ] = (
        (
            df[
                "persistent_low"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_emerging_hotspot"
            ]
            ==
            1
        )
    ).astype(int)

    df[
        "flag_regional_low_local_hotspot"
    ] = (
        (
            df[
                "persistent_low"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_hotspot"
            ]
            ==
            1
        )
    ).astype(int)

    df[
        "flag_regional_low_local_stable"
    ] = (
        (
            df[
                "persistent_low"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_stable"
            ]
            ==
            1
        )
    ).astype(int)

    # --------------------------------------------------------
    # Recent resurgence
    # --------------------------------------------------------

    df[
        "flag_regional_resurgence_local_worsening"
    ] = (
        (
            df[
                "recent_resurgence"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_worsening"
            ]
            ==
            1
        )
    ).astype(int)

    # --------------------------------------------------------
    # Emerging hotspot at ADM2
    # --------------------------------------------------------

    df[
        "flag_regional_emerging_local_worsening"
    ] = (
        (
            df[
                "emerging_hotspot"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_worsening"
            ]
            ==
            1
        )
    ).astype(int)

    # --------------------------------------------------------
    # Extreme exceptional regional case
    # --------------------------------------------------------

    df[
        "flag_regional_extreme_local_hotspot"
    ] = (
        (
            df[
                "extreme_exceptional_case"
            ]
            ==
            1
        )
        &
        (
            df[
                "ADM3_is_hotspot"
            ]
            ==
            1
        )
    ).astype(int)

    return df


# ============================================================================
# 11. PRIMARY CROSS-SCALE PATTERN
# ============================================================================

def classify_primary_pattern(
    row,
):
    """
    A single mutually exclusive label for mapping / summary.

    Important:
    The detailed binary flags remain the main analytical output.
    This primary class is only a compact presentation layer.
    """

    # --------------------------------------------------------
    # Strongest directional disagreements first
    # --------------------------------------------------------

    if (
        row[
            "flag_regional_decline_local_worsening"
        ]
        ==
        1
    ):

        return (
            "Regional decline + local worsening"
        )

    if (
        row[
            "flag_regional_chronic_local_decline"
        ]
        ==
        1
    ):

        return (
            "Regional chronic + local decline"
        )

    if (
        row[
            "flag_regional_accelerating_local_decline"
        ]
        ==
        1
    ):

        return (
            "Regional acceleration + local decline"
        )

    # --------------------------------------------------------
    # Hidden local risk
    # --------------------------------------------------------

    if (
        row[
            "flag_regional_neutral_local_hotspot"
        ]
        ==
        1
    ):

        return (
            "Regional neutral + local hotspot"
        )

    if (
        row[
            "flag_regional_low_local_emerging"
        ]
        ==
        1
    ):

        return (
            "Regional low + local emerging"
        )

    if (
        row[
            "flag_regional_low_local_hotspot"
        ]
        ==
        1
    ):

        return (
            "Regional low + local hotspot"
        )

    # --------------------------------------------------------
    # Consistent worsening / hotspot signal
    # --------------------------------------------------------

    if (
        row[
            "flag_regional_emerging_local_worsening"
        ]
        ==
        1
    ):

        return (
            "Regional emerging + local worsening"
        )

    if (
        row[
            "flag_regional_accelerating_local_increase"
        ]
        ==
        1
    ):

        return (
            "Regional accelerating + local increase"
        )

    if (
        row[
            "flag_regional_chronic_local_hotspot"
        ]
        ==
        1
    ):

        return (
            "Regional chronic + local hotspot"
        )

    if (
        row[
            "flag_regional_resurgence_local_worsening"
        ]
        ==
        1
    ):

        return (
            "Regional resurgence + local worsening"
        )

    if (
        row[
            "flag_regional_volatile_local_hotspot"
        ]
        ==
        1
    ):

        return (
            "Regional volatile + local hotspot"
        )

    # --------------------------------------------------------
    # Consistent decline / stable low
    # --------------------------------------------------------

    if (
        row[
            "flag_regional_decline_local_decline"
        ]
        ==
        1
    ):

        return (
            "Regional decline + local decline"
        )

    if (
        row[
            "flag_regional_low_local_stable"
        ]
        ==
        1
    ):

        return (
            "Regional low + local stable"
        )

    if (
        (
            row[
                "no_dominant_pattern"
            ]
            ==
            1
        )
        and
        (
            row[
                "ADM3_is_stable"
            ]
            ==
            1
        )
    ):

        return (
            "Regional neutral + local stable"
        )

    if (
        row[
            "flag_regional_volatile_local_volatile"
        ]
        ==
        1
    ):

        return (
            "Regional/local volatility alignment"
        )

    # --------------------------------------------------------
    # Remaining cases
    # --------------------------------------------------------

    if (
        row[
            "ADM3_is_volatile"
        ]
        ==
        1
    ):

        return (
            "Local volatile / mixed regional context"
        )

    return (
        "Multi-signal / mixed"
    )


# ============================================================================
# 12. LOAD INPUTS
# ============================================================================

adm2_labels = (
    load_adm2_labels()
)

adm2_features = (
    load_adm2_features()
)

adm2 = (
    build_adm2_table(
        adm2_labels,
        adm2_features,
    )
)

adm3 = (
    load_adm3()
)

adm3 = (
    add_adm3_dimensions(
        adm3
    )
)


# ============================================================================
# 13. SAVE ADM2 MULTI-LABEL TABLE
# ============================================================================

adm2.to_csv(
    OUTPUT_DIR
    / "01_adm2_multilabel_dimensions_v5.csv",
    index=False,
)


# ============================================================================
# 14. MERGE ADM2 -> ADM3
# ============================================================================

print("\n" + "=" * 80)
print("MERGE ADM2 MULTI-LABEL DIMENSIONS ONTO ADM3")
print("=" * 80)

adm2_for_merge = (
    adm2.copy()
)

# Remove/rename duplicate geography name columns
if (
    "ADM2_NAME"
    in adm2_for_merge.columns
):

    adm2_for_merge = (
        adm2_for_merge
        .rename(
            columns={
                "ADM2_NAME":
                    "ADM2_NAME_FROM_LABEL_TABLE"
            }
        )
    )


cross = adm3.merge(
    adm2_for_merge,
    on="ADM2_KEY",
    how="left",
    validate="many_to_one",
)


missing_parent = int(
    cross[
        "ADM2_FINAL_PATTERN"
    ]
    .isna()
    .sum()
)


print(
    f"Cross-scale ADM3 rows: {len(cross):,}"
)

print(
    f"ADM3 rows without matched ADM2 multi-label data: "
    f"{missing_parent:,}"
)


# Continuous-column QC: these names should exist without pandas suffixes.
required_adm3_metric_columns = [
    "ADM3_long_term_mean",
    "ADM3_recent_minus_early",
    "ADM3_trend_slope",
]

missing_adm3_metric_columns = [
    col
    for col in required_adm3_metric_columns
    if col not in cross.columns
]

if missing_adm3_metric_columns:
    raise ValueError(
        "\nADM3 metric columns missing after merge:\n"
        + "\n".join(missing_adm3_metric_columns)
        + "\n\nAvailable columns:\n"
        + str(list(cross.columns))
    )

print(
    "PASS: ADM3 continuous metrics preserved with explicit ADM3_ prefixes."
)


# ============================================================================
# 15. UNMATCHED ADM2 NAMES
# ============================================================================

unmatched = (
    cross.loc[
        cross[
            "ADM2_FINAL_PATTERN"
        ]
        .isna(),
        [
            "ADM2_NAME",
            "ADM2_KEY",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "ADM2_NAME"
    )
)


unmatched.to_csv(
    OUTPUT_DIR
    / "15_unmatched_adm2_names_v5.csv",
    index=False,
)


# ============================================================================
# 16. CROSS-SCALE FLAGS + PRIMARY CLASS
# ============================================================================

cross = (
    add_cross_scale_flags(
        cross
    )
)


cross[
    "primary_cross_scale_pattern"
] = cross.apply(
    classify_primary_pattern,
    axis=1,
)


# ============================================================================
# 17. CONTINUOUS DIRECTION FLAGS
# ============================================================================

adm2_slope_col = detect_column(
    cross.columns,
    [
        "ADM2_trend_slope",
        "trend_slope_feature",
        "trend_pp_per_year",
        "trend_slope",
    ],
)

# ADM3 slope is explicitly named ADM3_trend_slope, so any detected
# unprefixed slope here belongs to the ADM2 feature table.


def direction(
    value,
):

    if pd.isna(value):
        return "unknown"

    if value > 0:
        return "positive"

    if value < 0:
        return "negative"

    return "flat"


if (
    adm2_slope_col
    is not None
):

    cross[
        "ADM2_trend_direction"
    ] = (
        cross[
            adm2_slope_col
        ]
        .map(
            direction
        )
    )

    cross[
        "ADM3_trend_direction"
    ] = (
        cross[
            "ADM3_trend_slope"
        ]
        .map(
            direction
        )
    )

    cross[
        "continuous_trend_opposite"
    ] = (
        (
            cross[
                "ADM2_trend_direction"
            ]
            ==
            "positive"
        )
        &
        (
            cross[
                "ADM3_trend_direction"
            ]
            ==
            "negative"
        )
        |
        (
            cross[
                "ADM2_trend_direction"
            ]
            ==
            "negative"
        )
        &
        (
            cross[
                "ADM3_trend_direction"
            ]
            ==
            "positive"
        )
    ).astype(int)

else:

    cross[
        "continuous_trend_opposite"
    ] = 0


# ============================================================================
# 18. SAVE FULL ADM3 CROSS-SCALE TABLE
# ============================================================================

cross = (
    cross
    .sort_values(
        [
            "ADM2_NAME",
            "ADM3_NAME",
        ]
    )
)


cross.to_csv(
    OUTPUT_DIR
    / "02_adm3_cross_scale_comparison_v5.csv",
    index=False,
)


# ============================================================================
# 19. PRIMARY RELATIONSHIP COUNTS
# ============================================================================

primary_counts = (
    cross[
        "primary_cross_scale_pattern"
    ]
    .value_counts()
    .rename_axis(
        "primary_cross_scale_pattern"
    )
    .reset_index(
        name="adm3_count"
    )
)


primary_counts[
    "share_percent"
] = (
    primary_counts[
        "adm3_count"
    ]
    /
    len(
        cross
    )
    *
    100
)


primary_counts.to_csv(
    OUTPUT_DIR
    / "03_primary_relationship_counts_v5.csv",
    index=False,
)


# ============================================================================
# 20. DIMENSION ALIGNMENT COUNTS
# ============================================================================

flag_columns = [
    col
    for col in cross.columns
    if col.startswith(
        "flag_"
    )
]


dimension_counts = pd.DataFrame(
    {
        "comparison_flag":
            flag_columns,

        "adm3_count": [
            int(
                cross[
                    col
                ]
                .sum()
            )
            for col in flag_columns
        ],
    }
)


dimension_counts[
    "share_percent"
] = (
    dimension_counts[
        "adm3_count"
    ]
    /
    len(
        cross
    )
    *
    100
)


dimension_counts = (
    dimension_counts
    .sort_values(
        "adm3_count",
        ascending=False,
    )
)


dimension_counts.to_csv(
    OUTPUT_DIR
    / "13_dimension_alignment_counts_v5.csv",
    index=False,
)


# ============================================================================
# 20B. STRICT COLUMN QC BEFORE ADM2 SUMMARY
# ============================================================================

required_cross_columns = [
    "ADM3_long_term_mean",
    "ADM3_recent_minus_early",
    "ADM3_trend_slope",
    "ADM3_PATTERN",
    "ADM2_FINAL_PATTERN",
    "ADM2_NAME",
]

missing_cross_columns = [
    col
    for col in required_cross_columns
    if col not in cross.columns
]

if missing_cross_columns:
    raise ValueError(
        "\nRequired columns are missing before ADM2 summary:\n"
        + "\n".join(missing_cross_columns)
        + "\n\nAvailable columns:\n"
        + str(list(cross.columns))
    )

print(
    "PASS: All required ADM3 metric columns are available before ADM2 summary."
)


# ============================================================================
# 21. ADM2 CROSS-SCALE SUMMARY
# ============================================================================

summary_rows = []


for (
    adm2_name,
    group,
) in cross.groupby(
    "ADM2_NAME"
):

    n_adm3 = len(
        group
    )

    row = {
        "ADM2_NAME":
            adm2_name,

        "ADM2_FINAL_PATTERN":
            group[
                "ADM2_FINAL_PATTERN"
            ]
            .dropna()
            .iloc[0]
            if (
                group[
                    "ADM2_FINAL_PATTERN"
                ]
                .notna()
                .any()
            )
            else np.nan,

        "n_adm3":
            n_adm3,

        "adm3_dominant_pattern":
            dominant_value(
                group[
                    "ADM3_PATTERN"
                ]
            ),

        "adm3_dominant_pattern_share":
            dominant_share(
                group[
                    "ADM3_PATTERN"
                ]
            ),

        "adm3_pattern_entropy":
            normalized_entropy(
                group[
                    "ADM3_PATTERN"
                ]
            ),

        "adm3_unique_pattern_count":
            int(
                group[
                    "ADM3_PATTERN"
                ]
                .nunique()
            ),

        "adm3_hotspot_count":
            int(
                group[
                    "ADM3_is_hotspot"
                ]
                .sum()
            ),

        "adm3_hotspot_share":
            float(
                group[
                    "ADM3_is_hotspot"
                ]
                .mean()
            ),

        "adm3_worsening_count":
            int(
                group[
                    "ADM3_is_worsening"
                ]
                .sum()
            ),

        "adm3_worsening_share":
            float(
                group[
                    "ADM3_is_worsening"
                ]
                .mean()
            ),

        "adm3_declining_count":
            int(
                group[
                    "ADM3_is_declining"
                ]
                .sum()
            ),

        "adm3_declining_share":
            float(
                group[
                    "ADM3_is_declining"
                ]
                .mean()
            ),

        "adm3_stable_count":
            int(
                group[
                    "ADM3_is_stable"
                ]
                .sum()
            ),

        "adm3_stable_share":
            float(
                group[
                    "ADM3_is_stable"
                ]
                .mean()
            ),

        "continuous_trend_opposite_count":
            int(
                group[
                    "continuous_trend_opposite"
                ]
                .sum()
            ),

        "mean_adm3_long_term_mean":
            group[
                "ADM3_long_term_mean"
            ].mean(),

        "mean_adm3_recent_minus_early":
            group[
                "ADM3_recent_minus_early"
            ].mean(),

        "mean_adm3_trend_slope":
            group[
                "ADM3_trend_slope"
            ].mean(),

        "max_adm3_long_term_mean":
            group[
                "ADM3_long_term_mean"
            ].max(),

        "max_adm3_recent_minus_early":
            group[
                "ADM3_recent_minus_early"
            ].max(),
    }

    # Copy ADM2 dimensions
    for dim in ALL_ADM2_DIMENSIONS:

        row[
            f"ADM2_{dim}"
        ] = int(
            group[
                dim
            ]
            .dropna()
            .iloc[0]
        ) if (
            group[
                dim
            ]
            .notna()
            .any()
        ) else 0

    # Sum each comparison flag within parent ADM2
    for flag in flag_columns:

        row[
            flag.replace(
                "flag_",
                ""
            )
            +
            "_count"
        ] = int(
            group[
                flag
            ]
            .sum()
        )

    summary_rows.append(
        row
    )


adm2_summary = pd.DataFrame(
    summary_rows
)


adm2_summary = (
    adm2_summary
    .sort_values(
        [
            "adm3_pattern_entropy",
            "adm3_hotspot_share",
        ],
        ascending=[
            False,
            False,
        ],
    )
)


adm2_summary.to_csv(
    OUTPUT_DIR
    / "04_adm2_cross_scale_summary_v5.csv",
    index=False,
)


# ============================================================================
# 22. IMPORTANT CASE TABLES
# ============================================================================

regional_chronic_local_hotspot = (
    cross[
        cross[
            "flag_regional_chronic_local_hotspot"
        ]
        ==
        1
    ]
    .copy()
)

regional_chronic_local_hotspot.to_csv(
    OUTPUT_DIR
    / "05_regional_chronic_local_hotspot_v5.csv",
    index=False,
)


regional_neutral_local_hotspot = (
    cross[
        cross[
            "flag_regional_neutral_local_hotspot"
        ]
        ==
        1
    ]
    .copy()
)

regional_neutral_local_hotspot.to_csv(
    OUTPUT_DIR
    / "06_regional_neutral_local_hotspot_v5.csv",
    index=False,
)


regional_accelerating_local_increase = (
    cross[
        cross[
            "flag_regional_accelerating_local_increase"
        ]
        ==
        1
    ]
    .copy()
)

regional_accelerating_local_increase.to_csv(
    OUTPUT_DIR
    / "07_regional_accelerating_local_increase_v5.csv",
    index=False,
)


regional_decline_local_worsening = (
    cross[
        cross[
            "flag_regional_decline_local_worsening"
        ]
        ==
        1
    ]
    .copy()
)

regional_decline_local_worsening.to_csv(
    OUTPUT_DIR
    / "08_regional_decline_local_worsening_v5.csv",
    index=False,
)


regional_volatile_local_hotspot = (
    cross[
        cross[
            "flag_regional_volatile_local_hotspot"
        ]
        ==
        1
    ]
    .copy()
)

regional_volatile_local_hotspot.to_csv(
    OUTPUT_DIR
    / "09_regional_volatile_local_hotspot_v5.csv",
    index=False,
)


regional_low_local_emerging = (
    cross[
        cross[
            "flag_regional_low_local_emerging"
        ]
        ==
        1
    ]
    .copy()
)

regional_low_local_emerging.to_csv(
    OUTPUT_DIR
    / "10_regional_low_local_emerging_v5.csv",
    index=False,
)


# ============================================================================
# 23. CROSS-SCALE DISAGREEMENT
# ============================================================================

disagreement = (
    cross[
        (
            cross[
                "flag_regional_decline_local_worsening"
            ]
            ==
            1
        )
        |
        (
            cross[
                "flag_regional_chronic_local_decline"
            ]
            ==
            1
        )
        |
        (
            cross[
                "flag_regional_accelerating_local_decline"
            ]
            ==
            1
        )
        |
        (
            cross[
                "continuous_trend_opposite"
            ]
            ==
            1
        )
    ]
    .copy()
)


disagreement.to_csv(
    OUTPUT_DIR
    / "11_cross_scale_disagreement_v5.csv",
    index=False,
)


# ============================================================================
# 24. HIGH-HETEROGENEITY ADM2
# ============================================================================

if len(
    adm2_summary
) > 0:

    entropy_q75 = (
        adm2_summary[
            "adm3_pattern_entropy"
        ]
        .quantile(
            0.75
        )
    )

else:

    entropy_q75 = np.nan


high_heterogeneity = (
    adm2_summary[
        adm2_summary[
            "adm3_pattern_entropy"
        ]
        >=
        entropy_q75
    ]
    .copy()
)


high_heterogeneity.to_csv(
    OUTPUT_DIR
    / "12_high_heterogeneity_adm2_v5.csv",
    index=False,
)


# ============================================================================
# 25. QC
# ============================================================================

qc = pd.DataFrame(
    {
        "metric": [
            "adm2_label_rows",
            "adm2_feature_rows",
            "adm2_unique_after_merge",
            "adm3_rows",
            "adm3_unique",
            "adm3_parent_adm2_unique",
            "adm3_parent_missing_after_merge",
            "adm2_no_dominant_pattern_count",
            "adm2_chronic_hotspot_count",
            "adm2_persistent_low_count",
            "adm2_accelerating_count",
            "adm2_declining_count",
            "adm2_high_volatility_count",
        ],

        "value": [
            len(
                adm2_labels
            ),

            len(
                adm2_features
            ),

            adm2[
                "ADM2_KEY"
            ]
            .nunique(),

            len(
                adm3
            ),

            adm3[
                "ADM3_ID"
            ]
            .nunique(),

            adm3[
                "ADM2_KEY"
            ]
            .nunique(),

            missing_parent,

            int(
                adm2[
                    "no_dominant_pattern"
                ]
                .sum()
            ),

            int(
                adm2[
                    "chronic_hotspot"
                ]
                .sum()
            ),

            int(
                adm2[
                    "persistent_low"
                ]
                .sum()
            ),

            int(
                adm2[
                    "accelerating"
                ]
                .sum()
            ),

            int(
                adm2[
                    "declining"
                ]
                .sum()
            ),

            int(
                adm2[
                    "high_volatility"
                ]
                .sum()
            ),
        ],
    }
)


qc.to_csv(
    OUTPUT_DIR
    / "14_cross_scale_qc_v5.csv",
    index=False,
)


# ============================================================================
# 26. TERMINAL REPORT
# ============================================================================

print("\n" + "=" * 80)
print("V5 CROSS-SCALE QC")
print("=" * 80)

print(
    qc.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("PRIMARY CROSS-SCALE PATTERN COUNTS")
print("=" * 80)

print(
    primary_counts.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("DIMENSION-BY-DIMENSION COUNTS")
print("=" * 80)

print(
    dimension_counts.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("TOP ADM2 BY INTERNAL ADM3 HETEROGENEITY")
print("=" * 80)

display_cols = [
    "ADM2_NAME",
    "ADM2_FINAL_PATTERN",
    "n_adm3",
    "adm3_dominant_pattern",
    "adm3_dominant_pattern_share",
    "adm3_pattern_entropy",
    "adm3_hotspot_share",
    "adm3_worsening_share",
    "adm3_declining_share",
]


print(
    adm2_summary[
        display_cols
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("KEY CASE COUNTS")
print("=" * 80)

print(
    "Regional chronic + local hotspot:",
    len(
        regional_chronic_local_hotspot
    ),
)

print(
    "Regional neutral + local hotspot:",
    len(
        regional_neutral_local_hotspot
    ),
)

print(
    "Regional accelerating + local increase:",
    len(
        regional_accelerating_local_increase
    ),
)

print(
    "Regional decline + local worsening:",
    len(
        regional_decline_local_worsening
    ),
)

print(
    "Regional volatile + local hotspot:",
    len(
        regional_volatile_local_hotspot
    ),
)

print(
    "Regional low + local emerging:",
    len(
        regional_low_local_emerging
    ),
)

print(
    "Cross-scale disagreement cases:",
    len(
        disagreement
    ),
)

print(
    "High-heterogeneity ADM2 regions:",
    len(
        high_heterogeneity
    ),
)


# ============================================================================
# 27. HARD QC
# ============================================================================

print("\n" + "=" * 80)
print("QC INTERPRETATION")
print("=" * 80)


if (
    adm2[
        "ADM2_KEY"
    ]
    .nunique()
    ==
    79
):

    print(
        "PASS: 79 unique ADM2 units available."
    )

else:

    print(
        "WARNING: ADM2 count is not 79."
    )


if (
    adm3[
        "ADM3_ID"
    ]
    .nunique()
    ==
    512
):

    print(
        "PASS: 512 unique ADM3 units available."
    )

else:

    print(
        "WARNING: ADM3 count is not 512."
    )


if missing_parent == 0:

    print(
        "PASS: Every ADM3 row matched to its parent ADM2."
    )

else:

    print(
        f"WARNING: {missing_parent} ADM3 rows did not match a parent ADM2."
    )

    print(
        "Inspect:"
    )

    print(
        OUTPUT_DIR
        / "15_unmatched_adm2_names_v5.csv"
    )


# ============================================================================
# 28. OUTPUT REPORT
# ============================================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

output_files = [
    "01_adm2_multilabel_dimensions_v5.csv",
    "02_adm3_cross_scale_comparison_v5.csv",
    "03_primary_relationship_counts_v5.csv",
    "04_adm2_cross_scale_summary_v5.csv",
    "05_regional_chronic_local_hotspot_v5.csv",
    "06_regional_neutral_local_hotspot_v5.csv",
    "07_regional_accelerating_local_increase_v5.csv",
    "08_regional_decline_local_worsening_v5.csv",
    "09_regional_volatile_local_hotspot_v5.csv",
    "10_regional_low_local_emerging_v5.csv",
    "11_cross_scale_disagreement_v5.csv",
    "12_high_heterogeneity_adm2_v5.csv",
    "13_dimension_alignment_counts_v5.csv",
    "14_cross_scale_qc_v5.csv",
    "15_unmatched_adm2_names_v5.csv",
]


for filename in output_files:

    path = (
        OUTPUT_DIR
        / filename
    )

    print(
        f"{filename}: "
        f"{'OK' if path.exists() else 'MISSING'}"
    )


print(
    "\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print(
    "\nV5 multi-label ADM2–ADM3 cross-scale analysis completed."
)
