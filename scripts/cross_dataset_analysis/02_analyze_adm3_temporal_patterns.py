"""
02_analyze_adm3_temporal_patterns.py

South Sudan Flood Analysis
ADM3 Temporal Pattern Analysis, 2000-2025

Input
-----
processed_data/flood_analysis/adm3/
    adm3_annual_complete_panel_combined.csv

Outputs
-------
processed_data/flood_analysis/adm3/temporal_patterns/

    adm3_temporal_pattern_summary.csv
    adm3_temporal_pattern_classification.csv
    adm3_temporal_pattern_counts.csv
    adm3_top_increasing.csv
    adm3_top_declining.csv
    adm3_top_persistent.csv
    adm3_top_emerging.csv

Main metrics
------------
long_term_mean
early_mean
recent_mean
recent_minus_early
recent_5y_mean
previous_5y_mean
recent_5y_change
trend_slope
trend_pvalue
trend_r2
max_flooded_percent
flood_year_count
flood_year_share
std_flooded_percent
cv_flooded_percent
peak_year
peak_value

Pattern classes
---------------
Persistent hotspot
Emerging hotspot
Increasing
Declining
Episodic / volatile
Stable moderate
Stable low

Notes
-----
This script uses annual_flooded_percent as the main temporal variable.

The complete ADM3 x year panel is required so that zero-flood years are
retained in the trend analysis.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# 1. PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(
    "/Users/shibai/Documents/GitHub/DBL-Group-22"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm3"
    / "adm3_annual_complete_panel_combined.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_data"
    / "flood_analysis"
    / "adm3"
    / "temporal_patterns"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

START_YEAR = 2000
END_YEAR = 2025

EARLY_START = 2000
EARLY_END = 2009

RECENT_START = 2016
RECENT_END = 2025

PREVIOUS_5Y_START = 2016
PREVIOUS_5Y_END = 2020

RECENT_5Y_START = 2021
RECENT_5Y_END = 2025


# ============================================================================
# 2. OPTIONAL SCIPY
# ============================================================================

try:
    from scipy.stats import linregress

    SCIPY_AVAILABLE = True

except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# 3. HELPERS
# ============================================================================

def require_file(
    path: Path,
    description: str,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


def safe_mean(
    series: pd.Series,
) -> float:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(
        values.mean()
    )


def calculate_linear_trend(
    years: pd.Series,
    values: pd.Series,
):
    """
    Linear annual trend:

        flooded_percent = intercept + slope * year

    Returns:
        slope
        pvalue
        r2
    """

    x = pd.to_numeric(
        years,
        errors="coerce",
    )

    y = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid = (
        x.notna()
        & y.notna()
    )

    x = x[valid].astype(float)
    y = y[valid].astype(float)

    if len(x) < 3:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    if y.nunique() <= 1:
        return (
            0.0,
            1.0,
            0.0,
        )

    if SCIPY_AVAILABLE:

        result = linregress(
            x,
            y,
        )

        return (
            float(result.slope),
            float(result.pvalue),
            float(result.rvalue ** 2),
        )

    # Fallback if scipy is not installed.
    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    predicted = (
        intercept
        + slope * x
    )

    ss_res = np.sum(
        (
            y
            - predicted
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y
            - y.mean()
        ) ** 2
    )

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = (
            1
            - ss_res
            / ss_tot
        )

    return (
        float(slope),
        np.nan,
        float(r2),
    )


def quantile_or_zero(
    series: pd.Series,
    q: float,
) -> float:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return 0.0

    return float(
        values.quantile(q)
    )


# ============================================================================
# 4. LOAD DATA
# ============================================================================

print("=" * 78)
print("LOAD ADM3 COMPLETE ANNUAL PANEL")
print("=" * 78)

require_file(
    INPUT_FILE,
    "ADM3 annual complete panel",
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Input file:\n{INPUT_FILE}"
)

print(
    f"\nRows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns):,}"
)


required_columns = [
    "ADM3_ID",
    "ADM3_NAME",
    "year",
    "annual_flooded_percent",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )


# ============================================================================
# 5. BASIC CLEANING
# ============================================================================

df[
    "year"
] = pd.to_numeric(
    df[
        "year"
    ],
    errors="coerce",
)


df[
    "annual_flooded_percent"
] = pd.to_numeric(
    df[
        "annual_flooded_percent"
    ],
    errors="coerce",
)


df = df[
    df[
        "year"
    ].between(
        START_YEAR,
        END_YEAR,
    )
].copy()


df = df.sort_values(
    [
        "ADM3_ID",
        "year",
    ]
)


print(
    "\nADM3 units:",
    df[
        "ADM3_ID"
    ].nunique(),
)


print(
    "Years:",
    int(
        df[
            "year"
        ].min()
    ),
    "-",
    int(
        df[
            "year"
        ].max()
    ),
)


# ============================================================================
# 6. BUILD TEMPORAL SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("BUILD ADM3 TEMPORAL SUMMARY")
print("=" * 78)


summary_rows = []


identity_columns = [
    "ADM0_NAME",
    "ADM1_NAME",
    "ADM2_NAME",
    "ADM3_NAME",
]


for adm3_id, group in df.groupby(
    "ADM3_ID"
):

    group = group.sort_values(
        "year"
    ).copy()

    values = (
        group[
            "annual_flooded_percent"
        ]
        .astype(float)
    )

    years = (
        group[
            "year"
        ]
        .astype(int)
    )

    identity = {
        "ADM3_ID":
            adm3_id
    }

    for column in identity_columns:

        if column in group.columns:

            identity[
                column
            ] = (
                group[
                    column
                ]
                .dropna()
                .astype(str)
                .iloc[0]
                if group[
                    column
                ].notna()
                .any()
                else np.nan
            )


    long_term_mean = safe_mean(
        values
    )


    early_mask = years.between(
        EARLY_START,
        EARLY_END,
    )

    recent_mask = years.between(
        RECENT_START,
        RECENT_END,
    )


    early_mean = safe_mean(
        values[
            early_mask
        ]
    )

    recent_mean = safe_mean(
        values[
            recent_mask
        ]
    )


    recent_minus_early = (
        recent_mean
        - early_mean
    )


    previous_5y_mask = years.between(
        PREVIOUS_5Y_START,
        PREVIOUS_5Y_END,
    )

    recent_5y_mask = years.between(
        RECENT_5Y_START,
        RECENT_5Y_END,
    )


    previous_5y_mean = safe_mean(
        values[
            previous_5y_mask
        ]
    )

    recent_5y_mean = safe_mean(
        values[
            recent_5y_mask
        ]
    )


    recent_5y_change = (
        recent_5y_mean
        - previous_5y_mean
    )


    (
        trend_slope,
        trend_pvalue,
        trend_r2,
    ) = calculate_linear_trend(
        years,
        values,
    )


    flood_year_count = int(
        (
            values > 0
        ).sum()
    )


    flood_year_share = (
        flood_year_count
        / len(group)
    )


    max_flooded_percent = float(
        values.max()
    )


    min_flooded_percent = float(
        values.min()
    )


    median_flooded_percent = float(
        values.median()
    )


    std_flooded_percent = float(
        values.std(
            ddof=1
        )
    )


    if (
        long_term_mean is not None
        and
        not np.isnan(
            long_term_mean
        )
        and
        abs(
            long_term_mean
        ) > 1e-12
    ):

        cv_flooded_percent = (
            std_flooded_percent
            / long_term_mean
        )

    else:

        cv_flooded_percent = np.nan


    peak_index = (
        values.idxmax()
    )

    peak_year = int(
        group.loc[
            peak_index,
            "year",
        ]
    )


    peak_value = float(
        group.loc[
            peak_index,
            "annual_flooded_percent",
        ]
    )


    # Number of years >= ADM3's own long-term average.
    years_above_own_mean = int(
        (
            values
            >= long_term_mean
        ).sum()
    )


    summary_rows.append(
        {
            **identity,

            "n_years":
                len(group),

            "long_term_mean":
                long_term_mean,

            "median_flooded_percent":
                median_flooded_percent,

            "early_mean":
                early_mean,

            "recent_mean":
                recent_mean,

            "recent_minus_early":
                recent_minus_early,

            "previous_5y_mean":
                previous_5y_mean,

            "recent_5y_mean":
                recent_5y_mean,

            "recent_5y_change":
                recent_5y_change,

            "trend_slope":
                trend_slope,

            "trend_pvalue":
                trend_pvalue,

            "trend_r2":
                trend_r2,

            "max_flooded_percent":
                max_flooded_percent,

            "min_flooded_percent":
                min_flooded_percent,

            "std_flooded_percent":
                std_flooded_percent,

            "cv_flooded_percent":
                cv_flooded_percent,

            "flood_year_count":
                flood_year_count,

            "flood_year_share":
                flood_year_share,

            "years_above_own_mean":
                years_above_own_mean,

            "peak_year":
                peak_year,

            "peak_value":
                peak_value,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================================
# 7. DATA-DRIVEN THRESHOLDS
# ============================================================================

print("\n" + "=" * 78)
print("CALCULATE DATA-DRIVEN THRESHOLDS")
print("=" * 78)


# Exposure thresholds
LONG_TERM_HIGH = quantile_or_zero(
    summary[
        "long_term_mean"
    ],
    0.75,
)

RECENT_HIGH = quantile_or_zero(
    summary[
        "recent_mean"
    ],
    0.75,
)


# Change thresholds
CHANGE_HIGH = quantile_or_zero(
    summary[
        "recent_minus_early"
    ],
    0.75,
)

CHANGE_LOW = quantile_or_zero(
    summary[
        "recent_minus_early"
    ],
    0.25,
)


SLOPE_HIGH = quantile_or_zero(
    summary[
        "trend_slope"
    ],
    0.75,
)

SLOPE_LOW = quantile_or_zero(
    summary[
        "trend_slope"
    ],
    0.25,
)


VOLATILITY_HIGH = quantile_or_zero(
    summary[
        "std_flooded_percent"
    ],
    0.75,
)


PERSISTENCE_HIGH = quantile_or_zero(
    summary[
        "flood_year_share"
    ],
    0.75,
)


print(
    f"Long-term high threshold: "
    f"{LONG_TERM_HIGH:.4f}"
)

print(
    f"Recent high threshold: "
    f"{RECENT_HIGH:.4f}"
)

print(
    f"Recent-minus-early Q75: "
    f"{CHANGE_HIGH:.4f}"
)

print(
    f"Recent-minus-early Q25: "
    f"{CHANGE_LOW:.4f}"
)

print(
    f"Trend slope Q75: "
    f"{SLOPE_HIGH:.4f}"
)

print(
    f"Trend slope Q25: "
    f"{SLOPE_LOW:.4f}"
)

print(
    f"Volatility Q75: "
    f"{VOLATILITY_HIGH:.4f}"
)

print(
    f"Flood-year share Q75: "
    f"{PERSISTENCE_HIGH:.4f}"
)


# ============================================================================
# 8. TEMPORAL PATTERN CLASSIFICATION
# ============================================================================

print("\n" + "=" * 78)
print("CLASSIFY TEMPORAL PATTERNS")
print("=" * 78)


def classify_pattern(
    row,
) -> str:
    """
    Rule-based, data-driven classification.

    Priority matters:
    persistent and emerging patterns are evaluated before
    generic increasing/decreasing classes.
    """

    long_term = row[
        "long_term_mean"
    ]

    recent = row[
        "recent_mean"
    ]

    delta = row[
        "recent_minus_early"
    ]

    slope = row[
        "trend_slope"
    ]

    volatility = row[
        "std_flooded_percent"
    ]

    flood_share = row[
        "flood_year_share"
    ]


    # --------------------------------------------------------
    # Persistent hotspot
    # --------------------------------------------------------

    if (
        long_term >= LONG_TERM_HIGH
        and
        flood_share >= PERSISTENCE_HIGH
        and
        recent >= RECENT_HIGH
    ):

        return (
            "Persistent hotspot"
        )


    # --------------------------------------------------------
    # Emerging hotspot
    # --------------------------------------------------------

    if (
        recent >= RECENT_HIGH
        and
        delta >= CHANGE_HIGH
        and
        slope > 0
        and
        long_term < LONG_TERM_HIGH
    ):

        return (
            "Emerging hotspot"
        )


    # --------------------------------------------------------
    # Increasing
    # --------------------------------------------------------

    if (
        delta >= CHANGE_HIGH
        and
        slope >= SLOPE_HIGH
    ):

        return (
            "Increasing"
        )


    # --------------------------------------------------------
    # Declining
    # --------------------------------------------------------

    if (
        delta <= CHANGE_LOW
        and
        slope <= SLOPE_LOW
        and
        slope < 0
    ):

        return (
            "Declining"
        )


    # --------------------------------------------------------
    # Episodic / volatile
    # --------------------------------------------------------

    if (
        volatility >= VOLATILITY_HIGH
        and
        flood_share < PERSISTENCE_HIGH
    ):

        return (
            "Episodic / volatile"
        )


    # --------------------------------------------------------
    # Stable moderate
    # --------------------------------------------------------

    if (
        long_term > 0
        and
        long_term >= summary[
            "long_term_mean"
        ].median()
    ):

        return (
            "Stable moderate"
        )


    # --------------------------------------------------------
    # Stable low
    # --------------------------------------------------------

    return (
        "Stable low"
    )


summary[
    "temporal_pattern"
] = summary.apply(
    classify_pattern,
    axis=1,
)


# ============================================================================
# 9. ADD INTERPRETATION FLAGS
# ============================================================================

summary[
    "significant_positive_trend"
] = (
    (
        summary[
            "trend_slope"
        ] > 0
    )
    &
    (
        summary[
            "trend_pvalue"
        ] < 0.05
    )
)


summary[
    "significant_negative_trend"
] = (
    (
        summary[
            "trend_slope"
        ] < 0
    )
    &
    (
        summary[
            "trend_pvalue"
        ] < 0.05
    )
)


summary[
    "recently_worse"
] = (
    summary[
        "recent_minus_early"
    ] > 0
)


summary[
    "recent_5y_worse"
] = (
    summary[
        "recent_5y_change"
    ] > 0
)


# ============================================================================
# 10. SAVE MAIN RESULTS
# ============================================================================

summary = summary.sort_values(
    [
        "temporal_pattern",
        "recent_mean",
    ],
    ascending=[
        True,
        False,
    ],
)


summary_output = (
    OUTPUT_DIR
    / "adm3_temporal_pattern_summary.csv"
)


summary.to_csv(
    summary_output,
    index=False,
)


classification_columns = [
    column
    for column in [
        "ADM3_ID",
        "ADM0_NAME",
        "ADM1_NAME",
        "ADM2_NAME",
        "ADM3_NAME",
        "temporal_pattern",
        "long_term_mean",
        "early_mean",
        "recent_mean",
        "recent_minus_early",
        "trend_slope",
        "trend_pvalue",
        "flood_year_count",
        "flood_year_share",
        "max_flooded_percent",
        "peak_year",
    ]
    if column in summary.columns
]


summary[
    classification_columns
].to_csv(
    OUTPUT_DIR
    / "adm3_temporal_pattern_classification.csv",
    index=False,
)


# ============================================================================
# 11. PATTERN COUNTS
# ============================================================================

pattern_counts = (
    summary[
        "temporal_pattern"
    ]
    .value_counts()
    .rename_axis(
        "temporal_pattern"
    )
    .reset_index(
        name="adm3_count"
    )
)


pattern_counts[
    "share_percent"
] = (
    pattern_counts[
        "adm3_count"
    ]
    /
    len(summary)
    * 100
)


pattern_counts.to_csv(
    OUTPUT_DIR
    / "adm3_temporal_pattern_counts.csv",
    index=False,
)


# ============================================================================
# 12. TOP LISTS
# ============================================================================

top_increasing = (
    summary
    .sort_values(
        [
            "trend_slope",
            "recent_minus_early",
        ],
        ascending=False,
    )
    .head(30)
)


top_increasing.to_csv(
    OUTPUT_DIR
    / "adm3_top_increasing.csv",
    index=False,
)


top_declining = (
    summary
    .sort_values(
        [
            "trend_slope",
            "recent_minus_early",
        ],
        ascending=True,
    )
    .head(30)
)


top_declining.to_csv(
    OUTPUT_DIR
    / "adm3_top_declining.csv",
    index=False,
)


top_persistent = (
    summary[
        summary[
            "temporal_pattern"
        ]
        ==
        "Persistent hotspot"
    ]
    .sort_values(
        [
            "long_term_mean",
            "flood_year_share",
        ],
        ascending=False,
    )
)


top_persistent.to_csv(
    OUTPUT_DIR
    / "adm3_top_persistent.csv",
    index=False,
)


top_emerging = (
    summary[
        summary[
            "temporal_pattern"
        ]
        ==
        "Emerging hotspot"
    ]
    .sort_values(
        [
            "recent_minus_early",
            "recent_mean",
        ],
        ascending=False,
    )
)


top_emerging.to_csv(
    OUTPUT_DIR
    / "adm3_top_emerging.csv",
    index=False,
)


# ============================================================================
# 13. SAVE THRESHOLDS
# ============================================================================

thresholds = pd.DataFrame(
    {
        "threshold": [
            "LONG_TERM_HIGH_Q75",
            "RECENT_HIGH_Q75",
            "CHANGE_HIGH_Q75",
            "CHANGE_LOW_Q25",
            "SLOPE_HIGH_Q75",
            "SLOPE_LOW_Q25",
            "VOLATILITY_HIGH_Q75",
            "PERSISTENCE_HIGH_Q75",
        ],

        "value": [
            LONG_TERM_HIGH,
            RECENT_HIGH,
            CHANGE_HIGH,
            CHANGE_LOW,
            SLOPE_HIGH,
            SLOPE_LOW,
            VOLATILITY_HIGH,
            PERSISTENCE_HIGH,
        ],
    }
)


thresholds.to_csv(
    OUTPUT_DIR
    / "adm3_temporal_pattern_thresholds.csv",
    index=False,
)


# ============================================================================
# 14. QUALITY CHECK
# ============================================================================

print("\n" + "=" * 78)
print("QUALITY CHECK")
print("=" * 78)


print(
    "ADM3 summary rows:",
    len(summary),
)


print(
    "Unique ADM3:",
    summary[
        "ADM3_ID"
    ].nunique(),
)


print(
    "\nPattern counts:"
)


print(
    pattern_counts.to_string(
        index=False
    )
)


print(
    "\nPositive statistically significant trends:",
    int(
        summary[
            "significant_positive_trend"
        ].sum()
    ),
)


print(
    "Negative statistically significant trends:",
    int(
        summary[
            "significant_negative_trend"
        ].sum()
    ),
)


print(
    "\nLargest positive slope:"
)


display_cols = [
    column
    for column in [
        "ADM2_NAME",
        "ADM3_NAME",
        "trend_slope",
        "recent_minus_early",
        "recent_mean",
        "temporal_pattern",
    ]
    if column in summary.columns
]


print(
    summary
    .sort_values(
        "trend_slope",
        ascending=False,
    )[
        display_cols
    ]
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nLargest negative slope:"
)


print(
    summary
    .sort_values(
        "trend_slope",
        ascending=True,
    )[
        display_cols
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================================
# 15. OUTPUT SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("OUTPUT FILES")
print("=" * 78)


output_files = [
    "adm3_temporal_pattern_summary.csv",
    "adm3_temporal_pattern_classification.csv",
    "adm3_temporal_pattern_counts.csv",
    "adm3_temporal_pattern_thresholds.csv",
    "adm3_top_increasing.csv",
    "adm3_top_declining.csv",
    "adm3_top_persistent.csv",
    "adm3_top_emerging.csv",
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
    "\nADM3 temporal pattern analysis completed."
)
