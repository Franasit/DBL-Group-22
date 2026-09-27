"""
06_generate_ipc_flood_overlap_figures.py

Purpose
-------
Generate a full IPC × Flood figure suite at ADM2 and ADM3 level.

Boundary files are fixed to:

    raw_data/administrative_boundaries/ssd_admin2.geojson
    raw_data/administrative_boundaries/ssd_admin3.geojson

Main questions
--------------
1. Where are IPC Phase 3+ areas?
2. Where are flood hotspots?
3. Where do IPC and flood exposure overlap?
4. How do IPC phases change between Current / Projected periods?
5. Which ADM3 areas are most flood-exposed inside IPC P3+ ADM2 areas?
6. How do rainfall and runoff relate to flood exposure?
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

CROSS_DATA_DIR = (
    PROCESSED_DATA_DIR
    / "cross_dataset"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "output"
    / "figures"
)

FIGURE_DIR = (
    OUTPUT_ROOT
    / "ipc_flood_overlap"
)

ADM2_FIG_DIR = (
    FIGURE_DIR
    / "ADM2"
)

ADM3_FIG_DIR = (
    FIGURE_DIR
    / "ADM3"
)

COMPARISON_FIG_DIR = (
    FIGURE_DIR
    / "comparison"
)

for folder in [
    FIGURE_DIR,
    ADM2_FIG_DIR,
    ADM3_FIG_DIR,
    COMPARISON_FIG_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 2. FIXED BOUNDARY PATHS
# ============================================================

BOUNDARY_DIR = (
    RAW_DATA_DIR
    / "administrative_boundaries"
)

ADM2_BOUNDARY_FILE = (
    BOUNDARY_DIR
    / "ssd_admin2.geojson"
)

ADM3_BOUNDARY_FILE = (
    BOUNDARY_DIR
    / "ssd_admin3.geojson"
)


# ============================================================
# 3. CROSS-DATA INPUTS
# ============================================================

ADM2_IPC_FLOOD_FILE = (
    CROSS_DATA_DIR
    / "IPC_flood_ADM2_period.csv"
)

ADM3_IPC_FLOOD_FILE = (
    CROSS_DATA_DIR
    / "IPC_flood_ADM3_period.csv"
)

ADM2_FLOOD_SUMMARY_FILE = (
    CROSS_DATA_DIR
    / "flood_adm2_area_summary.csv"
)

ADM3_FLOOD_SUMMARY_FILE = (
    CROSS_DATA_DIR
    / "flood_adm3_area_summary.csv"
)


# ============================================================
# 4. HYDROLOGY FILE CANDIDATES
# ============================================================

HYDROLOGY_CANDIDATES = [

    PROJECT_ROOT
    / "rainfall_and_runoff_processed"
    / "ADM2_daily_rainfall_runoff.csv",

    PROCESSED_DATA_DIR
    / "rainfall_and_runoff"
    / "ADM2_daily_rainfall_runoff.csv",

    PROCESSED_DATA_DIR
    / "cross_dataset"
    / "ADM2_daily_rainfall_runoff.csv",
]


# ============================================================
# 5. IPC COLORS
# ============================================================

IPC_COLORS = {
    1: "#CDFACD",
    2: "#FFF200",
    3: "#F7941D",
    4: "#ED1C24",
    5: "#6D0019",
}

IPC_LABELS = {
    1: "P1 Minimal",
    2: "P2 Stressed",
    3: "P3 Crisis",
    4: "P4 Emergency",
    5: "P5 Catastrophe/Famine",
}


# ============================================================
# 6. GENERAL HELPERS
# ============================================================

def normalize_column_name(name: str) -> str:

    name = str(name).strip().lower()

    name = name.replace("%", " percentage ")
    name = name.replace("/", " ")
    name = name.replace("-", " ")
    name = name.replace(".", " ")

    name = re.sub(
        r"[^\w\s]",
        "",
        name,
    )

    name = re.sub(
        r"\s+",
        "_",
        name,
    )

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    return name.strip("_")


def make_unique_column_names(columns):

    seen = {}
    output = []

    for col in columns:

        if col not in seen:

            seen[col] = 1
            output.append(col)

        else:

            seen[col] += 1

            output.append(
                f"{col}_{seen[col]}"
            )

    return output


def normalize_columns(df):

    df = df.copy()

    cols = [
        normalize_column_name(c)
        for c in df.columns
    ]

    df.columns = (
        make_unique_column_names(
            cols
        )
    )

    return df


def save_figure(path):

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# 7. CHECK REQUIRED FILE
# ============================================================

def require_file(
    path: Path,
    description: str,
):

    if not path.exists():

        raise FileNotFoundError(
            f"\n{description} not found:\n"
            f"{path}"
        )

    print(
        f"\n{description}:"
    )

    print(
        f"  {path}"
    )


# ============================================================
# 8. DETECT BOUNDARY COLUMNS
# ============================================================

def detect_name_column(
    gdf,
    level,
):

    level_number = (
        "2"
        if level == "adm2"
        else "3"
    )

    candidates = [
        f"{level}_name",
        f"{level}_en",
        f"name_{level}",
        f"name_{level_number}",
        f"adm{level_number}_name",
        f"adm{level_number}_en",
        f"admin{level_number}_name",
        "name",
    ]

    for col in candidates:

        if col in gdf.columns:
            return col

    # fallback
    for col in gdf.columns:

        col_lower = col.lower()

        if (
            (
                level in col_lower
                or
                f"adm{level_number}"
                in col_lower
                or
                f"admin{level_number}"
                in col_lower
            )
            and
            (
                "name" in col_lower
                or
                "_en" in col_lower
            )
        ):

            return col

    raise ValueError(
        f"\nCould not detect {level.upper()} "
        f"name column.\n\n"
        f"Available columns:\n"
        f"{list(gdf.columns)}"
    )


def detect_id_column(
    gdf,
    level,
):

    level_number = (
        "2"
        if level == "adm2"
        else "3"
    )

    candidates = [
        f"{level}_id",
        f"{level}_code",
        f"{level}_pcode",

        f"adm{level_number}_id",
        f"adm{level_number}_code",
        f"adm{level_number}_pcode",

        f"admin{level_number}_id",
        f"admin{level_number}_code",

        f"gid_{level_number}",
        f"pcode_{level_number}",
    ]

    for col in candidates:

        if col in gdf.columns:
            return col

    return None


# ============================================================
# 9. LOAD BOUNDARY
# ============================================================

def load_boundary(
    path,
    level,
):

    require_file(
        path,
        f"{level.upper()} boundary",
    )

    gdf = gpd.read_file(
        path
    )

    gdf = normalize_columns(
        gdf
    )

    print(
        f"\n{level.upper()} raw columns:"
    )

    for col in gdf.columns:
        print(f"  {col}")

    if gdf.crs is None:

        print(
            "\nWARNING: Boundary CRS is missing."
        )

        print(
            "Assuming EPSG:4326."
        )

        gdf = gdf.set_crs(
            "EPSG:4326"
        )

    else:

        gdf = gdf.to_crs(
            "EPSG:4326"
        )

    name_col = detect_name_column(
        gdf,
        level,
    )

    id_col = detect_id_column(
        gdf,
        level,
    )

    print(
        f"\nDetected {level.upper()} name column:"
        f" {name_col}"
    )

    print(
        f"Detected {level.upper()} ID column:"
        f" {id_col}"
    )

    gdf[
        f"{level}_name"
    ] = (
        gdf[name_col]
        .astype(str)
        .str.strip()
    )

    if id_col is not None:

        gdf[
            f"{level}_id"
        ] = (
            gdf[id_col]
            .astype(str)
        )

    else:

        print(
            f"\nWARNING: No {level.upper()} ID found."
        )

        print(
            "Creating temporary technical IDs."
        )

        gdf[
            f"{level}_id"
        ] = [
            f"{level.upper()}_{i:03d}"
            for i
            in range(
                1,
                len(gdf) + 1,
            )
        ]

    print(
        f"\n{level.upper()} units loaded:"
        f" {len(gdf):,}"
    )

    return gdf


# ============================================================
# 10. LOAD CSV
# ============================================================

def load_csv(
    path,
    label,
    required=False,
):

    if not path.exists():

        print(
            f"\nWARNING: {label} not found:"
        )

        print(
            path
        )

        if required:

            raise FileNotFoundError(
                f"\nRequired file missing:\n"
                f"{path}"
            )

        return None

    print(
        f"\nLoading {label}:"
    )

    print(
        path
    )

    df = pd.read_csv(
        path
    )

    df = normalize_columns(
        df
    )

    print(
        f"Rows: {len(df):,}"
    )

    return df


# ============================================================
# 11. FIELD DETECTION
# ============================================================

def find_period_column(df):

    if df is None:
        return None

    candidates = [
        "period_type",
        "period",
        "ipc_period",
        "analysis_period",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


def find_phase_column(df):

    if df is None:
        return None

    candidates = [
        "phase",
        "phase_clean",
        "ipc_phase",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


def match_period_name(
    series,
    wanted,
):

    normalized = (
        series
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(
            " ",
            "_",
            regex=False,
        )
    )

    wanted = (
        wanted
        .lower()
        .replace(
            " ",
            "_",
        )
    )

    return (
        normalized
        == wanted
    )


# ============================================================
# 12. MERGE ADMIN TABLE
# ============================================================

def merge_admin_table(
    boundary,
    table,
    level,
):

    if table is None:
        return boundary.copy()

    id_col = (
        f"{level}_id"
    )

    name_col = (
        f"{level}_name"
    )

    # Prefer ID
    if (
        id_col in boundary.columns
        and
        id_col in table.columns
    ):

        left = boundary.copy()
        right = table.copy()

        left[id_col] = (
            left[id_col]
            .astype(str)
        )

        right[id_col] = (
            right[id_col]
            .astype(str)
        )

        return left.merge(
            right,
            on=id_col,
            how="left",
            suffixes=(
                "",
                "_data",
            ),
        )

    # Fallback to names
    if (
        name_col in boundary.columns
        and
        name_col in table.columns
    ):

        left = boundary.copy()
        right = table.copy()

        left[
            "_join_name"
        ] = (
            left[name_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        right[
            "_join_name"
        ] = (
            right[name_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        return left.merge(
            right,
            on="_join_name",
            how="left",
            suffixes=(
                "",
                "_data",
            ),
        )

    raise ValueError(
        f"\nCannot merge {level.upper()} data.\n"
        f"Boundary columns:\n"
        f"{list(boundary.columns)}\n\n"
        f"Data columns:\n"
        f"{list(table.columns)}"
    )


# ============================================================
# 13. IPC MAP
# ============================================================

def plot_ipc_phase_map(
    boundary,
    ipc,
    level,
    period,
    output_path,
):

    if ipc is None:
        return

    period_col = find_period_column(
        ipc
    )

    phase_col = find_phase_column(
        ipc
    )

    if (
        period_col is None
        or
        phase_col is None
    ):

        print(
            f"Skipping {period}: "
            "phase/period column missing."
        )

        return

    subset = ipc[
        match_period_name(
            ipc[
                period_col
            ],
            period,
        )
    ].copy()

    if len(subset) == 0:

        print(
            f"No IPC rows found for {period}"
        )

        return

    subset[
        phase_col
    ] = pd.to_numeric(
        subset[
            phase_col
        ],
        errors="coerce",
    )

    id_col = (
        f"{level}_id"
    )

    if id_col in subset.columns:

        subset = (
            subset
            .sort_values(
                phase_col,
                ascending=False,
            )
            .drop_duplicates(
                id_col
            )
        )

    gdf = merge_admin_table(
        boundary,
        subset,
        level,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            12,
        )
    )

    gdf.plot(
        ax=ax,
        color="#eeeeee",
        edgecolor="white",
        linewidth=0.4,
    )

    for phase in [
        1,
        2,
        3,
        4,
        5,
    ]:

        part = gdf[
            gdf[
                phase_col
            ]
            == phase
        ]

        if len(part) == 0:
            continue

        part.plot(
            ax=ax,
            color=IPC_COLORS[
                phase
            ],
            edgecolor="white",
            linewidth=0.5,
        )

    legend = [
        Patch(
            facecolor=IPC_COLORS[p],
            edgecolor="black",
            label=IPC_LABELS[p],
        )
        for p
        in IPC_COLORS
    ]

    ax.legend(
        handles=legend,
        loc="lower left",
        fontsize=8,
    )

    ax.set_title(
        f"{level.upper()} IPC Phase – "
        f"{period.replace('_', ' ').title()}",
        fontsize=14,
        weight="bold",
    )

    ax.axis("off")

    save_figure(
        output_path
    )


# ============================================================
# 14. CONTINUOUS FLOOD MAP
# ============================================================

def plot_continuous_map(
    boundary,
    table,
    level,
    variable,
    title,
    output_path,
):

    if table is None:
        return

    if variable not in table.columns:

        print(
            f"Skipping {title}: "
            f"{variable} missing."
        )

        return

    gdf = merge_admin_table(
        boundary,
        table,
        level,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            12,
        )
    )

    gdf.plot(
        column=variable,
        cmap="viridis",
        legend=True,
        ax=ax,
        edgecolor="white",
        linewidth=0.3,
        missing_kwds={
            "color": "#eeeeee",
            "label": "No data",
        },
    )

    ax.set_title(
        title,
        fontsize=14,
        weight="bold",
    )

    ax.axis("off")

    save_figure(
        output_path
    )


# ============================================================
# 15. IPC × FLOOD OVERLAP
# ============================================================

def plot_overlap_map(
    boundary,
    ipc,
    level,
    period,
    output_path,
):

    if ipc is None:
        return

    period_col = find_period_column(
        ipc
    )

    phase_col = find_phase_column(
        ipc
    )

    if (
        period_col is None
        or
        phase_col is None
    ):
        return

    subset = ipc[
        match_period_name(
            ipc[
                period_col
            ],
            period,
        )
    ].copy()

    if len(subset) == 0:
        return

    subset[
        phase_col
    ] = pd.to_numeric(
        subset[
            phase_col
        ],
        errors="coerce",
    )

    flood_col = next(
        (
            c
            for c
            in [
                "max_flooded_percentage",
                "mean_flooded_percentage",
                "flood_day_rate_percentage",
                "flood_days",
            ]
            if c
            in subset.columns
        ),
        None,
    )

    if flood_col is None:

        print(
            "No flood metric found "
            "inside IPC × flood table."
        )

        return

    subset[
        flood_col
    ] = pd.to_numeric(
        subset[
            flood_col
        ],
        errors="coerce",
    )

    flood_threshold = (
        subset[
            flood_col
        ]
        .median()
    )

    subset[
        "p3plus"
    ] = (
        subset[
            phase_col
        ]
        >= 3
    )

    subset[
        "high_flood"
    ] = (
        subset[
            flood_col
        ]
        >= flood_threshold
    )

    subset[
        "overlap_class"
    ] = "Neither"

    subset.loc[
        (
            subset[
                "p3plus"
            ]
            &
            ~subset[
                "high_flood"
            ]
        ),
        "overlap_class",
    ] = "P3+ only"

    subset.loc[
        (
            ~subset[
                "p3plus"
            ]
            &
            subset[
                "high_flood"
            ]
        ),
        "overlap_class",
    ] = "High flood only"

    subset.loc[
        (
            subset[
                "p3plus"
            ]
            &
            subset[
                "high_flood"
            ]
        ),
        "overlap_class",
    ] = "P3+ + High flood"

    id_col = (
        f"{level}_id"
    )

    if id_col in subset.columns:

        subset = (
            subset
            .drop_duplicates(
                id_col
            )
        )

    gdf = merge_admin_table(
        boundary,
        subset,
        level,
    )

    colors = {
        "Neither":
            "#eeeeee",

        "P3+ only":
            "#F7941D",

        "High flood only":
            "#4393c3",

        "P3+ + High flood":
            "#7b3294",
    }

    fig, ax = plt.subplots(
        figsize=(
            10,
            12,
        )
    )

    for (
        category,
        color,
    ) in colors.items():

        part = gdf[
            gdf[
                "overlap_class"
            ]
            == category
        ]

        if len(part) == 0:
            continue

        part.plot(
            ax=ax,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )

    handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            label=label,
        )
        for (
            label,
            color,
        )
        in colors.items()
    ]

    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=8,
    )

    ax.set_title(
        f"{level.upper()} IPC P3+ × Flood Overlap\n"
        f"{period.replace('_', ' ').title()}",
        fontsize=14,
        weight="bold",
    )

    ax.axis("off")

    save_figure(
        output_path
    )


# ============================================================
# 16. SCATTER
# ============================================================

def plot_scatter(
    df,
    x,
    y,
    label,
    title,
    xlabel,
    ylabel,
    output_path,
):

    if df is None:
        return

    required = [
        x,
        y,
        label,
    ]

    if not all(
        col in df.columns
        for col in required
    ):
        return

    temp = df[
        required
    ].copy()

    temp[x] = pd.to_numeric(
        temp[x],
        errors="coerce",
    )

    temp[y] = pd.to_numeric(
        temp[y],
        errors="coerce",
    )

    temp = temp.dropna(
        subset=[
            x,
            y,
        ]
    )

    if len(temp) == 0:
        return

    fig, ax = plt.subplots(
        figsize=(
            10,
            7,
        )
    )

    ax.scatter(
        temp[x],
        temp[y],
        alpha=0.7,
    )

    temp[
        "_rank"
    ] = (
        temp[x]
        .rank(
            pct=True
        )
        +
        temp[y]
        .rank(
            pct=True
        )
    )

    top = (
        temp
        .nlargest(
            min(
                8,
                len(temp),
            ),
            "_rank",
        )
    )

    for _, row in top.iterrows():

        ax.annotate(
            str(
                row[
                    label
                ]
            ),
            (
                row[x],
                row[y],
            ),
            fontsize=8,
            xytext=(
                4,
                4,
            ),
            textcoords="offset points",
        )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title,
        fontsize=14,
        weight="bold",
    )

    ax.grid(
        alpha=0.2
    )

    save_figure(
        output_path
    )


# ============================================================
# 17. IPC TRANSITION HEATMAP
# ============================================================

def plot_ipc_transition(
    df,
    output_path,
):

    if df is None:
        return

    period_col = find_period_column(
        df
    )

    phase_col = find_phase_column(
        df
    )

    if (
        period_col is None
        or
        phase_col is None
        or
        "adm2_name"
        not in df.columns
    ):
        return

    temp = df[
        [
            "adm2_name",
            period_col,
            phase_col,
        ]
    ].copy()

    temp[
        phase_col
    ] = pd.to_numeric(
        temp[
            phase_col
        ],
        errors="coerce",
    )

    pivot = temp.pivot_table(
        index="adm2_name",
        columns=period_col,
        values=phase_col,
        aggfunc="max",
    )

    wanted_order = [
        "current",
        "projected_1",
        "projected_2",
    ]

    ordered = []

    for wanted in wanted_order:

        for col in pivot.columns:

            standardized = (
                str(col)
                .lower()
                .replace(
                    " ",
                    "_",
                )
            )

            if standardized == wanted:

                ordered.append(
                    col
                )

    if ordered:

        pivot = (
            pivot[
                ordered
            ]
        )

    pivot[
        "_severity"
    ] = (
        pivot.max(
            axis=1
        )
    )

    pivot = (
        pivot
        .sort_values(
            "_severity",
            ascending=False,
        )
        .drop(
            columns=[
                "_severity"
            ]
        )
    )

    cmap = ListedColormap(
        [
            IPC_COLORS[1],
            IPC_COLORS[2],
            IPC_COLORS[3],
            IPC_COLORS[4],
            IPC_COLORS[5],
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            max(
                10,
                len(pivot)
                * 0.22,
            ),
        )
    )

    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        cmap=cmap,
        vmin=1,
        vmax=5,
    )

    ax.set_yticks(
        np.arange(
            len(pivot)
        )
    )

    ax.set_yticklabels(
        pivot.index,
        fontsize=7,
    )

    ax.set_xticks(
        np.arange(
            len(
                pivot.columns
            )
        )
    )

    ax.set_xticklabels(
        pivot.columns,
    )

    ax.set_title(
        "ADM2 IPC Phase Transition",
        fontsize=14,
        weight="bold",
    )

    plt.colorbar(
        image,
        ax=ax,
        ticks=[
            1,
            2,
            3,
            4,
            5,
        ],
        label="IPC Phase",
    )

    save_figure(
        output_path
    )


# ============================================================
# 18. CROSS-DATASET HEATMAP
# ============================================================

def zscore(
    series,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    std = values.std()

    if (
        pd.isna(std)
        or
        std == 0
    ):
        return values * 0

    return (
        values
        -
        values.mean()
    ) / std


def plot_cross_dataset_heatmap(
    df,
    level,
    output_path,
):

    if df is None:
        return

    name_col = (
        f"{level}_name"
    )

    if name_col not in df.columns:
        return

    candidates = [
        "phase",
        "phase_clean",
        "phase_3plus_percentage",
        "p3plus_percentage",

        "flood_days",
        "flood_day_rate_percentage",
        "max_flooded_percentage",
        "mean_flooded_percentage",

        "mean_rainfall_mm",
        "total_rainfall_mm",

        "mean_runoff_mm",
        "total_runoff_mm",
    ]

    variables = [
        c
        for c
        in candidates
        if c in df.columns
    ]

    if len(variables) < 2:
        return

    temp = df[
        [
            name_col,
        ]
        +
        variables
    ].copy()

    for col in variables:

        temp[
            col
        ] = pd.to_numeric(
            temp[
                col
            ],
            errors="coerce",
        )

    temp = (
        temp
        .groupby(
            name_col,
            as_index=False,
        )
        [variables]
        .mean()
    )

    for col in variables:

        temp[
            col
        ] = zscore(
            temp[
                col
            ]
        )

    temp[
        "_overall"
    ] = (
        temp[
            variables
        ]
        .mean(
            axis=1
        )
    )

    temp = (
        temp
        .sort_values(
            "_overall",
            ascending=False,
        )
        .head(
            30
        )
    )

    matrix = (
        temp[
            variables
        ]
        .to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                10,
                len(variables)
                * 1.4,
            ),
            12,
        )
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-2,
        vmax=2,
    )

    ax.set_yticks(
        np.arange(
            len(temp)
        )
    )

    ax.set_yticklabels(
        temp[
            name_col
        ],
        fontsize=8,
    )

    ax.set_xticks(
        np.arange(
            len(
                variables
            )
        )
    )

    ax.set_xticklabels(
        variables,
        rotation=45,
        ha="right",
    )

    ax.set_title(
        f"{level.upper()} "
        "Cross-Dataset Comparison",
        fontsize=14,
        weight="bold",
    )

    plt.colorbar(
        image,
        ax=ax,
        label="Z-score",
    )

    save_figure(
        output_path
    )


# ============================================================
# 19. ADM3 INSIDE P3+
# ============================================================

def plot_adm3_inside_p3(
    adm3,
    adm3_data,
    output_path,
):

    if adm3_data is None:
        return

    phase_col = find_phase_column(
        adm3_data
    )

    if phase_col is None:
        return

    flood_col = next(
        (
            c
            for c
            in [
                "max_flooded_percentage",
                "flood_day_rate_percentage",
                "flood_days",
            ]
            if c
            in adm3_data.columns
        ),
        None,
    )

    if flood_col is None:
        return

    temp = (
        adm3_data.copy()
    )

    temp[
        phase_col
    ] = pd.to_numeric(
        temp[
            phase_col
        ],
        errors="coerce",
    )

    temp[
        flood_col
    ] = pd.to_numeric(
        temp[
            flood_col
        ],
        errors="coerce",
    )

    temp = temp[
        temp[
            phase_col
        ]
        >= 3
    ]

    if len(temp) == 0:
        return

    gdf = merge_admin_table(
        adm3,
        temp,
        "adm3",
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            12,
        )
    )

    adm3.plot(
        ax=ax,
        color="#eeeeee",
        edgecolor="white",
        linewidth=0.2,
    )

    highlighted = gdf[
        gdf[
            phase_col
        ]
        >= 3
    ]

    if len(highlighted):

        highlighted.plot(
            column=flood_col,
            cmap="magma",
            legend=True,
            ax=ax,
            edgecolor="white",
            linewidth=0.2,
        )

    ax.set_title(
        "ADM3 Flood Exposure Inside ADM2 IPC P3+ Areas",
        fontsize=14,
        weight="bold",
    )

    ax.axis("off")

    save_figure(
        output_path
    )


# ============================================================
# 20. HYDROLOGY
# ============================================================

def find_hydrology_file():

    for path in HYDROLOGY_CANDIDATES:

        if path.exists():

            print(
                "\nHydrology file:"
            )

            print(
                path
            )

            return path

    for root in [
        PROJECT_ROOT,
        PROCESSED_DATA_DIR,
    ]:

        for path in root.rglob(
            "*.csv"
        ):

            name = (
                path.name
                .lower()
            )

            if (
                "adm2"
                in name
                and
                "rain"
                in name
                and
                "runoff"
                in name
            ):

                return path

    return None


def load_hydrology_summary():

    path = (
        find_hydrology_file()
    )

    if path is None:

        print(
            "\nHydrology file not found."
        )

        return None

    df = pd.read_csv(
        path
    )

    df = normalize_columns(
        df
    )

    area_col = next(
        (
            c
            for c
            in [
                "adm2_name",
                "area_name",
                "county",
                "county_name",
            ]
            if c
            in df.columns
        ),
        None,
    )

    if area_col is None:
        return None

    rainfall_col = next(
        (
            c
            for c
            in df.columns
            if (
                "rain" in c
                and
                pd.api.types.is_numeric_dtype(
                    df[c]
                )
            )
        ),
        None,
    )

    runoff_col = next(
        (
            c
            for c
            in df.columns
            if (
                "runoff" in c
                and
                pd.api.types.is_numeric_dtype(
                    df[c]
                )
            )
        ),
        None,
    )

    agg = {}

    if rainfall_col:

        agg[
            "mean_rainfall_mm"
        ] = (
            rainfall_col,
            "mean",
        )

    if runoff_col:

        agg[
            "mean_runoff_mm"
        ] = (
            runoff_col,
            "mean",
        )

    if not agg:
        return None

    return (
        df
        .groupby(
            area_col
        )
        .agg(
            **agg
        )
        .reset_index()
        .rename(
            columns={
                area_col:
                    "adm2_name"
            }
        )
    )


# ============================================================
# 21. MAIN
# ============================================================

def main():

    print(
        "=" * 75
    )

    print(
        "IPC × FLOOD FIGURE GENERATOR"
    )

    print(
        "=" * 75
    )


    # --------------------------------------------------------
    # Boundaries
    # --------------------------------------------------------

    adm2 = load_boundary(
        ADM2_BOUNDARY_FILE,
        "adm2",
    )

    adm3 = load_boundary(
        ADM3_BOUNDARY_FILE,
        "adm3",
    )


    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    adm2_ipc = load_csv(
        ADM2_IPC_FLOOD_FILE,
        "ADM2 IPC × flood data",
    )

    adm3_ipc = load_csv(
        ADM3_IPC_FLOOD_FILE,
        "ADM3 IPC × flood data",
    )

    adm2_flood = load_csv(
        ADM2_FLOOD_SUMMARY_FILE,
        "ADM2 flood summary",
    )

    adm3_flood = load_csv(
        ADM3_FLOOD_SUMMARY_FILE,
        "ADM3 flood summary",
    )


    # ========================================================
    # ADM2 IPC MAPS
    # ========================================================

    for number, period in [
        (
            "01",
            "current",
        ),
        (
            "02",
            "projected_1",
        ),
        (
            "03",
            "projected_2",
        ),
    ]:

        plot_ipc_phase_map(
            adm2,
            adm2_ipc,
            "adm2",
            period,
            ADM2_FIG_DIR
            /
            (
                f"{number}_IPC_"
                f"{period}_Phase.png"
            ),
        )


    # ========================================================
    # ADM2 FLOOD MAPS
    # ========================================================

    plot_continuous_map(
        adm2,
        adm2_flood,
        "adm2",
        "flood_days",
        "ADM2 Flood Days",
        ADM2_FIG_DIR
        /
        "04_Flood_Days.png",
    )

    plot_continuous_map(
        adm2,
        adm2_flood,
        "adm2",
        "max_flooded_percentage",
        "ADM2 Maximum Flooded Percentage",
        ADM2_FIG_DIR
        /
        "05_Max_Flooded_Percentage.png",
    )


    # ========================================================
    # IPC × FLOOD OVERLAP
    # ========================================================

    plot_overlap_map(
        adm2,
        adm2_ipc,
        "adm2",
        "current",
        ADM2_FIG_DIR
        /
        "06_IPC_P3Plus_Flood_Overlap.png",
    )


    # ========================================================
    # SCATTER
    # ========================================================

    if adm2_ipc is not None:

        period_col = find_period_column(
            adm2_ipc
        )

        if period_col:

            current = adm2_ipc[
                match_period_name(
                    adm2_ipc[
                        period_col
                    ],
                    "current",
                )
            ].copy()

        else:

            current = (
                adm2_ipc.copy()
            )

        phase_col = find_phase_column(
            current
        )

        if (
            phase_col
            and
            "flood_days"
            in current.columns
            and
            "adm2_name"
            in current.columns
        ):

            plot_scatter(
                current,
                phase_col,
                "flood_days",
                "adm2_name",
                "IPC Phase vs Flood Days",
                "IPC Phase",
                "Flood days",
                ADM2_FIG_DIR
                /
                "07_IPC_Phase_vs_Flood_Days.png",
            )

        p3plus = next(
            (
                c
                for c
                in [
                    "phase_3plus_percentage",
                    "p3plus_percentage",
                ]
                if c
                in current.columns
            ),
            None,
        )

        flood_pct = next(
            (
                c
                for c
                in [
                    "max_flooded_percentage",
                    "mean_flooded_percentage",
                ]
                if c
                in current.columns
            ),
            None,
        )

        if (
            p3plus
            and
            flood_pct
            and
            "adm2_name"
            in current.columns
        ):

            plot_scatter(
                current,
                p3plus,
                flood_pct,
                "adm2_name",
                "IPC P3+ vs Flood Exposure",
                "Population in P3+ (%)",
                "Flooded area (%)",
                ADM2_FIG_DIR
                /
                "08_IPC_P3Plus_vs_Flood_Percentage.png",
            )


    # ========================================================
    # HEATMAP
    # ========================================================

    plot_cross_dataset_heatmap(
        adm2_ipc,
        "adm2",
        ADM2_FIG_DIR
        /
        "09_ADM2_Cross_Dataset_Heatmap.png",
    )


    # ========================================================
    # IPC TRANSITION
    # ========================================================

    plot_ipc_transition(
        adm2_ipc,
        ADM2_FIG_DIR
        /
        "10_IPC_Transition_Heatmap.png",
    )


    # ========================================================
    # ADM3 FLOOD
    # ========================================================

    plot_continuous_map(
        adm3,
        adm3_flood,
        "adm3",
        "flood_days",
        "ADM3 Flood Days",
        ADM3_FIG_DIR
        /
        "11_ADM3_Flood_Days.png",
    )

    plot_continuous_map(
        adm3,
        adm3_flood,
        "adm3",
        "max_flooded_percentage",
        "ADM3 Maximum Flooded Percentage",
        ADM3_FIG_DIR
        /
        "12_ADM3_Max_Flooded_Percentage.png",
    )


    # ========================================================
    # ADM3 INSIDE IPC P3+
    # ========================================================

    plot_adm3_inside_p3(
        adm3,
        adm3_ipc,
        ADM3_FIG_DIR
        /
        "13_ADM3_Inside_P3Plus_ADM2.png",
    )


    # ========================================================
    # ADM3 HEATMAP
    # ========================================================

    plot_cross_dataset_heatmap(
        adm3_ipc,
        "adm3",
        ADM3_FIG_DIR
        /
        "14_ADM3_Hotspot_Comparison.png",
    )


    # ========================================================
    # HYDROLOGY
    # ========================================================

    hydro = load_hydrology_summary()

    if (
        hydro is not None
        and
        adm2_flood is not None
    ):

        combined = hydro.merge(
            adm2_flood,
            on="adm2_name",
            how="inner",
        )

        if (
            "mean_rainfall_mm"
            in combined.columns
            and
            "flood_days"
            in combined.columns
        ):

            plot_scatter(
                combined,
                "mean_rainfall_mm",
                "flood_days",
                "adm2_name",
                "Rainfall vs Flood Frequency",
                "Mean rainfall",
                "Flood days",
                COMPARISON_FIG_DIR
                /
                "15_Rainfall_vs_Flood.png",
            )

        if (
            "mean_runoff_mm"
            in combined.columns
            and
            "flood_days"
            in combined.columns
        ):

            plot_scatter(
                combined,
                "mean_runoff_mm",
                "flood_days",
                "adm2_name",
                "Runoff vs Flood Frequency",
                "Mean runoff",
                "Flood days",
                COMPARISON_FIG_DIR
                /
                "16_Runoff_vs_Flood.png",
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        "FIGURE GENERATION COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        "\nFigures saved to:"
    )

    print(
        FIGURE_DIR
    )

    print(
        "\nRecommended first figures to inspect:"
    )

    print(
        "  06_IPC_P3Plus_Flood_Overlap.png"
    )

    print(
        "  09_ADM2_Cross_Dataset_Heatmap.png"
    )

    print(
        "  10_IPC_Transition_Heatmap.png"
    )

    print(
        "  13_ADM3_Inside_P3Plus_ADM2.png"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()