#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_map_adm3_temporal_patterns.py
================================
South Sudan Flood Analysis — Final validated ADM3 temporal-pattern mapping

Purpose
-------
Map the FINAL validated ADM3 flood temporal typology produced by Script 14.
This script does not re-cluster, re-label, or alter the classification.

Pipeline
--------
10 build indicators -> 11 coverage audit -> 12 pattern discovery
-> 13 interpretation validation -> 14 final typology -> 15 mapping

Primary input
-------------
processed_data/cross_dataset_analysis/02_adm3_flood/
  14_finalize_adm3_temporal_pattern_classification/tables/
    14_adm3_final_temporal_pattern_classification_all512.csv

Boundary input
--------------
The script searches common project locations for an ADM3 boundary file and
joins using an existing ADM3 code/ID field. It NEVER fabricates ADM3 IDs from
polygon row order.

Outputs
-------
processed_data/cross_dataset_analysis/02_adm3_flood/
  15_map_adm3_temporal_patterns/
    figures/
      15_01_final_adm3_temporal_pattern_map.png
      15_02_adm3_long_term_trend_slope_map.png
      15_03_adm3_recent_vs_early_change_map.png
      15_04_adm3_long_term_mean_flood_map.png
      15_05_adm3_recent_5y_change_map.png
    tables/
      15_adm3_final_map_data.csv
      15_adm3_boundary_merge_audit.csv
      15_final_pattern_counts_map.csv
    qc/
      15_mapping_qc_summary.txt

Expected Script-14 population
-----------------------------
512 total ADM3
490 clustered ADM3
 10 Strong recent expansion
 39 Moderate recent expansion
218 Episodic / high-volatility
223 Recurrent / persistent
 13 Insufficient source coverage
  9 No detected flood record
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path("/Users/shibai/Documents/GitHub/DBL-Group-22")
ADM3_ROOT = (
    PROJECT_ROOT
    / "processed_data"
    / "cross_dataset_analysis"
    / "02_adm3_flood"
)

SCRIPT14_DIR = ADM3_ROOT / "14_finalize_adm3_temporal_pattern_classification"
FINAL_CLASSIFICATION_FILE = (
    SCRIPT14_DIR
    / "tables"
    / "14_adm3_final_temporal_pattern_classification_all512.csv"
)

OUTPUT_DIR = ADM3_ROOT / "15_map_adm3_temporal_patterns"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
QC_DIR = OUTPUT_DIR / "qc"
for directory in (FIG_DIR, TABLE_DIR, QC_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Common boundary locations. Add another candidate here if your repository
# stores the file elsewhere.
BOUNDARY_CANDIDATES = [
    PROJECT_ROOT / "raw_data" / "administrative_boundaries" / "ssd_admin3.geojson",
    PROJECT_ROOT / "raw_data" / "administrative_boundaries" / "ssd_admbnda_adm3.geojson",
    PROJECT_ROOT / "raw_data" / "administrative_boundaries" / "ssd_admbnda_adm3.shp",
    PROJECT_ROOT / "raw_data" / "administrative_boundaries" / "ssd_admin3.shp",
]

EXPECTED_TOTAL = 512
EXPECTED_CLUSTERED = 490
EXPECTED_NONCLUSTER = 22
EXPECTED_COUNTS = {
    "Strong recent expansion": 10,
    "Moderate recent expansion": 39,
    "Episodic / high-volatility": 218,
    "Recurrent / persistent": 223,
    "Insufficient source coverage": 13,
    "No detected flood record": 9,
}


# =============================================================================
# 2. VISUAL SETTINGS
# =============================================================================

FIGSIZE = (11, 10)
DPI = 320
BOUNDARY_COLOR = "#F3F1ED"
BOUNDARY_LINEWIDTH = 0.20
OUTLINE_COLOR = "#555555"
OUTLINE_LINEWIDTH = 0.55
TITLE_SIZE = 16
NOTE_SIZE = 8.5
LEGEND_FONT_SIZE = 9

# Final validated Script-14 classes only.
PATTERN_COLORS = {
    "Strong recent expansion": "#9E2A2B",
    "Moderate recent expansion": "#E76F51",
    "Episodic / high-volatility": "#E9C46A",
    "Recurrent / persistent": "#457B9D",
    "No detected flood record": "#D9D9D9",
    "Insufficient source coverage": "#FFFFFF",
}

PATTERN_ORDER = [
    "Strong recent expansion",
    "Moderate recent expansion",
    "Episodic / high-volatility",
    "Recurrent / persistent",
    "No detected flood record",
    "Insufficient source coverage",
]


# =============================================================================
# 3. HELPERS
# =============================================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"\n{description} not found:\n{path}")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    return re.sub(r"[^A-Z0-9]", "", text)


def detect_field(columns, candidates):
    lower_map = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def discover_boundary_file() -> Path:
    for path in BOUNDARY_CANDIDATES:
        if path.exists():
            return path

    search_roots = [
        PROJECT_ROOT / "raw_data" / "administrative_boundaries",
        PROJECT_ROOT / "raw_data",
    ]
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("*.geojson", "*.gpkg", "*.shp"):
            for path in root.rglob(pattern):
                name = path.name.lower()
                if "adm3" in name or "admin3" in name:
                    candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Could not locate an ADM3 boundary file under raw_data/.\n"
            "Expected a GeoJSON/GPKG/SHP filename containing 'adm3' or 'admin3'."
        )

    # Prefer filenames explicitly mentioning South Sudan/SSD, then shortest path.
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            0 if ("ssd" in p.name.lower() or "south" in p.name.lower()) else 1,
            len(str(p)),
        ),
    )
    return candidates[0]


def add_country_outline(gdf: gpd.GeoDataFrame, ax) -> None:
    gdf.dissolve().boundary.plot(
        ax=ax,
        color=OUTLINE_COLOR,
        linewidth=OUTLINE_LINEWIDTH,
        zorder=10,
    )


def clean_axis(ax) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal")


def add_note(fig, text: str) -> None:
    fig.text(0.5, 0.022, text, ha="center", va="bottom", fontsize=NOTE_SIZE)


def save_figure(fig, filename: str) -> None:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {path}")


def safe_diverging_norm(series: pd.Series) -> TwoSlopeNorm:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    vmax = float(np.nanpercentile(np.abs(values), 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


# =============================================================================
# 4. LOAD SCRIPT-14 FINAL CLASSIFICATION
# =============================================================================

def load_final_classification() -> pd.DataFrame:
    print("=" * 88)
    print("LOAD SCRIPT 14 FINAL ADM3 CLASSIFICATION")
    print("=" * 88)

    require_file(FINAL_CLASSIFICATION_FILE, "Script 14 final all-512 classification")
    df = pd.read_csv(FINAL_CLASSIFICATION_FILE)

    required = [
        "ADM3_ID",
        "ADM3_NAME",
        "cluster_id",
        "final_temporal_pattern",
        "analysis_status",
        "long_term_mean",
        "change_pp",
        "recent_5y_change_pp",
        "trend_slope",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Script 14 final classification is missing required columns:\n"
            + "\n".join(missing)
        )

    if len(df) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} Script-14 rows, found {len(df)}.")
    if df["ADM3_ID"].duplicated().any():
        dup = df.loc[df["ADM3_ID"].duplicated(False), "ADM3_ID"].astype(str).tolist()
        raise ValueError(f"Duplicate ADM3_ID values in Script 14: {dup[:20]}")

    observed_patterns = set(df["final_temporal_pattern"].dropna().astype(str))
    unexpected = observed_patterns - set(PATTERN_ORDER)
    if unexpected:
        raise ValueError(
            "Unexpected final temporal pattern(s) in Script 14:\n"
            + "\n".join(sorted(unexpected))
        )

    counts = df["final_temporal_pattern"].value_counts().to_dict()
    for label, expected in EXPECTED_COUNTS.items():
        observed = int(counts.get(label, 0))
        if observed != expected:
            raise ValueError(
                f"Final-count QC failed for '{label}': expected {expected}, observed {observed}."
            )

    clustered = int(df["cluster_id"].notna().sum())
    if clustered != EXPECTED_CLUSTERED:
        raise ValueError(
            f"Expected {EXPECTED_CLUSTERED} clustered ADM3, found {clustered}."
        )

    df["_join_id"] = df["ADM3_ID"].map(normalize_id)
    df["_join_name"] = df["ADM3_NAME"].map(normalize_text)

    print(f"Rows: {len(df):,}")
    print(f"Clustered: {clustered:,}")
    print(f"Non-clustered coverage categories: {len(df) - clustered:,}")
    print("\nFinal pattern counts:")
    print(df["final_temporal_pattern"].value_counts().to_string())
    return df


# =============================================================================
# 5. LOAD + PREPARE ADM3 BOUNDARY
# =============================================================================

def load_boundary() -> tuple[gpd.GeoDataFrame, Path, str, str | None]:
    print("\n" + "=" * 88)
    print("LOAD ADM3 BOUNDARY")
    print("=" * 88)

    boundary_file = discover_boundary_file()
    print(f"Boundary file: {boundary_file}")
    gdf = gpd.read_file(boundary_file)

    if gdf.empty:
        raise ValueError("ADM3 boundary file is empty.")
    if gdf.crs is None:
        raise ValueError("ADM3 boundary has no CRS; mapping cannot safely continue.")

    gdf = gdf.to_crs("EPSG:4326").reset_index(drop=True)

    id_field = detect_field(
        gdf.columns,
        [
            "ADM3_ID", "adm3_id", "ADM3_PCODE", "adm3_pcode",
            "ADM3_CODE", "adm3_code", "PCODE", "pcode",
            "GID_3", "gid_3",
        ],
    )
    if id_field is None:
        raise ValueError(
            "No ADM3 ID/code field was found in the boundary file.\n"
            "For safety, Script 15 will NOT create IDs from polygon row order.\n"
            f"Available fields: {list(gdf.columns)}"
        )

    name_field = detect_field(
        gdf.columns,
        ["ADM3_NAME", "adm3_name", "NAME_3", "name_3", "PAYAM", "payam"],
    )

    gdf["_join_id"] = gdf[id_field].map(normalize_id)
    if name_field is not None:
        gdf["_boundary_adm3_name"] = gdf[name_field].astype(str).str.strip()
        gdf["_join_name"] = gdf[name_field].map(normalize_text)
    else:
        gdf["_boundary_adm3_name"] = np.nan
        gdf["_join_name"] = ""

    if gdf["_join_id"].eq("").any():
        raise ValueError("At least one boundary polygon has a blank ADM3 join ID.")
    if gdf["_join_id"].duplicated().any():
        dup = gdf.loc[gdf["_join_id"].duplicated(False), id_field].astype(str).tolist()
        raise ValueError(f"Boundary ADM3 ID field is not unique. Examples: {dup[:20]}")

    print(f"Boundary polygons: {len(gdf):,}")
    print(f"Boundary ID field: {id_field}")
    print(f"Boundary name field: {name_field}")
    print(f"CRS: {gdf.crs}")
    return gdf, boundary_file, id_field, name_field


# =============================================================================
# 6. STRICT BOUNDARY ↔ SCRIPT-14 MERGE + AUDIT
# =============================================================================

def merge_and_audit(
    boundary: gpd.GeoDataFrame,
    final_df: pd.DataFrame,
    id_field: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    print("\n" + "=" * 88)
    print("STRICT ADM3 BOUNDARY ↔ SCRIPT 14 MERGE")
    print("=" * 88)

    boundary_ids = set(boundary["_join_id"])
    result_ids = set(final_df["_join_id"])
    missing_in_boundary = sorted(result_ids - boundary_ids)
    missing_in_results = sorted(boundary_ids - result_ids)

    if missing_in_boundary:
        print(f"WARNING: Script-14 ADM3 IDs absent from boundary: {len(missing_in_boundary)}")
    if missing_in_results:
        print(f"WARNING: Boundary ADM3 IDs absent from Script 14: {len(missing_in_results)}")

    result_cols = [c for c in final_df.columns if c not in {"_join_name"}]
    merged = boundary.merge(
        final_df[result_cols],
        on="_join_id",
        how="left",
        validate="one_to_one",
        suffixes=("_boundary", ""),
    )

    matched = merged["final_temporal_pattern"].notna()
    matched_count = int(matched.sum())

    # Name comparison is diagnostic, not a join fallback.
    if "_join_name" in boundary.columns:
        merged["adm3_name_match"] = np.where(
            matched,
            merged["_join_name"] == merged["ADM3_NAME"].map(normalize_text),
            False,
        )
    else:
        merged["adm3_name_match"] = np.nan

    audit = pd.DataFrame({
        "qc_metric": [
            "script14_rows",
            "script14_clustered_rows",
            "script14_noncluster_rows",
            "boundary_polygon_rows",
            "matched_boundary_polygons",
            "script14_ids_missing_in_boundary",
            "boundary_ids_missing_in_script14",
            "matched_name_mismatches",
        ],
        "value": [
            len(final_df),
            int(final_df["cluster_id"].notna().sum()),
            int(final_df["cluster_id"].isna().sum()),
            len(boundary),
            matched_count,
            len(missing_in_boundary),
            len(missing_in_results),
            int((matched & ~merged["adm3_name_match"]).sum()),
        ],
    })
    audit.to_csv(TABLE_DIR / "15_adm3_boundary_merge_audit.csv", index=False)

    print(audit.to_string(index=False))

    # Strict safety checks. We expect a complete 512-unit map.
    if len(boundary) != EXPECTED_TOTAL:
        raise ValueError(
            f"Boundary contains {len(boundary)} polygons; expected {EXPECTED_TOTAL}. "
            "Check that the correct South Sudan ADM3 boundary version is being used."
        )
    if matched_count != EXPECTED_TOTAL or missing_in_boundary or missing_in_results:
        raise ValueError(
            "ADM3 boundary/result merge is incomplete. Mapping stopped to prevent a misleading map.\n"
            f"Matched {matched_count}/{EXPECTED_TOTAL}; "
            f"results missing in boundary={len(missing_in_boundary)}; "
            f"boundary missing in results={len(missing_in_results)}."
        )

    # Names should normally agree. Stop on mismatches because wrong ID versions can map
    # a valid classification to the wrong polygon.
    mismatch_count = int((matched & ~merged["adm3_name_match"]).sum())
    if mismatch_count > 0:
        examples = merged.loc[
            matched & ~merged["adm3_name_match"],
            [id_field, "_boundary_adm3_name", "ADM3_ID", "ADM3_NAME"],
        ].head(20)
        examples.to_csv(QC_DIR / "15_adm3_name_mismatch_examples.csv", index=False)
        raise ValueError(
            f"Found {mismatch_count} ADM3 name mismatches after ID merge. "
            "Mapping stopped. See qc/15_adm3_name_mismatch_examples.csv."
        )

    return merged, audit


# =============================================================================
# 7. FINAL CATEGORICAL TYPOLOGY MAP
# =============================================================================

def plot_final_typology(gdf: gpd.GeoDataFrame) -> None:
    print("\nCreating final validated ADM3 temporal-pattern map...")
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Base layer ensures every ADM3 polygon is visible.
    gdf.plot(
        ax=ax,
        color="#F7F7F7",
        edgecolor=BOUNDARY_COLOR,
        linewidth=BOUNDARY_LINEWIDTH,
    )

    for pattern in PATTERN_ORDER:
        subset = gdf[gdf["final_temporal_pattern"] == pattern]
        if subset.empty:
            continue
        plot_kwargs = dict(
            ax=ax,
            color=PATTERN_COLORS[pattern],
            edgecolor="#B7B7B7" if pattern == "Insufficient source coverage" else BOUNDARY_COLOR,
            linewidth=0.45 if pattern == "Insufficient source coverage" else BOUNDARY_LINEWIDTH,
        )
        if pattern == "Insufficient source coverage":
            plot_kwargs["hatch"] = "////"
        subset.plot(**plot_kwargs)

    add_country_outline(gdf, ax)
    clean_axis(ax)
    ax.set_title(
        "Final Validated ADM3 Flood Temporal Patterns in South Sudan (2000–2025)",
        fontsize=TITLE_SIZE,
        pad=14,
        weight="semibold",
    )

    handles = []
    for pattern in PATTERN_ORDER:
        n = int((gdf["final_temporal_pattern"] == pattern).sum())
        if pattern == "Insufficient source coverage":
            handles.append(Patch(
                facecolor=PATTERN_COLORS[pattern], edgecolor="#777777", hatch="////",
                label=f"{pattern} (n={n})",
            ))
        else:
            handles.append(Line2D(
                [0], [0], marker="s", linestyle="", markersize=10,
                markerfacecolor=PATTERN_COLORS[pattern], markeredgecolor="none",
                label=f"{pattern} (n={n})",
            ))

    ax.legend(
        handles=handles,
        title="Final temporal pattern",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=10,
    )
    add_note(
        fig,
        "Final Script-14 typology. Hatched areas have insufficient source coverage; "
        "grey areas have complete source coverage but no detected flood record.",
    )
    plt.tight_layout(rect=[0.01, 0.05, 0.80, 0.97])
    save_figure(fig, "15_01_final_adm3_temporal_pattern_map.png")


# =============================================================================
# 8. CONTINUOUS DIAGNOSTIC MAPS
# =============================================================================

def plot_signed_metric(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    legend_label: str,
    note: str,
    filename: str,
) -> None:
    print(f"Creating {column} map...")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    norm = safe_diverging_norm(gdf[column])

    gdf.plot(
        ax=ax,
        column=column,
        cmap="RdBu_r",
        norm=norm,
        legend=True,
        edgecolor=BOUNDARY_COLOR,
        linewidth=BOUNDARY_LINEWIDTH,
        missing_kwds={"color": "#E6E6E6", "label": "Not available"},
        legend_kwds={"label": legend_label, "shrink": 0.72},
    )
    add_country_outline(gdf, ax)
    clean_axis(ax)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=14, weight="semibold")
    add_note(fig, note)
    plt.tight_layout(rect=[0.01, 0.05, 0.97, 0.97])
    save_figure(fig, filename)


def plot_long_term_mean(gdf: gpd.GeoDataFrame) -> None:
    print("Creating long-term mean flood map...")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    gdf.plot(
        ax=ax,
        column="long_term_mean",
        cmap="OrRd",
        legend=True,
        edgecolor=BOUNDARY_COLOR,
        linewidth=BOUNDARY_LINEWIDTH,
        missing_kwds={"color": "#E6E6E6", "label": "Not available"},
        legend_kwds={"label": "Mean annual flooded area (% of ADM3)", "shrink": 0.72},
    )
    add_country_outline(gdf, ax)
    clean_axis(ax)
    ax.set_title(
        "ADM3 Long-Term Mean Flooded Area (2000–2025)",
        fontsize=TITLE_SIZE, pad=14, weight="semibold",
    )
    add_note(
        fig,
        "Higher values indicate a larger mean share of the ADM3 affected by flooding across 2000–2025.",
    )
    plt.tight_layout(rect=[0.01, 0.05, 0.97, 0.97])
    save_figure(fig, "15_04_adm3_long_term_mean_flood_map.png")


# =============================================================================
# 9. SAVE MAP DATA + FINAL QC
# =============================================================================

def save_outputs(
    gdf: gpd.GeoDataFrame,
    boundary_file: Path,
    id_field: str,
    audit: pd.DataFrame,
) -> None:
    output_columns = [c for c in [
        "ADM3_ID", "ADM3_NAME", "ADM1_NAME", "ADM2_NAME",
        "cluster_id", "script12_preliminary_temporal_pattern",
        "final_temporal_pattern", "label_validation_status",
        "cluster_membership_changed", "analysis_status",
        "long_term_mean", "early_mean", "recent_mean", "change_pp",
        "recent_5y_mean", "recent_5y_change_pp", "trend_slope",
        "trend_pvalue", "trend_r2", "flood_year_count",
        "flood_year_fraction", "longest_consecutive_run",
        "max_flooded_percent", "candidate_change_point",
    ] if c in gdf.columns]

    gdf[output_columns].to_csv(
        TABLE_DIR / "15_adm3_final_map_data.csv", index=False
    )

    counts = (
        gdf["final_temporal_pattern"]
        .value_counts()
        .rename_axis("final_temporal_pattern")
        .reset_index(name="adm3_count")
    )
    counts["share_percent"] = counts["adm3_count"] / EXPECTED_TOTAL * 100
    counts.to_csv(TABLE_DIR / "15_final_pattern_counts_map.csv", index=False)

    qc_lines = [
        "SCRIPT 15 — FINAL ADM3 TEMPORAL PATTERN MAPPING QC",
        "=" * 70,
        f"Boundary file: {boundary_file}",
        f"Boundary ID field: {id_field}",
        f"Script 14 classification: {FINAL_CLASSIFICATION_FILE}",
        "",
        f"Expected total ADM3: {EXPECTED_TOTAL}",
        f"Mapped ADM3: {len(gdf)}",
        f"Clustered ADM3: {int(gdf['cluster_id'].notna().sum())}",
        f"Non-clustered coverage categories: {int(gdf['cluster_id'].isna().sum())}",
        f"Missing final labels after merge: {int(gdf['final_temporal_pattern'].isna().sum())}",
        f"Cluster membership changed by Script 15: False",
        "",
        "Final pattern counts:",
        counts.to_string(index=False),
        "",
        "Boundary merge audit:",
        audit.to_string(index=False),
        "",
        "Interpretation:",
        "Script 15 is visualization only. It does not re-cluster or revise Script-14 labels.",
        "Insufficient source coverage is visually separated from No detected flood record.",
    ]
    (QC_DIR / "15_mapping_qc_summary.txt").write_text(
        "\n".join(qc_lines), encoding="utf-8"
    )


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> None:
    final_df = load_final_classification()
    boundary, boundary_file, id_field, _ = load_boundary()
    merged, audit = merge_and_audit(boundary, final_df, id_field)

    plot_final_typology(merged)

    plot_signed_metric(
        merged,
        "trend_slope",
        "ADM3 Long-Term Flood Trend Slope (2000–2025)",
        "Annual change in flooded percentage points",
        "Red = increasing long-term flooded share; blue = decreasing; white = little long-term change.",
        "15_02_adm3_long_term_trend_slope_map.png",
    )

    plot_signed_metric(
        merged,
        "change_pp",
        "ADM3 Flood Change: Recent vs Early Period",
        "Recent mean − early mean (percentage points)",
        "Red = higher recent flooded share; blue = lower recent flooded share.",
        "15_03_adm3_recent_vs_early_change_map.png",
    )

    plot_long_term_mean(merged)

    plot_signed_metric(
        merged,
        "recent_5y_change_pp",
        "ADM3 Recent Five-Year Flood Change",
        "2021–2025 mean − 2016–2020 mean (percentage points)",
        "Red = higher flooded share in 2021–2025 than 2016–2020; blue = lower.",
        "15_05_adm3_recent_5y_change_map.png",
    )

    save_outputs(merged, boundary_file, id_field, audit)

    print("\n" + "=" * 88)
    print("SCRIPT 15 COMPLETED")
    print("=" * 88)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Final typology source: Script 14 validated all-512 classification")
    print("Expected/mapped ADM3: 512 / 512")
    print("No clustering or label revision performed in Script 15.")


if __name__ == "__main__":
    main()
