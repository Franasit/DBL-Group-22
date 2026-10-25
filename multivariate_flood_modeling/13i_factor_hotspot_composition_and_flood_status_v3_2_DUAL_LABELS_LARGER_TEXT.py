#!/usr/bin/env python3
"""13i | Flood-relevant factor-hotspot dashboard with selected and unselected severe-flood tables.

Purpose
-------
This script complements hazard-first analysis by asking the reverse question:

    "Where are the strongest exposure-factor concentrations, and how does flood
     hazard behave in those ADM3 areas?"

The script has two parts.

PART A — Existing factor-dense flood outcomes
---------------------------------------------
Factor-dense means an ADM3 has at least --min-factors saved high-factor flags.
The script preserves three national screening maps:
  1. Factor-dense areas with high recent flood activity.
  2. Factor-dense areas with worsening flood activity.
  3. Factor-dense areas with stable or improving flood conditions.

PART B — Flood-relevant factor-hotspot composition dashboard
-------------------------------------------------
For each exposure factor, the script finds the Top-N ADM3 by the factor's actual
saved ADM3 total, then takes the UNION across factors. One ADM3 can therefore be
selected by several factors simultaneously.

For every selected ADM3 the dashboard shows:
  - four independent mini-bars, all starting at zero;
  - bar length = national percentile (0-100) for that factor;
  - actual factor values and units;
  - which factors placed the ADM3 in the Top-N union;
  - a descriptive flood-status class.

The mini-bars are NOT stacked raw quantities because the factors use incompatible
units (facilities, km², km², people). Percentile bars make relative intensity
comparable without pretending that the raw units are additive.

Factor colours (colour-blind-aware + numeric labels)
----------------------------------------------------
  1 Health facilities : Indigo      #4C5FD7
  2 Cropland           : Golden yellow #D8B13F
  3 Rangeland          : Lake blue   #3A9FBF
  4 Population         : Violet      #9B6AD6

Every factor is also identified by 1/2/3/4, so interpretation does not rely on
colour alone.

Cropland quality rule
---------------------
Following the project QC decision, cropland values equal to 0 are treated as
N/A for DISPLAY, percentile calculation, and cropland Top-N ranking because the
available ASAP crop mask may miss local cultivation in some ADM3 areas. The raw
0 remains preserved in audit outputs. This rule applies ONLY to cropland.

Flood-status classes
--------------------
The script reuses saved 13d hazard flags; it does not create new hazard or factor
thresholds.

  - Persistent severe + worsening
      recent flood is high, worsening is high, and the earlier-period mean is
      also at/above the saved recent-severity cutoff.

  - Recently worsened + severe
      recent flood is high and worsening is high, but the earlier-period mean
      was below the saved recent-severity cutoff.

  - Severe, no clear worsening
      recent flood is high but the saved high-worsening flag is False.

  - Worsening, not yet severe
      saved high-worsening flag is True but recent flood is not high.

  - Stable / improving
      period-mean difference <= 0.

  - Moderate / mixed change
      all remaining positive-change cases.

"Severe" and "worsening" therefore mean high relative screening categories,
not disaster-loss classifications. "Persistent" uses the same saved recent
severity cutoff as a historical reference benchmark; it is descriptive only.

IMPORTANT
---------
- Saved factor thresholds are NEVER recalculated.
- Saved hazard thresholds are NEVER recalculated.
- Factor percentiles are descriptive display/ranking statistics, not new
  humanitarian thresholds.
- ADM3 totals are exposure context, not confirmed flood-affected quantities.
- This is retrospective screening, not a forecast or causal model.

Run
---
python scripts/multivariate_flood_modeling/13i_factor_hotspot_composition_and_flood_status.py

Useful options
--------------
--top-per-factor 10       Top-N ADM3 selected for each factor (default 10)
--rows-per-page 10        Dashboard rows per page (default 10)
--min-factors 2           Saved high-factor flags required for Part A
--no-part-a               Skip the three older factor-dense outcome maps
--input PATH              Exact 13d scenario folder
--out PATH                Custom output folder

Dependencies
------------
numpy pandas matplotlib geopandas shapely
"""
from __future__ import annotations

import argparse
import json
import math
import string
import textwrap
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import Rectangle
    import geopandas as gpd
    from shapely.ops import unary_union
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\nInstall with:\n"
        "python -m pip install numpy pandas matplotlib geopandas shapely"
    ) from exc


VERSION = "13i-3.0-factor-hotspot-composition-flood-status"
TABLE = "01_adm3_screening_full.csv"
META = "07_quality_and_methodology.json"

H = "historical_flood_days_mean_2019_2025"
D = "historical_flood_days_change_after_2019"
B = "mean_annual_flood_days_2000_2018"
PF = "population_for_screening_2025"

INK = "#1E293B"
MUTED = "#5B6673"
GRID = "#D8DEE6"
BAR_BG = "#EDF1F5"
SELECTED_EDGE = "#20242B"
LABEL_GUIDE = "#7C3AED"
SELECTED_LABEL = "#F28C28"  # orange for Table 1 labels
UNSELECTED_LABEL = "#74D7E3"  # light cyan for Table 2 labels
UNSELECTED_LABEL_TEXT = "#1A8A99"

# User-approved palette.
FACTOR_COLORS = {
    1: "#4C5FD7",  # Indigo
    2: "#D8B13F",  # Golden yellow
    3: "#3A9FBF",  # Lake blue
    4: "#9B6AD6",  # Violet
}

FACTOR_SPECS = [
    (1, "health", "Health facilities", "health_facilities_total", "facilities"),
    (2, "cropland", "Cropland", "cropland_area_km2", "km²"),
    (3, "rangeland", "Rangeland", "rangeland_area_km2", "km²"),
    (4, "population", "Population", PF, "people"),
]

STATUS_ORDER = [
    "Persistent severe + worsening",
    "Recently worsened + severe",
    "Severe, no clear worsening",
    "Worsening, not yet severe",
    "Stable / improving",
    "Moderate / mixed change",
]


# -----------------------------------------------------------------------------
# Paths and robust readers
# -----------------------------------------------------------------------------

def project_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for p in (start, *start.parents):
            if (p / "processed_data" / "multivariate_flood_modeling").is_dir():
                return p
    return Path.cwd()


def resolve(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def normalize_bool(series: pd.Series, name: str, nullable: bool = False) -> pd.Series:
    values = series.astype("string").str.strip().str.casefold()
    missing = values.isna() | values.isin(["", "nan", "none", "<na>"])
    valid = values.isin(["true", "false", "1", "0", "1.0", "0.0"]) | missing
    if not valid.all():
        raise ValueError(
            f"Unknown boolean values in {name}: {values[~valid].unique().tolist()[:6]}"
        )
    out = values.isin(["true", "1", "1.0"]).astype("boolean").mask(missing, pd.NA)
    if nullable:
        return out
    if out.isna().any():
        raise ValueError(f"Missing boolean values in required field {name}")
    return out.astype(bool)


def find_scenario(base: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        folder = explicit.parent if explicit.is_file() and explicit.name == TABLE else explicit
        if not (folder / TABLE).is_file() or not (folder / META).is_file():
            raise FileNotFoundError(f"Expected {TABLE} and {META} in {folder}")
        return folder

    population_root = base / "13d_adm3_humanitarian_priority_population"
    for folder in (
        population_root / "population_sensitivity",
        population_root,
        base / "13d_adm3_humanitarian_priority_screening",
    ):
        if (folder / TABLE).is_file() and (folder / META).is_file():
            return folder

    matches = [
        p.parent
        for p in base.glob("13d*/**/01_adm3_screening_full.csv")
        if (p.parent / META).is_file()
    ]
    if not matches:
        raise FileNotFoundError("No completed 13d full table + methodology JSON found.")
    return sorted(matches)[-1]


# -----------------------------------------------------------------------------
# Saved 13d analysis validation
# -----------------------------------------------------------------------------

def load_saved(folder: Path, min_factors: int) -> tuple[pd.DataFrame, dict, list[tuple]]:
    meta = json.loads((folder / META).read_text(encoding="utf-8"))
    cutoffs = meta.get("numeric_thresholds")
    if not isinstance(cutoffs, dict):
        raise ValueError("13d numeric_thresholds are missing. 13i will not recalculate them.")

    df = pd.read_csv(folder / TABLE, dtype={"adm_id": "string"})
    required = [
        "adm_id",
        "region_name",
        "parent_adm2_name",
        H,
        D,
        "high_recent_flood",
        "high_flood_increase",
    ]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError("Missing required saved fields: " + ", ".join(missing))

    df["adm_id"] = df.adm_id.astype("string").str.strip().str.upper()
    if df.adm_id.isna().any() or df.adm_id.eq("").any() or df.adm_id.duplicated().any():
        raise ValueError("ADM3 IDs must be nonempty and unique.")

    for c in (H, D):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if not np.isfinite(df[c]).all():
            raise ValueError(f"Non-finite values in {c}")

    # Earlier-period mean: use saved column when available; otherwise derive it
    # from the documented identity D = recent mean - earlier mean.
    if B in df:
        df[B] = pd.to_numeric(df[B], errors="coerce")
        if not np.isfinite(df[B]).all():
            raise ValueError(f"Non-finite values in {B}")
        if not np.allclose(df[H] - df[B], df[D], atol=1e-6, rtol=0):
            raise ValueError("Saved baseline, recent mean, and change are inconsistent.")
    else:
        df[B] = df[H] - df[D]

    df["high_recent_flood"] = normalize_bool(df.high_recent_flood, "high_recent_flood")
    df["high_flood_increase"] = normalize_bool(
        df.high_flood_increase, "high_flood_increase"
    )

    specs: list[tuple] = []
    high_cols: list[str] = []
    for code, short, label, value_col, unit in FACTOR_SPECS:
        flag = f"high_{short}"
        if flag not in df or value_col not in df or value_col not in cutoffs:
            if code == 4:
                continue
            raise ValueError(
                f"Saved factor {short} lacks value, flag, or cutoff; 13i will not infer it."
            )
        df[flag] = normalize_bool(df[flag], flag, nullable=(code == 4))
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        if code != 4 and not np.isfinite(df[value_col]).all():
            raise ValueError(f"Non-finite values in {value_col}")
        specs.append((code, short, label, value_col, unit, flag, float(cutoffs[value_col])))
        high_cols.append(flag)

    if not specs:
        raise ValueError("No saved factor flags available.")

    df["factor_count"] = df[high_cols].fillna(False).sum(axis=1).astype(int)
    df["factor_dense"] = df.factor_count.ge(min_factors)
    df["dense_high_recent"] = df.factor_dense & df.high_recent_flood
    df["dense_worsening"] = df.factor_dense & df.high_flood_increase
    df["dense_stable_or_improving"] = df.factor_dense & df[D].le(0)

    df["saved_factor_codes"] = df.apply(
        lambda r: "+".join(
            str(code)
            for code, short, *_ in specs
            if pd.notna(r[f"high_{short}"]) and bool(r[f"high_{short}"])
        ),
        axis=1,
    )

    return df, meta, specs


# -----------------------------------------------------------------------------
# Factor display/ranking statistics
# -----------------------------------------------------------------------------

def prepare_factor_statistics(df: pd.DataFrame, specs: list[tuple], top_n: int) -> pd.DataFrame:
    """Add QC-aware factor statistics and select flood-relevant factor hotspots.

    National percentiles are calculated across all ADM3 with usable factor values
    so the mini-bars retain a national interpretation.

    Factor Top-N ranks are calculated ONLY among ADM3 with a meaningful saved
    flood signal:

        flood_relevant = high_recent_flood OR high_flood_increase

    This prevents factor-rich but flood-unaffected ADM3 from occupying the core
    dashboard. No new hazard threshold is invented: the saved 13d hazard flags
    are reused.

    For cropland, a zero from the available ASAP crop mask is treated as N/A for
    display/ranking because it does not establish absence of local cultivation.
    The raw zero remains preserved in the audit output.
    """
    out = df.copy()

    out["flood_relevant"] = (
        out["high_recent_flood"].astype(bool)
        | out["high_flood_increase"].astype(bool)
    )

    for code, short, label, value_col, unit, flag, cutoff in specs:
        raw = pd.to_numeric(out[value_col], errors="coerce")
        usable = raw.copy()

        if code == 2:
            usable = usable.mask(usable.eq(0))

        out[f"factor_{code}_raw"] = raw
        out[f"factor_{code}_usable"] = usable
        out[f"factor_{code}_qc_na"] = raw.notna() & usable.isna()

        # National percentile for visual comparison in mini-bars.
        out[f"factor_{code}_percentile"] = (
            usable.rank(method="average", pct=True) * 100.0
        )

        # Top-N ranks only among flood-relevant ADM3.
        relevant_values = usable.where(out["flood_relevant"])
        rank = relevant_values.rank(method="min", ascending=False)
        out[f"factor_{code}_rank"] = rank.astype("Float64")
        out[f"factor_{code}_top_n"] = (
            out["flood_relevant"] & rank.le(top_n).fillna(False)
        )

    top_cols = [f"factor_{code}_top_n" for code, *_ in specs]
    out["selected_factor_hotspot"] = out[top_cols].any(axis=1)
    out["top_factor_count"] = out[top_cols].sum(axis=1).astype(int)
    out["top_factor_codes"] = out.apply(
        lambda r: "+".join(
            str(code)
            for code, *_ in specs
            if bool(r[f"factor_{code}_top_n"])
        ),
        axis=1,
    )

    return out

def factor_value_text(code: int, value, unit: str, qc_na: bool = False) -> str:
    if qc_na or pd.isna(value) or not np.isfinite(float(value)):
        return "N/A"
    x = float(value)
    if code == 1:
        return f"{int(round(x)):,} facilities"
    if code == 4:
        return f"{int(round(x)):,} people"
    if 0 < x < 0.1:
        return "<0.1 km²"
    return f"{x:,.1f} km²"


def compact_factor_value(code: int, value, qc_na: bool = False) -> str:
    """Compact value for the dashboard without losing units."""
    if qc_na or pd.isna(value) or not np.isfinite(float(value)):
        return "N/A"
    x = float(value)
    if code == 1:
        return f"{int(round(x)):,} fac."
    if code == 4:
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:.1f}M ppl"
        if abs(x) >= 1_000:
            return f"{x/1_000:.1f}k ppl"
        return f"{int(round(x)):,} ppl"
    if 0 < x < 0.1:
        return "<0.1 km²"
    if abs(x) >= 10_000:
        return f"{x/1_000:.1f}k km²"
    return f"{x:,.1f} km²"


# -----------------------------------------------------------------------------
# Flood-status classification
# -----------------------------------------------------------------------------

def classify_flood_status(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    cutoffs = meta.get("numeric_thresholds", {})
    if H not in cutoffs:
        raise ValueError(
            f"Saved hazard cutoff for {H} is missing; flood-status classes will not invent one."
        )
    severe_cutoff = float(cutoffs[H])

    recent_high = df.high_recent_flood.astype(bool)
    worsening_high = df.high_flood_increase.astype(bool)
    baseline_high = df[B].ge(severe_cutoff)
    nonpositive = df[D].le(0)

    status = pd.Series("Moderate / mixed change", index=df.index, dtype="string")
    status.loc[nonpositive & ~recent_high] = "Stable / improving"
    status.loc[recent_high & ~worsening_high] = "Severe, no clear worsening"
    status.loc[~recent_high & worsening_high] = "Worsening, not yet severe"
    status.loc[recent_high & worsening_high & ~baseline_high] = "Recently worsened + severe"
    status.loc[recent_high & worsening_high & baseline_high] = "Persistent severe + worsening"

    out = df.copy()
    out["historical_reference_high"] = baseline_high
    out["flood_status"] = status
    out["flood_status_severity_cutoff"] = severe_cutoff
    return out


# -----------------------------------------------------------------------------
# Geography
# -----------------------------------------------------------------------------

def load_geo(path: Path, df: pd.DataFrame) -> gpd.GeoDataFrame:
    geo = gpd.read_file(path)
    if geo.crs is None or "adm3_pcode" not in geo:
        raise ValueError("ADM3 GeoJSON must declare CRS and contain adm3_pcode.")
    geo = geo.copy()
    geo["adm_id"] = geo.adm3_pcode.astype("string").str.strip().str.upper()
    if geo.adm_id.duplicated().any():
        raise ValueError("Duplicate ADM3 PCODEs in boundary file.")
    if set(geo.adm_id) != set(df.adm_id):
        raise ValueError(
            "ADM3 IDs differ between saved analysis and boundary; no rows will be dropped."
        )
    return geo.merge(df, on="adm_id", how="left", validate="one_to_one")


# -----------------------------------------------------------------------------
# Part A: factor-dense outcome maps
# -----------------------------------------------------------------------------

def fmt_total(value, unit: str) -> str:
    if pd.isna(value) or not np.isfinite(float(value)):
        return "N/A"
    x = float(value)
    if unit == "facilities":
        return f"{int(round(x)):,} facilities"
    if unit == "people":
        return f"{int(round(x)):,} people"
    return f"{x:,.1f} {unit}"


def plot_factor_dense_outcome(
    geo: gpd.GeoDataFrame,
    df: pd.DataFrame,
    mask: pd.Series,
    specs: list[tuple],
    title: str,
    subtitle: str,
    metric: str,
    cmap: str,
    norm,
    outline_colour: str,
    outdir: Path,
    stem: str,
    dpi: int,
) -> list[str]:
    fig = plt.figure(figsize=(16.5, 9.8), facecolor="white")
    fig.text(.045, .95, title, fontsize=23, weight="bold", color=INK)
    fig.text(.045, .91, subtitle, fontsize=11, color=MUTED)

    ax = fig.add_axes([.04, .12, .66, .74])
    side = fig.add_axes([.74, .15, .225, .67])
    side.set_axis_off(); side.set_xlim(0, 1); side.set_ylim(0, 1)

    geo.plot(
        column=metric,
        cmap=cmap,
        norm=norm,
        ax=ax,
        edgecolor="#EDF1F4",
        linewidth=.25,
        zorder=1,
    )
    country = unary_union(geo.geometry)
    gpd.GeoSeries([country], crs=geo.crs).boundary.plot(
        ax=ax, color=INK, linewidth=1, zorder=3
    )

    selected = geo.loc[mask.to_numpy()]
    if len(selected):
        selected.boundary.plot(ax=ax, color="white", linewidth=3, zorder=4)
        selected.boundary.plot(ax=ax, color=outline_colour, linewidth=1.7, zorder=4.2)
    ax.set_axis_off(); ax.set_aspect("equal")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([.10, .075, .51, .018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(
        "Mean annual recorded flood-activity days (2019–2025)"
        if metric == H
        else "Difference in mean annual flood-activity days (2019–2025 − 2000–2018)",
        fontsize=9.5,
    )

    part = df.loc[mask]
    side.text(0, .99, "OUTCOME SUMMARY", va="top", fontsize=11, weight="bold", color=MUTED)
    side.text(0, .91, str(len(part)), va="top", fontsize=36, weight="bold", color=INK)
    side.text(
        .30,
        .875,
        f"ADM3 across {part.parent_adm2_name.nunique() if len(part) else 0} ADM2",
        va="top",
        fontsize=10,
        color=INK,
    )

    side.text(0, .76, "HIGH-FACTOR COUNTS", va="top", fontsize=10.5, weight="bold", color=INK)
    y = .69
    for code, short, label, value_col, unit, flag, cutoff in specs:
        side.text(
            .035, y, str(code), ha="center", va="center", fontsize=9, weight="bold", color="white",
            bbox={"boxstyle": "circle,pad=0.22", "fc": FACTOR_COLORS[code], "ec": "white", "lw": 1},
        )
        side.text(.10, y, label, va="center", fontsize=9.2, color=INK)
        side.text(
            .98, y, str(int((mask & df[flag].fillna(False)).sum())),
            ha="right", va="center", fontsize=10, weight="bold", color=INK,
        )
        y -= .067

    side.text(0, .38, "ADM3 TOTALS IN THIS OUTCOME", va="top", fontsize=10.5, weight="bold", color=INK)
    y = .32
    for code, short, label, value_col, unit, flag, cutoff in specs:
        vals = pd.to_numeric(part[value_col], errors="coerce")
        total = vals.sum(min_count=1)
        side.text(0, y, f"{code} {label}", va="top", fontsize=9, color=INK)
        side.text(
            .98, y, fmt_total(total, unit) if pd.notna(total) else "N/A",
            ha="right", va="top", fontsize=9, weight="bold", color=INK,
        )
        y -= .058

    fig.text(
        .045, .025,
        "Factor-dense uses saved 13d thresholds only; no thresholds are recalculated. "
        "ADM3 totals are exposure context, not confirmed affected people or losses.",
        fontsize=8.5, color=MUTED,
    )

    outputs = []
    for ext in ("png", "svg"):
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


# -----------------------------------------------------------------------------
# Part B: dashboard helpers
# -----------------------------------------------------------------------------

def add_map_labels(
    ax,
    page_geo: gpd.GeoDataFrame,
    ids: list[str],
    edge_color: str,
    text_color: str | None = None,
    line_color: str | None = None,
) -> None:
    """Add map IDs with coloured circle outlines for a given table group."""
    if text_color is None:
        text_color = edge_color
    points = page_geo.geometry.representative_point()
    for ident, point in zip(ids, points):
        ax.text(
            point.x,
            point.y,
            ident,
            ha="center",
            va="center",
            fontsize=10.2,
            weight="bold",
            color=text_color,
            bbox={
                "boxstyle": "circle,pad=0.28",
                "fc": "white",
                "ec": edge_color,
                "lw": 1.8,
            },
            zorder=8,
        )


def draw_factor_legend(fig, specs: list[tuple]) -> None:
    x = .045
    y = .085
    fig.text(x, y + .032, "FACTOR COLOURS", fontsize=9.5, weight="bold", color=INK)
    cursor = x
    for code, short, label, value_col, unit, flag, cutoff in specs:
        fig.text(
            cursor,
            y,
            str(code),
            ha="center",
            va="center",
            fontsize=8.5,
            weight="bold",
            color="white",
            bbox={
                "boxstyle": "circle,pad=0.20",
                "fc": FACTOR_COLORS[code],
                "ec": "white",
                "lw": 1,
            },
        )
        fig.text(cursor + .012, y, label, va="center", fontsize=8.5, color=INK)
        cursor += .14 if code != 1 else .16


def draw_mini_bars(
    ax,
    row: pd.Series,
    y_center: float,
    specs: list[tuple],
    x0: float,
    width: float,
    height: float,
) -> None:
    """Four independent percentile bars sharing the same 0-100 scale."""
    gap = height * 0.22
    bar_h = (height - gap * (len(specs) - 1)) / len(specs)
    top = y_center + height / 2

    for j, (code, short, label, value_col, unit, flag, cutoff) in enumerate(specs):
        y = top - (j + 1) * bar_h - j * gap
        pct = row.get(f"factor_{code}_percentile", np.nan)
        qc_na = bool(row.get(f"factor_{code}_qc_na", False))

        # Numeric label before each bar ensures colour is not the only encoding.
        ax.text(
            x0 - .010,
            y + bar_h / 2,
            str(code),
            ha="right",
            va="center",
            fontsize=7.3,
            weight="bold",
            color=INK,
            transform=ax.transAxes,
        )

        ax.add_patch(
            Rectangle(
                (x0, y), width, bar_h,
                transform=ax.transAxes,
                facecolor=BAR_BG,
                edgecolor="none",
                zorder=1,
            )
        )

        if not qc_na and pd.notna(pct) and np.isfinite(float(pct)):
            fill = max(0.0, min(100.0, float(pct))) / 100.0
            ax.add_patch(
                Rectangle(
                    (x0, y), width * fill, bar_h,
                    transform=ax.transAxes,
                    facecolor=FACTOR_COLORS[code],
                    edgecolor="none",
                    zorder=2,
                )
            )

        # Small marker if this factor is one of the Top-N selection reasons.
        if bool(row.get(f"factor_{code}_top_n", False)):
            ax.text(
                x0 + width + .006,
                y + bar_h / 2,
                "★",
                ha="left",
                va="center",
                fontsize=6.8,
                color=FACTOR_COLORS[code],
                transform=ax.transAxes,
            )


def value_lines(row: pd.Series, specs: list[tuple]) -> str:
    lines = []
    for code, short, label, value_col, unit, flag, cutoff in specs:
        raw = row.get(f"factor_{code}_raw", np.nan)
        qc_na = bool(row.get(f"factor_{code}_qc_na", False))
        lines.append(f"{code} {compact_factor_value(code, raw, qc_na)}")
    return "\n".join(lines)


def status_short(status: str) -> str:
    mapping = {
        "Persistent severe + worsening": "Persistent severe\n+ worsening",
        "Recently worsened + severe": "Recently worsened\n+ severe",
        "Severe, no clear worsening": "Severe\nno clear worsening",
        "Worsening, not yet severe": "Worsening\nnot yet severe",
        "Stable / improving": "Stable /\nimproving",
        "Moderate / mixed change": "Moderate /\nmixed change",
    }
    return mapping.get(str(status), str(status))



def not_selected_reason(row: pd.Series, specs: list[tuple]) -> str:
    """Explain why a flood-relevant ADM3 did not enter the selected factor-hotspot union."""
    if any(bool(row.get(f"factor_{code}_top_n", False)) for code, *_ in specs):
        return "Selected factor hotspot"

    qc_codes = [
        code for code, *_ in specs
        if bool(row.get(f"factor_{code}_qc_na", False))
    ]

    if qc_codes:
        return (
            "No factor Top-N; "
            + "N/A/QC limitation in factor "
            + ", ".join(map(str, qc_codes))
        )

    return "Flood-relevant, but no factor ranked in the selected Top-N"


def compact_all_factor_values(row: pd.Series, specs: list[tuple]) -> str:
    """Return all four actual/QC-aware factor quantities in a compact four-line block."""
    lines = []
    for code, short, label, value_col, unit, flag, cutoff in specs:
        raw = row.get(f"factor_{code}_raw", np.nan)
        qc_na = bool(row.get(f"factor_{code}_qc_na", False))
        lines.append(f"{code} {compact_factor_value(code, raw, qc_na)}")
    return "\n".join(lines)


def dashboard_pages(
    geo: gpd.GeoDataFrame,
    df: pd.DataFrame,
    specs: list[tuple],
    outdir: Path,
    top_per_factor: int,
    rows_per_page: int,
    dpi: int,
) -> list[str]:
    """Create a two-table dashboard with map labels for both tables."""
    selected = df.loc[df.selected_factor_hotspot].copy()
    if selected.empty:
        raise ValueError("No factor-hotspot ADM3 selected. Check factor values and --top-per-factor.")

    pct_cols = [f"factor_{code}_percentile" for code, *_ in specs]
    selected["best_factor_percentile"] = selected[pct_cols].max(axis=1, skipna=True)

    status_priority = {
        "Persistent severe + worsening": 0,
        "Recently worsened + severe": 1,
        "Severe, no clear worsening": 2,
        "Worsening, not yet severe": 3,
        "Moderate / mixed change": 4,
        "Stable / improving": 5,
    }
    selected["flood_status_priority"] = (
        selected["flood_status"].map(status_priority).fillna(99).astype(int)
    )

    selected = selected.sort_values(
        ["flood_status_priority", "top_factor_count", "best_factor_percentile", "adm_id"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    unselected = df.loc[df["flood_relevant"] & ~df["selected_factor_hotspot"]].copy()
    unselected["best_factor_percentile"] = unselected[pct_cols].max(axis=1, skipna=True)
    unselected["flood_status_priority"] = (
        unselected["flood_status"].map(status_priority).fillna(99).astype(int)
    )
    unselected["not_selected_reason"] = unselected.apply(
        lambda r: not_selected_reason(r, specs), axis=1
    )
    unselected = unselected.sort_values(
        ["flood_status_priority", H, D, "best_factor_percentile", "adm_id"],
        ascending=[True, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    selected.to_csv(outdir / "04_selected_factor_hotspots_full.csv", index=False)
    unselected.to_csv(outdir / "05_unselected_flood_relevant_full.csv", index=False)

    total_pages = max(
        math.ceil(len(selected) / rows_per_page),
        math.ceil(len(unselected) / rows_per_page) if len(unselected) else 1,
    )
    outputs: list[str] = []

    hmax = max(1, math.ceil(float(df[H].max()) / 25) * 25)
    hnorm = Normalize(0, hmax)

    geo_indexed = geo.set_index("adm_id")

    for page in range(total_pages):
        start_i = page * rows_per_page
        stop_i = start_i + rows_per_page

        table1 = selected.iloc[start_i:stop_i].copy().reset_index(drop=True)
        table2 = unselected.iloc[start_i:stop_i].copy().reset_index(drop=True)

        ids1 = list(string.ascii_uppercase[:len(table1)])
        ids2 = list(string.ascii_lowercase[:len(table2)])

        if len(table1):
            table1.insert(0, "map_id", ids1)
            page_geo1 = geo_indexed.loc[table1.adm_id].reset_index()
        else:
            page_geo1 = geo.iloc[0:0].copy()

        if len(table2):
            table2.insert(0, "map_id", ids2)
            page_geo2 = geo_indexed.loc[table2.adm_id].reset_index()
        else:
            page_geo2 = geo.iloc[0:0].copy()

        fig = plt.figure(figsize=(25, 15.5), facecolor="white")

        fig.text(
            .035, .958,
            "Flood-relevant factor hotspots and severe-flood context",
            fontsize=24, weight="bold", color=INK,
        )
        fig.text(
            .035, .927,
            f"Page {page + 1}/{total_pages}. Orange labels = Table 1 selected hotspots. Light-cyan labels = Table 2 unselected flood-relevant areas.",
            fontsize=11.5, color=MUTED,
        )

        # LEFT MAP
        ax = fig.add_axes([.025, .20, .52, .68])
        geo.plot(
            column=H,
            cmap="Blues",
            norm=hnorm,
            edgecolor="#F4F6F8",
            linewidth=.20,
            ax=ax,
            zorder=1,
        )

        current_ids = set()
        if len(page_geo1):
            current_ids |= set(page_geo1.adm_id)
        if len(page_geo2):
            current_ids |= set(page_geo2.adm_id)
        if current_ids:
            other = geo.loc[~geo.adm_id.isin(current_ids)]
            other.plot(
                ax=ax, facecolor="white", alpha=.24, edgecolor="none", zorder=1.4
            )

        country = unary_union(geo.geometry)
        gpd.GeoSeries([country], crs=geo.crs).boundary.plot(
            ax=ax, color=INK, linewidth=.9, zorder=3
        )

        if len(page_geo2):
            page_geo2.boundary.plot(ax=ax, color="white", linewidth=2.2, zorder=4.6)
            page_geo2.boundary.plot(ax=ax, color=UNSELECTED_LABEL, linewidth=1.35, zorder=4.8)
            add_map_labels(ax, page_geo2, ids2, edge_color=UNSELECTED_LABEL, text_color=UNSELECTED_LABEL_TEXT)

        if len(page_geo1):
            page_geo1.boundary.plot(ax=ax, color="white", linewidth=2.6, zorder=5.2)
            page_geo1.boundary.plot(ax=ax, color=SELECTED_LABEL, linewidth=1.6, zorder=5.4)
            add_map_labels(ax, page_geo1, ids1, edge_color=SELECTED_LABEL, text_color=SELECTED_LABEL)

        ax.set_axis_off()
        ax.set_aspect("equal")

        cax = fig.add_axes([.10, .165, .37, .014])
        cb = fig.colorbar(
            ScalarMappable(norm=hnorm, cmap="Blues"),
            cax=cax,
            orientation="horizontal",
        )
        cb.ax.tick_params(labelsize=9)
        cb.set_label(
            "Mean annual recorded flood-activity days (2019–2025)",
            fontsize=10,
        )

        # TABLE 1 — SELECTED
        t1 = fig.add_axes([.565, .515, .415, .39])
        t1.set_axis_off(); t1.set_xlim(0, 1); t1.set_ylim(0, 1)

        t1.text(0, .995, "TABLE 1 — SELECTED FLOOD-RELEVANT FACTOR HOTSPOTS", fontsize=13.5, weight="bold", color=INK, va="top")
        t1.text(0, .958, "Mini-bars show national percentiles; values show actual ADM3 totals.", fontsize=9.4, color=MUTED, va="top")

        x_id = .015; x_name = .075; x_profile = .34; x_values = .585; x_status = .805
        for x, label in [(x_id, "ID"), (x_name, "ADM3 / Parent ADM2"), (x_profile, "Factor percentile profile"), (x_values, "Actual factor totals"), (x_status, "Flood status")]:
            t1.text(x, .895, label, fontsize=9.6, weight="bold", color=INK, va="center")
        t1.plot([0, 1], [.865, .865], color=GRID, lw=1)

        step1 = .094 if rows_per_page <= 8 else .082
        y01 = .822
        if table1.empty:
            t1.text(.02, .80, "No selected factor hotspots on this page.", fontsize=10.5, color=MUTED, va="top")

        for i, row in table1.iterrows():
            y = y01 - i * step1
            if i % 2 == 0:
                t1.add_patch(Rectangle((0, y - step1 * .43), 1, step1 * .86, transform=t1.transAxes, facecolor="#F7F9FB", edgecolor="none", zorder=0))
            t1.text(x_id, y, row.map_id, fontsize=10.8, weight="bold", color=SELECTED_LABEL, va="center")
            t1.text(x_name, y, f"{row.region_name}\n({row.parent_adm2_name})", fontsize=8.7, color=INK, va="center", linespacing=1.05)
            draw_mini_bars(t1, row, y_center=y, specs=specs, x0=x_profile, width=.17, height=step1 * .62)
            t1.text(x_values, y, compact_all_factor_values(row, specs), fontsize=7.8, color=INK, va="center", linespacing=1.02)
            t1.text(x_status, y + .012, status_short(row.flood_status), fontsize=8.3, weight="bold", color=INK, va="center", linespacing=1.0)
            t1.text(x_status, y - .030, f"Recent {row[H]:.1f} d/y | Δ {row[D]:+.1f}", fontsize=7.2, color=MUTED, va="center")

        # TABLE 2 — UNSELECTED
        t2 = fig.add_axes([.565, .095, .415, .36])
        t2.set_axis_off(); t2.set_xlim(0, 1); t2.set_ylim(0, 1)

        t2.text(0, .995, "TABLE 2 — FLOOD-RELEVANT AREAS NOT SELECTED AS FACTOR HOTSPOTS", fontsize=12.4, weight="bold", color=INK, va="top")
        t2.text(0, .955, "These areas have a saved severe/worsening flood signal but did not enter any factor Top-N.", fontsize=9.0, color=MUTED, va="top")

        x_id2 = .015; x_name2 = .065; x_flood2 = .27; x_reason2 = .50; x_values2 = .79
        for x, label in [(x_id2, "ID"), (x_name2, "ADM3 / Parent ADM2"), (x_flood2, "Flood condition"), (x_reason2, "Why not selected"), (x_values2, "All factor totals")]:
            t2.text(x, .88, label, fontsize=9.0, weight="bold", color=INK, va="center")
        t2.plot([0, 1], [.845, .845], color=GRID, lw=1)

        step2 = .086 if rows_per_page <= 8 else .076
        y02 = .805
        if table2.empty:
            t2.text(.02, .80, "No unselected flood-relevant ADM3 on this page.", fontsize=10, color=MUTED, va="top")

        for i, row in table2.iterrows():
            y = y02 - i * step2
            if i % 2 == 0:
                t2.add_patch(Rectangle((0, y - step2 * .43), 1, step2 * .86, transform=t2.transAxes, facecolor="#F8F9FA", edgecolor="none", zorder=0))
            t2.text(x_id2, y, row.map_id, fontsize=10.2, weight="bold", color=UNSELECTED_LABEL_TEXT, va="center")
            t2.text(x_name2, y, f"{row.region_name}\n({row.parent_adm2_name})", fontsize=8.0, color=INK, va="center", linespacing=1.02)
            t2.text(x_flood2, y + .009, status_short(row.flood_status), fontsize=7.6, weight="bold", color=INK, va="center", linespacing=1.0)
            t2.text(x_flood2, y - .030, f"Recent {row[H]:.1f} d/y | Δ {row[D]:+.1f}", fontsize=6.9, color=MUTED, va="center")
            t2.text(x_reason2, y, textwrap.fill(str(row.not_selected_reason), width=26), fontsize=7.2, color=MUTED, va="center", linespacing=1.0)
            t2.text(x_values2, y, compact_all_factor_values(row, specs), fontsize=7.1, color=INK, va="center", linespacing=1.0)

        draw_factor_legend(fig, specs)
        fig.text(.565, .070, "Orange IDs/circles = Table 1 selected hotspots. Light-cyan IDs/circles = Table 2 unselected flood-relevant areas.", fontsize=8.8, color=MUTED)
        fig.text(.565, .051, "★ = factor used in the selected Top-N union. Blank/short mini-bars do not mean zero; read the actual factor totals beside them.", fontsize=8.2, color=MUTED)
        fig.text(.565, .033, "Cropland zero from the available ASAP mask is displayed as N/A and excluded from cropland percentile/Top-N ranking.", fontsize=8.2, color=MUTED)
        fig.text(.035, .021, "Descriptive exposure/hazard screening only. Factor totals are ADM3 administrative totals, not confirmed inundated assets, affected people, losses, or forecasts.", fontsize=8.5, color=MUTED)

        if len(table1):
            table1.to_csv(outdir / f"06_selected_table_page_{page + 1:02d}.csv", index=False)
        if len(table2):
            table2.to_csv(outdir / f"07_unselected_table_page_{page + 1:02d}.csv", index=False)

        for ext in ("png", "svg"):
            path = outdir / f"04_factor_hotspot_two_tables_page_{page + 1:02d}.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            outputs.append(path.name)
        plt.close(fig)

    return outputs


# -----------------------------------------------------------------------------
# Outputs and audits
# -----------------------------------------------------------------------------

def write_factor_hotspot_summary(df: pd.DataFrame, specs: list[tuple], out: Path) -> None:
    rows = []
    for code, short, label, value_col, unit, flag, cutoff in specs:
        part = df.loc[df[f"factor_{code}_top_n"]].copy()
        part = part.sort_values(
            [f"factor_{code}_usable", "adm_id"],
            ascending=[False, True],
            kind="stable",
        )
        for rank, (_, r) in enumerate(part.iterrows(), start=1):
            rows.append(
                {
                    "factor_number": code,
                    "factor": label,
                    "factor_rank": rank,
                    "adm_id": r.adm_id,
                    "region_name": r.region_name,
                    "parent_adm2_name": r.parent_adm2_name,
                    "raw_value": r[f"factor_{code}_raw"],
                    "usable_value": r[f"factor_{code}_usable"],
                    "unit": unit,
                    "national_percentile": r[f"factor_{code}_percentile"],
                    "saved_high_flag": r[flag],
                    "saved_cutoff": cutoff,
                    "flood_status": r.flood_status,
                    H: r[H],
                    D: r[D],
                    B: r[B],
                }
            )
    pd.DataFrame(rows).to_csv(out / "02_factor_topn_by_factor.csv", index=False)


def write_full_audit(df: pd.DataFrame, specs: list[tuple], out: Path) -> None:
    keep = [
        "adm_id",
        "region_name",
        "parent_adm2_name",
        H,
        D,
        B,
        "high_recent_flood",
        "high_flood_increase",
        "historical_reference_high",
        "flood_status",
        "factor_count",
        "factor_dense",
        "dense_high_recent",
        "dense_worsening",
        "dense_stable_or_improving",
        "saved_factor_codes",
        "flood_relevant",
        "selected_factor_hotspot",
        "top_factor_count",
        "top_factor_codes",
    ]
    for code, short, label, value_col, unit, flag, cutoff in specs:
        keep.extend(
            [
                value_col,
                flag,
                f"factor_{code}_raw",
                f"factor_{code}_usable",
                f"factor_{code}_qc_na",
                f"factor_{code}_percentile",
                f"factor_{code}_rank",
                f"factor_{code}_top_n",
            ]
        )
    df[list(dict.fromkeys(keep))].to_csv(out / "01_adm3_factor_hotspot_flood_status_audit.csv", index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        help="Exact 13d scenario folder containing CSV + methodology JSON",
    )
    parser.add_argument("--adm3-geojson", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--min-factors",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="Saved high-factor flags required for Part A factor-dense maps",
    )
    parser.add_argument(
        "--top-per-factor",
        type=int,
        default=10,
        choices=range(1, 21),
        help="Top-N ADM3 selected independently for each factor (default 10)",
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=8,
        choices=range(5, 13),
        help="Rows shown on each factor-hotspot dashboard page (default 8)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-part-a",
        action="store_true",
        help="Skip the three factor-dense outcome maps and create only the hotspot dashboard",
    )
    args = parser.parse_args(argv)

    if not 72 <= args.dpi <= 600:
        raise ValueError("--dpi must be between 72 and 600")

    root = args.root.expanduser().resolve() if args.root else project_root()
    base = root / "processed_data" / "multivariate_flood_modeling"
    folder = find_scenario(base, resolve(args.input, root))

    df, meta, specs = load_saved(folder, args.min_factors)
    df = classify_flood_status(df, meta)
    df = prepare_factor_statistics(df, specs, args.top_per_factor)

    boundary = (
        resolve(args.adm3_geojson, root)
        or root / "raw_data" / "administrative_boundaries" / "ssd_admin3.geojson"
    )
    geo = load_geo(boundary, df)

    out = (
        resolve(args.out, root)
        or base / "13i_factor_hotspot_two_tables_dual_labels"
    )
    maps = out / "maps"
    maps.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
            "savefig.facecolor": "white",
        }
    )

    generated: list[str] = []

    # Part A: preserve prior factor-dense outcome maps.
    if not args.no_part_a:
        hmax = max(1, math.ceil(float(df[H].max()) / 25) * 25)
        hnorm = Normalize(0, hmax)
        dlim = max(1, math.ceil(float(df[D].abs().max()) / 20) * 20)
        dnorm = TwoSlopeNorm(vmin=-dlim, vcenter=0, vmax=dlim)

        generated += plot_factor_dense_outcome(
            geo,
            df,
            df.dense_high_recent,
            specs,
            "Factor-dense areas with high recent flood activity",
            f"Factor-dense = at least {args.min_factors} saved high exposure-factor flags.",
            H,
            "Blues",
            hnorm,
            FACTOR_COLORS[1],
            maps,
            "01_factor_dense_high_recent_flood",
            args.dpi,
        )
        generated += plot_factor_dense_outcome(
            geo,
            df,
            df.dense_worsening,
            specs,
            "Factor-dense areas with worsening flood activity",
            f"Factor-dense = at least {args.min_factors} saved high exposure-factor flags; worsening uses the saved high-change screen.",
            D,
            "RdBu_r",
            dnorm,
            FACTOR_COLORS[4],
            maps,
            "02_factor_dense_worsening_flood",
            args.dpi,
        )
        # Stable/improving factor-dense areas remain in the audit CSV but are
        # intentionally omitted from the core maps. The main figure set focuses
        # on ADM3 with current severe or worsening flood signals.

    # Part B: new factor-hotspot dashboard.
    generated += dashboard_pages(
        geo,
        df,
        specs,
        maps,
        args.top_per_factor,
        args.rows_per_page,
        args.dpi,
    )

    write_full_audit(df, specs, out)
    write_factor_hotspot_summary(df, specs, out)

    selected = df.loc[df.selected_factor_hotspot].copy()
    status_counts = selected.flood_status.value_counts().reindex(STATUS_ORDER, fill_value=0)
    status_counts.rename_axis("flood_status").reset_index(name="selected_adm3").to_csv(
        out / "03_selected_hotspot_flood_status_counts.csv",
        index=False,
    )

    factor_summary = []
    for code, short, label, value_col, unit, flag, cutoff in specs:
        factor_summary.append(
            {
                "factor_number": code,
                "factor": label,
                "colour": FACTOR_COLORS[code],
                "saved_cutoff": cutoff,
                "unit": unit,
                "usable_adm3": int(df[f"factor_{code}_usable"].notna().sum()),
                "qc_na_adm3": int(df[f"factor_{code}_qc_na"].sum()),
                "top_n_selected": int(df[f"factor_{code}_top_n"].sum()),
            }
        )
    pd.DataFrame(factor_summary).to_csv(out / "04_factor_method_summary.csv", index=False)

    report = {
        "version": VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_scenario": str(folder),
        "thresholds_recalculated": False,
        "saved_numeric_thresholds": meta.get("numeric_thresholds"),
        "min_saved_high_factors_for_part_a": args.min_factors,
        "top_per_factor": args.top_per_factor,
        "factor_hotspot_selection_rule": "Top-N factor values are ranked only within ADM3 where saved high_recent_flood OR high_flood_increase is true; mini-bar percentiles remain national.",
        "dashboard_tables": {
            "table_1": "Selected flood-relevant factor hotspots with mini-bars, actual factor totals and flood status.",
            "table_2": "Flood-relevant ADM3 not selected as factor hotspots, with selection reason and all factor totals.",
            "map_label_scheme": "Orange uppercase IDs = Table 1 selected hotspots; light-cyan lowercase IDs = Table 2 unselected flood-relevant areas."
        },
        "rows_per_dashboard_page": args.rows_per_page,
        "flood_relevant_adm3": int(df.flood_relevant.sum()),
        "selected_union_adm3": int(df.selected_factor_hotspot.sum()),
        "multi_factor_topn_adm3": int((df.top_factor_count >= 2).sum()),
        "factor_colours": {str(k): v for k, v in FACTOR_COLORS.items()},
        "factor_bar_definition": "National percentile (0-100) among usable ADM3 values; four independent bars with common zero origin.",
        "factor_percentiles_are_new_thresholds": False,
        "cropland_quality_rule": {
            "raw_zero_preserved_in_audit": True,
            "zero_used_for_display": False,
            "zero_used_for_percentile": False,
            "zero_used_for_top_n_ranking": False,
            "display": "N/A",
            "reason": "Available ASAP crop mask may not reliably capture local cultivation in some ADM3 areas.",
        },
        "flood_status_definition": {
            "recent_severe": "saved high_recent_flood flag",
            "worsening": "saved high_flood_increase flag",
            "historical_reference_high": f"earlier-period mean >= saved {H} cutoff",
            "stable_or_improving": "period-mean difference <= 0",
        },
        "selected_hotspot_status_counts": status_counts.to_dict(),
        "generated_files": generated,
        "limitations": [
            "Top-N factor selection is descriptive ranking, not a humanitarian trigger.",
            "Factor percentiles compare unlike variables only after within-factor ranking; raw units are never added together.",
            "Factor-dense remains based on saved threshold flags and is not a validated vulnerability index.",
            "ADM3 factor totals are administrative totals, not confirmed inundated or affected quantities.",
            "Cropland zero is treated as N/A for this visualization/ranking under the current project QC decision; raw zeros remain available for audit.",
            "Flood-status labels are descriptive screening classes; no statistical significance or causal interpretation is implied.",
            "The analysis is retrospective and is not a forecast.",
        ],
    }
    (out / "05_methodology_and_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Source scenario: {folder}")
    print(f"[OK] Factor-dense ADM3 (>= {args.min_factors} saved high factors): {int(df.factor_dense.sum())}")
    print(f"[OK] Flood-relevant ADM3: {int(df.flood_relevant.sum())}")
    print(f"[OK] Factor-hotspot union among flood-relevant ADM3 (Top {args.top_per_factor} per factor): {int(df.selected_factor_hotspot.sum())}")
    print(f"[OK] Multi-factor Top-N overlaps: {int((df.top_factor_count >= 2).sum())}")
    print("[OK] Selected hotspot flood-status counts:")
    for status, count in status_counts.items():
        print(f"     {status}: {int(count)}")
    print(f"[OK] Output: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
