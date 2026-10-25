#!/usr/bin/env python3
"""13d v4 | Population-aware screening, raster audit, and explanatory maps.

Save in scripts/multivariate_flood_modeling/ and run without arguments.
Reads the existing 13c folder/ZIP; never edits 13b/13c or the original rasters.
Default output: processed_data/multivariate_flood_modeling/
                13d_adm3_humanitarian_priority_population/

1. Keep the three-asset baseline for comparison.
2. A low valid-value pixel fraction is diagnostic, not a missing-population rate.
3. Check 2025 population, optional local GeoTIFF and reviewed product metadata.
4. Population becomes factor 4 in the main scenario only after processing checks.
   Otherwise write a clearly labelled provisional four-factor sensitivity scenario.
5. Unknown population remains unknown; no fill with zero, no inverse-fraction scaling.

Map colour: recent activity (2019-2025) or the difference from 2000-2018.
Circle codes: 1 health; 2 cropland; 3 rangeland; 4 estimated population (2025).
No weighted score, causal factor claim, impact estimate, or future forecast.
Map 10 covers the union of the high-activity and high-change selection routes.
Map 11 covers ALL high-change areas with factors, not just the change-only group.

Required: numpy pandas matplotlib geopandas shapely pyproj.
Optional raster audit: rasterio. No web downloads or network calls are made.
References:
 https://www.worldpop.org/methods/top_down_constrained_vs_unconstrained/
 https://rasterio.readthedocs.io/en/stable/topics/masks.html
 https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html
 https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html
 https://geopandas.org/en/stable/docs/user_guide/mapping.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from itertools import combinations
import sys
import textwrap
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.patches import Circle, Patch
    from matplotlib.text import Text
    import geopandas as gpd
    from pyproj import Geod, Transformer
    from shapely.validation import make_valid
    from shapely.ops import unary_union
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}\nInstall in the active environment with:\n"
        "python -m pip install numpy pandas matplotlib geopandas shapely pyproj"
    ) from exc

VERSION = "13d-factor-maps-1.3-population-audit"
SOURCE = "01_adm3_flood_exposure_merged.csv"
H = "historical_flood_days_mean_2019_2025"
D = "historical_flood_days_change_after_2019"
B = "mean_annual_flood_days_2000_2018"
# (code, short key, source column, display name, unit)
FACTORS = (
    (1, "health", "health_facilities_total", "Health facilities", "facilities"),
    (2, "cropland", "cropland_area_km2", "Cropland", "km\u00b2"),
    (3, "rangeland", "rangeland_area_km2", "Rangeland", "km\u00b2"),
)
BASE_FACTORS = FACTORS
P = "population_estimate_2025"
PF = "population_for_screening_2025"
POPULATION_FACTOR = (4, "population", PF, "Population 2025", "people")
EXPOSURE = {column: short for _, short, column, _, _ in BASE_FACTORS}


def active_factors(df=None, cutoffs=None):
    enabled = PF in cutoffs if cutoffs is not None else bool(df["population_factor_enabled"].iloc[0])
    return BASE_FACTORS + (POPULATION_FACTOR,) if enabled else BASE_FACTORS

PROFILES = ["Multiple exposure overlap", "Single exposure overlap",
            "High flood activity only", "Increasing flood + high exposure",
            "Other / background"]
# A local equal-area projection, used for plotting only, not recomputing exposure areas.
MAP_CRS = "+proj=laea +lat_0=8 +lon_0=30 +datum=WGS84 +units=m +no_defs"
INK = "#1E293B"
MUTED = "#54616F"


def project_root() -> Path:
    """Prefer the script's project, then the current working directory's project."""
    here = Path(__file__).resolve()
    for start in (here.parent, Path.cwd()):
        for candidate in (start, *start.parents):
            if (candidate / "processed_data" / "multivariate_flood_modeling").is_dir():
                return candidate
    if here.parent.name == "multivariate_flood_modeling" and here.parent.parent.name == "scripts":
        return here.parents[2]
    return Path.cwd()


def resolve_path(path: Path, root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def usable_member(name: str, filename: str) -> bool:
    p = Path(name)
    return p.name == filename and "__MACOSX" not in p.parts and not p.name.startswith("._")


def pick_input(base: Path) -> Path:
    for path in (base / "13c_flood_exposure_screening", base / "13c_flood_exposure_screening.zip"):
        if path.is_dir() and any(path.rglob(SOURCE)):
            return path
        if path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                if any(usable_member(n, SOURCE) for n in z.namelist()):
                    return path
    raise FileNotFoundError(
        f"Cannot find {SOURCE} in the 13c output folder/ZIP under {base}. "
        "Run 13c first, or pass --input PATH."
    )


def load_input(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read bytes directly from ZIP; no unsafe extraction or name-based ADM joins."""
    if path.is_dir():
        matches = [p for p in path.rglob(SOURCE) if "__MACOSX" not in p.parts]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {SOURCE} in {path}; found {len(matches)}")
        content, selected = matches[0].read_bytes(), str(matches[0])
    elif path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if usable_member(n, SOURCE)]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one {SOURCE} in ZIP; found {len(names)}")
            content, selected = z.read(names[0]), f"{path}::{names[0]}"
    elif path.is_file() and path.suffix.lower() == ".csv":
        content, selected = path.read_bytes(), str(path)
    else:
        raise FileNotFoundError(f"Input must be a 13c directory, ZIP, or merged CSV: {path}")
    import io
    frame = pd.read_csv(io.BytesIO(content), dtype={"adm_id": "string"})
    return frame, {"source": selected, "source_csv_sha256": hashlib.sha256(content).hexdigest()}


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = frame.copy()
    required = ["adm_id", "region_name", "parent_adm2_name", H, D, *EXPOSURE]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError("Missing required 13c columns: " + ", ".join(missing))
    df["adm_id"] = normalize_ids(df["adm_id"])
    if df.empty or df.adm_id.isna().any() or df.adm_id.eq("").any() or df.adm_id.duplicated().any():
        raise ValueError("ADM3 IDs must be nonempty and unique; duplicate IDs must not be silently dropped.")
    core = [H, D, *EXPOSURE]
    for column in core:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    invalid = ~np.isfinite(df[core].to_numpy(dtype=float))
    if invalid.any():
        bad = {c: int(invalid[:, i].sum()) for i, c in enumerate(core) if invalid[:, i].any()}
        raise ValueError(f"Missing or non-finite core values: {bad}. They are not treated as zero.")
    if (df[[H, *EXPOSURE]] < 0).any().any():
        raise ValueError("Flood activity and ADM3 asset quantities cannot be negative.")
    if df[H].gt(366 + 1e-8).any():
        raise ValueError("Mean annual flood-activity days exceed 366. Check the metric and units upstream.")
    quality = {"adm3_rows": len(df), "unique_adm3": df.adm_id.nunique(),
               "change_formula_checked": False}
    if B in df:
        df[B] = pd.to_numeric(df[B], errors="coerce")
        if not np.isfinite(df[B]).all():
            raise ValueError("The baseline mean contains missing/non-finite values.")
        residual = (df[D] - (df[H] - df[B])).abs()
        if residual.max() > 1e-6:
            raise ValueError("Flood change is not mean(2019-2025) minus mean(2000-2018). Do not mislabel the map.")
        quality.update(change_formula_checked=True, change_formula_max_abs_error=float(residual.max()))
    else:
        warnings.warn("Baseline mean absent; cannot independently check the supplied change column.")
    for column in ("population_estimate_2025", "population_data_fraction_2025"):
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df, quality


def screening(df: pd.DataFrame, quantile: float, include_population: bool = False,
              scenario: str = "baseline") -> tuple[pd.DataFrame, dict]:
    """Preserve hazard rules; include population only in an explicit scenario.

    Missing population -> nullable high_population, never a zero population.
    Selection uses established positive flags; undecidable nonselections are labelled.
    """
    if not 0 < quantile < 1:
        raise ValueError("--quantile must lie strictly between 0 and 1.")
    out = df.copy()
    out["population_factor_enabled"] = bool(include_population)
    out["analysis_scenario"] = scenario
    factors = active_factors(out)
    columns = (H, D, *(f[2] for f in factors))
    cutoffs = {c: float(out[c].quantile(quantile)) for c in columns}
    if any(not np.isfinite(v) for v in cutoffs.values()):
        raise ValueError("A screening variable has no finite eligible values; no threshold can be computed.")
    for c, cutoff in cutoffs.items():
        known = out[c].notna() & np.isfinite(out[c])
        flag = (out[c].gt(0) if cutoff == 0 else out[c].ge(cutoff)).astype("boolean")
        out[c + "_screen_high"] = flag.where(known, pd.NA)
    out["high_recent_flood"] = out[H + "_screen_high"].astype(bool)
    out["high_flood_increase"] = out[D + "_screen_high"].astype(bool) & out[D].gt(0)
    for _, short, column, _, _ in factors:
        out[f"high_{short}"] = out[column + "_screen_high"]
        out[f"recent_flood_and_{short}"] = out.high_recent_flood & out[f"high_{short}"].fillna(False)
    if not include_population:
        out["high_population"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    flags = [f"high_{f[1]}" for f in factors]
    out["high_exposure_count"] = out[flags].fillna(False).sum(axis=1).astype(int)
    out["high_asset_count"] = out[[f"high_{f[1]}" for f in BASE_FACTORS]].fillna(False).sum(axis=1).astype(int)
    out["overlap_dimensions"] = out.high_exposure_count.where(out.high_recent_flood, 0).astype(int)
    out["increase_exposure_dimensions"] = out.high_exposure_count.where(out.high_flood_increase, 0).astype(int)
    out["increasing_flood_and_exposure"] = out.high_flood_increase & out.high_exposure_count.gt(0)
    out["candidate_via_recent_activity"] = out.high_recent_flood & out.high_exposure_count.gt(0)
    out["candidate_via_change_only"] = (~out.high_recent_flood) & out.increasing_flood_and_exposure
    out["priority_review_candidate"] = out.candidate_via_recent_activity | out.candidate_via_change_only
    out["population_factor_unknown"] = bool(include_population) & out[PF].isna()
    out["selection_undetermined_population"] = (
        out.population_factor_unknown & (out.high_recent_flood | out.high_flood_increase)
        & ~out.priority_review_candidate)
    out["candidate_status"] = np.select(
        [out.priority_review_candidate, out.selection_undetermined_population],
        ["selected", "undetermined_population"], default="not_selected_by_observed_rules")
    conditions = [out.high_recent_flood & out.overlap_dimensions.ge(2),
                  out.high_recent_flood & out.overlap_dimensions.eq(1),
                  out.high_recent_flood, out.increasing_flood_and_exposure]
    out["screening_profile"] = np.select(conditions, PROFILES[:-1], default=PROFILES[-1])
    def codes(row):
        return "+".join(str(code) for code, short, *_ in factors
                        if pd.notna(row[f"high_{short}"]) and bool(row[f"high_{short}"]))
    out["factor_codes"] = out.apply(codes, axis=1)
    out["population_reference_only"] = not include_population
    def reason(row):
        parts = []
        if row.high_recent_flood:
            parts.append("high recent recorded flood activity")
        if row.high_flood_increase:
            parts.append("high positive difference between period means")
        for _, short, _, label, _ in factors:
            value = row[f"high_{short}"]
            if pd.notna(value) and bool(value):
                parts.append(f"high ADM3 total: {label}")
        if row.population_factor_unknown:
            parts.append("population factor UNKNOWN, not below-threshold")
        return "; ".join(parts) if parts else "no observed screening threshold met"
    out["screening_reason"] = out.apply(reason, axis=1)
    out["_display_order"] = pd.Categorical(out.screening_profile, categories=PROFILES, ordered=True)
    out = out.sort_values(["_display_order", "overlap_dimensions", H, "adm_id"],
                          ascending=[True, False, False, True]).drop(columns="_display_order")
    return out.reset_index(drop=True), cutoffs


def factor_summary(df: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    rows = []
    candidate = df.priority_review_candidate
    for code, short, column, label, unit in active_factors(df):
        flag = df[f"high_{short}"]
        true = flag.fillna(False).astype(bool)
        rows.append({"factor_number": code, "factor": label, "source_column": column,
            "threshold": cutoffs[column], "unit": unit,
            "all_adm3_count": int(true.sum()), "all_adm3_denominator": len(df),
            "known_factor_denominator": int(flag.notna().sum()), "unknown_factor_count": int(flag.isna().sum()),
            "candidate_count": int((candidate & true).sum()), "candidate_denominator": int(candidate.sum()),
            "candidate_unknown_factor_count": int((candidate & flag.isna()).sum()),
            "high_recent_flood_and_factor": int((df.high_recent_flood & true).sum()),
            "high_increase_and_factor": int((df.high_flood_increase & true).sum())})
    return pd.DataFrame(rows)


def combination_summary(df: pd.DataFrame) -> pd.DataFrame:
    candidates = df[df.priority_review_candidate]
    rows = []
    numbers = [str(f[0]) for f in active_factors(df)]
    for size in range(1, len(numbers) + 1):
        for values in combinations(numbers, size):
            codes = "+".join(values)
            group = candidates[candidates.factor_codes == codes]
            rows.append({"factor_combination": codes, "candidate_count": len(group),
                         "recent_activity_route": int(group.candidate_via_recent_activity.sum()),
                         "change_only_route": int(group.candidate_via_change_only.sum()),
                         "population_unknown_in_combination": int(group.population_factor_unknown.sum())})
    return pd.DataFrame(rows)


def polygonal(geom):
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [polygonal(g) for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon", "GeometryCollection")]
        parts = [g for g in parts if g is not None and not g.is_empty]
        return unary_union(parts) if parts else None
    return None


def load_geography(path: Path, results: pd.DataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"ADM3 boundaries not found: {path}\nPass --adm3-geojson PATH, or use --skip-maps for tables only.")
    geo = gpd.read_file(path)
    required = ["adm3_pcode", "adm2_pcode", "geometry"]
    if any(c not in geo for c in required):
        raise ValueError(f"ADM3 boundaries require {required}; actual columns: {geo.columns.tolist()}")
    geo["adm_id"] = normalize_ids(geo.adm3_pcode)
    if geo.adm_id.isna().any() or geo.adm_id.eq("").any() or geo.adm_id.duplicated().any():
        raise ValueError("Boundary ADM3 IDs are missing or duplicated.")
    if geo.crs is None:
        raise ValueError("Boundary CRS is missing. The script will not guess it.")
    if geo.geometry.isna().any() or geo.geometry.is_empty.any():
        raise ValueError("Boundary file contains empty or missing geometries.")
    invalid = ~geo.geometry.is_valid
    repaired = geo.loc[invalid, "adm_id"].tolist()
    if invalid.any():
        geo.loc[invalid, "geometry"] = geo.loc[invalid, "geometry"].apply(lambda g: polygonal(make_valid(g)))
        warnings.warn(f"Repaired {len(repaired)} invalid ADM3 geometries for display; see audit JSON.")
    if (geo.geometry.isna().any() or geo.geometry.is_empty.any() or not geo.geometry.is_valid.all()
            or not geo.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Non-polygon or invalid geometry remains after repair.")
    left, right = set(results.adm_id), set(geo.adm_id)
    if left != right:
        raise ValueError(f"PCODE mismatch: {len(left-right)} analysis IDs without polygons {sorted(left-right)[:8]}; "
                         f"{len(right-left)} polygons without analysis {sorted(right-left)[:8]}. No rows were dropped.")
    source_crs = str(geo.crs)
    name_mismatches = 0
    if "adm3_name" in geo:
        names = geo.set_index("adm_id").adm3_name.reindex(results.adm_id).astype("string").str.strip().str.casefold()
        actual = results.set_index("adm_id").region_name.astype("string").str.strip().str.casefold()
        name_mismatches = int(names.ne(actual).fillna(True).sum())
    keep = [c for c in ("adm_id", "adm3_pcode", "adm2_pcode", "adm1_pcode", "adm1_name", "geometry") if c in geo]
    merged = geo[keep].merge(results, on="adm_id", how="left", validate="one_to_one")
    projected = merged.to_crs(MAP_CRS)
    if not np.isfinite(projected.total_bounds).all():
        raise ValueError("Non-finite projected map bounds.")
    return projected, {
        "boundary_source": str(path), "boundary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "boundary_crs": source_crs, "map_crs": MAP_CRS,
        "boundary_rows": len(geo), "matched_rows": len(merged),
        "name_mismatches_with_matching_pcode": name_mismatches,
        "repaired_geometry_ids": repaired,
    }


def outline_layer(path: Path, geo: gpd.GeoDataFrame, level: str) -> gpd.GeoDataFrame | None:
    code = f"{level}_pcode"
    if path.is_file():
        try:
            layer = gpd.read_file(path)
            if layer.crs is None:
                raise ValueError("missing CRS")
            if code in layer and code in geo:
                if set(normalize_ids(layer[code])) != set(normalize_ids(geo[code]).dropna()):
                    raise ValueError("PCODEs differ from ADM3 parent codes")
            if layer.geometry.isna().any() or layer.geometry.is_empty.any() or not layer.geometry.is_valid.all():
                raise ValueError("invalid outline geometries")
            return layer.to_crs(geo.crs)
        except (ValueError, OSError, RuntimeError) as exc:
            warnings.warn(f"Cannot use {path.name}: {exc}. Deriving outlines from matched ADM3 polygons.")
    if code in geo:
        keep = [code, "geometry"]
        if level == "adm1" and "adm1_name" in geo:
            keep.insert(1, "adm1_name")
        return geo[keep].dissolve(by=code, as_index=False)
    return None


def setup_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.titlesize": 15, "svg.fonttype": "none",
                         "savefig.facecolor": "white"})


def plot_diagnostics(df: pd.DataFrame, out: Path, dpi: int) -> list[str]:
    files = []
    profiles = df.screening_profile.value_counts().reindex(PROFILES, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.barh(PROFILES, profiles, color="#537B9C")
    ax.invert_yaxis(); ax.set_xlim(0, max(1, profiles.max()) * 1.2)
    ax.bar_label(bars, padding=4); ax.set_xlabel("Number of ADM3 areas")
    ax.set_title("13d | Screening profiles, not a humanitarian-risk ranking")
    fig.tight_layout(); fig.savefig(out / "04_screening_profiles.png", dpi=dpi); plt.close(fig)
    files.append("04_screening_profiles.png")
    factors = active_factors(df)
    labels = [f[3] for f in factors] + ["2+ factors", f"All {len(factors)} factors"]
    values = [int(df[f"recent_flood_and_{f[1]}"].sum()) for f in factors]
    values += [int(df.overlap_dimensions.ge(2).sum()), int(df.overlap_dimensions.eq(len(factors)).sum())]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(labels, values, color="#537B9C"); ax.invert_yaxis()
    ax.bar_label(bars, padding=4); ax.set_xlim(0, max(1, max(values)) * 1.25)
    ax.set_xlabel("Number of ADM3 areas; groups can overlap")
    ax.set_title("13d | High recent activity and high ADM3 exposure proxies")
    fig.tight_layout(); fig.savefig(out / "05_exposure_overlap_counts.png", dpi=dpi); plt.close(fig)
    files.append("05_exposure_overlap_counts.png")
    sample = df[df.priority_review_candidate].sort_values("adm_id")
    if not sample.empty:
        keys = ["high_recent_flood", "high_flood_increase"] + [f"high_{f[1]}" for f in factors]
        matrix = sample[keys].to_numpy(dtype=float, na_value=np.nan)
        fig, ax = plt.subplots(figsize=(11, max(5, len(sample) * .20 + 2)))
        cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#D1D5DB")
        ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        labels = ["Recent activity", "High positive change"] + [f"{f[0]} {f[3]}" for f in factors]
        ax.set_xticks(range(len(keys)), labels)
        ax.set_yticks(range(len(sample)), [f"{r.region_name} ({r.adm_id})" for r in sample.itertuples()])
        ax.tick_params(labelsize=7)
        ax.set_title(f"13d | {len(sample)} selected areas; PCODE order, not rank. ? = unknown")
        for i, j in np.ndindex(matrix.shape):
            v = matrix[i, j]
            ax.text(j, i, "?" if np.isnan(v) else str(int(v)), ha="center", va="center", fontsize=7,
                    color="white" if v == 1 else MUTED)
        fig.tight_layout(); fig.savefig(out / "06_candidate_indicator_matrix.png", dpi=dpi); plt.close(fig)
        files.append("06_candidate_indicator_matrix.png")
    return files


def map_canvas(title: str, subtitle: str):
    fig = plt.figure(figsize=(16, 10.8), dpi=120, facecolor="white")
    fig.text(.045, .967, "SOUTH SUDAN  /  ADM3  /  13d", size=10, weight="bold", color=MUTED)
    fig.text(.045, .925, title, size=24, weight="bold")
    fig.text(.045, .889, subtitle, size=11, color=MUTED)
    ax = fig.add_axes([.035, .285, .675, .575])
    side = fig.add_axes([.735, .275, .235, .58]); side.set_axis_off()
    side.set_xlim(0, 1); side.set_ylim(0, 1)
    cax = fig.add_axes([.085, .247, .57, .017])
    fig.add_artist(Line2D([.72, .72], [.28, .86], transform=fig.transFigure, color="#D8DFE6", lw=.9))
    return fig, ax, side, cax


def map_base(ax, geo, adm2, adm1, state_labels: bool) -> None:
    geo.boundary.plot(ax=ax, color="#F9FAFB", linewidth=.22, zorder=2)
    if adm2 is not None:
        adm2.boundary.plot(ax=ax, color="#697580", linewidth=.38, zorder=3)
    if adm1 is not None:
        adm1.boundary.plot(ax=ax, color="#34414D", linewidth=.68, zorder=4)
        if state_labels and "adm1_name" in adm1:
            for _, row in adm1.iterrows():
                point = row.geometry.representative_point()
                label = "\n".join(textwrap.wrap(str(row.adm1_name), width=19))
                ax.text(point.x, point.y, label, size=7.2, ha="center", va="center", color="#36424E",
                        alpha=.72, zorder=4, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    xmin, ymin, xmax, ymax = geo.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - .065 * dx, xmax + .065 * dx)
    ax.set_ylim(ymin - .10 * dy, ymax + .07 * dy)
    ax.set_aspect("equal"); ax.set_axis_off()
    cartographic_guides(ax, geo.crs)


def cartographic_guides(ax, crs) -> None:
    """Geodesic local scale bar and true-north direction at the arrow's location."""
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geod = Geod(ellps="WGS84")
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    sx, sy = xmin + .09 * width, ymin + .045 * height
    lon, lat = inverse.transform(sx, sy)
    end_lon, end_lat, _ = geod.fwd(lon, lat, 90, 200_000)
    ex, ey = forward.transform(end_lon, end_lat)
    ax.plot([sx, ex], [sy, ey], color=INK, lw=1.4, zorder=6)
    for x, y in ((sx, sy), (ex, ey)):
        ax.plot([x, x], [y - .005 * height, y + .005 * height], color=INK, lw=1.2, zorder=6)
    ax.text((sx + ex) / 2, min(sy, ey) - .015 * height, "200 km (local scale)",
            ha="center", va="top", size=8, zorder=6)
    nx, ny = xmin + .945 * width, ymin + .865 * height
    lon, lat = inverse.transform(nx, ny)
    nlon, nlat, _ = geod.fwd(lon, lat, 0, 60_000)
    tx, ty = forward.transform(nlon, nlat)
    ax.annotate("", xy=(tx, ty), xytext=(nx, ny),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2}, zorder=6)
    ax.text(tx, ty + .015 * height, "N", ha="center", size=10, weight="bold", zorder=6)


def side_text(ax, x: float, y: float, text: str, **kw) -> None:
    ax.text(x, y, text, transform=ax.transAxes, va="top", **kw)


def stat_line(ax, y: float, label: str, value, size: float = 11) -> None:
    side_text(ax, 0, y, label, fontsize=size)
    side_text(ax, .98, y, str(value), fontsize=size, ha="right", weight="bold")


def common_footer(fig, df=None) -> None:
    text = ("Sources: 08b / 13b via 13c; project administrative boundaries. "
            "ADM3 totals are not inundated assets, affected people or confirmed losses. Historical screening, not a forecast.")
    fig.text(.045, .022, text, size=8.0, color=MUTED)
    if df is not None:
        scenario = str(df.analysis_scenario.iloc[0])
        notice = {"baseline": "BASELINE | POPULATION NOT USED FOR SELECTION",
                  "population_sensitivity": "PROVISIONAL POPULATION SENSITIVITY | 2025 ESTIMATES",
                  "population_processing_checked": "POPULATION PROCESSING CHECKED | NOT CENSUS VALIDATION"}.get(scenario, scenario)
        fig.text(.97, .967, notice, ha="right", size=8.5, weight="bold",
                 color="#9A3412" if scenario == "population_sensitivity" else MUTED)


def threshold_text(cutoff: float, unit: str) -> str:
    comparison = ">" if cutoff == 0 else "\u2265"
    number = f"{cutoff:,.0f}" if float(cutoff).is_integer() else f"{cutoff:,.2f}"
    return f"{comparison} {number} {unit}"


def bottom_factor_legend(fig, cutoffs: dict, quantile: float) -> None:
    factors = active_factors(cutoffs=cutoffs)
    fig.text(.055, .182, "NUMBERED FACTORS | ADM3 totals, not causes of flooding or ranking positions", size=11, weight="bold")
    xs = (.065, .295, .525, .755) if len(factors) == 4 else (.065, .37, .675)
    for x, (code, _, column, label, unit) in zip(xs, factors):
        fig.text(x, .13, str(code), ha="center", va="center", size=12, weight="bold",
                 bbox={"boxstyle": "circle,pad=0.28", "fc": "white", "ec": INK, "lw": 1.1})
        fig.text(x + .022, .143, label, size=11, weight="bold", va="top")
        fig.text(x + .022, .115, threshold_text(cutoffs[column], unit) + " / ADM3", size=9.5, va="top")
    fig.text(.055, .076,
             f"High = Q{100*quantile:g} of available values, including ties; zero cutoffs require > 0. Multiple circles mean multiple factors.",
             size=9, color=MUTED)
    message = ("Hatching: population unknown; no circle 4 does not prove low population. 2025 population is a snapshot, not historical exposure."
               if len(factors) == 4 else "Population is reference-only in this baseline. Asset totals do not measure vulnerability or confirmed flood impact.")
    fig.text(.055, .051, message, size=8.5, color=MUTED)


def exposure_sidebar(side, df: pd.DataFrame, counts: pd.DataFrame) -> None:
    factor_sidebar(side, df, counts, worsening=False)


def worsening_sidebar(side, df: pd.DataFrame, counts: pd.DataFrame) -> None:
    factor_sidebar(side, df, counts, worsening=True)



def factor_sidebar(side, df: pd.DataFrame, counts: pd.DataFrame, worsening: bool) -> None:
    selected = df.increasing_flood_and_exposure if worsening else df.priority_review_candidate
    n = int(selected.sum())
    side_text(side, 0, .99, "HIGH-CHANGE REVIEW" if worsening else "REVIEW CANDIDATES", fontsize=11, weight="bold", color=MUTED)
    side_text(side, 0, .935, str(n), fontsize=35, weight="bold")
    side_text(side, .32, .905, f"of {len(df)} ADM3 areas", fontsize=11)
    side_text(side, 0, .80, "FACTOR COUNTS (ADM3)", fontsize=11, weight="bold")
    side_text(side, 0, .745, "Factor", fontsize=10, color=MUTED)
    side_text(side, .75, .745, "All", fontsize=9, color=MUTED, ha="right")
    side_text(side, .99, .745, "Selected", fontsize=9, color=MUTED, ha="right")
    summary = counts.set_index("factor_number", verify_integrity=True)
    for y, (code, short, _, label, _) in zip((.68, .615, .55, .485), active_factors(df)):
        flag = df[f"high_{short}"].fillna(False).astype(bool)
        all_count, selected_count = int(flag.sum()), int((selected & flag).sum())
        key = "high_increase_and_factor" if worsening else "candidate_count"
        if all_count != int(summary.loc[code, "all_adm3_count"]) or selected_count != int(summary.loc[code, key]):
            raise ValueError(f"Sidebar and CSV disagree for factor {code}.")
        side.text(.035, y-.014, str(code), transform=side.transAxes, ha="center", va="center",
                  size=9.5, weight="bold", bbox={"boxstyle":"circle,pad=0.22", "fc":"white", "ec":INK, "lw":1})
        side_text(side, .105, y, label, fontsize=9.5)
        side_text(side, .75, y, str(all_count), fontsize=10.5, ha="right")
        side_text(side, .99, y, str(selected_count), fontsize=10.5, ha="right", weight="bold")
    side_text(side, 0, .412, "Factors overlap; do not add the counts.", fontsize=9, color=MUTED)
    if worsening:
        stat_line(side, .355, "High positive change", int(df.high_flood_increase.sum()), 9.5)
        stat_line(side, .308, "Selected with 2+ factors", int((selected & df.high_exposure_count.ge(2)).sum()), 9.5)
    else:
        stat_line(side, .355, "High activity + factor(s)", int(df.candidate_via_recent_activity.sum()), 9.5)
        stat_line(side, .308, "Change-only + factor(s)", int(df.candidate_via_change_only.sum()), 9.5)
    values = df.loc[selected, PF]
    known = int(values.notna().sum())
    side_text(side, 0, .245, "2025 POPULATION IN SELECTED ADM3", fontsize=9.5, weight="bold")
    total = values.sum(min_count=1)
    value = "Unknown" if pd.isna(total) else f"{total:,.0f}"
    stat_line(side, .192, "Available estimates: subtotal", value, 9.5)
    side_text(side, 0, .145, f"Available for {known}/{n} selected ADM3;\nnot affected or inundated population.", fontsize=9, color=MUTED)
    unknown = int(df.population_factor_unknown.sum())
    note = (f"Population factor unknown: {unknown} ADM3."
            if bool(df.population_factor_enabled.iloc[0]) else "Population does not select areas here.")
    side_text(side, 0, .064, note + "\nNo validated humanitarian-risk ranking.", fontsize=9, color=MUTED)


def circle_group(codes: list[int], diameter: float) -> DrawingArea:
    gap = 1.8
    width = len(codes) * diameter + max(0, len(codes) - 1) * gap
    drawing = DrawingArea(width, diameter, 0, 0)
    for j, code in enumerate(codes):
        x, y = diameter / 2 + j * (diameter + gap), diameter / 2
        drawing.add_artist(Circle((x, y), diameter / 2 - .5, facecolor="white", edgecolor=INK, lw=.85))
        drawing.add_artist(Text(x, y, str(code), ha="center", va="center", fontsize=diameter * .62,
                                fontweight="bold", color=INK))
    return drawing


def intersection_area(a, b) -> float:
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def add_factor_badges(fig, ax, geo, diameter: float) -> pd.DataFrame:
    """Label ALL candidates; deterministic local collision reduction, with leader lines.

    Digits repeat factor codes. The leader-line anchor is inside the ADM3 polygon,
    not an observed facility/crop location. Unresolved collisions are audited.
    """
    candidates = geo.loc[geo.priority_review_candidate].sort_values("adm_id").copy()
    columns = ["adm_id", "region_name", "factor_codes", "anchor_lon", "anchor_lat",
               "offset_x_points", "offset_y_points", "unresolved_overlap", "leader_line"]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    points = candidates.geometry.representative_point()
    fig.canvas.draw()
    anchors = ax.transData.transform(np.column_stack([points.x, points.y]))
    px_per_point = fig.dpi / 72
    # Harder, locally crowded positions are placed first; this is NOT priority ordering.
    pairwise = np.sqrt(((anchors[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(pairwise, np.inf)
    order = np.argsort(pairwise.min(axis=1), kind="stable")
    offsets = [(0., 0.)]
    for radius in (12, 20, 30, 42, 56, 72, 92):
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            offsets.append((radius * math.cos(angle), radius * math.sin(angle)))
    occupied = []
    box = ax.get_window_extent()
    # Reserve the local scale bar and north arrow regions.
    obstacles = [(box.x0 + .04*box.width, box.y0, box.x0 + .34*box.width, box.y0 + .10*box.height),
                 (box.x0 + .89*box.width, box.y0 + .82*box.height, box.x1, box.y1)]
    to_wgs84 = Transformer.from_crs(geo.crs, "EPSG:4326", always_xy=True)
    records = []
    for i in order:
        row = candidates.iloc[i]
        point = points.iloc[i]
        codes = [int(v) for v in row.factor_codes.split("+") if v]
        width = (len(codes)*diameter + (len(codes)-1)*1.8 + 4) * px_per_point
        height = (diameter + 4) * px_per_point
        best = None
        for dx, dy in offsets:
            cx, cy = anchors[i] + np.array([dx, dy]) * px_per_point
            rect = (cx-width/2, cy-height/2, cx+width/2, cy+height/2)
            outside = (max(0, box.x0-rect[0]) + max(0, rect[2]-box.x1)
                       + max(0, box.y0-rect[1]) + max(0, rect[3]-box.y1))
            overlap = sum(intersection_area(rect, other) for other in occupied + obstacles)
            score = 1e6 * outside + 1e3 * overlap + dx*dx + dy*dy
            if best is None or score < best[0]:
                best = (score, dx, dy, rect, overlap, outside)
            if outside == 0 and overlap == 0:
                break
        _, dx, dy, rect, overlap, outside = best
        occupied.append(rect)
        moved = math.hypot(dx, dy) > 2
        if moved:
            target = ax.transData.inverted().transform(anchors[i] + np.array([dx, dy]) * px_per_point)
            ax.plot([point.x, target[0]], [point.y, target[1]], color="#414B55", lw=.55, alpha=.85, zorder=7)
            ax.plot(point.x, point.y, marker=".", markersize=2, color=INK, zorder=8)
        artist = AnnotationBbox(circle_group(codes, diameter), (point.x, point.y),
                                xybox=(dx, dy), xycoords="data", boxcoords="offset points",
                                box_alignment=(.5, .5), frameon=False, pad=0, zorder=10)
        ax.add_artist(artist)
        lon, lat = to_wgs84.transform(point.x, point.y)
        records.append({"adm_id": row.adm_id, "region_name": row.region_name,
                        "factor_codes": row.factor_codes, "anchor_lon": lon, "anchor_lat": lat,
                        "offset_x_points": dx, "offset_y_points": dy,
                        "unresolved_overlap": bool(overlap > 0 or outside > 0), "leader_line": moved})
    return pd.DataFrame(records, columns=columns).sort_values("adm_id")


def save_map(fig, directory: Path, stem: str, dpi: int, formats: list[str]) -> list[str]:
    paths = []
    for fmt in formats:
        path = directory / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        paths.append(str(path.name))
    plt.close(fig)
    return paths


def plot_spatial_maps(geo, adm2, adm1, df, cutoffs, counts, directory: Path, args) -> tuple[list[str], dict]:
    directory.mkdir(parents=True, exist_ok=True)
    max_h = max(1., math.ceil(float(df[H].max()) / 25.) * 25.)
    hnorm = Normalize(vmin=0, vmax=max_h)
    qlabel = f"Q{100 * args.quantile:g}"
    figure_paths = []

    # A: recent activity, without turning a continuous variable into factor digits.
    fig, ax, side, cax = map_canvas("Recent recorded flood activity",
        "Colour shows the mean annual number of days with recorded activity, 2019\u20132025; not full-area inundation duration.")
    geo.plot(column=H, cmap="Blues", norm=hnorm, ax=ax, edgecolor="none", zorder=1)
    map_base(ax, geo, adm2, adm1, not args.no_state_labels)
    cb = fig.colorbar(ScalarMappable(norm=hnorm, cmap="Blues"), cax=cax, orientation="horizontal")
    cb.set_label("Mean annual days with recorded flood activity, 2019\u20132025", fontsize=10)
    cax.axvline(cutoffs[H], color=INK, lw=1.1)
    cax.text(cutoffs[H], 1.4, qlabel, transform=cax.get_xaxis_transform(), ha="center", fontsize=8)
    side_text(side, 0, .99, "RECENT ACTIVITY", fontsize=11, weight="bold", color=MUTED)
    side_text(side, 0, .91, str(int(df.high_recent_flood.sum())), fontsize=37, weight="bold")
    side_text(side, 0, .79, f"ADM3 at/above the screening cutoff\nout of {len(df)} areas", fontsize=12)
    stat_line(side, .64, f"{qlabel} threshold (days)", f"{cutoffs[H]:.2f}")
    stat_line(side, .55, "Median (days)", f"{df[H].median():.2f}")
    stat_line(side, .46, "Maximum (days)", f"{df[H].max():.2f}")
    stat_line(side, .37, "Zero recorded activity", int(df[H].eq(0).sum()))
    side_text(side, 0, .24, "A recent-period historical summary,\nnot a 2000\u20132025 average or\na future flood prediction.", fontsize=11, color=MUTED, linespacing=1.6)
    fig.text(.055, .17, "READING THIS MAP", fontsize=11, weight="bold")
    fig.text(.055, .135, "Darker blue means more recorded flood-activity days. The same scale is used in the factor-overlay map.", fontsize=11)
    fig.text(.055, .10, "A high value alone does not establish greater humanitarian need. No exposure factors or population are encoded here.", fontsize=10)
    fig.text(.055, .067, "Thin outlines: ADM2; darker outlines and labels: ADM1. No factor circles on this map.", fontsize=9, color=MUTED)
    common_footer(fig, df)
    figure_paths += save_map(fig, directory, "08_recent_flood_activity_map", args.dpi, args.formats)

    # B: symmetric normalization; identical absolute differences have equal colour distance from zero.
    limit = max(1., math.ceil(float(df[D].abs().max()) / 20.) * 20.)
    dnorm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, ax, side, cax = map_canvas("Change in recorded flood activity",
        "Mean annual activity in 2019\u20132025 minus 2000\u20132018. This is a difference between period means, not an annual trend slope.")
    geo.plot(column=D, cmap="RdBu_r", norm=dnorm, ax=ax, edgecolor="none", zorder=1)
    map_base(ax, geo, adm2, adm1, not args.no_state_labels)
    cb = fig.colorbar(ScalarMappable(norm=dnorm, cmap="RdBu_r"), cax=cax, orientation="horizontal",
                      ticks=[-limit, -limit/2, 0, limit/2, limit])
    cb.set_label("Difference in mean annual flood-activity days: 2019\u20132025 minus 2000\u20132018", fontsize=10)
    positive, negative = int(df[D].gt(1e-9).sum()), int(df[D].lt(-1e-9).sum())
    side_text(side, 0, .99, "BETWEEN-PERIOD CHANGE", fontsize=11, weight="bold", color=MUTED)
    stat_line(side, .885, "Positive difference", positive)
    stat_line(side, .79, "Near-zero difference", len(df) - positive - negative)
    stat_line(side, .695, "Negative difference", negative)
    side_text(side, 0, .565, "HIGH POSITIVE CHANGE", fontsize=11, weight="bold")
    stat_line(side, .495, f"{qlabel} cutoff (days)", f"{cutoffs[D]:.2f}")
    stat_line(side, .405, "Qualifying ADM3", int(df.high_flood_increase.sum()))
    side_text(side, 0, .29, "High change must also be positive.\nIt is not the same as any increase.", fontsize=11, color=MUTED, linespacing=1.6)
    side_text(side, 0, .15, "No significance test, causal claim\nor future projection is implied.", fontsize=10, color=MUTED, linespacing=1.6)
    fig.text(.055, .17, "READING THIS MAP", fontsize=11, weight="bold")
    fig.text(.055, .135, "Blue: fewer days in the recent period. White: close to no difference. Red: more days in the recent period.", fontsize=11)
    fig.text(.055, .10, "The colour scale is symmetric around zero. A positive difference does not show that activity increased every year after 2019.", fontsize=10)
    fig.text(.055, .067, "Compare this map with recent activity to distinguish high levels from large changes.", fontsize=9, color=MUTED)
    common_footer(fig, df)
    figure_paths += save_map(fig, directory, "09_post2019_flood_change_map", args.dpi, args.formats)

    # C: keep recent-activity colour and overlay only numbered exposure factors.
    n = int(df.priority_review_candidate.sum())
    fig, ax, side, cax = map_canvas("Humanitarian review: hazard + exposure factors",
        f"All {n} review candidates are outlined and labelled. Circle numbers identify qualifying ADM3 totals, not their exact locations.")
    geo.plot(column=H, cmap="Blues", norm=hnorm, ax=ax, edgecolor="none", zorder=1)
    map_base(ax, geo, adm2, adm1, not args.no_state_labels)
    if bool(df.population_factor_enabled.iloc[0]):
        unknown = geo[geo.population_factor_unknown]
        if not unknown.empty:
            unknown.plot(ax=ax, facecolor="none", edgecolor="#5D6670", hatch="///", linewidth=.3, zorder=4.5)
    recent = geo[geo.candidate_via_recent_activity]
    change_only = geo[geo.candidate_via_change_only]
    if not recent.empty:
        recent.boundary.plot(ax=ax, color="#111111", linewidth=1.05, zorder=5)
    if not change_only.empty:
        change_only.boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--", zorder=5)
    cb = fig.colorbar(ScalarMappable(norm=hnorm, cmap="Blues"), cax=cax, orientation="horizontal")
    # Put the caption above this colour bar: leave a separate row for the outline key.
    fig.text(.37, .273, "Mean annual recorded flood-activity days, 2019\u20132025 (same scale as map 08)",
             ha="center", va="bottom", fontsize=10)
    exposure_sidebar(side, df, counts)
    bottom_factor_legend(fig, cutoffs, args.quantile)
    # Explanatory outline key sits above the bottom factor legend, outside the map.
    fig.legend(handles=[Line2D([0], [0], color="#111111", lw=1.2,
                               label="Solid: high activity + one or more exposure factors"),
                        Line2D([0], [0], color="#111111", lw=1.2, linestyle="--",
                               label="Dashed: high positive change + exposure factors, below high-activity cutoff")],
               loc="center left", bbox_to_anchor=(.05, .211), frameon=False, ncol=2,
               fontsize=8.6, handlelength=2.8, columnspacing=2)
    badges = add_factor_badges(fig, ax, geo, args.badge_size)
    badges.to_csv(directory / "13_marker_placement_audit.csv", index=False)
    conflicts = int(badges.unresolved_overlap.sum()) if len(badges) else 0
    if conflicts:
        warnings.warn(f"{conflicts} factor groups still have estimated label overlap. Inspect map 10; try --badge-size 10.")
    common_footer(fig, df)
    figure_paths += save_map(fig, directory, "10_humanitarian_review_factors_map", args.dpi, args.formats)

    # D: worsening map with the same factor-number system, focused on high positive change + assets.
    worsening_n = int(df.increasing_flood_and_exposure.sum())
    fig, ax, side, cax = map_canvas("Flood worsening + exposure factors",
        f"Colour shows the difference between period means, not an annual trend. Circles label all {worsening_n} high-change + exposure candidates.")
    geo.plot(column=D, cmap="RdBu_r", norm=dnorm, ax=ax, edgecolor="none", zorder=1)
    map_base(ax, geo, adm2, adm1, not args.no_state_labels)
    if bool(df.population_factor_enabled.iloc[0]):
        unknown = geo[geo.population_factor_unknown]
        if not unknown.empty:
            unknown.plot(ax=ax, facecolor="none", edgecolor="#5D6670", hatch="///", linewidth=.3, zorder=4.5)
    worsening = geo[geo.increasing_flood_and_exposure]
    if not worsening.empty:
        worsening.boundary.plot(ax=ax, color="#111111", linewidth=1.15, zorder=5)
    cb = fig.colorbar(ScalarMappable(norm=dnorm, cmap="RdBu_r"), cax=cax, orientation="horizontal",
                      ticks=[-limit, -limit/2, 0, limit/2, limit])
    fig.text(.37, .273, "Change in mean annual flood-activity days: 2019–2025 minus 2000–2018 (map 09 scale)",
             ha="center", va="bottom", fontsize=10)
    worsening_sidebar(side, df, counts)
    bottom_factor_legend(fig, cutoffs, args.quantile)
    fig.legend(handles=[Line2D([0], [0], color="#111111", lw=1.2,
                               label=f"Solid outline and circles: positive change {threshold_text(cutoffs[D], 'days')} ({qlabel}) + one or more exposure factors")],
               loc="center left", bbox_to_anchor=(.05, .211), frameon=False, ncol=1,
               fontsize=8.6, handlelength=2.8, columnspacing=2)
    worsening_geo = geo.copy()
    worsening_geo["priority_review_candidate"] = worsening_geo["increasing_flood_and_exposure"]
    worsening_badges = add_factor_badges(fig, ax, worsening_geo, args.badge_size)
    worsening_badges.to_csv(directory / "14_worsening_marker_placement_audit.csv", index=False)
    worsening_conflicts = int(worsening_badges.unresolved_overlap.sum()) if len(worsening_badges) else 0
    if worsening_conflicts:
        warnings.warn(f"{worsening_conflicts} worsening-factor groups still have estimated label overlap. Inspect map 11; try --badge-size 10.")
    common_footer(fig, df)
    figure_paths += save_map(fig, directory, "11_flood_worsening_factors_map", args.dpi, args.formats)

    quality = {"candidate_labels_drawn": len(badges), "candidate_labels_expected": n,
               "estimated_unresolved_label_overlaps": conflicts,
               "worsening_labels_drawn": len(worsening_badges), "worsening_labels_expected": worsening_n,
               "worsening_estimated_unresolved_label_overlaps": worsening_conflicts,
               "recent_activity_colour_limits": [0, max_h], "change_colour_limits": [-limit, limit],
               "marker_anchor": "representative point inside ADM3, not the location of an exposed asset",
               "factor_codes": {str(c): label for c, _, _, label, _ in active_factors(df)},
               "marker_audit": "maps/13_marker_placement_audit.csv",
               "worsening_marker_audit": "maps/14_worsening_marker_placement_audit.csv",
               "worsening_selection_rule": "high_flood_increase AND high_exposure_count > 0",
               "worsening_factor_counts": {str(code): int((df.increasing_flood_and_exposure & df[f"high_{short}"]).sum())
                                           for code, short, *_ in active_factors(df)}}
    if len(badges) != n:
        raise ValueError("Not all review candidates received a factor label.")
    if len(worsening_badges) != worsening_n:
        raise ValueError("Not all worsening candidates received a factor label.")
    return figure_paths, quality



def json_ready(value):
    """Convert numerical scalars to strict JSON; non-finite diagnostics become null."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(json_ready(data), indent=2, allow_nan=False), encoding="utf-8")


def prepare_population(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Audit population values independently of raster-validity fractions."""
    out = df.copy()
    legacy = [c for c in out if c.startswith("population") and
              ("incomplete_pixel_coverage" in c or "coverage_below_99" in c
               or c in ("population_full_coverage_flag", "population_use_for_priority"))]
    out = out.rename(columns={c: "legacy_13b13c__" + c for c in legacy})
    values = pd.to_numeric(out[P], errors="coerce")
    finite = values.notna() & np.isfinite(values)
    usable = finite & values.ge(0)
    out["population_original_numeric_2025"] = values
    out[PF] = values.where(usable)
    out["population_numeric_status"] = np.select(
        [values.isna(), ~finite, values.lt(0), values.eq(0)],
        ["missing_or_nonnumeric", "nonfinite", "negative_invalid", "valid_modelled_zero"],
        default="positive_estimate")
    fraction = pd.to_numeric(out.get("population_data_fraction_2025", np.nan), errors="coerce")
    out["population_valid_value_fraction_2025"] = fraction
    out["population_fraction_invalid"] = fraction.notna() & (~np.isfinite(fraction) | fraction.lt(0) | fraction.gt(1))
    out["population_sparse_valid_pixels_note"] = fraction.lt(.99) & ~out.population_fraction_invalid
    out["population_processing_status"] = "unverified_source_aggregate"
    out.loc[~usable, "population_processing_status"] = "unavailable_or_invalid_numeric_value"
    metadata = {"year": 2025, "source_field": P, "screening_field": PF,
        "adm3_rows": len(out), "valid_nonnegative_population_values": int(usable.sum()),
        "missing_or_invalid_values": int((~usable).sum()),
        "valid_zero_values": int((usable & values.eq(0)).sum()),
        "median_valid_value_fraction": float(fraction.median()),
        "low_valid_pixel_fraction_is_an_exclusion_rule": False,
        "values_divided_by_valid_fraction": False, "missing_values_filled_with_zero": False,
        "retired_legacy_flag_columns": legacy,
        "interpretation": "Valid-value raster fraction is not the fraction of people counted. Low fraction alone cannot prove incompleteness."}
    return out, metadata


def read_population_review(root: Path, path: Path | None) -> tuple[dict, dict]:
    """A product-specific human review is required; a filename is only a hint."""
    candidate = resolve_path(path, root) if path else root / "raw_data" / "population_review_2025.json"
    report = {"review_file": str(candidate), "review_file_found": candidate.is_file(),
              "product_metadata_reviewed": False}
    if not candidate.is_file():
        if path:
            raise FileNotFoundError(f"Population review file not found: {candidate}")
        return {}, report
    config = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Population review JSON must be an object.")
    report["declared_review"] = config
    if config.get("reviewed") is not True:
        return config, report
    required = ("raster_path", "year", "units", "nodata_semantics", "source_reference")
    if any(not config.get(k) for k in required):
        raise ValueError("A reviewed population JSON must supply raster_path, year, units, nodata_semantics and source_reference.")
    if config["year"] != 2025:
        raise ValueError("This version deliberately uses 2025 population; review year must be 2025.")
    if config["units"] not in ("people_per_pixel", "people_per_km2", "unknown"):
        raise ValueError("Population units must be people_per_pixel, people_per_km2 or unknown.")
    if config["nodata_semantics"] not in ("unsettled", "missing", "unknown"):
        raise ValueError("nodata_semantics must be unsettled, missing or unknown.")
    report["product_metadata_reviewed"] = (
        config["units"] == "people_per_pixel" and config["nodata_semantics"] in ("unsettled", "missing"))
    report["review_basis"] = "User-supplied product-specific review; script does not independently verify its bibliographic content."
    return config, report


def population_year_hint(path: Path) -> int | None:
    """Do not confuse a release tag such as R2025A with a population reference year."""
    match = re.search(r"(?:pop(?:ulation)?|ppp)[_-](20[0-9]{2})(?=[_.-]|$)", path.name.lower())
    if match:
        return int(match.group(1))
    years = re.findall(r"(?<![A-Za-z0-9])(20[0-9]{2})(?![A-Za-z0-9])", path.name)
    return int(years[0]) if len(set(years)) == 1 else None


def discover_population_raster(root: Path, explicit: Path | None, config: dict) -> tuple[Path | None, list[dict], str]:
    if explicit:
        selected = resolve_path(explicit, root)
        if not selected.is_file():
            raise FileNotFoundError(f"Population raster not found: {selected}")
        if config.get("reviewed") is True and config.get("raster_path"):
            if selected != resolve_path(Path(config["raster_path"]), root):
                raise ValueError("--population-raster differs from the reviewed raster_path; review the actual file.")
        return selected, [{"path": str(selected), "selected": True}], "explicit_argument"
    if config.get("raster_path"):
        selected = resolve_path(Path(config["raster_path"]), root)
        if not selected.is_file():
            raise FileNotFoundError(f"Reviewed population raster not found: {selected}")
        return selected, [{"path": str(selected), "selected": True}], "review_configuration"
    raw = root / "raw_data"
    paths = []
    if raw.exists():
        for path in raw.rglob("*"):
            name = path.name.lower()
            if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
                full = str(path.relative_to(raw)).lower()
                population_hint = any(k in full for k in ("worldpop", "population", "_pop_", "_ppp_"))
                year_hint = population_year_hint(path)
                separate_2025 = re.search(r"(?<![A-Za-z0-9])2025(?![A-Za-z0-9])", full) is not None
                if population_hint and (year_hint == 2025 or (year_hint is None and separate_2025)):
                    paths.append(path.resolve())
    paths = sorted(set(paths))
    selected = paths[0] if len(paths) == 1 else None
    records = [{"path": str(p), "selected": p == selected} for p in paths]
    reason = "one_candidate" if selected else ("no_candidate" if not paths else "ambiguous_multiple_candidates")
    return selected, records, reason


def raster_zonal_population_check(path: Path, boundary_path: Path, df: pd.DataFrame,
                                  config: dict) -> tuple[pd.DataFrame, dict]:
    """Masked, windowed sum using polygon-centre rasterization; never resample counts.

    Extent coverage and valid-cell fraction are separate diagnostics. All-NoData
    polygons return NaN even for a constrained product: no fabricated zero people.
    Units are not inferred from the file's numerical distribution or name.
    """
    hint = population_year_hint(path)
    if hint is not None and hint != 2025:
        raise ValueError(f"Population reference year appears to be {hint}, not 2025: {path.name}")
    import rasterio
    from rasterio.features import geometry_window, geometry_mask
    from rasterio.windows import Window
    from rasterio.errors import WindowError
    from shapely.geometry import Polygon, mapping
    boundary = gpd.read_file(boundary_path)
    if "adm3_pcode" not in boundary or boundary.crs is None:
        raise ValueError("Population audit needs georeferenced ADM3 polygons with adm3_pcode.")
    boundary["adm_id"] = normalize_ids(boundary.adm3_pcode)
    if boundary.adm_id.duplicated().any() or set(boundary.adm_id) != set(df.adm_id):
        raise ValueError("Population audit: boundary IDs do not uniquely match the 13c IDs.")
    invalid = ~boundary.geometry.is_valid
    if invalid.any():
        boundary.loc[invalid, "geometry"] = boundary.loc[invalid].geometry.apply(lambda g: polygonal(make_valid(g)))
    if boundary.geometry.isna().any() or boundary.geometry.is_empty.any() or not boundary.geometry.is_valid.all():
        raise ValueError("Population audit found unusable boundaries.")
    source_values = df.set_index("adm_id")[PF]
    with rasterio.open(path) as src:
        if src.crs is None or src.count != 1:
            raise ValueError("Population raster must have a CRS and exactly one band.")
        if abs(src.transform.b) > 1e-12 or abs(src.transform.d) > 1e-12:
            raise ValueError("Rotated population rasters need explicit review; no implicit reprojection is performed.")
        scale, offset = float(src.scales[0]), float(src.offsets[0])
        if not np.isfinite([scale, offset]).all() or scale <= 0:
            raise ValueError("Invalid population raster scale/offset.")
        tags = {**src.tags(), **src.tags(1)}
        for key in ("year", "population_year", "reference_year"):
            if key in tags and re.fullmatch(r"20[0-9]{2}", str(tags[key])) and int(tags[key]) != 2025:
                raise ValueError(f"Population raster metadata {key}={tags[key]} conflicts with year 2025.")
        declared_unit = str(src.units[0] or tags.get("units", tags.get("UNITTYPE", ""))).lower()
        density_hint = any(k in declared_unit.replace(" ", "") for k in ("/km2", "/km^2", "perkm2", "perkm²", "/km²"))
        unit_conflict = density_hint and config.get("units") == "people_per_pixel"
        if unit_conflict:
            raise ValueError("Raster metadata indicates density but the review says people_per_pixel.")
        metadata = {"path": str(path), "file_size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns, "width": src.width, "height": src.height,
            "crs": str(src.crs), "nodata": str(src.nodata), "dtype": str(src.dtypes[0]),
            "scale": scale, "offset": offset, "tags": tags, "band_unit": declared_unit,
            "metadata_indicates_density": density_hint,
            "read_method": "masked=True, finite nonnegative values only; native grid; windowed reads; all_touched=False",
            "sum_units": config.get("units", "unverified_raster_units"),
            "agreement_atol_people_or_native_units": .5, "agreement_rtol": 1e-4,
            "extent_fraction_note": "Geometric file footprint coverage, NOT valid-data or population completeness."}
        transform = src.transform
        footprint = Polygon([transform * (0, 0), transform * (src.width, 0),
                             transform * (src.width, src.height), transform * (0, src.height)])
        original_crs = boundary.crs
        projected = boundary.to_crs(src.crs) if boundary.crs != src.crs else boundary
        records = []
        for i, row in enumerate(projected.itertuples(), 1):
            geom = row.geometry
            if footprint.covers(geom):
                extent_fraction = 1.0
            elif not footprint.intersects(geom):
                extent_fraction = 0.0
            else:
                pieces = gpd.GeoSeries([geom.intersection(footprint), geom], crs=src.crs).to_crs(MAP_CRS)
                extent_fraction = float(pieces.area.iloc[0] / pieces.area.iloc[1])
            total = 0.0; inside_count = 0; valid_count = 0; negative = 0; nonfinite = 0
            minimum, maximum = np.inf, -np.inf
            try:
                window = geometry_window(src, [mapping(geom)], boundless=False)
            except WindowError:
                window = None
            if window is not None:
                r0, c0 = int(window.row_off), int(window.col_off)
                r1, c1 = r0 + int(window.height), c0 + int(window.width)
                for r in range(r0, r1, 512):
                    for c in range(c0, c1, 512):
                        w = Window(c, r, min(512, c1-c), min(512, r1-r))
                        shape = (int(w.height), int(w.width))
                        inside = geometry_mask([mapping(geom)], shape, src.window_transform(w),
                                               all_touched=False, invert=True)
                        if not inside.any():
                            continue
                        inside_count += int(inside.sum())
                        data = src.read(1, window=w, masked=True)
                        observed = inside & ~np.ma.getmaskarray(data)
                        values = data.data.astype("float64") * scale + offset
                        finite = np.isfinite(values)
                        negative += int((observed & finite & (values < 0)).sum())
                        nonfinite += int((observed & ~finite).sum())
                        valid = observed & finite & (values >= 0)
                        n = int(valid.sum())
                        if n:
                            selected = values[valid]
                            total += float(selected.sum(dtype="float64")); valid_count += n
                            minimum = min(minimum, float(selected.min())); maximum = max(maximum, float(selected.max()))
            recomputed = total if valid_count > 0 else np.nan
            supplied = source_values.loc[row.adm_id]
            comparable = pd.notna(supplied) and np.isfinite(recomputed)
            agreement = bool(np.isclose(recomputed, supplied, rtol=1e-4, atol=.5)) if comparable else False
            fraction = valid_count / inside_count if inside_count else np.nan
            structural = (pd.notna(supplied) and extent_fraction >= 1-1e-6 and valid_count > 0
                          and negative == 0 and nonfinite == 0 and agreement)
            semantics = config.get("nodata_semantics", "unknown")
            nodata_ok = semantics == "unsettled" or (semantics == "missing" and inside_count > 0 and valid_count == inside_count)
            records.append({"adm_id": row.adm_id, "raster_extent_fraction": extent_fraction,
                "included_pixel_centres_in_raster": inside_count, "valid_value_pixels": valid_count,
                "valid_value_fraction_within_raster": fraction, "raster_value_sum": recomputed,
                "negative_unmasked_pixels": negative, "nonfinite_unmasked_pixels": nonfinite,
                "min_valid_value": minimum if valid_count else np.nan,
                "max_valid_value": maximum if valid_count else np.nan,
                "source_population_estimate": supplied,
                "absolute_difference": abs(recomputed-supplied) if comparable else np.nan,
                "sum_agrees_with_13c": agreement, "structural_check_pass": bool(structural),
                "nodata_semantics_check_pass": bool(nodata_ok)})
            if i % 64 == 0 or i == len(projected):
                print(f"[AUDIT] Population raster: {i}/{len(projected)} ADM3", flush=True)
    return pd.DataFrame(records), metadata


def audit_population(df: pd.DataFrame, root: Path, boundary_path: Path, out: Path, args) -> tuple[pd.DataFrame, dict]:
    out.mkdir(parents=True, exist_ok=True)
    data, report = prepare_population(df)
    config, review_meta = read_population_review(root, args.population_review)
    report.update(review_meta)
    path, inventory, selection = discover_population_raster(root, args.population_raster, config)
    pd.DataFrame(inventory, columns=["path", "selected"]).to_csv(out / "02_raster_inventory.csv", index=False)
    report.update(raster_selection=selection, raster_path=str(path) if path else None,
                  raster_audit_status="not_run", population_ready_for_main_selection=False,
                  product_accuracy_validated=False, warnings=[])
    density = config.get("reviewed") is True and config.get("units") == "people_per_km2"
    original_usable = data[PF].notna()
    if args.skip_raster_audit:
        report["raster_audit_status"] = "explicitly_skipped"
    elif path is None:
        report["warnings"].append("No unique local 2025 population TIFF. Inspect 02_raster_inventory.csv; use --population-raster to disambiguate.")
    elif not boundary_path.is_file():
        report["warnings"].append("ADM3 boundary unavailable; source raster cannot be aggregated.")
    else:
        try:
            zonal, raster_meta = raster_zonal_population_check(path, boundary_path, data, config)
            zonal.to_csv(out / "03_raster_zonal_check.csv", index=False)
            report["raster_metadata"] = raster_meta
            report["raster_audit_status"] = "completed"
            density |= bool(raster_meta["metadata_indicates_density"])
            zone = zonal.set_index("adm_id").reindex(data.adm_id)
            structural = zone.structural_check_pass.to_numpy(dtype=bool)
            nodata_pass = zone.nodata_semantics_check_pass.to_numpy(dtype=bool)
            data["population_raster_structural_check"] = structural
            data["population_raster_nodata_check"] = nodata_pass
            data["population_raster_extent_fraction"] = zone.raster_extent_fraction.to_numpy()
            data["population_raster_valid_value_fraction"] = zone.valid_value_fraction_within_raster.to_numpy()
            data.loc[original_usable & structural, "population_processing_status"] = "raster_sum_agrees_product_review_pending"
            # Proven structural mismatch excludes only the affected value; no replacement with a guessed total.
            data.loc[original_usable & ~structural, "population_processing_status"] = "raster_check_failed_review_required"
            data.loc[~structural, PF] = np.nan
            reviewed = bool(report["product_metadata_reviewed"])
            if reviewed:
                data.loc[original_usable & structural & nodata_pass, "population_processing_status"] = "processing_checks_passed"
                data.loc[original_usable & structural & ~nodata_pass, "population_processing_status"] = "unresolved_missing_raster_values"
                data.loc[~nodata_pass, PF] = np.nan
            successful = structural & nodata_pass
            report["known_source_rows_structurally_consistent"] = int((original_usable & structural).sum())
            report["known_source_rows_failing_structure"] = int((original_usable & ~structural).sum())
            report["population_ready_for_main_selection"] = bool(
                reviewed and original_usable.any() and successful[original_usable.to_numpy()].all() and not density)
        except ImportError as exc:
            report["raster_audit_status"] = "dependency_missing"
            report["warnings"].append(f"Optional raster audit needs rasterio: {exc}. Install with python -m pip install rasterio.")
        except (ValueError, OSError, RuntimeError) as exc:
            report["raster_audit_status"] = "failed"
            report["warnings"].append(str(exc))
            if "density" in str(exc).lower():
                density = True
            if "year" in str(exc).lower() and ("not 2025" in str(exc).lower() or "conflicts with" in str(exc).lower()):
                data[PF] = np.nan
                data["population_processing_status"] = "wrong_population_reference_year"
    if density:
        data[PF] = np.nan
        data["population_processing_status"] = "density_not_population_count_requires_reprocessing"
        report["population_ready_for_main_selection"] = False
        report["warnings"].append("Density is not a pixel population count. Population factor suppressed; do not sum density as people.")
    if not report["product_metadata_reviewed"]:
        report["warnings"].append("Product-specific units and NoData semantics are not reviewed. Population may only enter a provisional sensitivity scenario.")
    if report["raster_audit_status"] == "failed":
        # Preserve source values as provisional, except explicitly identified density.
        report["warnings"].append("Raw raster processing could not be completed; no raw-data verification is claimed.")
    data["population_metadata_reviewed"] = bool(report["product_metadata_reviewed"])
    report["population_available_for_sensitivity"] = int(data[PF].notna().sum())
    report["population_unavailable_for_sensitivity"] = int(data[PF].isna().sum())
    report["population_known_estimate_subtotal"] = data[PF].sum(min_count=1)
    cols = ["adm_id", "region_name", "parent_adm2_name", P, PF] + [c for c in data if c.startswith("population_") and c not in (P, PF)]
    cols = list(dict.fromkeys(cols))
    data[cols].to_csv(out / "01_population_table_audit.csv", index=False)
    write_json(out / "04_population_review.json", report)
    return data, report


def write_analysis(results, cutoffs, geometry, adm2, adm1, out, args, source_meta, population_report):
    """Write a self-contained scenario; factors, labels and counts share one table."""
    out.mkdir(parents=True, exist_ok=True)
    counts = factor_summary(results, cutoffs)
    combinations_table = combination_summary(results)
    n = int(results.priority_review_candidate.sum())
    if int(combinations_table.candidate_count.sum()) != n:
        raise ValueError("Factor-combination counts do not sum to the selected-area count.")
    results.to_csv(out / "01_adm3_screening_full.csv", index=False)
    results.loc[results.priority_review_candidate].to_csv(out / "02_adm3_review_candidates.csv", index=False)
    results.loc[results.increasing_flood_and_exposure].to_csv(out / "15_adm3_worsening_candidates.csv", index=False)
    results.screening_profile.value_counts().reindex(PROFILES, fill_value=0).rename_axis("screening_profile").reset_index(name="adm3_count").to_csv(out / "03_profile_summary.csv", index=False)
    counts.to_csv(out / "11_factor_counts.csv", index=False)
    combinations_table.to_csv(out / "12_candidate_factor_combinations.csv", index=False)
    outputs = ["01_adm3_screening_full.csv", "02_adm3_review_candidates.csv", "15_adm3_worsening_candidates.csv",
               "03_profile_summary.csv", "11_factor_counts.csv", "12_candidate_factor_combinations.csv"]
    outputs += plot_diagnostics(results, out, args.dpi)
    spatial_meta = {}
    if geometry is not None and not args.skip_maps:
        keep = [c for c in ("adm_id", "adm3_pcode", "adm2_pcode", "adm1_pcode", "adm1_name", "geometry") if c in geometry]
        mapped = geometry[keep].merge(results, on="adm_id", how="left", validate="one_to_one")
        mapfiles, spatial_meta = plot_spatial_maps(mapped, adm2, adm1, results, cutoffs, counts, out / "maps", args)
        outputs += ["maps/" + f for f in mapfiles]
        outputs += ["maps/13_marker_placement_audit.csv", "maps/14_worsening_marker_placement_audit.csv"]
    metadata = {"version": VERSION, "scenario": results.analysis_scenario.iloc[0],
        "run_time_utc": datetime.now(timezone.utc).isoformat(), **source_meta,
        "adm3_rows": len(results), "review_candidates": n,
        "worsening_candidates": int(results.increasing_flood_and_exposure.sum()),
        "undetermined_selections_due_to_unknown_population": int(results.selection_undetermined_population.sum()),
        "quantile": args.quantile, "numeric_thresholds": cutoffs,
        "factor_codes": {str(f[0]): f[3] for f in active_factors(results)},
        "population_used_in_selection": bool(results.population_factor_enabled.iloc[0]),
        "population_processing_ready": bool(population_report["population_ready_for_main_selection"]),
        "selection_rule": "(high recent activity OR high positive period-mean difference) AND at least one observed high factor",
        "threshold_policy": ">= quantile including ties, >0 for a zero cutoff; positive change also requires >0; population quantile over eligible known values",
        "map_quality": spatial_meta, "maps_generated": bool(geometry is not None and not args.skip_maps),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Population data are modelled 2025 residents, not observed people affected by flooding.",
            "Valid pixel fraction is not population completeness and is never used to scale population.",
            "Units and NoData meanings are product-specific; filename inference alone is insufficient.",
            "Missing population is unknown; not zero and not automatically below the high-population threshold.",
            "High totals can reflect ADM3 area size; density, vulnerability and coping capacity are not modelled.",
            "Flood activity means recorded activity somewhere in an ADM3, not flooding across its whole area.",
            "Change is the difference between 2019-2025 and 2000-2018 period means, not an annual trend slope.",
            "This is retrospective co-location screening, not a validated risk ranking or a future forecast.",
            "A single 2025 population snapshot does not reconstruct historical population exposure.",
            "Raster aggregation uses cell-centre inclusion, with no fractional boundary weighting or count-raster resampling.",
            "Processing consistency is not validation of WorldPop against census or survey population.",
            "Marker collision detection is approximate; review actual local maps before publication."],
        "generated_files": outputs + ["07_quality_and_methodology.json"]}
    write_json(out / "07_quality_and_methodology.json", metadata)
    print(f"[OK] {metadata['scenario']}: {n} candidates; {metadata['worsening_candidates']} high-change candidates; {out}")
    print(counts[["factor_number", "factor", "all_adm3_count", "candidate_count", "unknown_factor_count"]].to_string(index=False))
    return metadata


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, help="Project root; normally detected automatically")
    p.add_argument("--input", type=Path, help="13c directory, ZIP or merged CSV; default: auto-detect")
    p.add_argument("--out", type=Path, help="Default: processed_data/multivariate_flood_modeling/13d_adm3_humanitarian_priority_population")
    p.add_argument("--adm3-geojson", type=Path, help="Default: raw_data/administrative_boundaries/ssd_admin3.geojson")
    p.add_argument("--adm2-geojson", type=Path, help="Optional ADM2 outlines; otherwise dissolved from ADM3")
    p.add_argument("--adm1-geojson", type=Path, help="Optional ADM1 outlines; otherwise dissolved from ADM3")
    p.add_argument("--quantile", type=float, default=.75, help="Exploratory relative cutoff, default 0.75")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--formats", nargs="+", choices=["png", "svg"], default=["png", "svg"])
    p.add_argument("--badge-size", type=float, default=11, help="Circle diameter in points, default 11")
    p.add_argument("--no-state-labels", action="store_true")
    p.add_argument("--skip-maps", action="store_true", help="Explicit diagnostic tables/charts-only run")
    p.add_argument("--population-raster", type=Path, help="Optional explicit 2025 count GeoTIFF; otherwise auto-discover")
    p.add_argument("--population-review", type=Path, help="Product-specific review JSON; default raw_data/population_review_2025.json")
    p.add_argument("--population-mode", choices=["auto", "reference-only"], default="auto",
                   help="auto: checked main result or provisional sensitivity; reference-only: no factor 4")
    p.add_argument("--skip-raster-audit", action="store_true", help="Never marks population verified; permits provisional sensitivity only")
    p.add_argument("--audit-only", action="store_true", help="Write population audit without screening or maps")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 72 <= args.dpi <= 1200:
        raise ValueError("--dpi must be between 72 and 1200.")
    if not 7 <= args.badge_size <= 24:
        raise ValueError("--badge-size must be between 7 and 24 points.")
    if not 0 < args.quantile < 1:
        raise ValueError("--quantile must be strictly between zero and one.")
    args.formats = list(dict.fromkeys(args.formats))
    root = args.root.expanduser().resolve() if args.root else project_root()
    base = root / "processed_data" / "multivariate_flood_modeling"
    src = resolve_path(args.input, root) if args.input else pick_input(base)
    out = resolve_path(args.out, root) if args.out else base / "13d_adm3_humanitarian_priority_population"
    raw, source_meta = load_input(src)
    data, quality = validate(raw)
    boundary_base = root / "raw_data" / "administrative_boundaries"
    boundary = resolve_path(args.adm3_geojson, root) if args.adm3_geojson else boundary_base / "ssd_admin3.geojson"
    geo = adm2 = adm1 = None
    boundary_quality = {}
    if boundary.is_file():
        geo, boundary_quality = load_geography(boundary, data)
        if not args.skip_maps and not args.audit_only:
            a2 = resolve_path(args.adm2_geojson, root) if args.adm2_geojson else boundary_base / "ssd_admin2.geojson"
            a1 = resolve_path(args.adm1_geojson, root) if args.adm1_geojson else boundary_base / "ssd_admin1.geojson"
            adm2, adm1 = outline_layer(a2, geo, "adm2"), outline_layer(a1, geo, "adm1")
    elif not args.skip_maps and not args.audit_only:
        raise FileNotFoundError(f"ADM3 boundary not found: {boundary}. Use --adm3-geojson PATH.")
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    data, pop_report = audit_population(data, root, boundary, out / "population_audit", args)
    for message in pop_report["warnings"]:
        print("[POPULATION REVIEW] " + message, flush=True)
    if args.audit_only:
        print(f"[OK] Population audit written to {out / 'population_audit'}")
        return 0
    baseline, base_cutoffs = screening(data, args.quantile, include_population=False, scenario="baseline")
    main_population = bool(pop_report["population_ready_for_main_selection"] and args.population_mode == "auto")
    has_population = bool(data[PF].notna().any() and args.population_mode == "auto")
    scenarios = []
    if main_population:
        augmented, aug_cutoffs = screening(data, args.quantile, True, "population_processing_checked")
        scenarios.append(write_analysis(augmented, aug_cutoffs, geo, adm2, adm1, out, args, source_meta, pop_report))
        baseline.to_csv(out / "16_baseline_three_factor_reference.csv", index=False)
        population_path = out
    else:
        scenarios.append(write_analysis(baseline, base_cutoffs, geo, adm2, adm1, out, args, source_meta, pop_report))
        augmented = None
        population_path = None
        if has_population:
            augmented, aug_cutoffs = screening(data, args.quantile, True, "population_sensitivity")
            population_path = out / "population_sensitivity"
            scenarios.append(write_analysis(augmented, aug_cutoffs, geo, adm2, adm1,
                                             population_path, args, source_meta, pop_report))
    if augmented is not None:
        before = baseline.set_index("adm_id")
        after = augmented.set_index("adm_id").reindex(before.index)
        compare = before[["region_name", "parent_adm2_name", P, PF]].copy()
        compare["baseline_candidate"] = before.priority_review_candidate
        compare["with_population_candidate"] = after.priority_review_candidate
        compare["added_with_population"] = ~before.priority_review_candidate & after.priority_review_candidate
        compare["baseline_worsening_candidate"] = before.increasing_flood_and_exposure
        compare["with_population_worsening_candidate"] = after.increasing_flood_and_exposure
        compare["high_population"] = after.high_population
        compare["population_factor_unknown"] = after.population_factor_unknown
        compare["selection_undetermined_population"] = after.selection_undetermined_population
        compare["factor_codes_with_population"] = after.factor_codes
        if (before.priority_review_candidate & ~after.priority_review_candidate).any():
            raise ValueError("Adding an exposure factor unexpectedly removed a baseline candidate.")
        compare.to_csv(out / "17_population_selection_comparison.csv")
    write_json(out / "00_run_manifest.json", {
        "version": VERSION, "data_quality": quality, "boundary_quality": boundary_quality,
        "population_processing_ready": pop_report["population_ready_for_main_selection"],
        "population_used_in_main_selection": main_population,
        "population_map_output": str(population_path) if population_path else None,
        "baseline_candidates": int(baseline.priority_review_candidate.sum()),
        "scenario_counts": [{k: m[k] for k in ("scenario", "review_candidates", "worsening_candidates")} for m in scenarios],
        "note": "This run did not overwrite 13b, 13c, raw data, or the older 13d output directory."})
    print(f"[OK] Complete output: {out}")
    if population_path:
        print(f"[OK] Population scenario tables: {population_path}")
        if not args.skip_maps:
            print(f"[OK] Maps with factor 4: {population_path / 'maps'}")
        if not main_population:
            print("[IMPORTANT] Population is provisional in the sensitivity folder, NOT a verified main result.")
    else:
        print("[NOTE] No eligible population scenario was generated; see population_audit/04_population_review.json.")
    print("[NOTE] Residents in selected ADM3 are not the population actually flooded or affected.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
