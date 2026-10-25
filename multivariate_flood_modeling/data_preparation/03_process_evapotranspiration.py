#!/usr/bin/env python3
"""JBG060: Aggregate daily FAO-56 reference ET0 NetCDF to ADM2/ADM3 annual panels.

Conservative pixel-centre assignment. A polygon with zero grid-cell centres receives
missing values rather than a fabricated zero. Results are ET0, NOT actual ET.
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

VAR = "ReferenceET_PenmanMonteith_FAO56"
YEARS = range(2000, 2026)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare_grid(ds):
    if VAR not in ds:
        raise ValueError(f"Variable {VAR} missing; available={list(ds.data_vars)}")
    da = ds[VAR]
    if da.attrs.get("units", "").replace(" ", "") not in ("mmd-1", "mm/day", "mmday-1"):
        raise ValueError(f"Unexpected ET0 units {da.attrs.get('units')!r}; expected mm d-1")
    if "time" not in da.dims or "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Unexpected dimensions {da.dims}")
    if da.sizes["time"] != 1:
        raise ValueError("Expected one daily timestep per input file")
    date = pd.Timestamp(da["time"].values[0]).normalize()
    lon = np.asarray(da["lon"].values, dtype=float)
    lat = np.asarray(da["lat"].values, dtype=float)
    if len(lon) < 2 or len(lat) < 2 or len(np.unique(lon)) != len(lon) or len(np.unique(lat)) != len(lat):
        raise ValueError("Invalid or nonunique lon/lat grid")
    if not (np.allclose(np.diff(np.sort(lon)), np.diff(np.sort(lon))[0]) and np.allclose(np.diff(np.sort(lat)), np.diff(np.sort(lat))[0])):
        raise ValueError("Irregular grid; this script requires regularly spaced coordinates")
    if not (-180 <= lon.min() <= lon.max() <= 180 and -90 <= lat.min() <= lat.max() <= 90):
        raise ValueError("Coordinates do not look like WGS84 longitude/latitude")
    # This script assumes genuine EPSG:4326 coordinates, not projected x/y.
    data = da.isel(time=0).transpose("lat", "lon")
    values = np.asarray(data.values, dtype=float)
    ii = np.argsort(lat)[::-1]
    jj = np.argsort(lon)
    return date, lat[ii], lon[jj], values[np.ix_(ii, jj)]


def geometry_masks(boundary_path, level, lat, lon):
    import geopandas as gpd
    from rasterio.features import rasterize
    from affine import Affine

    gdf = gpd.read_file(boundary_path)
    # South Sudan OCHA boundaries use adm2_pcode / adm3_pcode, while
    # some derived tables use ADM2_ID / ADM3_ID. Prefer source P-code.
    candidates = [f"adm{level}_pcode", f"ADM{level}_ID", f"adm{level}_id", f"ADM{level}_PCODE"]
    col = next((c for c in candidates if c in gdf.columns), None)
    if col is None:
        raise ValueError(
            f"ADM{level} boundary has no recognizable region code. "
            f"Tried {candidates}; found {list(gdf.columns)}"
        )
    print(f"ADM{level} boundary ID column: {col}", flush=True)
    if gdf.crs is None:
        raise ValueError(f"{boundary_path} has no declared CRS; cannot safely rasterize")
    gdf = gdf.to_crs("EPSG:4326")
    if gdf[col].isna().any():
        raise ValueError(f"Missing {col} values in boundaries")
    gdf[col] = gdf[col].astype(str).str.strip()
    if gdf[col].eq("").any() or gdf[col].str.lower().isin(["nan", "none"]).any():
        raise ValueError(f"Blank {col} values in boundaries")
    # Dissolve fragmented polygons with same administrative ID.
    gdf = gdf[[col, "geometry"]].dissolve(by=col, as_index=False)
    if gdf.geometry.is_empty.any() or gdf.geometry.isna().any():
        raise ValueError("Empty geometry in administrative boundary")
    dy = float(abs(lat[0] - lat[1])); dx = float(abs(lon[1] - lon[0]))
    transform = Affine.translation(float(lon[0]) - dx / 2, float(lat[0]) + dy / 2) * Affine.scale(dx, -dy)
    # Separate mask for every polygon; this avoids silently assigning overlap
    # based on polygon order. Memory is small for a 360x140 source grid.
    masks = []
    for geom in gdf.geometry:
        masks.append(rasterize([(geom, 1)], out_shape=(len(lat), len(lon)), transform=transform, fill=0, dtype="uint8", all_touched=False).astype(bool))
    masks = np.asarray(masks)
    n = masks.sum(axis=(1, 2))
    if not np.any(n):
        raise ValueError(
            f"ADM{level} polygons contain no ET0 grid-cell centres. "
            "Check coordinate reference system and spatial coverage."
        )
    print(
        f"ADM{level}: {len(gdf)} regions; "
        f"{int(np.sum(n > 0))} with >=1 ET0 grid-cell centre; "
        f"{int(np.sum(n == 0))} with none.", flush=True
    )
    # geodesic area is not computed here; cos(latitude) is an approximation
    # for latitude-dependent pixel area on a regular geographic grid.
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    return gdf[col].tolist(), masks, n, weights


def load_one(path):
    import xarray as xr
    with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
        return prepare_grid(ds)


def aggregate_level(level, ids, masks, n_cells, weights, values, minimum_coverage):
    out = []
    ok = np.isfinite(values) & (values >= 0)
    for k, adm_id in enumerate(ids):
        polygon = masks[k]
        available = polygon & ok
        denom = float(weights[polygon].sum())
        num = float(weights[available].sum())
        frac = num / denom if denom else 0.0
        et0 = float(np.sum(values[available] * weights[available]) / num) if num > 0 and frac >= minimum_coverage else np.nan
        out.append({"adm_id": adm_id, "et0_mm_day": et0, "valid_area_fraction": frac, "grid_cells": int(n_cells[k])})
    return out


def run(root, min_coverage, min_year_fraction, maximum_files, allow_partial):
    et_dir = root / "raw_data" / "evapotranspiration"
    files = sorted(et_dir.glob("ET_*/*.nc"))
    if not files:
        raise FileNotFoundError(f"No ET0 NetCDF found under {et_dir}/ET_*/*.nc")
    if maximum_files:
        files = files[:maximum_files]
    out_dir = root / "processed_data" / "multivariate_flood_modeling" / "03_evapotranspiration"
    if maximum_files:
        out_dir = out_dir / f"smoke_test_first_{maximum_files}_files"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in files:
        parent_year = re.fullmatch(r"ET_(\d{4})", p.parent.name)
        if parent_year is None:
            raise ValueError(f"Unrecognized source folder: {p}")
        manifest.append((p, int(parent_year.group(1))))
    # Fail fast before processing thousands of daily files.
    boundary_paths = {}
    for level in (2, 3):
        gp = root / "raw_data" / "administrative_boundaries" / f"ssd_admin{level}.geojson"
        if not gp.is_file():
            raise FileNotFoundError(f"ADM{level} boundary file not found: {gp}")
        boundary_paths[level] = gp
    print("Boundary files found: " + ", ".join(str(p) for p in boundary_paths.values()), flush=True)
    date_seen = {}
    daily = {2: [], 3: []}
    masks_by_level = {}
    grid_sig = None
    qc_errors = []
    for idx, (path, folder_year) in enumerate(manifest):
        try:
            date, lat, lon, values = load_one(path)
            if date.year != folder_year or date.year not in YEARS:
                raise ValueError(f"NetCDF date {date.date()} disagrees with folder ET_{folder_year} or supported period")
            if date in date_seen:
                raise ValueError(f"Duplicate ET0 date {date.date()} in {path} and {date_seen[date]}")
            current_sig = (lat.tobytes(), lon.tobytes())
            if grid_sig is None:
                grid_sig = current_sig
                for level in (2, 3):
                    masks_by_level[level] = geometry_masks(boundary_paths[level], level, lat, lon)
            elif current_sig != grid_sig:
                raise ValueError(f"ET0 grid changed in file {path}; manual regridding is required")
            for level in (2, 3):
                ids, masks, n_cells, weights = masks_by_level[level]
                for record in aggregate_level(level, ids, masks, n_cells, weights, values, min_coverage):
                    record["date"] = date
                    daily[level].append(record)
            date_seen[date] = str(path)
        except Exception as exc:
            qc_errors.append({"file": str(path.relative_to(root)), "error": str(exc)})
            if grid_sig is None or len(masks_by_level) != 2:
                pd.DataFrame(qc_errors, columns=["file", "error"]).to_csv(
                    out_dir / "source_file_errors.csv", index=False
                )
                raise RuntimeError(
                    f"ET0 initialization failed on {path.name}: {exc}. "
                    "Check the boundary GeoJSON and the ET0 data dimensions; "
                    "this run was stopped early to avoid wasting time."
                ) from exc
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1}/{len(files)} ET0 files", flush=True)
    pd.DataFrame(qc_errors, columns=["file", "error"]).to_csv(out_dir / "source_file_errors.csv", index=False)
    calendar_all = pd.date_range("2000-01-01", "2025-12-31", freq="D")
    missing_dates = calendar_all.difference(pd.DatetimeIndex(sorted(date_seen)))
    pd.DataFrame({"missing_date": missing_dates.strftime("%Y-%m-%d")}).to_csv(out_dir / "source_missing_dates.csv", index=False)
    if not date_seen or any(not daily[level] for level in (2, 3)):
        raise ValueError("No successfully aggregated ET0 observations; review source_file_errors.csv")
    summaries = {}
    for level in (2, 3):
        df = pd.DataFrame(daily[level]); df["year"] = pd.to_datetime(df["date"]).dt.year
        df["valid_day"] = df["et0_mm_day"].notna()
        df["et0_for_sum"] = df["et0_mm_day"].fillna(0.0)
        year = df.groupby(["adm_id", "year"], as_index=False).agg(
            et0_sum_observed_mm=("et0_for_sum", "sum"),
            et0_mean_valid_day_mm=("et0_mm_day", "mean"),
            et0_days_valid=("valid_day", "sum"),
            et0_days_seen=("date", "nunique"),
            et0_min_daily_coverage=("valid_area_fraction", "min"),
            et0_mean_daily_coverage=("valid_area_fraction", "mean"),
            et0_grid_cells=("grid_cells", "first"),
        )
        year["days_expected"] = year["year"].map(lambda y: 366 if calendar.isleap(int(y)) else 365)
        year["et0_valid_day_fraction"] = year["et0_days_valid"] / year["days_expected"]
        year["et0_annual_mm"] = year["et0_sum_observed_mm"].where(year["et0_valid_day_fraction"] >= min_year_fraction)
        year["et0_mean_daily_mm"] = year["et0_mean_valid_day_mm"].where(year["et0_valid_day_fraction"] >= min_year_fraction)
        year["et0_qc_status"] = np.select([
            year["et0_grid_cells"].eq(0),
            year["et0_days_valid"].eq(0),
            year["et0_valid_day_fraction"].lt(min_year_fraction),
        ], ["NO_GRID_CELL_CENTRE", "NO_VALID_ET0", "INSUFFICIENT_DAILY_COVERAGE"], default="PASS")
        # Reindex against all administrative units, even with no valid ET0.
        ids = masks_by_level[level][0]
        keys = pd.MultiIndex.from_product([ids, YEARS], names=["adm_id", "year"]).to_frame(index=False)
        year = keys.merge(year, on=["adm_id", "year"], how="left", validate="one_to_one")
        year["et0_qc_status"] = year["et0_qc_status"].fillna("NO_SOURCE_YEAR")
        year["et0_grid_cells"] = year["et0_grid_cells"].fillna(0).astype(int)
        annual_path = out_dir / f"adm{level}_annual_et0.csv"
        year.to_csv(annual_path, index=False)
        year.groupby("et0_qc_status", dropna=False).size().rename("rows").reset_index().to_csv(out_dir / f"adm{level}_et0_qc_summary.csv", index=False)
        pd.DataFrame({"adm_id": ids, "grid_cells": masks_by_level[level][2]}).to_csv(out_dir / f"adm{level}_grid_coverage.csv", index=False)
        summaries[f"ADM{level}"] = {"annual_rows": len(year), "accepted_rows": int((year.et0_qc_status == "PASS").sum()), "zero_grid_regions": int(np.sum(masks_by_level[level][2] == 0)), "annual_file": str(annual_path.relative_to(root))}
    summary = {"source_files_found": len(files), "unique_dates_read": len(date_seen), "source_errors": len(qc_errors), "dates_missing_2000_2025": len(missing_dates), "min_daily_spatial_coverage": min_coverage, "min_annual_valid_day_fraction": min_year_fraction, "incomplete_scan": bool(maximum_files), "ADM2_ADM3": summaries, "scientific_warning": "Pixel-centre spatial assignment; no fractional polygons. Annual ET0 is an observed-day sum only when >= required coverage; incomplete years are not gap-filled. Actual evapotranspiration is NOT measured."}
    (out_dir / "et0_processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({"output_dir": str(out_dir), **summary}, indent=2))
    if qc_errors or missing_dates.size:
        print("WARNING: Not all expected daily files were processed. Check QC reports; missing dates were NOT filled with zeros.")
    if (qc_errors or missing_dates.size) and not allow_partial and not maximum_files:
        print("NOTE: Partial output was saved for diagnosis; it is not approved as a complete ET0 source.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=project_root(), help="DBL-Group-22 project root")
    p.add_argument("--min-daily-coverage", type=float, default=0.8)
    p.add_argument("--min-year-fraction", type=float, default=0.95)
    p.add_argument("--max-files", type=int, default=0, help="Small smoke test; outputs intentionally incomplete")
    p.add_argument("--allow-partial", action="store_true", help="Acknowledge incomplete source; does NOT fill missing data")
    a = p.parse_args()
    if not 0 < a.min_daily_coverage <= 1 or not 0 < a.min_year_fraction <= 1:
        p.error("Coverage thresholds must be in (0,1]")
    run(a.root.resolve(), a.min_daily_coverage, a.min_year_fraction, a.max_files, a.allow_partial)


if __name__ == "__main__":
    main()
