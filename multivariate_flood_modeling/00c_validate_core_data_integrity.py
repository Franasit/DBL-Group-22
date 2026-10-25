#!/usr/bin/env python3
"""JBG060 | 00C: independent core-data integrity and panel-readiness validation.

Read-only validation. Does NOT build modelling features or impute missing values.
Run from anywhere: python scripts/multivariate_flood_modeling/00c_validate_core_data_integrity.py
Optional: --root /path/to/DBL-Group-22 --full-et --max-et-files 0

Outputs to processed_data/multivariate_flood_modeling/00c_core_data_integrity/
The script reports limitations instead of silently declaring untested files valid.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

START, END = 2000, 2025
EXPECTED_ADM = {"adm2": 79, "adm3": 512}  # reference totals, not proof all units have observations
EXT = {".csv", ".parquet", ".xlsx", ".xls"}
REPORTS = []


def report(section, item, status, detail, path="", scope=""):
    REPORTS.append(dict(section=section, item=item, status=status, detail=str(detail), path=str(path), validation_scope=scope))


def find_root(explicit):
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "raw_data").is_dir():
            raise SystemExit(f"Not a project root (raw_data missing): {root}")
        return root
    for candidate in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (candidate / "raw_data").is_dir() and (candidate / "processed_data").is_dir():
            return candidate
    raise SystemExit("Cannot locate project. Pass --root /path/to/DBL-Group-22")


def relative(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def save(out, name, rows, columns=None):
    df = pd.DataFrame(rows)
    if columns and df.empty:
        df = pd.DataFrame(columns=columns)
    df.to_csv(out / name, index=False)
    return df


def check_calendar(value):
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        ts = pd.DatetimeIndex(ts).dropna()
        return ts
    except Exception:
        return pd.DatetimeIndex([])


def et_files(root, out, max_files, full_et):
    """Independently verify ET filename/year and NetCDF time fields when present.

    Default reads all ET file names and samples at most 52 headers to bound runtime.
    With --full-et opens every NetCDF header (NOT every raster cell).
    """
    parent = root / "raw_data" / "evapotranspiration"
    files = sorted(p for p in parent.rglob("*.nc") if p.is_file()) if parent.exists() else []
    all_rows = []
    by_year = Counter()
    for p in files:
        # Search parent and basename for an unambiguous YYYY token.
        tokens = re.findall(r"(?<!\d)(?:19\d{2}|20\d{2})(?!\d)", str(p.relative_to(parent)))
        unique = sorted(set(int(x) for x in tokens if START <= int(x) <= END))
        declared_year = unique[0] if len(unique) == 1 else None
        if declared_year is not None:
            by_year[declared_year] += 1
        all_rows.append(dict(file=relative(p, root), declared_year=declared_year, year_token_issue="" if declared_year else "MISSING_OR_AMBIGUOUS_YEAR"))
    save(out, "01_et_file_year_inventory.csv", all_rows, ["file", "declared_year", "year_token_issue"])
    days = pd.date_range(f"{START}-01-01", f"{END}-12-31", freq="D")
    expected = Counter(int(y) for y in days.year)
    year_rows = [dict(year=y, expected_daily_files=expected[y], observed_files=by_year[y], count_matches=by_year[y] == expected[y],
                      note="File count agreement does not prove unique valid dates within files") for y in range(START, END+1)]
    save(out, "02_et_annual_file_counts.csv", year_rows)
    if not files:
        report("ET", "source inventory", "MISSING", "No ET NetCDF files", parent)
        return
    bad = sum(1 for r in all_rows if r["year_token_issue"])
    report("ET", "annual file counts", "PASS" if bad == 0 and all(r["count_matches"] for r in year_rows) else "REVIEW",
           f"{len(files)} NetCDF; {bad} ambiguous filename years; counts matching={sum(r['count_matches'] for r in year_rows)}/26", parent,
           "FILENAME_AND_COUNT_ONLY")
    try:
        import xarray as xr
    except ImportError as e:
        report("ET", "NetCDF metadata", "NOT_CHECKED", f"xarray unavailable: {e}", parent)
        save(out, "03_et_netcdf_metadata_checks.csv", [], ["file", "status", "detail"])
        return
    if full_et:
        chosen = files if max_files == 0 else files[:max_files]
    else:
        # Two representative files per year, evenly spread in each year's files.
        per_year = {}
        for r, p in zip(all_rows, files):
            if r["declared_year"] is not None:
                per_year.setdefault(r["declared_year"], []).append(p)
        chosen = sorted(set(q for ps in per_year.values() for q in [ps[0], ps[-1]]))
        chosen += [p for r, p in zip(all_rows, files) if r["declared_year"] is None][:5]
        chosen = sorted(set(chosen))
        if max_files:
            chosen = chosen[:max_files]
    results = []
    for p in chosen:
        result = dict(file=relative(p, root), status="OK", dimensions="", variables="", time_start="", time_end="", time_len="",
                      time_year_matches_folder="", units="", detail="")
        try:
            with xr.open_dataset(p, decode_times=True) as ds:
                result["dimensions"] = json.dumps(dict(ds.sizes))
                result["variables"] = json.dumps(list(ds.data_vars))
                result["units"] = json.dumps({v: ds[v].attrs.get("units", "") for v in ds.data_vars})
                coord = next((c for c in ("time", "valid_time", "date") if c in ds.coords or c in ds.variables), None)
                if coord:
                    times = check_calendar(ds[coord].values.ravel())
                    result["time_len"] = int(ds[coord].size)
                    if len(times):
                        result["time_start"] = str(times.min())
                        result["time_end"] = str(times.max())
                        years = re.findall(r"(?<!\d)20\d{2}(?!\d)", str(p.relative_to(parent)))
                        years = {int(y) for y in years if START <= int(y) <= END}
                        if len(years) == 1:
                            result["time_year_matches_folder"] = all(int(y) in years for y in times.year)
                            if not result["time_year_matches_folder"]:
                                result["status"] = "REVIEW"
                    else:
                        result["status"] = "REVIEW"
                        result["detail"] = "Time coordinate present but could not decode"
                else:
                    result["detail"] = "No explicit time coordinate: date may be encoded only in filename"
                    result["status"] = "REVIEW"
        except Exception as exc:
            result.update(status="ERROR", detail=f"{type(exc).__name__}: {exc}")
        results.append(result)
    save(out, "03_et_netcdf_metadata_checks.csv", results)
    failures = sum(r["status"] == "ERROR" for r in results)
    report("ET", "NetCDF header/time sample", "REVIEW" if failures or len(chosen) < len(files) else "PASS",
           f"Examined {len(chosen)}/{len(files)} files; errors={failures}; all values/pixels NOT validated", parent,
           "ALL_HEADERS" if len(chosen) == len(files) else "SAMPLED_HEADERS")


def era5_validation(root, out, max_files):
    parent = root / "raw_data" / "rainfall_and_runoff"
    files = sorted(p for p in parent.rglob("*.nc") if p.is_file()) if parent.exists() else []
    rows = []
    try:
        import xarray as xr
    except ImportError:
        report("ERA5", "NetCDF availability", "NOT_CHECKED", "xarray not installed", parent)
        save(out, "04_era5_raw_netcdf_metadata.csv", [], ["file", "status"])
        return
    chosen = files[:max_files] if max_files else files
    for p in chosen:
        rec = dict(file=relative(p, root), status="OK", dimensions="", variables="", units="", first_time="", last_time="", time_length="", error="")
        try:
            with xr.open_dataset(p, decode_times=True) as ds:
                rec["dimensions"] = json.dumps(dict(ds.sizes))
                rec["variables"] = json.dumps(list(ds.data_vars))
                rec["units"] = json.dumps({v: ds[v].attrs.get("units", "") for v in ds.data_vars})
                axis = next((t for t in ("valid_time", "time", "date") if t in ds.variables or t in ds.coords), None)
                if axis:
                    ts = check_calendar(ds[axis].values.ravel())
                    rec["time_length"] = int(ds[axis].size)
                    if len(ts):
                        rec["first_time"] = str(ts.min())
                        rec["last_time"] = str(ts.max())
                    else:
                        rec["status"] = "REVIEW"
                        rec["error"] = "Time decode failed"
                else:
                    rec["status"] = "REVIEW"
                    rec["error"] = "No recognizable time coordinate"
        except Exception as exc:
            rec.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        rows.append(rec)
    save(out, "04_era5_raw_netcdf_metadata.csv", rows, ["file", "status"])
    report("ERA5", "Raw ERA5 NetCDF metadata", "REVIEW" if not rows or any(r["status"] != "OK" for r in rows) else "PASS",
           f"{len(rows)}/{len(files)} NetCDF headers checked. Check units and accumulation conventions before deriving rainfall/runoff.", parent,
           "NETCDF_HEADERS_ONLY")


def read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format {suffix}")


def pick_column(columns, candidates):
    exact = {str(c).strip().lower(): c for c in columns}
    for c in candidates:
        if c in exact:
            return exact[c]
    return None


def year_series(frame):
    col = pick_column(frame.columns, ["year", "flood_year", "calendar_year", "observation_year", "hydro_year", "date", "datetime", "time"])
    if col is None:
        return None, None
    vals = frame[col]
    # Some years stored numerically, others dates.
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.notna().mean() > .8 and numeric.dropna().between(1900, 2100).mean() > .9:
        return col, numeric.astype("Int64")
    return col, pd.to_datetime(vals, errors="coerce").dt.year.astype("Int64")


def detect_level(path):
    match = re.search(r"(?<![a-z0-9])adm([23])(?![a-z0-9])", str(path).lower())
    return f"adm{match.group(1)}" if match else None


def panels(root, out, max_tables, max_rows):
    base = root / "processed_data"
    priority_fragments = ("annual_flood_indicators", "annual_complete_panel", "temporal_features", "annual_rainfall", "annual_runoff")
    candidates = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in EXT and
                  detect_level(p) and (any(x in p.name.lower() for x in priority_fragments) or "flood_analysis" in str(p).lower())]
    # Most relevant first and unique exact path; avoid browsing CSVs created by audit itself.
    candidates = sorted(set(candidates), key=lambda p: ("annual_flood_indicators_all" not in p.name.lower(),
                                                        "annual_complete_panel" not in p.name.lower(), len(str(p)), str(p)))
    all_count = len(candidates)
    if max_tables:
        candidates = candidates[:max_tables]
    panel_rows, yearly_rows = [], []
    for p in candidates:
        level = detect_level(p)
        rec = dict(file=relative(p, root), level=level, status="", rows="", admin_field="", year_field="",
                   unique_units="", duplicate_unit_year_keys="", invalid_year_rows="", year_min="", year_max="", years_2000_2025="",
                   observed_unit_year_pairs="", theoretical_pairs="", missing_pairs_vs_observed_units="",
                   important_note="", error="")
        try:
            size = p.stat().st_size
            if size > 200_000_000:
                rec.update(status="NOT_CHECKED", error="File above 200MB limit; assess separately")
                panel_rows.append(rec)
                continue
            df = read_table(p)
            if max_rows and len(df) > max_rows:
                df = df.head(max_rows)
                rec["important_note"] = "SAMPLED_ROWS; completeness claims disallowed"
            rec["rows"] = len(df)
            yfield, years = year_series(df)
            rec["year_field"] = str(yfield or "")
            candidate_fields = [f"{level}_pcode", f"{level}_code", f"{level}_id", f"{level}_name", "adm_pcode", "admin_id"]
            afield = pick_column(df.columns, candidate_fields)
            rec["admin_field"] = str(afield or "")
            if afield is None or years is None:
                rec["status"] = "SCHEMA_REVIEW"
                rec["error"] = "Missing detectable admin identifier and/or year. May be features, not annual panel."
            else:
                valid = df[afield].notna() & years.notna() & years.between(START, END)
                pairs = pd.DataFrame({"unit": df[afield].astype("string").str.strip(), "year": years})
                pairs = pairs.loc[valid].drop_duplicates()
                counts = pairs.groupby("year")["unit"].nunique()
                rec["unique_units"] = pairs["unit"].nunique()
                rec["duplicate_unit_year_keys"] = int(df.loc[valid].shape[0] - len(pairs))
                rec["invalid_year_rows"] = int((~valid).sum())
                rec["year_min"] = int(years.min()) if years.notna().any() else ""
                rec["year_max"] = int(years.max()) if years.notna().any() else ""
                rec["years_2000_2025"] = int(counts.shape[0])
                rec["observed_unit_year_pairs"] = len(pairs)
                rec["theoretical_pairs"] = EXPECTED_ADM[level] * (END - START + 1)
                rec["missing_pairs_vs_observed_units"] = int(rec["unique_units"] * (END - START + 1) - len(pairs))
                rec["status"] = "KEY_REVIEW" if rec["duplicate_unit_year_keys"] else "KEYS_UNIQUE"
                for yr in range(START, END + 1):
                    yearly_rows.append(dict(file=relative(p, root), level=level, year=yr, observed_units=int(counts.get(yr, 0)),
                                            reference_unit_count=EXPECTED_ADM[level]))
        except Exception as exc:
            rec.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        panel_rows.append(rec)
    save(out, "05_adm_panel_key_checks.csv", panel_rows)
    save(out, "06_adm_panel_year_coverage.csv", yearly_rows)
    report("ADM panels", "candidate table key/coverage", "REVIEW" if any(r["status"] not in ("KEYS_UNIQUE",) for r in panel_rows) else "PASS",
           f"{len(panel_rows)}/{all_count} relevant table files checked; unique keys alone do not establish flood indicator correctness.", base,
           "FULL_TABLE_WHEN_SMALL; OTHERWISE_ROW_SAMPLE")


def driver_sources(root, out):
    """Source availability is not spatial joinability: record that distinction."""
    definitions = {
        "flood_masks": ("raw_data/flood_masks", "pixel/tile/day -> administrative year aggregation required"),
        "rainfall_and_runoff": ("raw_data/rainfall_and_runoff", "ERA5 grid -> administrative polygons; verify accumulation units"),
        "evapotranspiration": ("raw_data/evapotranspiration", "ET0 reference ET, not measured actual ET; aggregate grid to polygons"),
        "water_levels_lakes": ("raw_data/water_levels_lakes", "Upstream lake-level station series; hydrological linkage not polygon assignment"),
        "darthmouth_flood_observatory": ("raw_data/darthmouth_flood_observatory", "Discharge station/event hydrograph; check hydrological routing"),
        "administrative_boundaries": ("raw_data/administrative_boundaries", "Validate exact ADM2 and ADM3 region identifiers/geometry"),
        "worldpop": ("raw_data/worldpop", "Exposure only where time coverage valid (2015–2025 available)") ,
        "farmland": ("raw_data/farmland", "Exposure/use data with dataset-specific temporal coverage"),
        "health_facilities": ("raw_data/health_facilities", "Spatial exposure features; do not assume yearly history"),
        "GDP": ("raw_data/GDP", "Country-level time series, not automatically ADM2/ADM3 GDP"),
        "IPC": ("raw_data/IPC", "IPC temporal/geographic reporting periods must match annual panel"),
    }
    rows = []
    for key, (name, caution) in definitions.items():
        path = root / name
        count = sum(p.is_file() and not p.name.startswith(".") for p in path.rglob("*")) if path.is_dir() else 0
        rows.append(dict(dataset=key, raw_path=name, files=count, directory_exists=path.is_dir(), join_status="NOT_YET_VALIDATED", caveat=caution))
    save(out, "07_source_joinability_inventory.csv", rows)
    report("Drivers", "joinability inventory", "NOT_CHECKED", "Existence audited; no geospatial joins, lagged forecasting, or target-leakage checks performed", root,
           "FILE_EXISTENCE_ONLY")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Project folder containing raw_data and processed_data")
    parser.add_argument("--full-et", action="store_true", help="Inspect every ET NetCDF header (can take a long time)")
    parser.add_argument("--max-et-files", type=int, default=0, help="Limit ET inspected headers, 0 means default sample or all with --full-et")
    parser.add_argument("--max-era5-files", type=int, default=0, help="Limit ERA5 headers, 0 = all")
    parser.add_argument("--max-panel-files", type=int, default=100, help="Limit panel candidate files, 0 = all")
    parser.add_argument("--max-panel-rows", type=int, default=0, help="0 = full table, otherwise first N rows")
    args = parser.parse_args()
    root = find_root(args.root)
    out = root / "processed_data" / "multivariate_flood_modeling" / "00c_core_data_integrity"
    out.mkdir(parents=True, exist_ok=True)
    et_files(root, out, args.max_et_files, args.full_et)
    era5_validation(root, out, args.max_era5_files)
    panels(root, out, args.max_panel_files, args.max_panel_rows)
    driver_sources(root, out)
    summary = save(out, "08_validation_summary.csv", REPORTS)
    metadata = {"run_utc": datetime.now(timezone.utc).isoformat(), "project_root": str(root), "study_years": [START, END],
                "adm_reference_counts": EXPECTED_ADM, "full_et": args.full_et,
                "notes": ["No data were modified", "Reading a file does not validate all numeric values",
                          "ET0 reference ET is not actual evapotranspiration", "Spatial joins and data leakage are not checked here"]}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["# JBG060 — 00C Core Data Integrity Report", "", f"Run: {metadata['run_utc']}", "",
             "This is a **core integrity screening**, not a certification that all data or joins are correct.", "",
             "| Section | Test | Status | What was checked |", "|---|---|---|---|"]
    for row in REPORTS:
        lines.append("| " + " | ".join(str(row[k]).replace("|", "/") for k in ("section", "item", "status", "detail")) + " |")
    lines += ["", "## Mandatory before forecasting", "", "- Verify true observation dates across *all* ET files if important; use --full-et for all headers.",
              "- Verify NetCDF time encoding, ERA5 accumulated-unit semantics, ET0 meaning, and grid-to-ADM spatial weights.",
              "- Resolve ADM2/ADM3 keys, missing administrative areas, duplicates, and year coverage.",
              "- Define forecast origin and restrict predictors to data available at prediction time.",
              "- Validate Dartmouth and upstream lake connectivity, and audit missingness by model feature.",
              "- Compare raw annual flood masks and derived flood indicator methodology before modelling."]
    (out / "integrity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Completed core integrity screening. Reports: {out}")
    print(summary[["section", "item", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
