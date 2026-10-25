#!/usr/bin/env python3
"""JBG060 | Independent audit-completeness verification (read-only).

Run from DBL-Group-22:
 python scripts/multivariate_flood_modeling/00b_verify_audit_completeness.py
 python scripts/multivariate_flood_modeling/00b_verify_audit_completeness.py --root /path/to/DBL-Group-22

This verifier does NOT import the existing audit script, reuse its discovery rules,
or claim that sampled data checks prove complete content quality. It independently
enumerates every filesystem entry under raw_data and processed_data and reconciles
against the prior audit's inventories. Its own output directory is excluded to
prevent self-generated results changing the inventory on each run.

Required: Python 3.10+ only. Optional: xarray/netCDF4/h5netcdf, rasterio,
pandas, pyarrow for targeted content checks. All missing optional checks are logged.
"""
from __future__ import annotations
import argparse, collections, csv, datetime as dt, json, os, re, stat, sys, traceback
from pathlib import Path

SOURCES = ("raw_data", "processed_data")
DATASETS = ("flood_masks", "rainfall_and_runoff", "evapotranspiration", "water_levels_lakes",
            "darthmouth_flood_observatory", "administrative_boundaries", "worldpop",
            "farmland", "health_facilities", "GDP", "IPC")
DATA_EXT = {".csv", ".tsv", ".txt", ".parquet", ".xlsx", ".xls", ".json", ".geojson",
            ".shp", ".shx", ".dbf", ".prj", ".gpkg", ".tif", ".tiff", ".nc",
            ".nc4", ".grib", ".grb", ".zip", ".h5", ".hdf5"}
IGNORABLE = {".ds_store", "thumbs.db", "desktop.ini"}
YEAR_RE = re.compile(r"(?<!\d)(20(?:0\d|1\d|2[0-5]))(?!\d)")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_inventory(path):
    if not path.exists():
        return None, f"Missing audit report: {path}"
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if "relative_path" not in (reader.fieldnames or []):
                return None, "Audit report lacks relative_path column"
            rows = list(reader)
        return rows, ""
    except Exception as e:
        return None, f"Audit report cannot be read: {type(e).__name__}: {e}"


def norm_path(p):
    return str(p).replace("\\", "/").lstrip("./")


def source_family(rel):
    chunks = Path(rel).parts
    if len(chunks) >= 3 and chunks[0] == "raw_data":
        return chunks[1]
    return "PROCESSED_OR_OTHER"


def scan_files(root, outdir):
    """Use os.walk, including hidden files, unsupported extensions and links."""
    found, special, warnings = [], [], []
    for source in SOURCES:
        base = root / source
        if not base.is_dir():
            warnings.append(f"Missing source directory: {source}")
            continue
        def onerror(err):
            warnings.append(f"Traversal error: {err}")
        for parent, dirnames, filenames in os.walk(base, topdown=True, followlinks=False, onerror=onerror):
            pdir = Path(parent)
            # Ignore only this verifier's new output; do not ignore other analysis output.
            dirnames[:] = [d for d in dirnames if (pdir / d).resolve() != outdir.resolve()]
            for d in dirnames:
                target = pdir / d
                if target.is_symlink():
                    special.append({"relative_path": target.relative_to(root).as_posix(), "issue": "DIRECTORY_SYMLINK_NOT_FOLLOWED", "target": os.readlink(target)})
            for name in filenames:
                file = pdir / name
                rel = file.relative_to(root).as_posix()
                try:
                    st = file.lstat()
                except OSError as e:
                    warnings.append(f"Stat failure {rel}: {e}")
                    continue
                if stat.S_ISLNK(st.st_mode):
                    kind = "SYMLINK"
                elif stat.S_ISREG(st.st_mode):
                    kind = "REGULAR"
                else:
                    kind = "SPECIAL_FILE"
                ext = file.suffix.lower()
                hidden = any(part.startswith(".") for part in file.relative_to(root).parts)
                entry = dict(relative_path=rel, source=source, raw_dataset_folder=source_family(rel),
                             extension=ext or "(none)", size_bytes=st.st_size, file_kind=kind,
                             hidden=hidden, known_data_extension=ext in DATA_EXT,
                             discovery_note="", audit_presence="")
                found.append(entry)
                if kind != "REGULAR":
                    special.append({"relative_path": rel, "issue": kind, "target": os.readlink(file) if kind == "SYMLINK" else ""})
    return found, special, warnings


def choose_samples(paths, per_year):
    """Deterministic per-year selection, not full validation."""
    groups = collections.defaultdict(list)
    for p in paths:
        years = YEAR_RE.findall(str(p))
        yr = int(years[-1]) if years else "UNKNOWN_YEAR"
        groups[yr].append(p)
    return [p for yr in sorted(groups, key=str) for p in sorted(groups[yr])[:per_year]]


def check_netcdf(path, rel):
    row = {"relative_path": rel, "check": "NETCDF_METADATA_AND_SMALL_SAMPLES", "status": "", "details": ""}
    try:
        import xarray as xr
        with xr.open_dataset(path, decode_times=False) as ds:
            dimensions = {str(k): int(v) for k,v in ds.sizes.items()}
            varnames = list(ds.data_vars)
            dateinfo = {}
            for name in ds.variables:
                if "time" in name.lower() or name.lower() in {"date", "valid_time"}:
                    v = ds[name]
                    dateinfo[name] = dict(shape=list(v.shape), units=str(v.attrs.get("units", "")), calendar=str(v.attrs.get("calendar", "")))
            # Only one value per numeric variable: full grid integrity is NOT checked.
            sample = {}
            for name in varnames[:8]:
                arr = ds[name]
                try:
                    item = arr.isel({d: 0 for d in arr.dims}).values
                    sample[name] = str(item)[:100]
                except Exception as exc:
                    sample[name] = "SAMPLE_ERROR: " + str(exc)[:100]
            row["status"] = "CHECKED_SAMPLE_ONLY"
            row["details"] = json.dumps({"dimensions": dimensions, "variables": varnames[:30], "time_metadata": dateinfo, "single_cell_samples": sample}, default=str)
    except ImportError as e:
        row["status"] = "NOT_CHECKED_DEPENDENCY_MISSING"; row["details"] = str(e)
    except Exception as e:
        row["status"] = "READ_ERROR"; row["details"] = f"{type(e).__name__}: {e}"
    return row


def check_raster(path, rel):
    row = {"relative_path": rel, "check": "RASTER_METADATA_AND_BLOCK_SAMPLE", "status": "", "details": ""}
    try:
        import rasterio
        with rasterio.open(path) as ds:
            band = ds.read(1, window=((0, min(64,ds.height)), (0,min(64,ds.width))), masked=True)
            details = {"crs": str(ds.crs), "bounds": list(ds.bounds), "width": ds.width,
                       "height": ds.height, "count": ds.count, "nodata": ds.nodata,
                       "sample_valid_pixels": int(band.count())}
            row["status"] = "CHECKED_SAMPLE_ONLY"; row["details"] = json.dumps(details)
    except ImportError as e:
        row["status"] = "NOT_CHECKED_DEPENDENCY_MISSING"; row["details"] = str(e)
    except Exception as e:
        row["status"] = "READ_ERROR"; row["details"] = f"{type(e).__name__}: {e}"
    return row


def check_panel(path, rel):
    row = {"relative_path": rel, "check": "PANEL_SCHEMA_FIRST_5000_ROWS", "status": "", "details": ""}
    try:
        import pandas as pd
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            df = table.slice(0, 5000).to_pandas()
            total = table.num_rows
        else:
            df = pd.read_csv(path, nrows=5000, low_memory=False)
            total = "NOT_COUNTED"
        cols = [str(c) for c in df.columns]
        yearcols = [c for c in cols if re.search(r"(^year$|date|time)", c, re.I)]
        admcols = [c for c in cols if re.search(r"adm[23]|admin[23]|county|payam|pcode", c, re.I)]
        details = {"rows_sampled": len(df), "total_rows": total, "columns": cols[:80],
                   "potential_year_columns": yearcols, "potential_adm_columns": admcols,
                   "duplicate_complete_rows_sample": int(df.duplicated().sum()),
                   "missing_cell_fraction_sample": float(df.isna().to_numpy().mean()) if df.size else None}
        row["status"] = "CHECKED_SAMPLE_ONLY"; row["details"] = json.dumps(details, default=str)
    except ImportError as e:
        row["status"] = "NOT_CHECKED_DEPENDENCY_MISSING"; row["details"] = str(e)
    except Exception as e:
        row["status"] = "READ_ERROR"; row["details"] = f"{type(e).__name__}: {e}"
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--audit-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--netcdf-samples-per-year", type=int, default=2)
    p.add_argument("--skip-content-samples", action="store_true")
    a = p.parse_args()
    if a.netcdf_samples_per_year < 0: p.error("--netcdf-samples-per-year must be nonnegative")
    if a.root is None:
        possible = [Path.cwd(), *Path(__file__).resolve().parents]
        a.root = next((q for q in possible if (q / "raw_data").is_dir() and (q / "processed_data").is_dir()), Path.cwd())
    root = a.root.resolve()
    audit = (a.audit_dir or (root / "processed_data/multivariate_flood_modeling/00_data_audit")).resolve()
    out = (a.output_dir or (root / "processed_data/multivariate_flood_modeling/00b_audit_completeness_verification")).resolve()
    if not ((root / "raw_data").is_dir() and (root / "processed_data").is_dir()):
        p.error(f"Expected raw_data and processed_data under {root}")
    if out == audit or out == root or out in (root / "raw_data", root / "processed_data"):
        p.error("Output folder cannot be root, a source folder, or existing audit folder")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Independent scan: {root}", flush=True)
    found, special, scan_errors = scan_files(root, out)
    by_path = collections.Counter(x["relative_path"] for x in found)
    expected, err1 = read_inventory(audit / "00_expected_file_inventory.csv")
    audited, err2 = read_inventory(audit / "01_audited_file_inventory.csv")
    expected_set = {norm_path(x["relative_path"]) for x in expected} if expected is not None else set()
    audited_set = {norm_path(x["relative_path"]) for x in audited} if audited is not None else set()
    current_set = set(by_path)
    # V4 audited inventory may contain auto-generated reports from previous audit run.
    old_out_prefix = "processed_data/multivariate_flood_modeling/00b_audit_completeness_verification/"
    for row in found:
        rel = row["relative_path"]
        row["audit_presence"] = "IN_AUDITED_REPORT" if rel in audited_set else "NOT_IN_AUDITED_REPORT"
        if rel not in expected_set:
            row["discovery_note"] = "NOT_IN_EXPECTED_REPORT"
        if row["file_kind"] != "REGULAR": row["discovery_note"] += " NONREGULAR_OR_LINK"
        if not row["known_data_extension"]: row["discovery_note"] += " UNRECOGNIZED_EXTENSION"
    write_csv(out / "01_independent_filesystem_inventory.csv", found,
              ["relative_path", "source", "raw_dataset_folder", "extension", "size_bytes", "file_kind", "hidden", "known_data_extension", "audit_presence", "discovery_note"])
    write_csv(out / "02_files_not_in_audit.csv", [x for x in found if x["relative_path"] not in audited_set],
              ["relative_path", "source", "raw_dataset_folder", "extension", "size_bytes", "file_kind", "hidden", "known_data_extension", "audit_presence", "discovery_note"])
    absent = [{"relative_path": s, "issue": "LISTED_BUT_NOT_ON_DISK"} for s in sorted(audited_set-current_set)]
    write_csv(out / "03_audit_entries_not_on_disk.csv", absent, ["relative_path", "issue"])
    # Exclusion candidates must be reviewed; no non-data extension is assumed disposable.
    omitted = [dict(x, priority=("REVIEW" if x["file_kind"] != "REGULAR" or
                                   (x["extension"] not in DATA_EXT and Path(x["relative_path"]).name.lower() not in IGNORABLE)
                                   else "LOW")) for x in found if x["relative_path"] not in audited_set]
    write_csv(out / "04_excluded_or_omitted_review.csv", omitted,
              ["relative_path", "source", "raw_dataset_folder", "extension", "size_bytes", "file_kind", "hidden", "known_data_extension", "audit_presence", "discovery_note", "priority"])
    counts = collections.Counter(x["raw_dataset_folder"] for x in found if x["source"] == "raw_data" and x["file_kind"] == "REGULAR")
    raw_coverage = [{"dataset": name, "independent_regular_file_count": counts[name],
                     "in_audited_inventory": sum(1 for x in found if x["source"] == "raw_data" and x["raw_dataset_folder"] == name and x["relative_path"] in audited_set),
                     "unknown_directory": name not in DATASETS} for name in sorted(set(DATASETS)|set(counts))]
    write_csv(out / "05_raw_dataset_directory_counts.csv", raw_coverage,
              ["dataset", "independent_regular_file_count", "in_audited_inventory", "unknown_directory"])
    # Date-in-path checks are diagnostics; filename date does not guarantee internal time coverage.
    temporal = []
    for name in DATASETS:
        group = [x for x in found if x["source"] == "raw_data" and x["raw_dataset_folder"] == name and x["file_kind"] == "REGULAR"]
        yrs = collections.Counter()
        for r in group:
            years = set(YEAR_RE.findall(r["relative_path"]))
            for year in years: yrs[year] += 1
        observed = sorted(yrs)
        temporal.append({"dataset": name, "files": len(group), "years_visible_in_paths": ",".join(observed),
                         "missing_years_in_filenames_2000_2025": ",".join(str(y) for y in range(2000,2026) if str(y) not in yrs),
                         "year_file_counts": json.dumps(dict(sorted(yrs.items()))),
                         "interpretation": "FILENAME_ONLY_NOT_TIME_COORDINATE_VALIDATION"})
    write_csv(out / "06_years_visible_in_raw_paths.csv", temporal,
              ["dataset", "files", "years_visible_in_paths", "missing_years_in_filenames_2000_2025", "year_file_counts", "interpretation"])
    content = []
    if not a.skip_content_samples:
        et_files = [root / x["relative_path"] for x in found if x["raw_dataset_folder"] == "evapotranspiration" and x["extension"] in (".nc", ".nc4")]
        sample_paths = choose_samples(et_files, a.netcdf_samples_per_year)
        print(f"Checking {len(sample_paths)} stratified ET NetCDF samples, plus other raw NetCDF and rasters...", flush=True)
        for file in sample_paths:
            content.append(check_netcdf(file, file.relative_to(root).as_posix()))
        for x in found:
            if x["source"] != "raw_data" or x["file_kind"] != "REGULAR": continue
            if x["raw_dataset_folder"] == "evapotranspiration": continue
            f = root / x["relative_path"]
            if x["extension"] in (".nc", ".nc4"):
                content.append(check_netcdf(f, x["relative_path"]))
            elif x["extension"] in (".tif", ".tiff"):
                content.append(check_raster(f, x["relative_path"]))
        # Candidate existing annual indicator/panel tables, not all tables.
        candidates = [x for x in found if x["source"] == "processed_data" and
                      x["file_kind"] == "REGULAR" and x["extension"] in (".csv", ".parquet") and
                      re.search(r"adm[23].*(?:annual|panel|indicator|temporal)|(?:annual|panel).*adm[23]", x["relative_path"], re.I)]
        for x in sorted(candidates, key=lambda r: r["relative_path"])[:60]:
            content.append(check_panel(root/x["relative_path"], x["relative_path"]))
    write_csv(out / "07_targeted_content_checks.csv", content, ["relative_path", "check", "status", "details"])
    write_csv(out / "08_filesystem_special_entries.csv", special, ["relative_path", "issue", "target"])
    summaries = []
    def add(key, value, meaning): summaries.append({"metric": key, "value": value, "interpretation": meaning})
    add("independent_file_entries",len(found),"Includes hidden and unsupported extensions; does not follow directory symlinks")
    add("prior_expected_rows", len(expected) if expected is not None else "UNAVAILABLE", "Prior audit inventory row count")
    add("prior_audited_rows", len(audited) if audited is not None else "UNAVAILABLE", "Prior audited inventory row count")
    add("entries_not_in_previous_audit",len(current_set-audited_set),"Includes exclusions, previous output updates and non-data files")
    add("audit_entries_missing_on_disk",len(audited_set-current_set),"Could reflect files moved/removed since previous run")
    add("previous_audit_duplicate_paths",sum(v-1 for v in collections.Counter(norm_path(r["relative_path"]) for r in (audited or [])).values() if v>1),"Duplicated rows in previous audited inventory")
    add("independent_duplicate_paths",sum(v-1 for v in by_path.values() if v>1),"Unexpected multiple traversal paths")
    add("unrecognized_extension_entries",sum(not r["known_data_extension"] for r in found),"Must be reviewed, not automatically deemed irrelevant")
    add("symlinks_and_special_entries", len(special),"Directory links are not followed")
    add("filesystem_traversal_errors",len(scan_errors),"Cannot claim completeness if nonzero")
    add("targeted_content_checks",len(content),"These are samples or metadata; not full grid/cell audit")
    add("content_read_errors",sum(x["status"] == "READ_ERROR" for x in content),"Errors in targeted verification checks")
    add("content_checks_blocked_by_dependencies",sum(x["status"] == "NOT_CHECKED_DEPENDENCY_MISSING" for x in content),"Install optional readers for these checks")
    write_csv(out / "09_verification_summary.csv", summaries, ["metric", "value", "interpretation"])
    errors = [s for s in [err1,err2,*scan_errors] if s]
    status = "NOT_VERIFIED" if errors else ("REVIEW_REQUIRED" if current_set-audited_set or audited_set-current_set or any(x["status"] == "READ_ERROR" for x in content) else "INVENTORY_RECONCILED_ONLY")
    report = ["# Independent Audit Completeness Verification", "",
              f"Generated (UTC): {dt.datetime.now(dt.timezone.utc).isoformat()}",
              f"Project: `{root}`", f"Previous audit: `{audit}`", "",
              f"**Result: {status}**", "",
              "## Independent reconciliation", "",
              f"- Filesystem entries: **{len(found):,}** (including hidden/unsupported files).",
              f"- Previous audit entries: **{len(audited) if audited is not None else 'UNAVAILABLE'}**.",
              f"- Filesystem entries absent from previous audit: **{len(current_set-audited_set):,}**.",
              f"- Prior audit entries now absent from filesystem: **{len(audited_set-current_set):,}**.",
              f"- Traversal warnings: **{len(scan_errors)}**.",
              "", "## Key limitations", "",
              "- This script enumerates all files independently from the prior audit, but does not follow directory symlinks.",
              "- Added files after the previous audit (including its own outputs) appear as newly discovered; inspect rather than automatically calling them omissions.",
              "- ET NetCDF files are sampled by year; **not all data cells or dates are checked**.",
              "- Filename year completeness does not establish that internal timestamps are complete.",
              "- Raster integrity checks use a small pixel window. No full-raster nodata coverage guarantee.",
              "- Panel schema checks use limited rows and do not validate admin-year uniqueness, spatial joins, forecast leakage or causal validity.",
              "- Unsupported extensions may be important; consult `04_excluded_or_omitted_review.csv`.",
              "- Existing audit generated files can inflate filesystem counts if comparing to a prior snapshot.",
              "", "## Errors / blockers", ""]
    report += [f"- {e}" for e in errors] if errors else ["- No inventory-read or traversal errors detected."]
    report += ["", "## Raw source-folder counts", "", "| Dataset | Files | Previously audited |", "|---|---:|---:|"]
    report += [f"| {r['dataset']} | {r['independent_regular_file_count']} | {r['in_audited_inventory']} |" for r in raw_coverage]
    report += ["", "## Suggested next actions", "",
               "1. Review **02_files_not_in_audit.csv** and **04_excluded_or_omitted_review.csv** to distinguish intentional exclusions from actual missed data.",
               "2. Resolve missing optional readers and examine errors in **07_targeted_content_checks.csv**.",
               "3. Perform separate complete year-month and timestamp checks for selected critical ET/ERA5 files.",
               "4. Validate ADM2/ADM3 region-year keys, joins, units and leakage before fitting models.", ""]
    (out/"verification_report.md").write_text("\n".join(report), encoding="utf-8")
    (out/"verification_metadata.json").write_text(json.dumps({"root":str(root),"audit_dir":str(audit),"out":str(out),"status":status,
         "scan_errors":scan_errors,"independent_entries":len(found),"prior_audited_rows":len(audited) if audited is not None else None},indent=2),encoding="utf-8")
    print(f"Result: {status}")
    print(f"Filesystem entries: {len(found)} | Not in previous audit: {len(current_set-audited_set)} | Prior entries absent: {len(audited_set-current_set)}")
    print(f"Output: {out}")
    return 1 if errors else 0

if __name__ == "__main__":
    sys.exit(main())
