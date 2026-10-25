#!/usr/bin/env python3
"""13a | South Sudan impact-source audit (read-only).

Place in scripts/multivariate_flood_modeling/ and run with VS Code or:
    python scripts/multivariate_flood_modeling/13a_impact_data_audit.py

Audits raw_data/{worldpop,health_facilities,farmland,IPC,GDP} using metadata
and sampled rows. Outputs are written ONLY to
processed_data/multivariate_flood_modeling/13a_impact_data_audit/.

IMPORTANT: This is an inventory and data-readiness audit, NOT an ADM3
exposure estimate or a verification that flood observations are complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FOLDERS = ("worldpop", "health_facilities", "farmland", "IPC", "GDP")
RASTER = {".tif", ".tiff", ".img", ".vrt", ".nc"}
VECTOR = {".shp", ".gpkg", ".geojson", ".json", ".fgb", ".kml", ".gdb"}
TABLE = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".txt"}
ARCHIVE = {".zip"}
SKIP_SUFFIX = {".dbf", ".shx", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".xml", ".aux", ".ovr", ".tfw", ".lock", ".ds_store"}
COORD_KEYWORDS = ["latitude", "longitude", "lat", "lon", "lng", "x_coord", "y_coord", "easting", "northing"]
GEONAME_KEYWORDS = ["adm3", "adm2", "pcode", "admin3", "admin2", "county", "payam", "state", "area", "location", "geometry", "wkt"]
YEAR_RE = re.compile(r"(?<!\d)(?:19[89]\d|20[0-3]\d)(?!\d)")


def project_root() -> Path:
    f = Path(__file__).resolve()
    if f.parent.name == "multivariate_flood_modeling" and f.parent.parent.name == "scripts":
        return f.parents[2]
    return Path.cwd()


def year_tokens(text: str) -> list[int]:
    return sorted(set(int(m.group()) for m in YEAR_RE.finditer(str(text))))


def short(value, n=300):
    text = str(value)
    return text[: n - 3] + "..." if len(text) > n else text


def optional_import(module: str):
    try:
        return __import__(module)
    except ImportError:
        return None


def sample_table(path: Path, max_rows: int) -> dict:
    ext = path.suffix.lower()
    details = {}
    if ext in (".csv", ".tsv", ".txt"):
        delim = "\t" if ext == ".tsv" else ","
        if ext in (".csv", ".txt"):
            with path.open("r", encoding="utf-8-sig", errors="replace") as f:
                snippet = f.read(8192)
            try:
                delim = csv.Sniffer().sniff(snippet, delimiters=",;\t|").delimiter
            except csv.Error:
                pass
        try:
            df = pd.read_csv(path, sep=delim, nrows=max_rows, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=delim, nrows=max_rows, encoding="latin-1", low_memory=False)
        details["delimiter"] = repr(delim)
        details["sample_only"] = True
    elif ext in (".xlsx", ".xls"):
        excel = pd.ExcelFile(path)
        details["sheets"] = excel.sheet_names
        df = pd.read_excel(excel, sheet_name=excel.sheet_names[0], nrows=max_rows)
        details["sample_sheet"] = excel.sheet_names[0]
        details["sample_only"] = True
    elif ext == ".parquet":
        pa = optional_import("pyarrow")
        if pa is not None:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            details["total_rows_from_metadata"] = pf.metadata.num_rows
            first = next(pf.iter_batches(batch_size=max_rows), None)
            df = first.to_pandas() if first is not None else pd.DataFrame(columns=pf.schema.names)
        else:
            df = pd.read_parquet(path).head(max_rows)  # fallback if alternative engine installed
        details["sample_only"] = True
    else:
        raise ValueError("Unsupported tabular extension")
    cols = [str(c) for c in df.columns]
    details.update({
        "sample_rows": len(df),
        "column_count": len(cols),
        "columns": cols[:100],
        "sample_missing_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
        "sample_missing_pct": {str(k): round(float(v) * 100, 2) for k, v in df.isna().mean().items()},
        "potential_coordinates": [c for c in cols if c.lower().strip() in COORD_KEYWORDS],
        "potential_admin_fields": [c for c in cols if any(k in c.lower() for k in GEONAME_KEYWORDS)],
    })
    # Do not assert that year-like values indicate complete temporal coverage.
    date_cols = [c for c in cols if any(k in c.lower() for k in ("year", "date", "period", "month", "time"))]
    details["possible_time_columns"] = date_cols[:20]
    for c in date_cols[:8]:
        nonnull = df[c].dropna().astype(str).head(100)
        years = set()
        for val in nonnull:
            years.update(year_tokens(val))
        if years:
            details.setdefault("years_seen_in_sample", {})[c] = sorted(years)
    return details


def inspect_raster(path: Path) -> dict:
    rio = optional_import("rasterio")
    if rio is None:
        return {"status": "SKIPPED_MISSING_RASTERIO", "notes": "pip install rasterio"}
    with rio.open(path) as src:
        bounds = src.bounds
        crs = src.crs
        out = {
            "driver": src.driver, "crs": str(crs) if crs else None,
            "width": src.width, "height": src.height, "bands": src.count,
            "pixel_resolution": list(src.res), "nodata": short(src.nodata),
            "dtype": list(src.dtypes), "bounds_native_crs": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "band_descriptions": list(src.descriptions)[:20],
            "has_georeference": bool(crs),
            "raster_tags": {str(k): short(v, 150) for k, v in list(src.tags().items())[:20]},
            "sample_only": False,
        }
        if crs:
            try:
                from rasterio.warp import transform_bounds
                out["bounds_wgs84"] = list(transform_bounds(crs, "EPSG:4326", *out["bounds_native_crs"], densify_pts=21))
            except Exception as err:
                out["bounds_wgs84_error"] = short(err)
        # No reading whole raster or counting observed/NoData pixels: prohibit false coverage inference.
        return out


def inspect_vector(path: Path, max_rows: int) -> dict:
    gpd = optional_import("geopandas")
    if gpd is None:
        return {"status": "SKIPPED_MISSING_GEOPANDAS", "notes": "pip install geopandas"}
    pyogrio = optional_import("pyogrio")
    layers = []
    if pyogrio is not None:
        try:
            layers = [r[0] for r in pyogrio.list_layers(path)]
        except Exception:
            pass
    layers = layers or [None]
    result = {"layers": [], "sample_only": True}
    for layer in layers[:15]:
        kwargs = {"layer": layer} if layer else {}
        if pyogrio is not None:
            try:
                inf = pyogrio.read_info(path, **kwargs)
                layer_result = {"layer": layer or "default", "feature_count": int(inf["features"]) if inf.get("features") is not None else None,
                                "crs_from_metadata": str(inf.get("crs")), "geometry_type": str(inf.get("geometry_type")),
                                "columns": [str(c) for c in inf.get("fields", [])]}
            except Exception:
                layer_result = {"layer": layer or "default"}
        else:
            layer_result = {"layer": layer or "default"}
        sample = gpd.read_file(path, rows=max_rows, **kwargs)
        columns = [str(c) for c in sample.columns if c != sample.geometry.name]
        layer_result.update({"sample_rows": int(len(sample)), "crs": str(sample.crs) if sample.crs else None,
                             "columns": columns[:100], "possible_admin_fields": [c for c in columns if any(k in c.lower() for k in GEONAME_KEYWORDS)],
                             "sample_missing_pct": {c: round(float(sample[c].isna().mean())*100, 2) for c in columns[:100]},
                             "sample_missing_geometry": int(sample.geometry.isna().sum())})
        result["layers"].append(layer_result)
    if len(layers) > 15:
        result["notes"] = "Only first 15 layers sampled."
    return result


def inspect_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        members = [x for x in z.infolist() if not x.is_dir()]
        suffixes = Counter(Path(x.filename).suffix.lower() or "(none)" for x in members)
        return {"archived_file_count": len(members), "archive_suffix_counts": dict(suffixes),
                "archive_members_preview": [short(m.filename, 150) for m in members[:35]],
                "status": "ARCHIVE_LISTED_NOT_EXTRACTED", "notes": "Contents not audited until extracted."}


def inspect_file(path: Path, root: Path, max_rows: int) -> dict:
    ext = path.suffix.lower()
    item = {"source_group": path.relative_to(root).parts[1], "file": str(path.relative_to(root)),
            "extension": ext, "size_mb": round(path.stat().st_size/1048576, 3),
            "filename_years": year_tokens(path.name), "status": "OK", "details": {}}
    try:
        if ext in RASTER:
            item["format_kind"] = "raster"
            item["details"] = inspect_raster(path)
        elif ext in VECTOR:
            item["format_kind"] = "vector"
            item["details"] = inspect_vector(path, max_rows)
        elif ext in TABLE:
            item["format_kind"] = "table"
            item["details"] = sample_table(path, max_rows)
        elif ext in ARCHIVE:
            item["format_kind"] = "archive"
            item["details"] = inspect_zip(path)
        else:
            item["format_kind"] = "other"
            item["status"] = "METADATA_ONLY"
        if item["details"].get("status", "").startswith("SKIPPED"):
            item["status"] = item["details"]["status"]
    except Exception as exc:
        item["status"] = "READ_ERROR"
        item["error"] = f"{type(exc).__name__}: {short(exc)}"
    return item


def files_in_group(base: Path):
    # Do not descend into .gdb directories. Exclude sidecars and hidden artefacts.
    if not base.exists():
        return []
    selected = []
    for f in sorted(base.rglob("*")):
        if any(part.lower().endswith(".gdb") for part in f.relative_to(base).parts):
            continue
        if not f.is_file() or f.name.startswith(".") or f.suffix.lower() in SKIP_SUFFIX:
            continue
        selected.append(f)
    for gdb in sorted(base.rglob("*.gdb")):
        if gdb.is_dir():
            selected.append(gdb)
    return sorted(selected)


def group_assessment(group, items):
    if not items:
        return {"source_group": group, "file_count": 0, "read_errors": 0,
                "status": "EMPTY_OR_MISSING", "recommended_next_step": "Locate source files."}
    bad = sum(x["status"] == "READ_ERROR" for x in items)
    unknown_crs = 0
    for x in items:
        if x["format_kind"] == "raster" and x["details"].get("crs") is None:
            unknown_crs += 1
        if x["format_kind"] == "vector" and any(y.get("crs") is None for y in x["details"].get("layers", [])):
            unknown_crs += 1
    next_step = {
        "worldpop": "Validate year, persons-per-cell units, raster NoData and ADM3 zonal population sums.",
        "health_facilities": "Validate coordinates, facility type, duplicates and whether facilities were operational in target years.",
        "farmland": "Validate agricultural class definitions, raster/vector units and reference year before ADM3 overlay.",
        "IPC": "Validate issue date, validity period, phase definition, spatial unit, and boundary crosswalk.",
        "GDP": "Validate dataset meaning, currency/unit, reference year, and suitability for ADM3 vulnerability proxy.",
    }[group]
    return {"source_group": group, "file_count": len(items), "read_errors": bad,
            "unknown_crs_or_metadata": unknown_crs,
            "status": "REQUIRES_SEMANTIC_VALIDATION" if bad == 0 else "HAS_READ_ERRORS",
            "recommended_next_step": next_step}


def run(root: Path, output: Path, max_rows: int):
    raw = root / "raw_data"
    if not raw.is_dir():
        raise FileNotFoundError(f"Expected project data folder: {raw}")
    output.mkdir(parents=True, exist_ok=True)
    items = []
    for group in FOLDERS:
        files = files_in_group(raw/group)
        print(f"{group}: {len(files)} primary files", flush=True)
        for i, file in enumerate(files, 1):
            result = inspect_file(file, root, max_rows)
            items.append(result)
            print(f"  [{i}/{len(files)}] {file.name}: {result['status']}", flush=True)
    groups = [group_assessment(g, [x for x in items if x["source_group"] == g]) for g in FOLDERS]
    flat = []
    for it in items:
        d = it["details"]
        cols = d.get("columns", [])
        if not cols and d.get("layers"):
            cols = d["layers"][0].get("columns", [])
        flat.append({
            "source_group": it["source_group"], "file": it["file"], "format_kind": it["format_kind"],
            "extension": it["extension"], "size_mb": it["size_mb"],
            "filename_years": ", ".join(map(str, it["filename_years"])),
            "status": it["status"], "crs": d.get("crs", d.get("layers", [{}])[0].get("crs") if d.get("layers") else None),
            "sample_rows": d.get("sample_rows", d.get("layers", [{}])[0].get("sample_rows") if d.get("layers") else None),
            "column_preview": ", ".join(cols[:30]), "read_error": it.get("error", ""),
        })
    pd.DataFrame(flat, columns=["source_group", "file", "format_kind", "extension", "size_mb", "filename_years", "status", "crs", "sample_rows", "column_preview", "read_error"]).to_csv(output/"01_impact_file_inventory.csv", index=False)
    pd.DataFrame(groups).to_csv(output/"02_impact_group_readiness.csv", index=False)
    errors = [{"source_group": x["source_group"], "file": x["file"], "error": x.get("error", "")} for x in items if x["status"] == "READ_ERROR"]
    pd.DataFrame(errors, columns=["source_group", "file", "error"]).to_csv(output/"03_read_errors.csv", index=False)
    (output/"04_impact_metadata_details.json").write_text(json.dumps(items, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    meta = {
        "purpose": "Read-only source inventory and sampled quality audit; NOT a flood impact computation.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root), "sample_rows_per_table_or_vector_layer": max_rows,
        "groups": groups,
        "limitations": [
            "Non-null flood event-date proxy values cannot certify actual remote-sensing observation coverage.",
            "Table null percentages are sample estimates, not complete-file missingness rates.",
            "Vector geometry statistics use sampled features; raster metadata does not prove NoData coverage.",
            "Filename and sampled-column years do not establish complete time coverage.",
            "Archives are inventoried but not extracted or internally quality-audited.",
            "No ADM3 spatial joins, population-at-risk calculations, facility-exposure estimates, or priority scores are produced.",
            "Any prioritization requires documented unit semantics, alignment, spatial coverage and forecast-time availability.",
        ],
        "next_step": "Validate source semantics and temporal coverage, then develop separate ADM3 integration script 13b.",
    }
    (output/"05_methodology_and_limitations.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 13a — South Sudan Impact Data Audit", "", "This script audits only source-file metadata and samples; it does **not** estimate people affected or prioritize humanitarian actions.", "", "## Folder readiness", "", "| Source group | Files | Read errors | Status |", "|---|---:|---:|---|"]
    lines += [f"| {g['source_group']} | {g['file_count']} | {g['read_errors']} | {g['status']} |" for g in groups]
    lines += ["", "## Interpretation safeguards", "", *[f"- {x}" for x in meta["limitations"]], "", "## Follow-up", "", "Use the inventory and JSON column details to verify variable definitions and administrative coverage before joining or ranking ADM3 areas."]
    (output/"00_README_audit_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"\nDONE — {len(items)} source files audited. Results: {output}")
    print(pd.DataFrame(groups)[["source_group", "file_count", "read_errors", "status"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Read-only inventory and audit of five impact data groups")
    parser.add_argument("--root", type=Path, default=project_root(), help="Repository root")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--sample-rows", type=int, default=500, help="Max sampled tabular/vector rows per file/layer")
    args = parser.parse_args()
    if args.sample_rows < 1:
        parser.error("--sample-rows must be positive")
    root = args.root.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else root / "processed_data/multivariate_flood_modeling/13a_impact_data_audit")
    run(root, output, args.sample_rows)


if __name__ == "__main__":
    main()
