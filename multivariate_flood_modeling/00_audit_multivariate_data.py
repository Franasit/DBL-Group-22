#!/usr/bin/env python3
"""JBG060 | Comprehensive, non-destructive raw + processed data audit (v4).

Run:
  python scripts/multivariate_flood_modeling/00_audit_multivariate_data.py
  python scripts/multivariate_flood_modeling/00_audit_multivariate_data.py --root /path/to/DBL-Group-22

Dependencies: pandas, numpy; recommended: xarray, netCDF4, openpyxl,
pyarrow, rasterio, geopandas. Missing optional readers are reported, not hidden.
NetCDF checks read metadata and limited samples by default, not full rasters.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = [
    'flood_masks', 'rainfall_and_runoff', 'evapotranspiration',
    'water_levels_lakes', 'darthmouth_flood_observatory',
    'administrative_boundaries', 'worldpop', 'farmland',
    'health_facilities', 'GDP', 'IPC'
]
EXTENSIONS = {'.csv', '.tsv', '.txt', '.parquet', '.xlsx', '.xls', '.json',
              '.geojson', '.shp', '.shx', '.dbf', '.prj', '.gpkg', '.tif',
              '.tiff', '.nc', '.nc4', '.grib', '.grb', '.zip', '.h5', '.hdf5'}
PATTERN = re.compile(r'recurr|unusual|persist|cluster|hotspot|cross[_-]?scale|trend|flood[_-]?pattern|composition', re.I)
TIME_NAME = re.compile(r'^(year|date|datetime|timestamp|time|valid_time|observation_date|event_date|year_\d+)$', re.I)
ADM_NAME = re.compile(r'adm[_ ]?[123]|admin[_ ]?[123]|pcode|county|district|shapeid|shape_id', re.I)
MISSING = {'', 'NA', 'N/A', 'NULL', 'NONE', 'NAN', '-9999', '-999'}


# Authoritative top-level source directories. Never override these with filename keywords.
RAW_DIR_MAP = {
    'flood_masks': 'flood_masks',
    'rainfall_and_runoff': 'rainfall_and_runoff',
    'evapotranspiration': 'evapotranspiration',
    'water_levels_lakes': 'water_levels_lakes',
    'darthmouth_flood_observatory': 'darthmouth_flood_observatory',
    'dartmouth_flood_observatory': 'darthmouth_flood_observatory',
    'administrative_boundaries': 'administrative_boundaries',
    'worldpop': 'worldpop',
    'farmland': 'farmland',
    'health_facilities': 'health_facilities',
    'gdp': 'GDP',
    'ipc': 'IPC',
}

# Source-specific folders in cross-dataset outputs: use only unambiguous markers.
# Multiple independent source markers are flagged as mixed-source, not forced into one category.
PROCESSED_SOURCE_MARKERS = [
    (r'dartmouth|darthmouth|discharge', 'darthmouth_flood_observatory'),
    (r'lake[_-]?water|lake[_-]?level|water[_-]?levels[_-]?lakes', 'water_levels_lakes'),
    (r'evapotranspiration|penmanmonteith|referenceet|(?<![a-z])et[_-]20\d\d', 'evapotranspiration'),
    (r'rainfall|runoff|era5|precipitation', 'rainfall_and_runoff'),
    (r'worldpop|ssd[_-]?pop[_-]?20\d\d|population[_-]?exposure', 'worldpop'),
    (r'health[_-]?facilit|health[_-]?centre|health[_-]?center', 'health_facilities'),
    (r'farmland|cropland|rangeland|cattle|agricultur', 'farmland'),
    (r'(?<![a-z])gdp(?![a-z])|gross[_-]?domestic', 'GDP'),
    (r'(?<![a-z])ipc(?![a-z])|food[_-]?insecur', 'IPC'),
    (r'administrative[_-]?boundar|admin[_-]?boundar', 'administrative_boundaries'),
    (r'flood[_-]?mask|flood[_-]?analysis|adm[23][_/-]flood|annual[_-]?flood|flood[_-]?hazard', 'flood_masks'),
]


def classify(path: Path, source: str) -> dict:
    """Return explicit source lineage and confidence; never guess from generic 'population'."""
    parts = path.parts
    if len(parts) < 2:
        return dict(dataset='UNCLASSIFIED', classification_rule='INSUFFICIENT_PATH',
                    classification_confidence='NONE', review_status='REVIEW_REQUIRED',
                    source_datasets='', classification_evidence='')
    top = parts[1].lower()
    # IMPORTANT: raw_data/farmland/cattle-population... stays farmland.
    if source == 'raw_data':
        dataset = RAW_DIR_MAP.get(top)
        return dict(dataset=dataset or 'UNCLASSIFIED', classification_rule='RAW_PARENT_DIRECTORY',
                    classification_confidence='HIGH' if dataset else 'NONE',
                    review_status='CONFIRMED_BY_DIRECTORY' if dataset else 'REVIEW_REQUIRED',
                    source_datasets=dataset or '', classification_evidence=parts[1])
    # Dedicated processed-data directories likewise take precedence over keywords.
    if top in RAW_DIR_MAP:
        dataset = RAW_DIR_MAP[top]
        return dict(dataset=dataset, classification_rule='PROCESSED_PARENT_DIRECTORY',
                    classification_confidence='HIGH', review_status='CONFIRMED_BY_DIRECTORY',
                    source_datasets=dataset, classification_evidence=parts[1])
    # In a cross-dataset folder, an explicit source-specific subdirectory is authoritative
    # only if it denotes one source (the folder can contain additional analysis files).
    text = '/'.join(parts[2:]).lower()
    hits = {dataset for pattern, dataset in PROCESSED_SOURCE_MARKERS
            if re.search(pattern, text, re.I)}
    if len(hits) == 1:
        dataset = next(iter(hits))
        return dict(dataset=dataset, classification_rule='PROCESSED_EXPLICIT_MARKER',
                    classification_confidence='MEDIUM', review_status='REVIEW_SOURCE_LINEAGE',
                    source_datasets=dataset, classification_evidence=text[:240])
    if len(hits) > 1:
        return dict(dataset='MULTI_SOURCE_REVIEW', classification_rule='MULTIPLE_SOURCE_MARKERS',
                    classification_confidence='LOW', review_status='REVIEW_REQUIRED',
                    source_datasets=';'.join(sorted(hits)), classification_evidence=text[:240])
    # Existing cross-dataset products are analytical outputs, not a twelfth raw source.
    if top in {'cross_dataset_analysis', 'multivariate_flood_modeling'} or PATTERN.search(text):
        return dict(dataset='DERIVED_CROSS_DATASET_ANALYSIS',
                    classification_rule='DERIVED_ANALYSIS_WITH_UNCONFIRMED_LINEAGE',
                    classification_confidence='MEDIUM', review_status='REVIEW_SOURCE_LINEAGE',
                    source_datasets='', classification_evidence=text[:240])
    return dict(dataset='UNCLASSIFIED', classification_rule='NO_EXPLICIT_SOURCE',
                classification_confidence='NONE', review_status='REVIEW_REQUIRED',
                source_datasets='', classification_evidence=text[:240])


LAKE_COLUMNS = [
    'mission','cycle','date_raw','hour','minute','level_anomaly_m','estimated_error_m',
    'backscatter_db','wet_tropo','ionosphere','dry_tropo','mode1','mode2',
    'frozen_flag','level_egm2008_m','source_flag'
]


def inspect_lake_altimetry_txt(path: Path) -> dict:
    """Parse USDA/USGS-style TPJOJS satellite lake altimetry, skipping prose header.

    Invalid markers are explicitly excluded. Duplicate dates may reflect overlapping
    missions and are *reported*, not removed at the audit stage.
    """
    records = []
    metadata = {}
    rejected = Counter()
    with path.open('r', encoding='utf-8-sig', errors='replace') as fh:
        for line in fh:
            tokens = line.strip().split()
            if len(tokens) == 16 and re.fullmatch(r'\d{8}', tokens[2]):
                records.append(tokens)
            elif 'Latitude and longitude' in line:
                metadata['location_text'] = line.strip()
            elif 'Target database id number and name' in line:
                metadata['target_text'] = line.strip()
    if not records:
        raise ValueError('No valid 16-column satellite observation rows were found')
    df = pd.DataFrame(records, columns=LAKE_COLUMNS)
    dates = pd.to_datetime(df.date_raw, format='%Y%m%d', errors='coerce')
    numeric_cols = ['level_anomaly_m','estimated_error_m','level_egm2008_m']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    invalid_date = dates.isna()
    invalid_anomaly = df.level_anomaly_m.eq(999.99) | ~np.isfinite(df.level_anomaly_m)
    invalid_error = df.estimated_error_m.eq(99.999) | ~np.isfinite(df.estimated_error_m)
    invalid_egm = df.level_egm2008_m.eq(9999.99) | ~np.isfinite(df.level_egm2008_m)
    valid = ~invalid_date & ~invalid_anomaly & ~invalid_error & ~invalid_egm
    study = dates.dt.year.between(2000, 2025)
    yy = sorted(dates.loc[valid & study].dt.year.unique().astype(int).tolist())
    return dict(check_level='FULL_TABULAR_CONTENT', check_method='LAKE_TPJOJS_16_COLUMN_PARSER',
                rows_examined=len(df), observation_rows=len(df),
                invalid_dates=int(invalid_date.sum()), invalid_level_anomaly=int(invalid_anomaly.sum()),
                invalid_level_egm2008=int(invalid_egm.sum()), invalid_error=int(invalid_error.sum()),
                valid_observations=int(valid.sum()), valid_observations_2000_2025=int((valid & study).sum()),
                duplicate_valid_dates=int(dates.loc[valid].duplicated().sum()),
                year_counts_2000_2025=json.dumps({str(y):int(n) for y,n in dates.loc[valid & study].dt.year.value_counts().sort_index().items()}),
                min_year=min(yy) if yy else None, max_year=max(yy) if yy else None,
                years_found=json.dumps(yy), columns=json.dumps(LAKE_COLUMNS),
                lake_metadata=json.dumps(metadata),
                warning='Lake site is not an ADM2/ADM3 observation; watershed linkage and mission harmonization required')


def inspect_text(path: Path) -> dict:
    # These are satellite-altimetry tables with a prose header, not comma-separated CSV.
    if re.fullmatch(r'water_level_(kyoga|victoria)(?:\(\d+\))?\.txt', path.name, re.I):
        return inspect_lake_altimetry_txt(path)
    with path.open('r', encoding='utf-8-sig', errors='replace') as handle:
        lines = [handle.readline().rstrip('\n') for _ in range(30)]
    return dict(check_level='SAMPLED_CONTENT', check_method='TEXT_FIRST_30_LINES',
                rows_examined=sum(bool(x) for x in lines),
                text_preview=json.dumps(lines[:8], ensure_ascii=False),
                warning='Unstructured text preview only; values and chronology not validated')


def read_csv_with_header_detection(path: Path, sep: str, limit: int) -> tuple[pd.DataFrame, str]:
    # World Bank CSV exports often have a few non-tabular preamble lines.
    if re.search(r'API_.*DS2.*\.csv$', path.name, re.I):
        for skip in range(0, 8):
            try:
                df = pd.read_csv(path, sep=sep, skiprows=skip, nrows=limit,
                                 low_memory=False, encoding='utf-8-sig')
                if any('country name' in str(c).lower() for c in df.columns):
                    return df, f'WORLD_BANK_HEADER_SKIP_{skip}'
            except (pd.errors.ParserError, UnicodeError):
                continue
    df = pd.read_csv(path, sep=sep, nrows=limit, low_memory=False, encoding='utf-8-sig')
    return df, 'FIRST_N_ROWS'


def inspect_table(path: Path, suffix: str, limit: int) -> dict:
    if suffix in ('.csv', '.tsv'):
        sep = '\t' if suffix == '.tsv' else ','
        df, method = read_csv_with_header_detection(path, sep, limit)
    elif suffix in ('.xlsx', '.xls'):
        book = pd.ExcelFile(path)
        df = pd.read_excel(book, sheet_name=book.sheet_names[0], nrows=limit)
        method = 'FIRST_SHEET_FIRST_N_ROWS'
    elif suffix == '.parquet':
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        batches = pf.iter_batches(batch_size=min(limit, 8192))
        parts, read_count = [], 0
        for batch in batches:
            parts.append(batch.to_pandas())
            read_count += batch.num_rows
            if read_count >= limit:
                break
        df = pd.concat(parts, ignore_index=True).head(limit) if parts else pd.DataFrame(columns=pf.schema_arrow.names)
        method = 'FIRST_N_ROWS_PARQUET_BATCHED'
    elif suffix in ('.json', '.geojson'):
        if suffix == '.geojson':
            import geopandas as gpd
            df = pd.DataFrame(gpd.read_file(path).drop(columns='geometry', errors='ignore')).head(limit)
        else:
            df = pd.read_json(path).head(limit)
        method = 'FIRST_N_ROWS_JSON_LOADED'
    elif suffix in ('.gpkg', '.shp'):
        import geopandas as gpd
        df = pd.DataFrame(gpd.read_file(path, rows=limit).drop(columns='geometry', errors='ignore'))
        method = 'FIRST_N_SPATIAL_FEATURES'
    else:
        return {}
    columns = [str(c) for c in df.columns]
    date_candidates = [c for c in columns if TIME_NAME.search(c)]
    adm_candidates = [c for c in columns if ADM_NAME.search(c)]
    years = set()
    time_info = []
    for c in date_candidates[:8]:
        ser = df[c].dropna()
        if ser.empty:
            continue
        if re.fullmatch(r'year|year_\d+', c, re.I):
            numeric = pd.to_numeric(ser, errors='coerce')
            valid = numeric[numeric.between(1900, 2100) & (numeric % 1 == 0)]
            yy = set(valid.astype(int).tolist())
        else:
            # Avoid numeric timestamps interpreted as epoch nanoseconds.
            strings = ser.astype(str).str.strip()
            strings = strings[~strings.str.fullmatch(r'[+-]?\d+(\.\d+)?')]
            dates = pd.to_datetime(strings, errors='coerce', utc=True)
            yy = set(dates.dropna().dt.year.astype(int).tolist())
            yy = {y for y in yy if 1900 <= y <= 2100}
        if yy:
            years |= yy
            time_info.append({'column': c, 'min_year': min(yy), 'max_year': max(yy)})
    null_pct = round(float(df.isna().mean().mean() * 100), 3) if df.size else None
    return dict(check_level='SAMPLED_CONTENT', check_method=method, rows_examined=int(len(df)),
                columns=json.dumps(columns), time_columns=json.dumps(date_candidates),
                admin_columns=json.dumps(adm_candidates),
                pattern_columns=json.dumps([c for c in columns if PATTERN.search(c)]),
                years_found=json.dumps(sorted(years)), min_year=min(years) if years else None,
                max_year=max(years) if years else None,
                missing_cell_pct_sample=null_pct,
                duplicate_rows_sample=int(df.duplicated().sum()) if len(df) else 0,
                time_field_details=json.dumps(time_info),
                warning='Sample statistics are not full-dataset quality guarantees')


def inspect_netcdf(path: Path, sample_cells: int) -> dict:
    import xarray as xr
    with xr.open_dataset(path, decode_times=True, chunks=None) as ds:
        dims = {str(k): int(v) for k,v in ds.sizes.items()}
        variables = {}
        times = []
        sampled = {}
        for name, var in ds.data_vars.items():
            variables[str(name)] = {'dims': list(var.dims), 'units': str(var.attrs.get('units', 'NOT_SPECIFIED')),
                                    'dtype': str(var.dtype)}
            if np.issubdtype(var.dtype, np.number) and sample_cells > 0:
                # Strided sub-selection: never read entire gridded array.
                indices = {}
                for dim in var.dims:
                    n = int(var.sizes[dim]); indices[dim] = slice(0, n, max(1, n // 8))
                view = var.isel(indices)
                # cap selection sizes further when there are >2 dimensions
                for dim in view.dims:
                    if view.size <= sample_cells: break
                    view = view.isel({dim: slice(0, min(int(view.sizes[dim]), 2))})
                if view.size <= sample_cells:
                    try:
                        a = np.asarray(view.values)
                        sampled[str(name)] = {'cells': int(a.size), 'nulls': int(pd.isna(a).sum())}
                    except Exception as exc:
                        sampled[str(name)] = {'sample_error': str(exc)[:160]}
        for name in ds.coords:
            if re.search(r'time|date', name, re.I):
                c = ds.coords[name]
                if c.size:
                    try:
                        vals = c.values
                        # xarray/cftime calendars are represented as strings when needed.
                        first, last = str(vals[0]), str(vals[-1])
                        times.append({'coordinate': str(name), 'first': first, 'last': last, 'count': int(c.size)})
                    except Exception as exc:
                        times.append({'coordinate': str(name), 'error': str(exc)[:160]})
        return dict(check_level='METADATA_AND_SAMPLED_CONTENT', check_method='XARRAY_LAZY_OPEN_AND_BOUNDED_SAMPLE',
                    dimensions=json.dumps(dims), variables=json.dumps(variables),
                    netcdf_time_coordinates=json.dumps(times), netcdf_sample=json.dumps(sampled),
                    warning='Full-grid missingness, continuity and spatial alignment not yet verified')


def inspect_raster(path: Path) -> dict:
    import rasterio
    with rasterio.open(path) as ds:
        return dict(check_level='METADATA_ONLY', check_method='RASTERIO_METADATA',
                    raster_crs=str(ds.crs), raster_bounds=str(tuple(ds.bounds)),
                    raster_resolution=str(ds.res), raster_shape=f'{ds.height}x{ds.width}',
                    raster_nodata=str(ds.nodata), warning='Pixel-level validity not audited')


def inspect_one(path: Path, root: Path, origin: str, max_rows: int, nc_sample: int) -> dict:
    relative = path.relative_to(root).as_posix()
    ext = path.suffix.lower()
    row = dict(relative_path=relative, source=origin, **classify(path.relative_to(root), origin),
               extension=ext, size_bytes=path.stat().st_size, check_level='INVENTORY_ONLY',
               check_method='FILE_STAT', result='OK', warning='')
    try:
        if ext == '.txt':
            row.update(inspect_text(path))
        elif ext in {'.csv','.tsv','.xlsx','.xls','.parquet','.json','.geojson','.gpkg','.shp'}:
            row.update(inspect_table(path, ext, max_rows))
        elif ext in {'.nc','.nc4'}:
            row.update(inspect_netcdf(path, nc_sample))
        elif ext in {'.tif','.tiff'}:
            row.update(inspect_raster(path))
        else:
            row.update(result='NOT_CONTENT_CHECKED', warning='No content reader configured for this file type')
    except (ImportError, ModuleNotFoundError) as exc:
        row.update(result='READER_UNAVAILABLE', check_level='INVENTORY_ONLY', warning=str(exc)[:250])
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        backend_missing = ext in {'.nc', '.nc4'} and ('dependencies may not be installed' in message or 'unrecognized engine' in message)
        row.update(result='READER_UNAVAILABLE' if backend_missing else 'READ_ERROR',
                   check_level='INVENTORY_ONLY', warning=message[:300])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--max-rows', type=int, default=25000)
    p.add_argument('--nc-sample-cells', type=int, default=1024)
    p.add_argument('--include-raw', action='store_true', help='Compatibility option; raw_data is included by default')
    p.add_argument('--processed-only', action='store_true', help='Skip raw_data, not recommended for full coverage')
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    out = root/'processed_data'/'multivariate_flood_modeling'/'00_data_audit'
    out.mkdir(parents=True, exist_ok=True)
    sources = ([('raw_data', root/'raw_data')] if not args.processed_only else []) + [('processed_data', root/'processed_data')]
    expected = []
    ignored = []
    for source, base in sources:
        if not base.is_dir():
            ignored.append({'path':str(base.relative_to(root)), 'reason':'SOURCE_DIRECTORY_MISSING'})
            continue
        for folder, dirs, files in os.walk(base):
            fpath = Path(folder)
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in {'__pycache__', '.git', '.venv', 'venv'} and
                             not (fpath/d).resolve().is_relative_to(out.resolve()))
            for name in sorted(files):
                file = fpath/name
                if name.startswith('.') or file.is_symlink():
                    ignored.append({'path':file.relative_to(root).as_posix(), 'reason':'HIDDEN_OR_SYMLINK'}); continue
                if file.suffix.lower() not in EXTENSIONS:
                    ignored.append({'path':file.relative_to(root).as_posix(), 'reason':'NON_DATA_EXTENSION'}); continue
                expected.append((file, source))
    expected.sort(key=lambda z: z[0].as_posix())
    print(f'Found {len(expected)} input data files; auditing (this may take time for large files).', flush=True)
    rows = []
    for i,(path,source) in enumerate(expected, 1):
        try:
            rows.append(inspect_one(path, root, source, args.max_rows, args.nc_sample_cells))
        except Exception as exc:
            rows.append(dict(relative_path=path.relative_to(root).as_posix(), source=source,
                             **classify(path.relative_to(root), source), result='READ_ERROR', check_level='INVENTORY_ONLY',
                             warning=f'Unexpected error: {exc}'[:300]))
        if i % 100 == 0: print(f'  Audited {i}/{len(expected)} files', flush=True)
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        inventory = pd.DataFrame(columns=['relative_path','source','dataset','result','check_level'])
    inventory.to_csv(out/'file_inventory.csv', index=False)
    exp_df = pd.DataFrame([{'relative_path':f.relative_to(root).as_posix(), 'source':src,
                            'size_bytes':f.stat().st_size} for f,src in expected],
                          columns=['relative_path','source','size_bytes'])
    exp_df.to_csv(out/'00_expected_file_inventory.csv', index=False)
    inventory.to_csv(out/'01_audited_file_inventory.csv', index=False)
    classification_cols = ['relative_path','source','dataset','source_datasets',
                           'classification_rule','classification_confidence',
                           'review_status','classification_evidence']
    for col in classification_cols:
        if col not in inventory.columns:
            inventory[col] = ''
    inventory[classification_cols].to_csv(out/'dataset_classification_validation.csv', index=False)
    inventory.loc[inventory.review_status.astype(str).str.contains('REVIEW', na=False),
                  classification_cols].to_csv(out/'classification_review_required.csv', index=False)

    found = set(inventory.relative_path.tolist())
    missing = [{'relative_path':f.relative_to(root).as_posix(), 'reason':'NOT_AUDITED'} for f,_ in expected if f.relative_to(root).as_posix() not in found]
    missing += inventory.loc[inventory.result.isin(['READ_ERROR','READER_UNAVAILABLE','NOT_CONTENT_CHECKED']), ['relative_path','result','warning']].fillna('').to_dict('records')
    pd.DataFrame(missing, columns=['relative_path','reason','result','warning']).to_csv(out/'02_missing_or_skipped_files.csv', index=False)
    pd.DataFrame(ignored, columns=['path','reason']).to_csv(out/'excluded_files.csv', index=False)
    coverage = []
    for category in DATASETS + ['DERIVED_CROSS_DATASET_ANALYSIS', 'MULTI_SOURCE_REVIEW', 'UNCLASSIFIED']:
        sub = inventory[inventory.dataset == category]
        coverage.append({'dataset':category, 'raw_files':int((sub.source=='raw_data').sum()),
                         'processed_files':int((sub.source=='processed_data').sum()),
                         'total_files':len(sub),
                         'content_or_metadata_checked':int(sub.check_level.isin(['SAMPLED_CONTENT','METADATA_AND_SAMPLED_CONTENT','METADATA_ONLY']).sum()),
                         'reader_errors':int(sub.result.isin(['READ_ERROR','READER_UNAVAILABLE']).sum()),
                         'status': 'NOT_FOUND' if sub.empty else ('REQUIRES_REVIEW' if any(sub.result!='OK') else 'DISCOVERED_NOT_FULLY_VALIDATED')})
    pd.DataFrame(coverage).to_csv(out/'03_dataset_coverage.csv', index=False)
    patterns = inventory[inventory.relative_path.str.contains(PATTERN, na=False)].copy()
    patterns.to_csv(out/'flood_pattern_inventory.csv', index=False)
    keys = inventory[inventory.relative_path.str.contains('adm2|adm3|annual_flood|flood_indicator', case=False, regex=True, na=False)].copy()
    keys.to_csv(out/'04_adm2_adm3_candidate_files.csv', index=False)
    for col in ['check_method','warning']:
        if col not in inventory.columns: inventory[col] = ''
    inventory[['relative_path','dataset','result','check_level','check_method','warning']].to_csv(out/'05_quality_check_summary.csv', index=False)
    checklist = [
        ('file_discovery','Compare expected, audited, excluded and missing lists','FILE_LIST_CHECK'),
        ('source_coverage','Confirm all 11 source datasets and lineage','MANUAL_REVIEW'),
        ('netcdf_ET','Verify ET0 variable, unit, monthly/year continuity, spatial extent and validity','NOT_FULLY_CHECKED'),
        ('adm2_adm3','Validate actual region-year coverage and unique administrative codes','NOT_YET_CHECKED'),
        ('lake_altimetry','Verify satellite mission biases, lake connectivity and lake-to-ADM causal linkage','NOT_YET_CHECKED'),
        ('panel_join','Verify spatial joins, units, granularity and missing observations','NOT_YET_CHECKED'),
        ('predictive_leakage','Recompute history-dependent features inside each training window','NOT_YET_CHECKED'),
    ]
    pd.DataFrame(checklist,columns=['check','requirement','status']).to_csv(out/'modeling_readiness_checklist.csv',index=False)
    report = [
        '# Comprehensive Multivariate Data Audit', '',
        f'- Run (UTC): {datetime.now(timezone.utc).isoformat()}',
        f'- Project root: `{root}`',
        '- Inputs: ' + ', '.join(f'`{source}`' for source,_ in sources),
        f'- Expected data files: **{len(expected)}**',
        f'- Inventory entries: **{len(inventory)}**',
        f'- Missing inventory entries: **{len(set(r["relative_path"] for r in missing if r.get("reason")=="NOT_AUDITED"))}**',
        f'- Content/metadata read failures: **{int(inventory.result.isin(["READ_ERROR","READER_UNAVAILABLE"]).sum())}**',
        f'- Unclassified: **{int((inventory.dataset=="UNCLASSIFIED").sum())}**',
        '', '## Important interpretation',
        'Inventory coverage is not equivalent to comprehensive quality validation.',
        'CSV/Excel checks are sampled, NetCDF checks are metadata + bounded samples, raster checks are metadata-only.',
        'Missing year folders, temporal gaps, spatial alignment, geographic validity, and source lineage require follow-up.',
        'The audit intentionally excludes its own output directory to prevent recursive audit inflation.',
        '', '## Dataset counts', '',
        '| Dataset | Raw | Processed | Issues |', '|---|---:|---:|---:|'
    ]
    for row in coverage:
        report.append(f'| {row["dataset"]} | {row["raw_files"]} | {row["processed_files"]} | {row["reader_errors"]} |')
    report.extend(['', '## Before panel modelling',
                   '1. Review `02_missing_or_skipped_files.csv` and `excluded_files.csv`.',
                   '2. Check `dataset_classification_validation.csv`, `classification_review_required.csv`, and all `UNCLASSIFIED` and `MULTI_SOURCE_REVIEW` paths.',
                   '3. Validate ADM2/ADM3 region-year indicators, and establish temporal availability of each driver.',
                   '4. Rebuild lagged pattern features separately within every backtest split to prevent leakage.',
                   '5. Distinguish reference ET (ET0) from actual ET.', ''])
    (out/'audit_report.md').write_text('\n'.join(report),encoding='utf-8')
    meta = {'utc_run':datetime.now(timezone.utc).isoformat(), 'project_root':str(root),
            'inputs':[source for source,_ in sources], 'files_discovered':len(expected),
            'files_audited':len(inventory), 'classification_version':'parent-first-v4-lake-parser', 'max_rows_per_table':args.max_rows,
            'nc_max_sample_cells':args.nc_sample_cells,
            'audit_note':'No input files were modified; audit is not full validation.'}
    (out/'audit_run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'Completed. Outputs: {out}', flush=True)


if __name__ == '__main__':
    main()
