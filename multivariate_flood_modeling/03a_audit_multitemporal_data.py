#!/usr/bin/env python3
"""JBG060: non-destructive temporal inventory across all raw-data families.

Run from repository root:
 python scripts/multivariate_flood_modeling/03a_audit_multitemporal_data.py

Uses file/path evidence and available time coordinates. Reports unknowns explicitly;
does not claim daily suitability from filenames or event timestamps.
"""
from __future__ import annotations
import argparse
import calendar
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

FAMILIES = ["flood_masks", "rainfall_and_runoff", "evapotranspiration",
            "water_levels_lakes", "darthmouth_flood_observatory",
            "administrative_boundaries", "worldpop", "farmland",
            "health_facilities", "GDP", "IPC"]
TABLE_EXT = {'.csv', '.tsv', '.parquet', '.xlsx', '.xls'}
DATE_NAMES = ('valid_time', 'time', 'date', 'datetime', 'timestamp',
              'observation_date', 'event_date', 'start_date', 'end_date')
YEAR_NAMES = ('year', 'calendar_year', 'observation_year')
YEAR_TOKEN = re.compile(r'(?<!\d)(?:19\d\d|20\d\d)(?!\d)')


def freq_label(dts):
    """Observed distinct-time spacing; does not assert regularly sampled coverage."""
    ix = pd.DatetimeIndex(dts).dropna().unique().sort_values()
    if len(ix) < 2:
        return 'UNDETERMINED', None, 0, 0
    ns = ix.asi8
    diffs = np.diff(ns) / 1e9 / 86400
    med = float(np.median(diffs))
    if med <= 1 / 24 + 0.002:
        label = 'HOURLY_OR_FINER_OBSERVED'
    elif med <= 1.5:
        label = 'DAILY_OBSERVED'
    elif med <= 10:
        label = 'WEEKLY_OR_IRREGULAR_OBSERVED'
    elif med <= 45:
        label = 'MONTHLY_OR_IRREGULAR_OBSERVED'
    elif med <= 110:
        label = 'QUARTERLY_OR_IRREGULAR_OBSERVED'
    elif med <= 400:
        label = 'ANNUAL_OR_IRREGULAR_OBSERVED'
    else:
        label = 'SPARSE_OR_IRREGULAR_OBSERVED'
    return label, round(med, 3), int((diffs <= 0).sum()), int(len(ix))


def parse_time(vals):
    try:
        arr = np.asarray(vals).ravel()
        if np.issubdtype(arr.dtype, np.number):
            return pd.DatetimeIndex([])
        if len(arr) > 100000:
            # Representative bounded sample; clearly flagged by caller
            arr = arr[np.linspace(0, len(arr)-1, 100000, dtype=int)]
        return pd.DatetimeIndex(pd.to_datetime(arr, errors='coerce', utc=True)).tz_localize(None)
    except Exception:
        return pd.DatetimeIndex([])


def inspect_netcdf(p):
    import xarray as xr
    with xr.open_dataset(p, decode_times=True) as ds:
        axis = next((c for c in DATE_NAMES if c in ds.coords and ds[c].ndim == 1), None)
        if axis is None:
            return dict(evidence='NETCDF_METADATA', frequency='UNKNOWN_NO_TIME_COORD',
                        time_count=0, distinct_times=0, coverage_start='', coverage_end='',
                        median_interval_days=None, note=f'dims={dict(ds.sizes)}; vars={list(ds.data_vars)[:12]}')
        coord = ds[axis]
        ix = parse_time(coord.values)
        label, median, _, unique = freq_label(ix)
        return dict(evidence='NETCDF_TIME_COORDINATE', frequency=label, time_count=int(coord.size),
                    distinct_times=unique or int(ix.nunique()), coverage_start=str(ix.min()) if len(ix) else '',
                    coverage_end=str(ix.max()) if len(ix) else '', median_interval_days=median,
                    note=f'axis={axis}; variables={list(ds.data_vars)[:12]}; full_cell_validity_NOT_checked')


def inspect_table(p, max_rows):
    if p.suffix.lower() == '.csv':
        df = pd.read_csv(p, nrows=max_rows, low_memory=False)
    elif p.suffix.lower() == '.tsv':
        df = pd.read_csv(p, sep='\t', nrows=max_rows, low_memory=False)
    elif p.suffix.lower() == '.parquet':
        df = pd.read_parquet(p).head(max_rows)
    else:
        df = pd.read_excel(p, nrows=max_rows)
    cols = {str(c).strip().lower():c for c in df.columns}
    for key in DATE_NAMES:
        if key in cols:
            ix = parse_time(df[cols[key]].values)
            if len(ix.dropna()):
                label, median, _, n = freq_label(ix)
                return dict(evidence='TABLE_DATE_VALUES_SAMPLE', frequency=label,
                            time_count=len(df), distinct_times=n, coverage_start=str(ix.min()),
                            coverage_end=str(ix.max()), median_interval_days=median,
                            note=f'date_col={cols[key]}; first_{max_rows}_rows_only; event_dates_do_NOT_establish_continuous_sampling')
    for key in YEAR_NAMES:
        if key in cols:
            yrs = pd.to_numeric(df[cols[key]], errors='coerce').dropna().astype(int)
            yrs = yrs[yrs.between(1900, 2100)]
            return dict(evidence='TABLE_YEAR_VALUES_SAMPLE', frequency='ANNUAL_LABELS_ONLY',
                        time_count=len(df), distinct_times=int(yrs.nunique()),
                        coverage_start=str(yrs.min()) if len(yrs) else '',
                        coverage_end=str(yrs.max()) if len(yrs) else '', median_interval_days=None,
                        note=f'year_col={cols[key]}; first_{max_rows}_rows_only; no_intra_year_resolution')
    return dict(evidence='TABLE_SCHEMA_SAMPLE', frequency='UNKNOWN_NO_DATE_COLUMN', time_count=len(df),
                distinct_times=0, coverage_start='', coverage_end='', median_interval_days=None,
                note=f'columns={list(df.columns)[:20]}; first_{max_rows}_rows_only')


def inspect_file(p, family, root, max_rows):
    rel = p.relative_to(root).as_posix()
    base = dict(dataset=family, file=rel, ext=p.suffix.lower(), bytes=p.stat().st_size,
                filename_years=','.join(sorted(set(YEAR_TOKEN.findall(rel)))))
    try:
        if p.suffix.lower() in {'.nc','.nc4'}:
            info = inspect_netcdf(p)
        elif p.suffix.lower() in TABLE_EXT:
            info = inspect_table(p, max_rows)
        else:
            info = dict(evidence='FILENAME_ONLY', frequency='UNKNOWN_NOT_CONTENT_CHECKED',
                        time_count=None, distinct_times=None, coverage_start='', coverage_end='',
                        median_interval_days=None,
                        note='Filename years are not proof of observation frequency. Inspect source documentation/time bands.')
        return dict(**base, **info, status='OK')
    except Exception as exc:
        return dict(**base, evidence='READ_ERROR', frequency='UNKNOWN', time_count=None,
                    distinct_times=None, coverage_start='', coverage_end='', median_interval_days=None,
                    note=f'{type(exc).__name__}: {exc}', status='ERROR')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--max-table-rows', type=int, default=10000)
    ap.add_argument('--max-files-per-family', type=int, default=0,
                    help='0 = all files (large ET directories may take time); sample cap for quick scan')
    ap.add_argument('--include-processed', action='store_true', help='Include processed_data files in addition to raw_data')
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    if not (root/'raw_data').is_dir():
        ap.error('Not a repository root: raw_data missing')
    out = root/'processed_data/multivariate_flood_modeling/03a_multitemporal_data_audit'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in FAMILIES:
        paths = sorted(p for p in (root/'raw_data'/family).rglob('*') if p.is_file()) if (root/'raw_data'/family).exists() else []
        count = len(paths)
        if args.max_files_per_family:
            # Spread across sorted paths/years rather than first N only.
            k = min(count, args.max_files_per_family)
            paths = [paths[i] for i in np.linspace(0, count-1, k, dtype=int)] if k else []
        print(f'{family}: inspecting {len(paths)}/{count} files', flush=True)
        for p in paths:
            rows.append(inspect_file(p, family, root, args.max_table_rows))
    data = pd.DataFrame(rows)
    data.to_csv(out/'file_temporal_inventory.csv', index=False)
    summary = []
    for family in FAMILIES:
        sub = data[data.dataset == family] if not data.empty else pd.DataFrame()
        fdir = root/'raw_data'/family
        total = sum(x.is_file() for x in fdir.rglob('*')) if fdir.is_dir() else 0
        labels = sub.frequency.value_counts().to_dict() if len(sub) else {}
        summary.append(dict(dataset=family, source_files=total, inspected_files=len(sub),
                            read_errors=int((sub.status=='ERROR').sum()) if len(sub) else 0,
                            observed_frequencies=json.dumps(labels),
                            evidence_status='PARTIAL_SAMPLE' if len(sub)<total else ('NO_FILES' if total==0 else 'INVENTORY_ONLY_NOT_MODEL_READY'),
                            note='Observed frequency is not proof of regularity, spatial compatibility, or valid flood observations'))
    pd.DataFrame(summary).to_csv(out/'dataset_temporal_summary.csv',index=False)
    (out/'README.md').write_text(
        '# JBG060 Temporal Audit\n\nThis is a read-only temporal inventory, not a certification of daily/weekly/model readiness.\n\n'
        'Review `file_temporal_inventory.csv` and `dataset_temporal_summary.csv`.\n\n'
        'Important: archived rasters, GeoTIFF, TXT, GeoJSON and other formats remain UNKNOWN until their '
        'internal dates/bands are checked using source-specific readers. Date fields in event catalogues '
        'are event timestamps, not necessarily regularly sampled time series.\n\n'
        'Checks still required: unique dates and calendar gaps across files; valid-cell/day coverage; '
        'flood-mask observation cadence; hourly ERA5 accumulation semantics; ADM geographic alignment; '
        'period-level leakage/forecast-origin constraints.\n', encoding='utf-8')
    print(f'Output: {out}')
    print(pd.DataFrame(summary)[['dataset','source_files','inspected_files','read_errors']].to_string(index=False))


if __name__ == '__main__':
    main()
