#!/usr/bin/env python3
"""JBG060 | 03b: Full-date audit of flood-event parquet files (read-only).

Run from the DBL-Group-22 repository root:
  python scripts/multivariate_flood_modeling/03b_validate_flood_temporal_coverage.py

IMPORTANT: Event dates are NOT verified observation dates. Absence of an event
is not evidence of a flood-free observation. This script does not create daily
flood-days targets or claim that daily/weekly prediction is already possible.

Dependencies: pandas, pyarrow, numpy.
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FILENAME = re.compile(r'flood_events_(h\d+v\d+)_(20\d{2})\.parquet$', re.I)
DATE_CANDIDATES = ('date', 'observation_date', 'event_date', 'datetime', 'timestamp', 'time')


def inspect(path: Path, root: Path, batch_size: int):
    match = FILENAME.fullmatch(path.name)
    tile = match.group(1).lower() if match else 'UNKNOWN_TILE'
    filename_year = int(match.group(2)) if match else None
    pf = pq.ParquetFile(path)
    cols = pf.schema_arrow.names
    names = {c.lower(): c for c in cols}
    date_col = next((names[x] for x in DATE_CANDIDATES if x in names), None)
    base = {'file': path.relative_to(root).as_posix(), 'tile': tile,
            'filename_year': filename_year, 'rows': pf.metadata.num_rows,
            'date_field': date_col or '', 'columns': json.dumps(cols),
            'status': 'OK', 'error': ''}
    if not date_col:
        return dict(base, status='NO_DATE_FIELD'), [], []
    counts = Counter()
    null_dates = 0
    time_of_day = False
    for batch in pf.iter_batches(batch_size=batch_size, columns=[date_col]):
        ser = pd.Series(batch.column(0).to_pandas())
        dt = pd.to_datetime(ser, errors='coerce', utc=True)
        null_dates += int(dt.isna().sum())
        good = dt.dropna()
        if len(good):
            time_of_day |= bool(((good.dt.hour != 0) | (good.dt.minute != 0) | (good.dt.second != 0)).any())
            counts.update(good.dt.strftime('%Y-%m-%d').value_counts().to_dict())
    dates = pd.DatetimeIndex(pd.to_datetime(list(counts.keys()), utc=True)).tz_convert(None).normalize().sort_values()
    diffs = np.diff(dates.asi8) / 86400e9 if len(dates)>1 else np.array([])
    expected = pd.date_range(f'{filename_year}-01-01', f'{filename_year}-12-31', freq='D') if filename_year else pd.DatetimeIndex([])
    in_year = dates.intersection(expected)
    missing_event_days = expected.difference(in_year)
    outside = dates.difference(expected) if filename_year else dates[:0]
    row = dict(base,
        dated_rows=sum(counts.values()), invalid_date_rows=null_dates,
        unique_event_dates=len(dates), event_dates_in_filename_year=len(in_year),
        days_in_filename_year=len(expected), event_day_fraction=(round(len(in_year)/len(expected),6) if len(expected) else None),
        days_without_recorded_events=len(missing_event_days),
        unique_dates_outside_filename_year=len(outside),
        first_event_date=dates[0].date().isoformat() if len(dates) else '',
        last_event_date=dates[-1].date().isoformat() if len(dates) else '',
        median_gap_between_event_dates=(round(float(np.median(diffs)),3) if len(diffs) else None),
        longest_gap_between_event_dates=(round(float(diffs.max()),3) if len(diffs) else None),
        duplicate_rows_on_same_event_date=sum(v-1 for v in counts.values()),
        non_midnight_timestamps_observed=time_of_day,
        note='Days without records are UNKNOWN OBSERVATION STATUS, not zero-flood days.')
    monthly = []
    for month in range(1,13):
        idx = in_year[in_year.month == month]
        monthly.append({'file':base['file'], 'tile':tile, 'year':filename_year, 'month':month,
                        'calendar_days':calendar.monthrange(filename_year,month)[1] if filename_year else None,
                        'distinct_event_dates':len(idx), 'status':'EVENT_DATE_COUNT_ONLY'})
    years = [{'file':base['file'], 'tile':tile, 'filename_year':filename_year,
              'event_date':str(d.date()), 'event_rows':counts[d.strftime('%Y-%m-%d')]}
             for d in dates]
    return row, monthly, years


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--batch-size', type=int, default=250000)
    args = ap.parse_args()
    root = args.root.resolve()
    inp = root/'raw_data'/'flood_masks'
    out = root/'processed_data'/'multivariate_flood_modeling'/'03b_flood_temporal_coverage'
    if not inp.is_dir(): ap.error(f'Missing {inp}')
    paths = sorted(inp.rglob('*.parquet'))
    if not paths: ap.error(f'No Parquet flood masks found under {inp}')
    out.mkdir(parents=True,exist_ok=True)
    report, monthly, dates = [], [], []
    for i,path in enumerate(paths,1):
        print(f'[{i}/{len(paths)}] {path.relative_to(root)}', flush=True)
        try:
            r,m,d = inspect(path,root,args.batch_size)
            report.append(r);monthly.extend(m);dates.extend(d)
        except Exception as exc:
            report.append({'file':path.relative_to(root).as_posix(), 'status':'ERROR',
                           'error':f'{type(exc).__name__}: {exc}'})
    files = pd.DataFrame(report)
    files.to_csv(out/'01_full_file_date_audit.csv',index=False)
    pd.DataFrame(monthly).to_csv(out/'02_monthly_event_date_coverage.csv',index=False)
    daily = pd.DataFrame(dates, columns=['file','tile','filename_year','event_date','event_rows'])
    daily.to_csv(out/'03_recorded_event_dates.csv',index=False)
    if len(daily):
        union = (daily.groupby(['filename_year','event_date'],dropna=False)
                 .agg(tiles_with_event=('tile','nunique'),event_rows=('event_rows','sum'))
                 .reset_index())
        union.to_csv(out/'04_union_event_dates_by_year.csv',index=False)
    summary = []
    if 'filename_year' in files.columns:
        for yr,grp in files.dropna(subset=['filename_year']).groupby('filename_year'):
            yr = int(yr)
            d = daily[daily.filename_year == yr] if len(daily) else daily
            in_year = d[d.event_date.str.startswith(str(yr))] if len(d) else d
            unique_days = in_year.event_date.nunique() if len(in_year) else 0
            n = 366 if calendar.isleap(yr) else 365
            summary.append({'year':yr, 'parquet_files':len(grp),
                            'tiles':grp.tile.nunique() if 'tile' in grp else 0,
                            'distinct_dates_with_any_recorded_event':unique_days,
                            'days_without_recorded_event':n-unique_days,
                            'date_event_fraction':round(unique_days/n,6),
                            'file_errors':int((grp.status!='OK').sum()),
                            'observational_denominator_verified':False})
    pd.DataFrame(summary).to_csv(out/'05_yearly_event_date_summary.csv',index=False)
    read_errors = sum(r.get('status')!='OK' for r in report)
    text = [
        '# JBG060 — 03b Flood Event Temporal Coverage', '',
        f'- Parquet files examined: {len(paths)}',
        f'- Files with errors / missing date fields: {read_errors}',
        '- Date columns read across ALL Parquet row groups, not merely the first 10,000 rows.',
        '- Dates with recorded events are NOT a complete daily observation calendar.',
        '- A missing event date must NOT be coded as a non-flood day.',
        '- Counts by tile cannot be simply summed into region flood-days without spatial and observational coverage handling.',
        '', '## Decision on time-scale modeling',
        'Daily/weekly/monthly/quarterly flood targets: **NOT YET VERIFIED**.',
        'Event dates can be summarized at these periods, but that is NOT equivalent to measuring flood-days or flood-free days.',
        'Next: inspect original flood-mask product metadata and processing scripts for observation opportunity, pixel-level valid-mask, and exact spatial/time units.',
        'Confirm what a row represents (pixel, component, event), tile coverage, QA flags and temporal sampling before aggregation to ADM2/ADM3.',
        'If only event detections exist, model event detections with explicit effort/coverage limitations instead of pretending to observe all days.',
        '', '## Outputs',
        '- 01_full_file_date_audit.csv: full per-file event date check',
        '- 02_monthly_event_date_coverage.csv: monthly distinct EVENT dates (not observed coverage)',
        '- 03_recorded_event_dates.csv: event dates per file and row counts',
        '- 04_union_event_dates_by_year.csv: union across tiles; NOT region counts',
        '- 05_yearly_event_date_summary.csv: union-year summaries',
    ]
    (out/'README.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print('\nOUTPUT:',out, '\nFile read issues:', read_errors)
    if summary:
        print(pd.DataFrame(summary).tail(5).to_string(index=False))

if __name__=='__main__':
    main()
