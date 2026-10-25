#!/usr/bin/env python3
"""JBG060 | 03c Flood event structure and observation-opportunity audit.

READ ONLY: source Parquet files are never modified. Inspect ALL rows in batches.
Recurring and unusual rows are deduplicated by (UTC calendar date, tile, lat, lon).
Exact numeric coordinates are used; no spatial tolerance is silently assumed.

Produces overlap/duplication statistics, cloud-fraction diagnostics, monthly
unique event-key counts and an explicit observation-denominator decision.
An event-only dataset cannot establish valid flood-free observation days.

Run from DBL-Group-22 repository root:
  python scripts/multivariate_flood_modeling/03c_validate_flood_observation_and_event_structure.py
  python scripts/multivariate_flood_modeling/03c_validate_flood_observation_and_event_structure.py --year 2025

Requires pandas, numpy, pyarrow; uses standard-library sqlite3 for disk-backed
exact-key deduplication, one year at a time. Large years may take some time.
"""
from __future__ import annotations
import argparse
import calendar
import json
import re
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NAME = re.compile(r'flood_events_(h\d+v\d+)_(20\d{2})\.parquet$', re.I)
REQUIRED = {'date', 'lat', 'lon', 'tile', 'cloud_frac'}


def new_db(path):
    con = sqlite3.connect(str(path))
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('PRAGMA synchronous=OFF')
    con.execute('PRAGMA temp_store=MEMORY')
    con.execute('''CREATE TABLE event_keys (
       event_date TEXT NOT NULL, tile TEXT NOT NULL,
       lat REAL NOT NULL, lon REAL NOT NULL,
       kind_mask INTEGER NOT NULL,
       PRIMARY KEY(event_date,tile,lat,lon)) WITHOUT ROWID''')
    return con


def inspect_file(path, kind, tile_expected, year, con, batch_size):
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    if not REQUIRED.issubset(names):
        raise ValueError(f'{path}: missing required columns {sorted(REQUIRED-names)}')
    rows, malformed, wrong_year, bad_cloud, outside_cloud = 0, 0, 0, 0, 0
    cloud_count = 0
    cloud_sum = 0.0
    cloud_min = np.inf
    cloud_max = -np.inf
    cloud_hist = np.zeros(5, dtype=np.int64)
    tile_mismatch = 0
    mask = 1 if kind == 'recurring' else 2
    for batch in pf.iter_batches(batch_size=batch_size, columns=['date','lat','lon','tile','cloud_frac']):
        frame = batch.to_pandas()
        rows += len(frame)
        date = pd.to_datetime(frame['date'], errors='coerce', utc=True)
        lat = pd.to_numeric(frame['lat'], errors='coerce')
        lon = pd.to_numeric(frame['lon'], errors='coerce')
        tiles = frame['tile'].astype('string').str.strip().str.lower()
        cloud = pd.to_numeric(frame['cloud_frac'], errors='coerce')
        valid = date.notna() & np.isfinite(lat) & np.isfinite(lon) & lat.between(-90,90) & lon.between(-180,180) & tiles.notna() & tiles.ne('')
        malformed += int((~valid).sum())
        wrong_year += int((valid & date.dt.year.ne(year)).sum())
        tile_mismatch += int((valid & tiles.ne(tile_expected)).sum())
        good_cloud = np.isfinite(cloud)
        bad_cloud += int((~good_cloud).sum())
        outside_cloud += int((good_cloud & (~cloud.between(0,1))).sum())
        cv = cloud[good_cloud & cloud.between(0,1)].to_numpy(dtype=float)
        if len(cv):
            cloud_count += len(cv)
            cloud_sum += float(cv.sum())
            cloud_min = min(cloud_min, float(cv.min()))
            cloud_max = max(cloud_max, float(cv.max()))
            cloud_hist += np.histogram(cv, bins=[0,.2,.4,.6,.8,1.0000000001])[0]
        ok = valid & date.dt.year.eq(year) & tiles.eq(tile_expected)
        keys = pd.DataFrame({
            'event_date':date[ok].dt.strftime('%Y-%m-%d'),
            'tile':tiles[ok], 'lat':lat[ok].astype(float), 'lon':lon[ok].astype(float)
        }).drop_duplicates()
        # For equal keys in the two event types, accumulate a bit mask, not the row counts.
        values = [(r.event_date, r.tile, float(r.lat), float(r.lon), mask)
                  for r in keys.itertuples(index=False)]
        con.executemany('''INSERT INTO event_keys(event_date,tile,lat,lon,kind_mask)
           VALUES(?,?,?,?,?) ON CONFLICT(event_date,tile,lat,lon)
           DO UPDATE SET kind_mask=(event_keys.kind_mask | excluded.kind_mask)''', values)
    return dict(file=str(path), kind=kind, tile=tile_expected, year=year,
                source_rows=int(rows), invalid_key_rows=malformed,
                rows_outside_filename_year=wrong_year, mismatched_tile_rows=tile_mismatch,
                missing_cloud_rows=bad_cloud, cloud_outside_0_1_rows=outside_cloud,
                valid_cloud_rows=cloud_count,
                mean_cloud_frac=(cloud_sum/cloud_count if cloud_count else None),
                min_cloud_frac=(cloud_min if cloud_count else None),
                max_cloud_frac=(cloud_max if cloud_count else None),
                **{f'cloud_frac_bin_{i}':int(v) for i,v in enumerate(cloud_hist)})


def year_summary(con, year):
    aggregates = con.execute('''SELECT tile, count(*),
        sum(kind_mask=1), sum(kind_mask=2), sum(kind_mask=3),
        count(DISTINCT event_date)
        FROM event_keys GROUP BY tile ORDER BY tile''').fetchall()
    summaries = []
    for tile, total, recurring, unusual, overlap, dates in aggregates:
        summaries.append(dict(year=year,tile=tile,unique_event_keys=total,
                              recurring_only_keys=recurring, unusual_only_keys=unusual,
                              overlapping_both_types_keys=overlap,
                              distinct_dates_with_event=dates,
                              overlap_share_of_unique_keys=overlap/total if total else None))
    by_month = [dict(year=year,month=int(month),tile=tile,unique_event_keys=cnt,
                     distinct_dates_with_event=days)
                for month,tile,cnt,days in con.execute('''SELECT substr(event_date,6,2),tile,count(*),
                   count(DISTINCT event_date) FROM event_keys
                   GROUP BY substr(event_date,6,2),tile ORDER BY 1,2''')]
    union_dates = con.execute('SELECT count(DISTINCT event_date) FROM event_keys').fetchone()[0]
    ncalendar = 366 if calendar.isleap(year) else 365
    return summaries,by_month,dict(year=year,calendar_days=ncalendar,
                                   dates_with_any_recorded_event=union_dates,
                                   dates_without_recorded_event=ncalendar-union_dates,
                                   verified_valid_observation_days=None,
                                   non_event_days_proven_flood_free=False,
                                   target_daily_flood_days_ready=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--year',type=int,default=None,help='Optional smoke test for one year')
    p.add_argument('--batch-size',type=int,default=100000)
    args=p.parse_args()
    if args.year is not None and not 2000 <= args.year <= 2025: p.error('year must be 2000..2025')
    if args.batch_size<=0: p.error('batch-size must be positive')
    root=args.root.expanduser().resolve()
    inp=root/'raw_data/flood_masks'
    out=root/'processed_data/multivariate_flood_modeling/03c_flood_observation_structure'
    paths=[]
    for path in sorted(inp.rglob('*.parquet')):
        match=NAME.fullmatch(path.name)
        kind='recurring' if path.parent.name=='compact_recurring' else 'unusual' if path.parent.name=='compact_unusual' else None
        if match and kind:
            year=int(match.group(2))
            if args.year is None or year==args.year:
                paths.append((path,kind,match.group(1).lower(),year))
    if not paths: p.error(f'No recognized event files in {inp}')
    out.mkdir(parents=True,exist_ok=True)
    files=[]; overlaps=[]; monthly=[]; annual=[]; failures=[]
    years=sorted({r[3] for r in paths})
    with tempfile.TemporaryDirectory(prefix='jbg060_03c_') as temp:
        for year in years:
            con=new_db(Path(temp)/f'event_{year}.sqlite')
            year_paths=[r for r in paths if r[3]==year]
            try:
                for path,kind,tile,_ in year_paths:
                    print(f'{year} {kind} {tile}: {path.name}',flush=True)
                    # A failed year's outputs are not accepted as complete.
                    record=inspect_file(path,kind,tile,year,con,args.batch_size)
                    record['file']=path.relative_to(root).as_posix()
                    files.append(record)
                con.commit()
                o,m,a=year_summary(con,year)
                overlaps.extend(o);monthly.extend(m);annual.append(a)
            except Exception as exc:
                failures.append({'year':year,'error':f'{type(exc).__name__}: {exc}'})
                print(f'ERROR {year}: {exc}',flush=True)
            finally:
                con.close()
    pd.DataFrame(files).to_csv(out/'01_file_schema_cloud_quality.csv',index=False)
    pd.DataFrame(overlaps).to_csv(out/'02_recurring_unusual_exact_key_overlap.csv',index=False)
    pd.DataFrame(monthly).to_csv(out/'03_monthly_unique_event_keys.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'04_annual_event_dates_and_observation_limit.csv',index=False)
    pd.DataFrame(failures,columns=['year','error']).to_csv(out/'05_processing_errors.csv',index=False)
    decision={
      'complete_full_dataset_scan':args.year is None and not failures and len(years)==26,
      'years_examined':years,
      'parquet_files_examined':len(files),
      'failed_years':failures,
      'event_key_definition':'UTC calendar date + tile + exact lat + exact lon',
      'file_cloud_frac_role':'attribute of event records; NOT independently verified observation opportunity',
      'unobserved_or_non_event_days':'UNKNOWN; cannot distinguish cloud obscuration, absent coverage and true no-flood',
      'daily_weekly_monthly_flood_days_targets':'NOT VERIFIED',
      'scientific_next_step':'Locate original pixel-level flood and valid-observation masks or product QA documentation; check geographic key consistency and compare existing annual ADM flood indicator methodology.'
    }
    (out/'audit_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    (out/'README.md').write_text('''# 03c — Flood event structure and observation audit

Read-only inspection of all rows of each recognized flood event Parquet.
- `01_file_schema_cloud_quality.csv`: event record counts, invalid keys, cloud fractions.
- `02_recurring_unusual_exact_key_overlap.csv`: exact space-time overlap of categories.
- `03_monthly_unique_event_keys.csv`: recorded event keys, **not** flood-days.
- `04_annual_event_dates_and_observation_limit.csv`: event dates and explicit unknown observation denominators.
- `05_processing_errors.csv`: any failures by year.
- `audit_decision.json`: eligibility limitations.

**Important:** Exact float coordinate matching assumes stable grid coordinates and does not prove geometrically equal locations if inputs use different coordinate precision. The audit does not generate negative/non-flood observations or validated ADM2/ADM3 targets. Cloud fractions on event rows cannot establish all valid observation opportunities. The 2025 ERA5 missing last day issue is separate from these flood event files.
''',encoding='utf-8')
    print('Output:',out)
    print('Read errors:',len(failures))
    if overlaps: print(pd.DataFrame(overlaps).tail(6).to_string(index=False))

if __name__=='__main__': main()
