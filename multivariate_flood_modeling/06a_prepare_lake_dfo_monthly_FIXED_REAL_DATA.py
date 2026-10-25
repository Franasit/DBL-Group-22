#!/usr/bin/env python3
"""JBG060: lake levels and DFO station discharge -> monthly values and ADM panels.

Safe design:
- Data sources are measured AT stations/lakes, not automatically at ADM units.
- A reviewed mapping CSV (source_type,source_id,level,adm_id,relationship)
  is mandatory before producing ADM-level features.
- Never assign a nearest station as though hydrologically upstream.
- No missing measurement is replaced with zero.
- Outputs contain reporting-period aggregates, which MUST be lagged for forecasting.

Install: pip install pandas numpy xarray netCDF4 openpyxl
Run: python scripts/multivariate_flood_modeling/06a_prepare_lake_dfo_monthly.py
Optional mapping: --mapping scripts/multivariate_flood_modeling/hydrology_adm_mapping.csv
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def timestamp(series):
    """Strictly parse conventional dates; do not mistake numbers for Unix nanoseconds."""
    s = pd.Series(series)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors='coerce')
    txt = s.astype('string').str.strip()
    a = pd.to_datetime(txt, errors='coerce', format='mixed', dayfirst=False)
    # YYYYMMDD / YYYY-MM-DD / ISO only; numeric serial dates are unsupported without explicit origin
    acceptable = txt.str.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$|^\d{8}$', na=False)
    a = a.where(acceptable)
    return a.dt.tz_localize(None) if a.dt.tz is not None else a


def get_column(df, names, excluded=()):
    names = [v.lower() for v in names]
    for col in df.columns:
        key = re.sub('[^a-z0-9]', '',str(col).lower())
        if key in [re.sub('[^a-z0-9]', '',n) for n in names] and col not in excluded:
            return col
    return None


def time_series_frame(dates, values, source_type, source_id, source_file):
    t = timestamp(dates)
    v = pd.to_numeric(pd.Series(values), errors='coerce')
    out = pd.DataFrame({'date':t.values, 'value':v.values})
    out = out.dropna(subset=['date','value']).copy()
    out['source_type'],out['source_id'],out['source_file'] = source_type,str(source_id),str(source_file)
    return out


def load_dfo(path):
    df = pd.read_csv(path, low_memory=False)
    dc = get_column(df, ['date', 'datetime', 'observation_date'])
    vc = get_column(df, ['discharge', 'discharge_m3_s', 'discharge (m3/s)',
                         'flow', 'flow_m3_s', 'flowrate', 'Q', 'value'])
    if dc is None or vc is None:
        return None, f'Cannot identify date/discharge columns. Available: {list(df.columns)}'
    station = path.stem.split('_')[0]
    result = time_series_frame(df[dc], df[vc], 'dfo', station, path.name)
    result = result[result.value.ge(0)].copy()  # negative discharge is invalid here
    if result.empty:
        return None, 'No valid nonnegative discharge observations'
    return result, ''

def load_albert(path):
    with xr.open_dataset(path) as ds:
        tc = next((c for c in ['datetime','time','date','valid_time'] if c in ds.variables),None)
        vc = next((c for c in ['water_level','level','waterlevel'] if c in ds.variables),None)
        if not tc or not vc:
            return None, f'Missing water level or time variable: {list(ds.variables)}'
        dates=np.asarray(ds[tc].values).ravel()
        values=np.asarray(ds[vc].values).ravel()
        if len(dates)!=len(values):
            return None, f'Time / water-level dimensions do not match: {len(dates)}, {len(values)}'
        if np.issubdtype(dates.dtype, np.datetime64):
            result=time_series_frame(pd.to_datetime(dates),values,'lake','Albert',path.name)
        else:
            # Only parse correctly decoded datetime values; do not guess numeric epochs
            result=time_series_frame(dates,values,'lake','Albert',path.name)
        if result.empty: return None,'Albert has no successfully parsed time-value pairs'
        return result,''


def load_lake_txt(path):
    """Parse USDA/LEGOS satellite-lake height product documented in uploaded TXT.

    Column 3: YYYYMMDD date; column 15: EGM2008 height above MSL in meters.
    Sentinel 9999.99 = missing. Column 6 instead provides reference-relative height;
    we intentionally use column 15, preserving a common declared datum for these TXT files.
    """
    lake = path.stem.removeprefix('water_level_')
    dates = []
    values = []
    candidate_rows = 0
    invalid_rows = 0
    for raw in path.read_text(encoding='utf-8-sig', errors='replace').splitlines():
        fields = raw.strip().split()
        # exactly 16 columns for observations, mission identifier first
        if len(fields) != 16 or not re.fullmatch(r'\d{8}', fields[2]):
            continue
        candidate_rows += 1
        try:
            d = pd.to_datetime(fields[2], format='%Y%m%d', errors='raise')
            level = float(fields[14])
            if not np.isfinite(level) or np.isclose(level, 9999.99) or level >= 9990:
                invalid_rows += 1
                continue
            dates.append(d)
            values.append(level)
        except (ValueError, OverflowError):
            invalid_rows += 1
    if not dates:
        return None, f'No valid date + EGM2008 height pairs; candidate_rows={candidate_rows}, invalid_rows={invalid_rows}'
    result = time_series_frame(dates, values, 'lake', lake, path.name)
    if result.empty:
        return None, 'No valid lake observations after parsing'
    return result, f'EGM2008_m; candidate_rows={candidate_rows}; invalid_rows={invalid_rows}'

def monthly_aggregate(records):
    if not records:return pd.DataFrame(columns=['source_type','source_id','year','month','hydro_mean','hydro_max','hydro_min','n_measurements','n_distinct_dates','observed_month_fraction'])
    d=pd.concat(records,ignore_index=True)
    d['day']=d['date'].dt.normalize()
    # repeated measurements on a day can be legitimate; daily averaging stops heavily sampled days dominating mean
    daily=d.groupby(['source_type','source_id','day'],as_index=False).agg(value=('value','mean'),measurement_count=('value','size'))
    daily['year']=daily.day.dt.year
    daily['month']=daily.day.dt.month
    monthly=daily.groupby(['source_type','source_id','year','month'],as_index=False).agg(
        hydro_mean=('value','mean'),hydro_max=('value','max'),hydro_min=('value','min'),
        n_distinct_dates=('day','nunique'),n_measurements=('measurement_count','sum'))
    dates=pd.to_datetime(dict(year=monthly.year,month=monthly.month,day=1))
    monthly['observed_month_fraction']=monthly.n_distinct_dates/dates.dt.days_in_month
    return monthly.sort_values(['source_type','source_id','year','month'])


def read_mapping(path):
    mapping=pd.read_csv(path, dtype=str).fillna('')
    req=['source_type','source_id','level','adm_id','relationship']
    missing=set(req)-set(mapping.columns)
    if missing: raise ValueError(f'Mapping CSV missing {missing}')
    for c in req:mapping[c]=mapping[c].str.strip()
    bad=mapping[(~mapping.source_type.isin(['lake','dfo']))|(~mapping.level.isin(['ADM2','ADM3']))|(mapping.adm_id=='')|(mapping.relationship=='')]
    if len(bad):raise ValueError('Invalid mapping entries: type must be lake/dfo; level ADM2/ADM3; adm_id/relationship required')
    if mapping.duplicated(['source_type','source_id','level','adm_id']).any():raise ValueError('Duplicate mapping keys')
    return mapping


def mapped_monthly(monthly,mapping):
    x=monthly.merge(mapping,on=['source_type','source_id'],how='inner',validate='many_to_many')
    if x.empty:return pd.DataFrame(columns=['level','adm_id','year','month','lake_level_mean','lake_level_max','lake_obs_fraction','dfo_discharge_mean','dfo_discharge_max','dfo_obs_fraction','lake_n_sources','dfo_n_sources'])
    # Never blend distinct sources as if one measurement. If multiple sources map to an ADM,
    # refuse aggregation until a scientifically justified combination is selected.
    count=x.groupby(['level','adm_id','year','month','source_type']).source_id.nunique()
    if count.gt(1).any():
        ex=count[count.gt(1)].head(5).to_dict()
        raise ValueError(f'Multiple same-type sources mapped to one ADM-month; aggregation needs expert decision. Examples: {ex}')
    out=[]
    for typ,prefix in [('lake','lake'),('dfo','dfo')]:
        z=x.loc[x.source_type==typ,['level','adm_id','year','month','hydro_mean','hydro_max','observed_month_fraction','source_id']].copy()
        if z.empty:continue
        z=z.rename(columns={'hydro_mean':f'{prefix}_'+('level_mean' if typ=='lake' else 'discharge_mean'),
                            'hydro_max':f'{prefix}_'+('level_max' if typ=='lake' else 'discharge_max'),
                            'observed_month_fraction':f'{prefix}_obs_fraction',
                            'source_id':f'{prefix}_source_id'})
        out.append(z)
    if not out:return pd.DataFrame()
    merged=out[0]
    for z in out[1:]:merged=merged.merge(z,on=['level','adm_id','year','month'],how='outer',validate='1:1')
    return merged


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--mapping',type=Path,default=None,help='Reviewed hydrological relation CSV, required for ADM features')
    parser.add_argument('--min-coverage',type=float,default=0.7,
                        help='Minimum observed-day fraction for DFO daily discharge only; lake is irregular ~10 days and is NOT thresholded this way')
    args=parser.parse_args()
    root=args.root.resolve()
    inp=root/'raw_data'
    out=root/'processed_data/multivariate_flood_modeling/06a_lake_dfo_monthly'
    out.mkdir(parents=True,exist_ok=True)
    records=[];audit=[]
    for p in sorted((inp/'darthmouth_flood_observatory').glob('*_discharge.csv')):
        try: data,err=load_dfo(p)
        except Exception as exc:data,err=None,f'{type(exc).__name__}: {exc}'
        audit.append(dict(file=str(p.relative_to(root)),status='OK' if data is not None else 'REVIEW',reason=err,rows=len(data) if data is not None else 0))
        if data is not None:records.append(data)
    lakes=inp/'water_levels_lakes'
    for p in sorted(list(lakes.glob('*.txt'))+list(lakes.glob('*.nc'))):
        try:data,err=load_albert(p) if p.suffix=='.nc' else load_lake_txt(p)
        except Exception as exc:data,err=None,f'{type(exc).__name__}: {exc}'
        audit.append(dict(file=str(p.relative_to(root)),status='OK' if data is not None else 'REVIEW',reason=err,rows=len(data) if data is not None else 0))
        if data is not None:records.append(data)
    # DFO station metadata is spatial context only; NOT a hydrological ADM mapping.
    info = inp/'darthmouth_flood_observatory'/'information.xlsx'
    if info.exists():
        try:
            loc = pd.read_excel(info)
            cols = {str(c).strip().lower(): c for c in loc.columns}
            required = ['area id', 'latitude', 'longitude']
            if all(k in cols for k in required):
                station_locations = loc[[cols[k] for k in required]].rename(columns={
                    cols['area id']: 'source_id',
                    cols['latitude']: 'latitude',
                    cols['longitude']: 'longitude'
                })
                station_locations['source_id'] = station_locations['source_id'].astype(str)
                station_locations['source_type'] = 'dfo'
                station_locations.to_csv(out/'dfo_station_locations.csv', index=False)
        except Exception as exc:
            print('Warning: DFO station metadata could not be read:', exc)
    pd.DataFrame(audit).to_csv(out/'source_read_audit.csv',index=False)
    monthly=monthly_aggregate(records)
    monthly.to_csv(out/'station_lake_monthly.csv', index=False)
    # Also export explicitly complete calendar to show missing measurement months as NaN.
    full_records = []
    for (source_type, source_id), part in monthly.groupby(['source_type', 'source_id']):
        mindex = pd.period_range(
            f'{int(part.year.min())}-{int(part.month[part.year.eq(part.year.min())].min()):02d}',
            f'{int(part.year.max())}-{int(part.month[part.year.eq(part.year.max())].max()):02d}',
            freq='M'
        )
        base = pd.DataFrame({'year': mindex.year, 'month': mindex.month})
        base['source_type'], base['source_id'] = source_type, source_id
        full_records.append(base.merge(part, how='left', on=['source_type', 'source_id', 'year', 'month'], validate='1:1'))
    if full_records:
        pd.concat(full_records, ignore_index=True).to_csv(out/'station_lake_monthly_calendar.csv', index=False)
    print('Monthly source rows:',len(monthly))
    print(pd.DataFrame(audit).to_string(index=False))
    notes={'mapping_status':'NOT_PROVIDED','monthly_source_rows':len(monthly),'min_coverage':args.min_coverage,
           'warning':'Victoria/Kyoga TXT parsed as EGM2008 height (m), DFO discharge m3/s; check Albert units separately. Lake sparse observations are not held to daily coverage. No spatial mapping assumed. All features must be lagged for forecasting.'}
    if args.mapping is not None:
        path=args.mapping if args.mapping.is_absolute() else root/args.mapping
        mapping=read_mapping(path)
        matches=mapped_monthly(monthly,mapping)
        # Satellite lake heights are irregular (~10-day cycles): valid sparse months
        # are retained with their actual count. Do not apply 70% DAILY coverage to them.
        # DFO discharge is expected to be daily and receives the daily coverage check.
        c = 'dfo_obs_fraction'
        if c in matches:
            for v in ['dfo_discharge_mean', 'dfo_discharge_max']:
                if v in matches.columns:
                    matches.loc[matches[c] < args.min_coverage, v] = np.nan
        matches.to_csv(out/'adm_monthly_hydrological_features.csv',index=False)
        # Merge mapped features into existing monthly panel, leave others missing
        for level in ['ADM2','ADM3']:
            input_panel=root/f'processed_data/multivariate_flood_modeling/05a_monthly_panel/{level.lower()}_monthly_panel.csv'
            if not input_panel.exists():continue
            panel=pd.read_csv(input_panel,dtype={'adm_id':str})
            mapped=matches[matches.level==level].drop(columns='level')
            augmented=panel.merge(mapped,on=['adm_id','year','month'],how='left',validate='1:1')
            augmented.to_csv(out/f'{level.lower()}_monthly_augmented.csv',index=False)
            print(level,'panel rows',len(augmented),'with lake:',int(augmented.get('lake_level_mean',pd.Series(dtype=float)).notna().sum()),'with DFO:',int(augmented.get('dfo_discharge_mean',pd.Series(dtype=float)).notna().sum()))
        notes['mapping_status']='REVIEWED_EXTERNAL_MAPPING_USED'
        notes['mapping_csv']=str(path)
        notes['mapped_monthly_rows']=len(matches)
    else:
        (out/'hydrology_adm_mapping_TEMPLATE.csv').write_text('source_type,source_id,level,adm_id,relationship\n',encoding='utf-8')
        print('No reviewed mapping: exported source-level monthly series only. Fill hydrology_adm_mapping_TEMPLATE.csv using hydrological evidence before modeling.')
    (out/'methodology.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print('Output:',out)

if __name__=='__main__':main()
