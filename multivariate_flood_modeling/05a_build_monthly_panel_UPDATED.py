#!/usr/bin/env python3
"""JBG060: construct REAL region-month inputs for 05b monthly Random Forest.

Default target: exploratory --event-proxy creates a count of calendar
DAYS with >=1 recorded flood event in each polygon (NOT validated flood days).
The latter is not a complete flood observation denominator. It is inappropriate
for claims about actual flood-free days and should be described as an event-date
proxy throughout results.

Hydrology: regional mean of vetted daily ERA5 precipitation/runoff grid depths,
using grid-cell CENTRE raster assignment (coarser than polygon-area weighting).
ET0: regional mean of daily FAO56 reference ET0 grids, using pixel centres.
All monthly totals require >= --min-valid-day-fraction valid days (default .95).
Daily ERA5 2025-12-31 missing stays missing; December 2025 is flagged PARTIAL
although it may pass a 95% completeness threshold. No imputation is performed.

Memory note: flood Parquet can be large; batch processing and regional point
lookup caching are used. Runtime depends on data size (2000-2025 ~ many rows).

Examples:
 python scripts/multivariate_flood_modeling/05a_build_monthly_panel.py
 python scripts/multivariate_flood_modeling/05a_build_monthly_panel.py --event-proxy
 python scripts/multivariate_flood_modeling/05a_build_monthly_panel.py \
    --flood-monthly-csv processed_data/validated_monthly_flood_days.csv
"""
from __future__ import annotations
import argparse, calendar, json, re
from pathlib import Path
import numpy as np
import pandas as pd

YEARS = range(2000, 2026)
ET_VAR = 'ReferenceET_PenmanMonteith_FAO56'
HYDRO_NAMES = {'rainfall':('precipitation_mm','rainfall_mm','daily_precipitation_mm','tp_mm','tp'),
               'runoff':('runoff_mm','daily_runoff_mm','ro_mm','ro')}

def root_from_script():
    return Path(__file__).resolve().parents[2]

def boundaries(root, level):
    import geopandas as gpd
    path=root/f'raw_data/administrative_boundaries/ssd_admin{level}.geojson'
    g=gpd.read_file(path)
    cols=[f'adm{level}_pcode',f'ADM{level}_ID',f'adm{level}_id',f'ADM{level}_PCODE']
    col=next((c for c in cols if c in g.columns),None)
    if not col or g.crs is None: raise ValueError(f'{path}: admin ID or CRS missing')
    g=g.to_crs('EPSG:4326')
    g['adm_id']=g[col].astype('string').str.strip()
    g=g[['adm_id','geometry']].dissolve(by='adm_id',as_index=False)
    if g.adm_id.isna().any() or g.adm_id.duplicated().any(): raise ValueError('Duplicate admin IDs')
    return g.sort_values('adm_id').reset_index(drop=True)

def grid_mapping(g,lat,lon):
    """Single centre-to-region mapping, no double counting across polygons."""
    import geopandas as gpd
    from shapely.geometry import Point
    yy,xx=np.meshgrid(lat,lon,indexing='ij')
    pts=gpd.GeoDataFrame({'cell':np.arange(xx.size)},geometry=gpd.points_from_xy(xx.ravel(),yy.ravel()),crs='EPSG:4326')
    join=gpd.sjoin(pts,g[['adm_id','geometry']],how='left',predicate='within')
    if join.duplicated('cell').any(): raise ValueError('A grid centre belongs to multiple admin regions')
    mapping=np.full(xx.size,-1,dtype=np.int32)
    ids={v:i for i,v in enumerate(g.adm_id.astype(str))}
    valid=join.adm_id.notna()
    mapping[join.loc[valid,'cell'].to_numpy(int)]=join.loc[valid,'adm_id'].map(ids).to_numpy(int)
    return mapping

def weighted_region_daily(values,mapping,nregions):
    """Mean depth per admin from all valid centre-assigned pixels for each day."""
    ntime=values.shape[0]
    x=values.reshape(ntime,-1)
    mask=mapping>=0
    group=mapping[mask]
    out=np.full((ntime,nregions),np.nan)
    for k in range(ntime):
        row=x[k,mask]; good=np.isfinite(row)&(row>=0)
        sums=np.bincount(group[good],weights=row[good],minlength=nregions)
        counts=np.bincount(group[good],minlength=nregions)
        out[k,counts>0]=sums[counts>0]/counts[counts>0]
    return out

def monthly_from_day_frames(frames,field,threshold):
    d=pd.concat(frames,ignore_index=True)
    d['year']=d.date.dt.year; d['month']=d.date.dt.month
    grp=d.groupby(['adm_id','year','month'],as_index=False).agg(
        sum_valid=(field,lambda x:x.sum(min_count=1)),valid_days=(field,'count'),
        seen_dates=('date','nunique'))
    grp['calendar_days']=pd.to_datetime(dict(year=grp.year,month=grp.month,day=1)).dt.days_in_month
    grp['valid_fraction']=grp.valid_days/grp.calendar_days
    grp[f'{field}_monthly']=grp.sum_valid.where(grp.valid_fraction>=threshold)
    return grp

def process_hydro(root,admins,threshold):
    import xarray as xr
    paths=sorted((root/'processed_data/rainfall_and_runoff_processed/daily_grids').glob('ERA5_daily_spatial_*.nc'))
    if not paths: raise FileNotFoundError('No vetted ERA5 daily grids in processed_data/rainfall_and_runoff_processed/daily_grids')
    all_outputs={}
    for level,g in admins.items(): all_outputs[level]=[]
    mapping={}; grid_ref=None
    for file in paths:
        with xr.open_dataset(file,decode_times=True) as ds:
            tn=next((x for x in ['time','valid_time','date'] if x in ds.coords),None)
            yn=next((x for x in ['latitude','lat'] if x in ds.coords),None)
            xn=next((x for x in ['longitude','lon'] if x in ds.coords),None)
            if not all([tn,yn,xn]): raise ValueError(f'{file}: time/lat/lon not recognized')
            lat=np.asarray(ds[yn].values);lon=np.asarray(ds[xn].values)
            if grid_ref is None:
                grid_ref=(lat.copy(),lon.copy())
                mapping={lvl:grid_mapping(g,lat,lon) for lvl,g in admins.items()}
            elif not (np.array_equal(lat,grid_ref[0]) and np.array_equal(lon,grid_ref[1])):
                raise ValueError(f'{file}: ERA5 grid changed')
            dates=pd.to_datetime(ds[tn].values).normalize()
            for role,options in HYDRO_NAMES.items():
                match=next((x for x in options if x in ds.data_vars),None)
                if not match: raise ValueError(f'{file}: no verified {role} variable, available={list(ds.data_vars)}')
                da=ds[match]
                units=str(da.attrs.get('units','')).strip().lower().replace(' ','')
                if units not in ('mm','mm/day','m','m/day','mmd-1','mmd^-1','md-1'):
                    raise ValueError(f'{file}: {role} units {units!r} not verified')
                factor=1000 if units in ('m','m/day','md-1') else 1
                arr=np.asarray(da.transpose(tn,yn,xn).values,dtype=float)*factor
                if np.any(np.isfinite(arr)&(arr < -1e-5)):
                    raise ValueError(f'{file}: negative {role} daily depth')
                for level,g in admins.items():
                    vals=weighted_region_daily(arr,mapping[level],len(g))
                    tmp=pd.DataFrame(vals,columns=g.adm_id.astype(str));tmp['date']=dates
                    long=tmp.melt(id_vars='date',var_name='adm_id',value_name=role+'_mm')
                    all_outputs[level].append((role,long))
        print('ERA5:',file.name,flush=True)
    result={}
    for level in admins:
        byrole={role:[] for role in HYDRO_NAMES}
        for role,df in all_outputs[level]: byrole[role].append(df)
        pieces=[]
        for role,frames in byrole.items():
            monthly=monthly_from_day_frames(frames,role+'_mm',threshold)
            monthly=monthly.rename(columns={f'{role}_mm_monthly':f'{role}_mm',
                'valid_days':f'{role}_valid_days','valid_fraction':f'{role}_valid_fraction',
                'seen_dates':f'{role}_seen_dates'})
            pieces.append(monthly[['adm_id','year','month',f'{role}_mm',f'{role}_valid_days',f'{role}_valid_fraction']])
        result[level]=pieces[0].merge(pieces[1],on=['adm_id','year','month'],validate='one_to_one')
    return result

def process_et0(root,admins,threshold):
    import xarray as xr
    paths=sorted((root/'raw_data/evapotranspiration').glob('ET_*/*.nc'))
    if not paths: raise FileNotFoundError('No ET0 files under raw_data/evapotranspiration/ET_*/*.nc')
    refs=None;mapping={};frames={k:[] for k in admins};errors=[]
    for idx,file in enumerate(paths,1):
        try:
            with xr.open_dataset(file,decode_times=True) as ds:
                da=ds[ET_VAR]
                if set(da.dims)!={'time','lat','lon'} or da.sizes['time']!=1:
                    raise ValueError('Unexpected ET0 shape')
                units=str(da.attrs.get('units','')).replace(' ','')
                if units not in ('mmd-1','mm/day','mmday-1'):
                    raise ValueError(f'ET0 units {units!r} invalid')
                lat=np.asarray(ds.lat.values);lon=np.asarray(ds.lon.values)
                if refs is None:
                    refs=(lat.copy(),lon.copy())
                    mapping={lvl:grid_mapping(g,lat,lon) for lvl,g in admins.items()}
                elif not (np.array_equal(lat,refs[0]) and np.array_equal(lon,refs[1])):
                    raise ValueError('ET0 grid changed')
                date=pd.Timestamp(da.time.values[0]).normalize()
                arr=np.asarray(da.transpose('time','lat','lon').values,dtype=float)
                for level,g in admins.items():
                    vals=weighted_region_daily(arr,mapping[level],len(g))[0]
                    frames[level].append(pd.DataFrame({'adm_id':g.adm_id.astype(str),'date':date,'et0_mm':vals}))
        except Exception as exc:
            errors.append((str(file),str(exc)))
        if idx%1000==0: print(f'ET0: {idx}/{len(paths)}',flush=True)
    if not any(frames.values()): raise ValueError(f'No ET0 rows generated. First errors: {errors[:3]}')
    output={}
    for level,series in frames.items():
        m=monthly_from_day_frames(series,'et0_mm',threshold)
        output[level]=m.rename(columns={'et0_mm_monthly':'et0_mm','valid_days':'et0_valid_days',
           'valid_fraction':'et0_valid_fraction'})[['adm_id','year','month','et0_mm','et0_valid_days','et0_valid_fraction']]
    return output,errors

def flood_from_external(file):
    x=pd.read_csv(file,low_memory=False)
    needed=['level','adm_id','year','month','flood_days']
    if not set(needed).issubset(x): raise ValueError(f'{file} must have {needed}')
    x=x[needed].copy()
    x['level']=x.level.astype(str).str.upper().str.strip()
    x['adm_id']=x.adm_id.astype(str).str.strip()
    for c in ['year','month','flood_days']: x[c]=pd.to_numeric(x[c],errors='raise')
    if x.duplicated(['level','adm_id','year','month']).any():raise ValueError('Duplicate validated flood month records')
    n=pd.to_datetime(dict(year=x.year.astype(int),month=x.month.astype(int),day=1)).dt.days_in_month
    if (x.flood_days<0).any() or (x.flood_days>n).any():raise ValueError('Invalid monthly flood-days')
    return {k:x.loc[x.level.eq(k),['adm_id','year','month','flood_days']] for k in ['ADM2','ADM3']}

def event_proxy(root,admins,batch_size):
    """Explore unique dates with ANY event in polygon; NOT regional flooded-day truth."""
    import geopandas as gpd
    import pyarrow.parquet as pq
    files=sorted((root/'raw_data/flood_masks').glob('compact_*/*.parquet'))
    if not files: raise FileNotFoundError('No compact flood events Parquet')
    # Cached mapping for distinct event pixel coordinates; may grow large.
    outputs={}; location_cache={lvl:{} for lvl in admins}
    for lvl in admins: outputs[lvl]=set()
    for idx,file in enumerate(files,1):
        pf=pq.ParquetFile(file)
        if not {'date','lat','lon','tile'}.issubset(pf.schema_arrow.names):
            raise ValueError(f'{file}: event schema missing expected keys')
        for chunk in pf.iter_batches(batch_size=batch_size,columns=['date','lat','lon','tile']):
            df=chunk.to_pandas()
            df['date']=pd.to_datetime(df.date,errors='coerce',utc=True).dt.tz_localize(None).dt.normalize()
            df['lat']=pd.to_numeric(df.lat,errors='coerce')
            df['lon']=pd.to_numeric(df.lon,errors='coerce')
            if df[['date','lat','lon']].isna().any().any(): raise ValueError(f'{file}: invalid dates or coordinates')
            # Coordinates within the same tile identify one pixel. Mapping determined once.
            pos=df[['tile','lat','lon']].drop_duplicates()
            keys=list(pos.itertuples(index=False,name=None))
            for level,g in admins.items():
                cache=location_cache[level]
                fresh=[key for key in keys if key not in cache]
                if fresh:
                    pts=gpd.GeoDataFrame({'key':range(len(fresh))},
                        geometry=gpd.points_from_xy([t[2] for t in fresh],[t[1] for t in fresh]),crs='EPSG:4326')
                    found=gpd.sjoin(pts,g[['adm_id','geometry']],how='left',predicate='within')
                    if found.duplicated('key').any():raise ValueError('Flood pixel maps to multiple regions')
                    for row in found.itertuples(index=False):
                        cache[fresh[int(row.key)]]=None if pd.isna(row.adm_id) else str(row.adm_id)
                # avoid materializing all source rows as accumulated time-event keys
                part=df[['date','tile','lat','lon']].drop_duplicates().copy()
                part['adm_id']=[cache[k] for k in part[['tile','lat','lon']].itertuples(index=False,name=None)]
                part=part.dropna(subset=['adm_id'])[['date','adm_id']].drop_duplicates()
                outputs[level].update((a,d.year,d.month,d.day) for d,a in part.itertuples(index=False,name=None))
        print(f'Flood events: {idx}/{len(files)} {file.name}',flush=True)
    result={}
    for level in admins:
        data=pd.DataFrame(list(outputs[level]),columns=['adm_id','year','month','day'])
        if data.empty:raise ValueError(f'No events spatially assigned to {level}')
        result[level]=data.groupby(['adm_id','year','month']).size().rename('flood_days').reset_index()
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=None)
    p.add_argument('--flood-monthly-csv',type=Path,help='Validated flood-days month table: level,adm_id,year,month,flood_days')
    p.add_argument('--event-proxy',action='store_true',help='EXPLORATORY only: use unique event-recorded dates as proxy')
    p.add_argument('--min-valid-day-fraction',type=float,default=.95)
    p.add_argument('--batch-size',type=int,default=150000)
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    # No command-line flags: use the explicitly labelled exploratory event proxy.
    # Providing a validated monthly CSV switches to verified mode automatically.
    if a.event_proxy and a.flood_monthly_csv:
        p.error('Do not combine --event-proxy with --flood-monthly-csv')
    if not a.flood_monthly_csv:
        a.event_proxy = True
    if not 0<a.min_valid_day_fraction<=1:p.error('--min-valid-day-fraction must be (0,1]')
    root=(a.root or root_from_script()).resolve()
    out=a.output or root/'processed_data/multivariate_flood_modeling/05a_monthly_panel'
    out.mkdir(parents=True,exist_ok=True)
    admins={f'ADM{lvl}':boundaries(root,lvl) for lvl in (2,3)}
    target=flood_from_external(a.flood_monthly_csv) if a.flood_monthly_csv else event_proxy(root,admins,a.batch_size)
    print('Aggregating ERA5 daily hydrology...',flush=True)
    hydro=process_hydro(root,admins,a.min_valid_day_fraction)
    print('Aggregating ET0 daily grids...',flush=True)
    et0,et_errors=process_et0(root,admins,a.min_valid_day_fraction)
    summary={'target_semantics':'VALIDATED_FLOOD_DAYS' if a.flood_monthly_csv else 'EXPLORATORY_EVENT_DATE_PROXY',
             'note':'A zero event date count is NOT independently verified as zero flooding.',
             'hydrology_spatial_method':'ERA5 grid-cell-centre region mean (approximate)',
             'et0_spatial_method':'ET0 grid-cell-centre region mean (approximate)',
             'min_valid_day_fraction':a.min_valid_day_fraction,'et0_file_errors':et_errors[:50],
             'et0_file_error_count':len(et_errors),'levels':{}}
    for lvl,g in admins.items():
        periods=pd.period_range('2000-01','2025-12',freq='M')
        period=pd.DataFrame({'year':periods.year,'month':periods.month})
        skeleton=g[['adm_id']].assign(_key=1).merge(period.assign(_key=1),on='_key').drop(columns='_key')
        panel=skeleton.merge(target[lvl],on=['adm_id','year','month'],how='left',validate='one_to_one')
        if a.event_proxy:
            # ZERO IS ONLY absence of EVENT RECORDS, not verified flood-free observation.
            panel['flood_days']=panel.flood_days.fillna(0)
        panel=panel.merge(hydro[lvl],on=['adm_id','year','month'],how='left',validate='one_to_one')
        panel=panel.merge(et0[lvl],on=['adm_id','year','month'],how='left',validate='one_to_one')
        panel['target_status']=summary['target_semantics']
        panel['hydrology_partial_month']=((panel.rainfall_valid_days<
            pd.to_datetime(dict(year=panel.year,month=panel.month,day=1)).dt.days_in_month) |
            (panel.runoff_valid_days<pd.to_datetime(dict(year=panel.year,month=panel.month,day=1)).dt.days_in_month))
        cols=['adm_id','year','month','flood_days','rainfall_mm','runoff_mm','et0_mm','target_status',
              'rainfall_valid_days','rainfall_valid_fraction','runoff_valid_days','runoff_valid_fraction',
              'et0_valid_days','et0_valid_fraction','hydrology_partial_month']
        path=out/f'{lvl.lower()}_monthly_panel.csv'
        panel[cols].to_csv(path,index=False)
        summary['levels'][lvl]={'rows':len(panel),'target_missing':int(panel.flood_days.isna().sum()),
             'rainfall_missing':int(panel.rainfall_mm.isna().sum()),'runoff_missing':int(panel.runoff_mm.isna().sum()),
             'et0_missing':int(panel.et0_mm.isna().sum()),'output':str(path)}
        print(lvl,summary['levels'][lvl],flush=True)
    (out/'methodology.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Output:',out)
    if a.event_proxy:print('WARNING: EVENT-DATE PROXY ONLY. DO NOT CLAIM VERIFIED FLOOD DAYS.')

if __name__=='__main__':main()
