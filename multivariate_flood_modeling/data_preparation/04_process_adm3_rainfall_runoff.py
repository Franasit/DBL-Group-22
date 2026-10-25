#!/usr/bin/env python3
"""JBG060: annual ADM3 rainfall/runoff from vetted daily ERA5 spatial grids.

This script deliberately never interprets unknown units or undiagnosed cumulative
fields. It does not resample original hourly ERA5. Source data must already be
validated daily cell-depths; consult the ERA5 quality audit and processing script.
"""
from __future__ import annotations
import argparse
import calendar
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.sparse import csr_matrix
from shapely.geometry import box
from shapely.ops import transform

YEARS = range(2000, 2026)
VARIABLE_OPTIONS = {
    'rainfall': ('precipitation_mm', 'rainfall_mm', 'daily_precipitation_mm',
                 'daily_rainfall_mm', 'tp_mm', 'total_precipitation', 'tp', 'rainfall', 'precipitation'),
    'runoff': ('runoff_mm', 'daily_runoff_mm', 'ro_mm', 'total_runoff', 'ro', 'runoff'),
}


def project_root(current: Path) -> Path:
    for folder in [current.parent, *current.parents]:
        if (folder / 'raw_data').is_dir() and (folder / 'processed_data').is_dir():
            return folder
    raise FileNotFoundError('Cannot identify repository root; supply --root.')


def pick_coord(ds, aliases, label):
    matches = [name for name in aliases if name in ds.coords and ds[name].ndim == 1]
    if len(matches) != 1:
        raise ValueError(f'Cannot identify unique {label} coordinate: {matches}. Available: {list(ds.coords)}')
    return matches[0]


def pick_variable(ds, role, supplied):
    if supplied:
        if supplied not in ds.data_vars:
            raise ValueError(f'{role} variable {supplied!r} absent; available: {list(ds.data_vars)}')
        return supplied
    matches = [n for n in VARIABLE_OPTIONS[role] if n in ds.data_vars]
    if not matches:
        raise ValueError(f'No recognized {role} variable; available: {list(ds.data_vars)}. Set --{role}-var.')
    if len(matches) > 1:
        raise ValueError(f'Ambiguous {role} variables {matches}; set --{role}-var explicitly.')
    return matches[0]


def unit_to_mm_factor(da, role):
    raw = str(da.attrs.get('units', '')).strip()
    unit = re.sub(r'\s+', '', raw.casefold()).replace('millimeters', 'mm').replace('millimetres', 'mm')
    if unit in ('mm', 'mm/day', 'mmd-1', 'mmday-1', 'mmd^-1'):
        return 1.0, raw
    if unit in ('m', 'm/day', 'md-1', 'mday-1', 'md^-1'):
        return 1000.0, raw
    raise ValueError(f'{role} unit {raw!r} is not an accepted daily depth. STOP: do not guess units.')


def grid_edges(values, label):
    v = np.asarray(values, dtype='float64')
    if v.ndim != 1 or len(v) < 2 or not np.isfinite(v).all():
        raise ValueError(f'{label} must be a finite, 1-D grid with >=2 values')
    d = np.diff(v)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError(f'{label} must be strictly monotonic')
    edges = np.empty(len(v) + 1)
    edges[1:-1] = (v[:-1] + v[1:]) / 2
    edges[0] = v[0] - d[0] / 2
    edges[-1] = v[-1] + d[-1] / 2
    return edges


def load_admin(root):
    path = root / 'raw_data/administrative_boundaries/ssd_admin3.geojson'
    if not path.is_file():
        raise FileNotFoundError(f'ADM3 boundary missing: {path}')
    gdf = gpd.read_file(path)
    col = next((c for c in ['adm3_pcode', 'ADM3_ID', 'adm3_id', 'ADM3_PCODE'] if c in gdf.columns), None)
    if not col:
        raise ValueError(f'ADM3 boundary has no ADM3 code. Columns: {list(gdf.columns)}')
    if gdf.crs is None:
        raise ValueError('Administrative boundaries lack CRS')
    gdf = gdf.to_crs('EPSG:4326')
    gdf['adm_id'] = gdf[col].astype('string').str.strip()
    if gdf.adm_id.isna().any() or gdf.adm_id.duplicated().any() or (gdf.adm_id == '').any():
        raise ValueError('Missing or duplicate ADM3 IDs')
    if len(gdf) != 512:
        raise ValueError(f'Expected 512 ADM3 polygons; got {len(gdf)}')
    if gdf.geometry.isna().any() or (~gdf.geometry.is_valid).any():
        raise ValueError('ADM3 has null/invalid polygon geometry; repair boundaries separately first')
    return gdf.sort_values('adm_id').reset_index(drop=True), path


def make_area_weights(gdf, lats, lons):
    """Generate polygon overlap areas in EPSG:6933; preserve native ERA5 resolution."""
    lat_edges = grid_edges(lats, 'latitude')
    lon_edges = grid_edges(lons, 'longitude')
    ny, nx = len(lats), len(lons)
    to_area = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True).transform
    rows, cols, weights = [], [], []
    static = []
    for i, (adm_id, geom) in enumerate(zip(gdf['adm_id'], gdf.geometry)):
        area_geom = transform(to_area, geom)
        polygon_area = area_geom.area
        if polygon_area <= 0:
            raise ValueError(f'ADM3 {adm_id} has zero area in equal-area projection')
        xmin, ymin, xmax, ymax = geom.bounds
        y_select = [j for j in range(ny) if max(lat_edges[j], lat_edges[j+1]) > ymin and min(lat_edges[j], lat_edges[j+1]) < ymax]
        x_select = [k for k in range(nx) if max(lon_edges[k], lon_edges[k+1]) > xmin and min(lon_edges[k], lon_edges[k+1]) < xmax]
        intersect_area = 0.0
        for j in y_select:
            for k in x_select:
                cell = box(min(lon_edges[k],lon_edges[k+1]), min(lat_edges[j],lat_edges[j+1]),
                           max(lon_edges[k],lon_edges[k+1]), max(lat_edges[j],lat_edges[j+1]))
                if not geom.intersects(cell):
                    continue
                area = transform(to_area, cell).intersection(area_geom).area
                if area > 0:
                    rows.append(i); cols.append(j * nx + k); weights.append(area)
                    intersect_area += area
        frac = min(1.0, intersect_area / polygon_area)
        static.append({'adm_id':adm_id, 'grid_cells_intersected':sum(1 for n in rows if n == i),
                       'geometry_area_m2':polygon_area, 'grid_intersection_area_m2':intersect_area,
                       'domain_coverage_fraction':frac})
        if i % 100 == 0:
            print(f'Computing spatial intersections: {i+1}/{len(gdf)}', flush=True)
    sparse = csr_matrix((np.asarray(weights), (np.asarray(rows), np.asarray(cols))),
                        shape=(len(gdf), ny * nx), dtype='float64')
    return sparse, pd.DataFrame(static)


def inspect_one(path, a):
    with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
        tn = pick_coord(ds, ('time','valid_time','date'), 'time')
        yn = pick_coord(ds, ('latitude','lat'), 'latitude')
        xn = pick_coord(ds, ('longitude','lon'), 'longitude')
        rain = pick_variable(ds,'rainfall', a.rainfall_var)
        run = pick_variable(ds,'runoff', a.runoff_var)
        if rain == run: raise ValueError('Rainfall and runoff resolved to same variable')
        info = {}
        for role,name in [('rainfall',rain),('runoff',run)]:
            da = ds[name]
            if set(da.dims) != {tn,yn,xn}:
                raise ValueError(f'{name} dimensions {da.dims} not exactly (time,latitude,longitude); unexpected extra dims')
            factor, units = unit_to_mm_factor(da,role)
            attrs = ' '.join(str(v) for v in da.attrs.values()).casefold()
            if 'cumulative since' in attrs or 'forecast step' in attrs or 'accumulated since' in attrs:
                raise ValueError(f'{name} looks like a forecast accumulation, not a daily depth; STOP')
            info[role] = {'name':name,'factor_to_mm':factor,'source_units':units}
        lat = np.asarray(ds[yn].values,dtype=float)
        lon = np.asarray(ds[xn].values,dtype=float)
        dates = pd.to_datetime(ds[tn].values)
        if dates.isna().any(): raise ValueError('Unparseable daily dates')
        return tn,yn,xn,lat,lon,info


def analyze_year(path, year, axes, info, w, static, min_daily_coverage, min_days_fraction):
    tn,yn,xn,ref_lat,ref_lon = axes
    with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
        if not np.array_equal(ds[yn].values,ref_lat) or not np.array_equal(ds[xn].values,ref_lon):
            raise ValueError(f'{path.name}: grid changed from first year; stop rather than reuse wrong spatial weights')
        dates = pd.to_datetime(ds[tn].values)
        if dates.has_duplicates or len(dates) != (366 if calendar.isleap(year) else 365):
            raise ValueError(f'{path.name}: duplicate or missing daily dates, observed {len(dates)}')
        expected = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
        if not dates.equals(expected):
            raise ValueError(f'{path.name}: daily date sequence is not the complete calendar year')
        total_weight = np.asarray(w.sum(axis=1)).ravel()
        day_count = len(dates)
        out = {'adm_id':static.adm_id.to_numpy(), 'year':np.full(len(static),year),
               'domain_coverage_fraction':static.domain_coverage_fraction.to_numpy(),
               'grid_cells_intersected':static.grid_cells_intersected.to_numpy()}
        valid_days_all = []
        for role, value_name in [('rainfall','rainfall'),('runoff','runoff')]:
            name = info[role]['name']
            if name not in ds: raise ValueError(f'{path.name} missing variable {name}')
            factor, units = unit_to_mm_factor(ds[name],role)
            if factor != info[role]['factor_to_mm']:
                raise ValueError(f'{path.name}: inconsistent source units for {name}')
            arr = np.asarray(ds[name].transpose(tn,yn,xn).values,dtype='float64') * factor
            if arr.shape != (day_count,len(ref_lat),len(ref_lon)):
                raise ValueError(f'Unexpected array shape for {name}: {arr.shape}')
            if np.any(np.isfinite(arr) & (arr < -1e-5)):
                raise ValueError(f'{path.name}: negative daily {role} values; verify whether daily increments are correctly produced')
            arr = np.where((arr >= 0) & np.isfinite(arr),arr,np.nan)
            valid = np.isfinite(arr).reshape(day_count,-1).T
            values = np.nan_to_num(arr, nan=0).reshape(day_count,-1).T
            numerator = np.asarray(w @ values)
            denominator = np.asarray(w @ valid.astype('float64'))
            with np.errstate(divide='ignore',invalid='ignore'):
                mean_daily = np.where(denominator > 0, numerator / denominator, np.nan).T
                coverage = np.where(total_weight[:,None] > 0, denominator / total_weight[:,None], 0).T
            accepted = (coverage >= min_daily_coverage) & (static.domain_coverage_fraction.to_numpy()[None,:] >= 0.95)
            mean_daily[~accepted] = np.nan
            valid_days = np.isfinite(mean_daily).sum(axis=0)
            year_complete = valid_days >= np.ceil(day_count * min_days_fraction)
            sums = np.nansum(mean_daily,axis=0)
            sums[~year_complete] = np.nan
            daily_mean = np.nanmean(np.where(np.isfinite(mean_daily),mean_daily,np.nan),axis=0)
            daily_mean[~year_complete] = np.nan
            out[f'{value_name}_annual_mm'] = sums
            out[f'{value_name}_mean_daily_mm'] = daily_mean
            out[f'{value_name}_max_daily_mm'] = np.where(year_complete, np.nanmax(np.where(np.isfinite(mean_daily), mean_daily, -np.inf),axis=0),np.nan)
            out[f'{value_name}_valid_days'] = valid_days
            out[f'{value_name}_valid_day_fraction'] = valid_days/day_count
            valid_days_all.append(year_complete)
        out['hydrology_qc_status'] = np.where(np.logical_and(*valid_days_all), 'PASS', 'INSUFFICIENT_COVERAGE')
        return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=None)
    p.add_argument('--rainfall-var',default=None)
    p.add_argument('--runoff-var',default=None)
    p.add_argument('--min-daily-coverage',type=float,default=0.80)
    p.add_argument('--min-valid-day-fraction',type=float,default=0.95)
    p.add_argument('--audit-only',action='store_true', help='Inspect first/middle/last daily grid and exit, no aggregation')
    p.add_argument('--confirm-daily-increments',action='store_true',
                   help='Confirm you verified ERA5 processing semantics: input cells are daily depths, not cumulative forecasts')
    a=p.parse_args()
    if not (0 < a.min_daily_coverage <= 1 and 0 < a.min_valid_day_fraction <= 1):
        p.error('Coverage thresholds must be in (0,1]')
    root = a.root.resolve() if a.root else project_root(Path(__file__).resolve())
    folder = root/'processed_data/rainfall_and_runoff_processed/daily_grids'
    files = sorted(folder.glob('ERA5_daily_spatial_*.nc'))
    years = [int(re.search(r'_(\d{4})\.nc$',f.name).group(1)) for f in files]
    if years != list(YEARS):
        raise ValueError(f'Expected exactly one daily spatial NetCDF per year 2000–2025, found years: {years}')
    first = files[0]
    tn,yn,xn,lat,lon,info = inspect_one(first,a)
    print('Daily grid:',len(files),'years;',len(lat),'latitudes;',len(lon),'longitudes')
    print('Variables:', json.dumps(info,indent=2))
    print('Input:',folder)
    audit = root/'processed_data/rainfall_and_runoff_processed/ERA5_quality_audit.csv'
    if audit.is_file(): print('Existing ERA5 audit:',audit,'(review manually for semantic correctness)')
    if a.audit_only:
        for f in [files[0],files[len(files)//2],files[-1]]:
            aa,bb,cc,ll,oo,ii=inspect_one(f,a)
            if not (np.array_equal(lat,ll) and np.array_equal(lon,oo)):
                raise ValueError(f'Grid mismatch in {f.name}')
            print(f'{f.name}: variable units and grid OK; rainfall={ii["rainfall"]["name"]}, runoff={ii["runoff"]["name"]}')
        print('AUDIT COMPLETE. Not a certification of the original hourly accumulation semantics.')
        return
    if not a.confirm_daily_increments:
        p.error('Before processing, review source ERA5 quality_audit and daily-processing methodology; rerun with --confirm-daily-increments only if inputs truly represent daily rainfall/runoff depths.')
    admin,boundary=load_admin(root)
    weights,static=make_area_weights(admin,lat,lon)
    if (static.grid_cells_intersected == 0).any():
        print('WARNING: regions with no overlapping ERA5 grid:',int((static.grid_cells_intersected==0).sum()))
    print('ADM3 polygons:',len(admin),'; with grid intersection:',int((static.grid_cells_intersected>0).sum()))
    axes=(tn,yn,xn,lat,lon)
    frames=[]
    for idx,(file,year) in enumerate(zip(files,YEARS),1):
        frames.append(analyze_year(file,year,axes,info,weights,static,a.min_daily_coverage,a.min_valid_day_fraction))
        print(f'Processed ERA5 daily grid year {year} ({idx}/26)',flush=True)
    panel=pd.concat(frames,ignore_index=True).sort_values(['adm_id','year'])
    assert len(panel)==512*26 and not panel.duplicated(['adm_id','year']).any()
    out=root/'processed_data/multivariate_flood_modeling/04_adm3_rainfall_runoff'
    out.mkdir(parents=True,exist_ok=True)
    panel.to_csv(out/'adm3_annual_rainfall_runoff.csv',index=False)
    static.to_csv(out/'adm3_grid_coverage.csv',index=False)
    panel.groupby('hydrology_qc_status',dropna=False).size().rename('region_years').reset_index().to_csv(out/'adm3_hydrology_qc.csv',index=False)
    pd.DataFrame([{'variable':r,'source_column':d['name'],'original_units':d['source_units'],'mm_factor':d['factor_to_mm']} for r,d in info.items()]).to_csv(out/'source_variable_lineage.csv',index=False)
    summary={'input':str(folder),'boundary':str(boundary),'source_files':len(files),'years':[2000,2025],
             'regions':len(admin),'region_year_rows':len(panel),'qc_pass_rows':int((panel.hydrology_qc_status=='PASS').sum()),
             'grid_intersection_zero_regions':int((static.grid_cells_intersected==0).sum()),
             'units_verified_as_daily_depth':True,'hourly_accumulation_semantics_external_confirmation':True,
             'method':'EPSG:6933 polygon-by-cell intersected area weighted daily means; annual sum of valid daily regional means',
             'warning':'Native ERA5 grid resolution unchanged; small ADM3 share coarse climate information. QC is not independent station validation.'}
    (out/'processing_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output_dir':str(out),'panel_rows':len(panel),'qc_pass_rows':summary['qc_pass_rows']},indent=2))

if __name__=='__main__':
    main()
