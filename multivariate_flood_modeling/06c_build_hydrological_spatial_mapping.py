#!/usr/bin/env python3
"""JBG060 Script 06c: hydrology source location and ADM2/ADM3 candidate mapping.

This script creates GEOGRAPHIC CANDIDATES, not verified hydrological connectivity.
It NEVER writes an approved 06a hydrology mapping or assigns source values to areas.

Usage (project root, no arguments required):
  python scripts/multivariate_flood_modeling/06c_build_hydrological_spatial_mapping.py

Dependencies: pandas, geopandas, shapely, matplotlib, openpyxl (optional), xarray (optional)
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ADMIN_DIR = Path('raw_data/administrative_boundaries')
HYDRO_DIR = Path('processed_data/multivariate_flood_modeling/06a_lake_dfo_monthly')
LAKE_DIR = Path('raw_data/water_levels_lakes')
OUTPUT_DIR = Path('processed_data/multivariate_flood_modeling/06c_hydrological_spatial_mapping')

# Use the supplied lake-file metadata when available. These are lake target points,
# not gauges located in South Sudan nor proof of downstream connectivity.
FALLBACK_LAKES = {
    'Kyoga': (1.484, 32.778),
    'victoria': (-0.507, 33.488),
}


def lake_coordinates_from_txt(path: Path):
    if not path.exists():
        return None
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines()[:25]:
        if 'Latitude and longitude' in line:
            nums = re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?', line.split(':')[0])
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])
    return None


def lake_albert_coordinates(path: Path):
    """Use NC metadata/coord scalars only; no guessed lake-centre coordinate."""
    if not path.exists():
        return None
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            lat_options = ['latitude', 'lat', 'lake_latitude', 'lake_lat']
            lon_options = ['longitude', 'lon', 'lake_longitude', 'lake_lon']
            def extract(names):
                for name in names:
                    if name in ds.attrs:
                        val = float(ds.attrs[name]); return val
                    if name in ds and ds[name].size == 1:
                        return float(ds[name].values.item())
                return None
            lat, lon = extract(lat_options), extract(lon_options)
            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    except (ImportError, OSError, ValueError, TypeError):
        pass
    return None


def station_locations(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    warnings_rows = []
    loc = root / HYDRO_DIR / 'dfo_station_locations.csv'
    records = []
    if not loc.exists():
        warnings_rows.append({'issue': 'DFO station location file missing', 'detail': str(loc)})
    else:
        df = pd.read_csv(loc, dtype={'source_id': str})
        needed = {'source_id', 'latitude', 'longitude'}
        if not needed.issubset(df.columns):
            raise ValueError(f'{loc} needs columns {sorted(needed)}; found {list(df.columns)}')
        for row in df.itertuples(index=False):
            records.append({'source_type': 'dfo', 'source_id': str(row.source_id),
                            'latitude': row.latitude, 'longitude': row.longitude,
                            'coordinate_origin': '06a_dfo_station_locations',
                            'location_status': 'provided'})

    lake_folder = root / LAKE_DIR
    for lake_id, filename in [('Kyoga', 'water_level_Kyoga.txt'),
                              ('victoria', 'water_level_victoria.txt')]:
        coords = lake_coordinates_from_txt(lake_folder / filename)
        if coords is None:
            coords = FALLBACK_LAKES[lake_id]
            origin = 'manual_transcription_from_supplied_lake_header_check_against_original'
            warnings_rows.append({'issue': 'Lake coords fallback', 'detail': filename})
        else:
            origin = 'lake_file_header_target_midpoint'
        records.append({'source_type': 'lake', 'source_id': lake_id,
                        'latitude': coords[0], 'longitude': coords[1],
                        'coordinate_origin': origin, 'location_status': 'provided'})

    albert_nc = lake_folder / 'water_level_altimetry_Albert.nc'
    albert = lake_albert_coordinates(albert_nc)
    if albert is not None:
        records.append({'source_type': 'lake', 'source_id': 'Albert',
                        'latitude': albert[0], 'longitude': albert[1],
                        'coordinate_origin': 'lake_netcdf_metadata',
                        'location_status': 'provided'})
    else:
        records.append({'source_type': 'lake', 'source_id': 'Albert',
                        'latitude': None, 'longitude': None,
                        'coordinate_origin': 'unresolved_nc_coordinate_metadata',
                        'location_status': 'NEEDS_LOCATION_REVIEW'})
        warnings_rows.append({'issue': 'Lake Albert coordinates unresolved',
                              'detail': 'Provide verified coordinates via --extra-locations-csv; no coordinate guessed.'})

    result = pd.DataFrame(records)
    result['latitude'] = pd.to_numeric(result['latitude'], errors='coerce')
    result['longitude'] = pd.to_numeric(result['longitude'], errors='coerce')
    result['source_id'] = result['source_id'].astype(str)
    invalid = ~result['latitude'].between(-90,90) | ~result['longitude'].between(-180,180)
    result.loc[invalid, 'location_status'] = 'NEEDS_LOCATION_REVIEW'
    return result, pd.DataFrame(warnings_rows, columns=['issue','detail'])


def geometry_file(root: Path, level: str, explicit: str | None) -> Path:
    if explicit:
        f = Path(explicit)
        f = f if f.is_absolute() else root / f
        if not f.exists():
            raise FileNotFoundError(f'Boundary file not found: {f}')
        return f
    directory = root / ADMIN_DIR
    # Require level in filename: never fall back from ADM2 to ADM3 silently.
    files = [f for f in directory.rglob('*') if f.suffix.lower() in {'.geojson','.gpkg','.shp'}
             and any(k in f.stem.lower() for k in (level.lower(), 'admin'+level[-1]))]
    if not files:
        raise FileNotFoundError(f'No {level} boundary file under {directory}. Pass --{level.lower()}-boundaries.')
    files.sort(key=lambda f: (0 if f.suffix.lower()=='.geojson' else 1, len(str(f))))
    return files[0]


def boundary_id_column(gdf: gpd.GeoDataFrame, level: str, specified: str | None) -> str:
    if specified:
        if specified not in gdf.columns:
            raise ValueError(f'Boundary ID column {specified} missing; columns are {list(gdf.columns)}')
        return specified
    level_num = level[-1]
    candidates = [f'adm{level_num}_pcode',f'ADM{level_num}_PCODE',f'ADM{level_num}_Pcode',
                  f'adm{level_num}_code', 'adm_id', 'pcode']
    matches = [c for c in candidates if c in gdf.columns]
    if not matches:
        raise ValueError(f'Cannot identify {level} pcode from {list(gdf.columns)}; pass --{level.lower()}-id-column')
    return matches[0]


def load_admin(path: Path, level: str, id_col: str | None) -> gpd.GeoDataFrame:
    raw = gpd.read_file(path)
    if raw.crs is None:
        raise ValueError(f'Boundary CRS undefined: {path}; define it from source metadata before proceeding')
    col = boundary_id_column(raw, level, id_col)
    raw['adm_id'] = raw[col].astype('string').str.strip()
    raw = raw[raw['adm_id'].notna() & raw.geometry.notna()].copy()
    if raw['adm_id'].duplicated().any():
        raw = raw.dissolve(by='adm_id', as_index=False)
    return raw[['adm_id','geometry']].to_crs('EPSG:4326')


def candidates_for_level(sources: gpd.GeoDataFrame, adm: gpd.GeoDataFrame,
                         level: str, top_k: int, max_distance_km: float) -> pd.DataFrame:
    if sources.empty:
        return pd.DataFrame()
    # Equal-area projected units are metres; values here are approximate geographic distances.
    # Point-to-polygon distance, not station-to-polygon-centroid distance.
    sm = sources.to_crs('EPSG:6933')
    am = adm.to_crs('EPSG:6933')
    records = []
    for src in sm.itertuples():
        distances = am.geometry.distance(src.geometry) / 1000.0
        nearest = distances.nsmallest(min(top_k, len(am)))
        for rank, (index, km) in enumerate(nearest.items(), 1):
            if km > max_distance_km:
                continue
            records.append({
                'source_type': src.source_type,
                'source_id': src.source_id,
                'level': level,
                'adm_id': str(am.loc[index, 'adm_id']),
                'rank_by_distance': rank,
                'distance_to_polygon_km_approx': round(float(km), 2),
                'station_inside_polygon': bool(km < 1e-6),
                'candidate_method': 'GEOGRAPHIC_PROXIMITY_ONLY',
                'hydrological_connection_verified': False,
                'review_status': 'PENDING_HYDROLOGICAL_REVIEW',
                'relationship': '',
                'evidence_source': '',
                'reviewer_notes': '',
            })
    return pd.DataFrame(records, columns=[
        'source_type','source_id','level','adm_id','rank_by_distance',
        'distance_to_polygon_km_approx','station_inside_polygon','candidate_method',
        'hydrological_connection_verified','review_status','relationship',
        'evidence_source','reviewer_notes'])


def plot_sources(adm2: gpd.GeoDataFrame, adm3: gpd.GeoDataFrame,
                 src: gpd.GeoDataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(10,11), dpi=160)
    adm2.plot(ax=ax, facecolor='#edf3f7', edgecolor='#657688', linewidth=.8)
    adm3.boundary.plot(ax=ax, color='#b7c4d0', linewidth=.2)
    styles = {'dfo': ('#ec8840','^'), 'lake': ('#1565a8','o')}
    for t, (color,marker) in styles.items():
        part = src[src.source_type == t]
        if len(part):
            part.plot(ax=ax, color=color, marker=marker, markersize=40, label=t.upper(), zorder=5)
            for row in part.itertuples():
                ax.annotate(str(row.source_id), (row.geometry.x,row.geometry.y),
                            fontsize=6, xytext=(3,3), textcoords='offset points')
    ax.set_title('Lake and DFO station locations vs South Sudan administrative boundaries')
    ax.set_xlabel('Longitude (degrees East)')
    ax.set_ylabel('Latitude (degrees North)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.text(.01,.01,'Map shows location only; NOT evidence of upstream hydrological connectivity.',
            transform=ax.transAxes,fontsize=8,va='bottom',
            bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=None,help='Repo root (auto-detect from script path by default)')
    parser.add_argument('--adm2-boundaries',default=None)
    parser.add_argument('--adm3-boundaries',default=None)
    parser.add_argument('--adm2-id-column',default=None)
    parser.add_argument('--adm3-id-column',default=None)
    parser.add_argument('--extra-locations-csv',default=None,
                        help='Optional manually verified source_type,source_id,latitude,longitude CSV')
    parser.add_argument('--nearest',type=int,default=5)
    parser.add_argument('--max-distance-km',type=float,default=400)
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    root=Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    out=Path(args.output).resolve() if args.output else root / OUTPUT_DIR
    out.mkdir(parents=True,exist_ok=True)
    adm2_file=geometry_file(root,'ADM2',args.adm2_boundaries)
    adm3_file=geometry_file(root,'ADM3',args.adm3_boundaries)
    adm2=load_admin(adm2_file,'ADM2',args.adm2_id_column)
    adm3=load_admin(adm3_file,'ADM3',args.adm3_id_column)
    loc,warnings_df=station_locations(root)
    if args.extra_locations_csv:
        f=Path(args.extra_locations_csv)
        f=f if f.is_absolute() else root/f
        custom=pd.read_csv(f,dtype={'source_id':str})
        mandatory={'source_type','source_id','latitude','longitude'}
        if not mandatory.issubset(custom.columns):
            raise ValueError(f'Extra locations needs {mandatory}')
        custom=custom[list(mandatory)].copy()
        custom['location_status']='provided'
        custom['coordinate_origin']='user_supplied_verified_location'
        loc=pd.concat([loc,custom],ignore_index=True).drop_duplicates(['source_type','source_id'],keep='last')
    loc.to_csv(out/'01_hydrology_source_locations.csv',index=False)
    warnings_df.to_csv(out/'00_location_warnings.csv',index=False)
    valid=loc.dropna(subset=['latitude','longitude']).copy()
    valid=valid[valid.latitude.between(-90,90)&valid.longitude.between(-180,180)]
    if valid.empty:
        raise ValueError('No valid lake/DFO coordinates available')
    src=gpd.GeoDataFrame(valid,geometry=gpd.points_from_xy(valid.longitude,valid.latitude),crs='EPSG:4326')
    p2=candidates_for_level(src,adm2,'ADM2',args.nearest,args.max_distance_km)
    p3=candidates_for_level(src,adm3,'ADM3',args.nearest,args.max_distance_km)
    p2.to_csv(out/'02_adm2_geographic_candidates_REVIEW.csv',index=False)
    p3.to_csv(out/'03_adm3_geographic_candidates_REVIEW.csv',index=False)
    src.to_file(out/'04_source_locations.geojson',driver='GeoJSON')
    plot_sources(adm2,adm3,src,out/'05_station_lake_locations_map.png')
    # DO NOT output files that 06a or 06b might mistake for an approved mapping.
    summary={
        'purpose':'GEOGRAPHIC REVIEW ONLY. No verified source-to-administration linkage is generated.',
        'adm2_file':str(adm2_file),'adm3_file':str(adm3_file),
        'adm2_regions':len(adm2),'adm3_regions':len(adm3),
        'sources_geolocated':len(src),'sources_unlocated':len(loc)-len(src),
        'adm2_candidate_rows':len(p2),'adm3_candidate_rows':len(p3),
        'distance_units':'approximate kilometres in EPSG:6933 (point to polygon)',
        'water_routing_verified':False,
        'approved_mappings_created':0,
        'required_next_step':'Cross-check sources against river network and flow direction, document evidence, THEN explicitly approve each region-station relationship before modifying 06a mapping template.',
    }
    (out/'methodology.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('Output:',out)

if __name__=='__main__':
    main()
