#!/usr/bin/env python3
"""JBG060 / 06f: exploratory HydroBASINS + HydroRIVERS connectivity.

No automatic hydrological approvals or ADM forecasting assignments are made.
Outputs are CANDIDATES for expert review, not an operational routing model.

Designed for the user's directories:
  raw_data/HydroRIVERS_v10_af/
  raw_data/hybas_af_lev01-12_v1c/
  raw_data/administrative_boundaries/
  processed_data/multivariate_flood_modeling/06c_hydrological_spatial_mapping/

Usage: python scripts/multivariate_flood_modeling/06f_build_river_basin_connectivity.py
Optional: --basin-level 08 --max-snap-km 20
Dependencies: geopandas shapely pandas numpy matplotlib pyogrio (recommended)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

WGS84 = 'EPSG:4326'
METRIC = 'EPSG:6933'  # area calculations; approximate distances
LEVEL = '08'


def root_default():
    p = Path(__file__).resolve()
    return p.parents[2] if p.parent.name == 'multivariate_flood_modeling' else Path.cwd()


def locate_file(directory, *, include, exclude=()):
    """Locate geospatial datasets including ESRI FileGDB directories."""
    if not directory.exists():
        raise FileNotFoundError(f'Directory does not exist: {directory}')
    vector_ext = {'.shp', '.gpkg', '.geojson'}
    items = [p for p in directory.rglob('*')
             if (p.is_file() and p.suffix.lower() in vector_ext)
             or (p.is_dir() and p.suffix.lower() == '.gdb')]
    matches = [p for p in items if all(x in p.stem.lower() for x in include)
               and not any(x in p.stem.lower() for x in exclude)]
    if not matches and len(items) == 1:
        matches = items
        print(f'NOTE: using sole available vector dataset: {items[0]}')
    if not matches:
        archives = [p.name for p in directory.rglob('*') if p.is_file()
                    and p.suffix.lower() in {'.zip','.7z','.rar','.tar','.gz'}]
        listing = [p.name for p in directory.iterdir()]
        raise FileNotFoundError(
            f'No vector dataset matching {include} found in {directory}. '
            f'Contents: {listing[:20]}. '
            f'Found vector datasets: {[str(p) for p in items[:12]]}. '
            f'Archives: {archives[:8]}. For FileGDB, extract the entire .gdb directory.'
        )
    matches.sort(key=lambda p: (0 if p.suffix.lower()=='.gdb' else 1, len(str(p))))
    return matches[0]


def hydro_files(root, level):
    river_dir = root / 'raw_data/HydroRIVERS_v10_af'
    basin_dir = root / 'raw_data/hybas_af_lev01-12_v1c'
    river = locate_file(river_dir, include=('hydrorivers',))
    basin = locate_file(basin_dir, include=('hybas', f'lev{level}'))
    return river, basin


def adm_file(root, level):
    root = root / 'raw_data/administrative_boundaries'
    files = [f for f in root.rglob('*') if f.suffix.lower() in {'.geojson', '.gpkg', '.shp'}
             and (f'adm{level}' in f.stem.lower() or f'admin{level}' in f.stem.lower())]
    if not files:
        raise FileNotFoundError(f'ADM{level} boundary file not found in {root}')
    files.sort(key=lambda f: (0 if f.suffix.lower() == '.geojson' else 1, len(str(f))))
    return files[0]


def load_adm(root, level):
    path = adm_file(root, level)
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError(f'ADM{level} missing CRS: {path}')
    name = next((c for c in [f'adm{level}_pcode', f'ADM{level}_PCODE', 'adm_id'] if c in g), None)
    if name is None:
        raise ValueError(f'No ADM{level} ID column in {path}: {list(g.columns)}')
    g = g[[name, 'geometry']].rename(columns={name:'adm_id'}).dropna(subset=['geometry']).copy()
    g['adm_id'] = g['adm_id'].astype(str).str.strip()
    if g.adm_id.duplicated().any():
        g = g.dissolve(by='adm_id', as_index=False)
    return g.to_crs(WGS84), path


def load_sources(root):
    f = root / 'processed_data/multivariate_flood_modeling/06c_hydrological_spatial_mapping/01_hydrology_source_locations.csv'
    if not f.exists():
        raise FileNotFoundError(f'Missing 06c source list: {f}. Run 06c first.')
    d = pd.read_csv(f, dtype={'source_id':str})
    for col in ['latitude', 'longitude']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d[d.latitude.between(-90,90) & d.longitude.between(-180,180)].copy()
    d['source_type'] = d.source_type.astype(str).str.lower().str.strip()
    d['source_id'] = d.source_id.astype(str).str.strip()
    d = d.drop_duplicates(['source_type','source_id'])
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.longitude, d.latitude), crs=WGS84)


def select_rivers(path, bbox):
    g = gpd.read_file(path, bbox=bbox)
    if g.crs is None:
        raise ValueError(f'River CRS missing in {path}')
    g = g.to_crs(WGS84)
    required = {'HYRIV_ID','NEXT_DOWN'}
    if not required.issubset(g.columns):
        raise ValueError(f'HydroRIVERS topology columns unavailable: {required - set(g.columns)}. Found {list(g.columns)}')
    return g


def select_basins(path, bbox):
    # Read whole level for downstream tracing; bbox-cropping would discard parent links.
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError(f'Basin CRS missing in {path}')
    g = g.to_crs(WGS84)
    required = {'HYBAS_ID','NEXT_DOWN'}
    if not required.issubset(g.columns):
        raise ValueError(f'HydroBASINS topology columns unavailable: {required - set(g.columns)}. Found {list(g.columns)}')
    return g


def geom_nearest(point, layer_metric):
    if layer_metric.empty:
        return None, float('nan')
    # Spatial index queries only nearby candidates, not every Africa geometry.
    idx = layer_metric.sindex.nearest(point, return_all=True)
    ids = sorted(set(map(int, idx[1])))
    if not ids:
        return None, float('nan')
    ds = layer_metric.iloc[ids].geometry.distance(point)
    best = ds.idxmin()
    return best, float(ds.loc[best] / 1000.0)


def as_int(value):
    try:
        if pd.isna(value): return None
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return None


def walk_downstream(start, edges, max_hops=200):
    """Within SAME basin level, follow NEXT_DOWN; stop on missing/cycle/0."""
    visited = {start}
    chain = [start]
    reason = 'max_hops'
    for _ in range(max_hops):
        nxt = edges.get(chain[-1])
        if nxt in {None, 0, -1}: reason = 'outlet_or_missing'; break
        if nxt == chain[-1] or nxt in visited: reason = 'cycle'; break
        if nxt not in edges: reason = 'outside_dataset_or_missing'; break
        chain.append(nxt)
        visited.add(nxt)
    return chain, reason


def river_snaps(sources, rivers, max_km):
    rm = rivers.to_crs(METRIC)
    sm = sources.to_crs(METRIC)
    rows = []
    for s in sm.itertuples():
        idx, km = geom_nearest(s.geometry, rm)
        r = rm.loc[idx] if idx is not None else None
        rows.append({
            'source_type': s.source_type, 'source_id':s.source_id,
            'nearest_hyriv_id': as_int(r['HYRIV_ID']) if r is not None else None,
            'river_next_down': as_int(r['NEXT_DOWN']) if r is not None else None,
            'river_dist_km_approx': round(km, 3) if math.isfinite(km) else None,
            'river_snap_status': ('REVIEW_NEARBY_RIVER' if math.isfinite(km) and km <= max_km else 'UNMATCHED_OR_DISTANT'),
            'river_matching_basis': 'NEAREST_RIVER_GEOMETRY_ONLY_NOT_VERIFIED',
        })
    return pd.DataFrame(rows)


def basin_assign(sources, basins):
    # Use within-intersect then nearest fallback; fallback is NOT a topology-verified match.
    b = basins.to_crs(METRIC)
    s = sources.to_crs(METRIC)
    rows=[]
    for item in s.itertuples():
        hits = list(b.sindex.query(item.geometry, predicate='intersects'))
        if hits:
            idx = int(hits[0]); km=0.0; status='POINT_INTERSECTS_BASIN'
        else:
            idx, km = geom_nearest(item.geometry, b)
            status='NEAREST_BASIN_FALLBACK'
        b_id = as_int(b.iloc[idx]['HYBAS_ID']) if idx is not None else None
        rows.append({'source_type':item.source_type, 'source_id':item.source_id,
                     'source_hybas_id':b_id, 'basin_assignment_status':status,
                     'distance_to_basin_km_approx':round(km,3) if math.isfinite(km) else None})
    return pd.DataFrame(rows)


def adm_basin_overlaps(adm, basins, level, min_fraction=0.01):
    # Spatial join prunes Africa-wide basin polygons to those intersecting South Sudan.
    a = adm.to_crs(METRIC)
    b = basins.to_crs(METRIC)
    matches = gpd.sjoin(a, b[['HYBAS_ID','geometry']], predicate='intersects', how='inner')
    if matches.empty:
        return pd.DataFrame(columns=['level','adm_id','hybas_id','area_fraction'])
    rows=[]
    for r in matches.itertuples():
        basin_geom = b.loc[r.index_right, 'geometry']
        frac = float(r.geometry.intersection(basin_geom).area / r.geometry.area) if r.geometry.area > 0 else 0
        if frac >= min_fraction:
            rows.append({'level':level, 'adm_id':str(r.adm_id), 'hybas_id':as_int(r.HYBAS_ID),
                         'area_fraction':round(frac,5)})
    return pd.DataFrame(rows,columns=['level','adm_id','hybas_id','area_fraction'])


def make_candidates(assignments, overlaps, edges, max_hops):
    if overlaps.empty: return pd.DataFrame()
    # source basin + downstream basins. These are topology candidates, not confirmed impact.
    by_basin = {bid: chunk for bid, chunk in overlaps.groupby('hybas_id')}
    rows=[]
    for s in assignments.itertuples():
        if pd.isna(s.source_hybas_id): continue
        start=int(s.source_hybas_id)
        chain, stop=walk_downstream(start,edges,max_hops)
        for hop,bid in enumerate(chain):
            if bid not in by_basin: continue
            for r in by_basin[bid].itertuples():
                rows.append({'source_type':s.source_type,'source_id':s.source_id,
                    'source_hybas_id':start,'target_hybas_id':bid,'level':r.level,'adm_id':r.adm_id,
                    'downstream_hops':hop,'adm_area_fraction_in_basin':r.area_fraction,
                    'basin_assignment_status':s.basin_assignment_status,
                    'chain_stop_reason':stop,
                    'candidate_basis':'HYDROBASINS_NEXT_DOWN_SAME_LEVEL',
                    'review_status':'PENDING_NETWORK_AND_GAUGE_VALIDATION',
                    'hydrological_connection_verified':False,
                    'predictor_use':'DO_NOT_USE_UNTIL_REVIEWED'})
    return pd.DataFrame(rows).drop_duplicates(['source_type','source_id','level','adm_id','target_hybas_id']) if rows else pd.DataFrame(columns=[
        'source_type','source_id','source_hybas_id','target_hybas_id','level','adm_id','downstream_hops',
        'adm_area_fraction_in_basin','basin_assignment_status','chain_stop_reason','candidate_basis',
        'review_status','hydrological_connection_verified','predictor_use'])


def plot_review(adm2, rivers, basins, sources, output):
    # Clip display to South Sudan + upstream lakes; never render Africa-wide full geometry.
    minx, miny, maxx, maxy = adm2.total_bounds
    x0=min(minx, float(sources.longitude.min()))-1
    y0=min(miny, float(sources.latitude.min()))-1
    x1=max(maxx, float(sources.longitude.max()))+1
    y1=max(maxy, float(sources.latitude.max()))+1
    fig,ax=plt.subplots(figsize=(12,9))
    subset_basins = basins.cx[x0:x1,y0:y1]
    subset_rivers = rivers.cx[x0:x1,y0:y1]
    if len(subset_basins): subset_basins.boundary.plot(ax=ax,color='#c6c4c0',linewidth=.3)
    if len(subset_rivers): subset_rivers.plot(ax=ax,color='#439bc4',linewidth=.45,alpha=.55)
    adm2.boundary.plot(ax=ax,color='#404a5a',linewidth=.75)
    for t,color,marker in [('dfo','#d35400','^'),('lake','#1462a4','o')]:
        part=sources[sources.source_type==t]
        if len(part):
            part.plot(ax=ax,color=color,marker=marker,markersize=42,label=t.upper(),zorder=5)
    ax.set_xlim(x0,x1); ax.set_ylim(y0,y1)
    ax.set_title('South Sudan: HydroBASINS / HydroRIVERS / hydrology sources (review map)')
    ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
    ax.legend();ax.set_aspect('equal')
    fig.text(.5,.02,'Visual comparison only; topology candidates require watercourse and data-timing validation.',ha='center',fontsize=9)
    fig.savefig(output,dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(root_default()))
    p.add_argument('--basin-level',choices=[f'{i:02d}' for i in range(1,13)],default=LEVEL)
    p.add_argument('--max-snap-km',type=float,default=20.0)
    p.add_argument('--max-downstream-hops',type=int,default=150)
    p.add_argument('--min-adm-overlap-fraction',type=float,default=0.01)
    p.add_argument('--output',default='processed_data/multivariate_flood_modeling/06f_river_basin_connectivity')
    args=p.parse_args()
    root=Path(args.root).resolve()
    out=Path(args.output)
    if not out.is_absolute():out=root/out
    out.mkdir(parents=True,exist_ok=True)

    src=load_sources(root)
    adm2,path2=load_adm(root,2)
    adm3,path3=load_adm(root,3)
    river_file,basin_file=hydro_files(root,args.basin_level)
    print('River file:',river_file)
    print('Basin file:',basin_file)
    # Read near South Sudan and observed stations, including upstream Uganda.
    bounds=adm2.total_bounds
    bbox=(min(bounds[0],float(src.longitude.min()))-2,
          min(bounds[1],float(src.latitude.min()))-2,
          max(bounds[2],float(src.longitude.max()))+2,
          max(bounds[3],float(src.latitude.max()))+2)
    print('Reading HydroRIVERS in regional window',bbox)
    rivers=select_rivers(river_file,bbox)
    print('Reading basin level',args.basin_level,'across Africa (needed for NEXT_DOWN links)')
    basins=select_basins(basin_file,bbox)
    print('River segments read:',len(rivers),'Basin polygons read:',len(basins))

    snaps=river_snaps(src,rivers,args.max_snap_km)
    assign=basin_assign(src,basins)
    edges={as_int(r.HYBAS_ID):as_int(r.NEXT_DOWN) for r in basins[['HYBAS_ID','NEXT_DOWN']].itertuples(index=False)}
    overlaps=pd.concat([
        adm_basin_overlaps(adm2,basins,'ADM2',args.min_adm_overlap_fraction),
        adm_basin_overlaps(adm3,basins,'ADM3',args.min_adm_overlap_fraction),
    ],ignore_index=True)
    candidates=make_candidates(assign,overlaps,edges,args.max_downstream_hops)
    snaps.to_csv(out/'01_source_river_snaps_REVIEW.csv',index=False)
    assign.to_csv(out/'02_source_basin_assignments_REVIEW.csv',index=False)
    overlaps.to_csv(out/'03_admin_basin_overlaps.csv',index=False)
    candidates.to_csv(out/'04_downstream_admin_candidates_REVIEW.csv',index=False)
    plot_review(adm2,rivers,basins,src,out/'05_river_basin_sources_map.png')

    summary={
        'river_file':str(river_file),'basin_file':str(basin_file),
        'adm2_file':str(path2),'adm3_file':str(path3),
        'basin_level':args.basin_level,'river_segments_in_read_window':len(rivers),
        'africa_basin_count':len(basins),'sources_geolocated':len(src),
        'source_river_snap_candidates':int((snaps.river_snap_status=='REVIEW_NEARBY_RIVER').sum()),
        'adm_basin_overlap_rows':len(overlaps),'source_downstream_adm_candidate_rows':len(candidates),
        'approved_mappings':0,
        'disclaimer':'Unverified watercourse positions/flow network/candidate land intersections. No linkage should be used as a predictive feature until reviewed. A hydrobasin downstream chain is a coarse basin topology, not evidence of flood propagation or forecast lag.',
        'next_action':'Review river snaps against original DFO metadata, trace basin and river network, then explicitly approve links for ADM feature construction. Never automatically feed candidate table to 06b.',
    }
    (out/'06_methodology.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('Output:',out)

if __name__=='__main__': main()
