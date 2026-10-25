#!/usr/bin/env python3
"""Verify local HydroBASINS/HydroRIVERS structure without loading full Africa geometries.

Place in scripts/multivariate_flood_modeling and run directly in VS Code.
Outputs: processed_data/multivariate_flood_modeling/06e_hydro_data_check/
This script is read-only with respect to raw data and does not infer connectivity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd


def root_from_script() -> Path:
    script = Path(__file__).resolve()
    if script.parent.name == 'multivariate_flood_modeling' and script.parent.parent.name == 'scripts':
        return script.parent.parent.parent
    return Path.cwd().resolve()


def choose_files(root: Path, level: int) -> list[Path]:
    base = root / 'raw_data' / 'hybas_af_lev01-12_v1c'
    # Prefer expected exact level; walk nested folders, too.
    return sorted(base.rglob(f'hybas_af_lev{level:02d}_v1c.shp')) if base.exists() else []


def read_sample(file: Path, limit: int = 3) -> dict:
    result = {'path': str(file), 'status': 'OK', 'crs': None, 'columns': [], 'rows_sampled': 0,
              'sample_attributes': [], 'missing_companions': []}
    for ext in ('.dbf', '.shx', '.prj'):
        if not file.with_suffix(ext).exists():
            result['missing_companions'].append(file.with_suffix(ext).name)
    try:
        frame = gpd.read_file(file, rows=limit)
        result['crs'] = str(frame.crs)
        result['columns'] = [str(c) for c in frame.columns]
        result['rows_sampled'] = len(frame)
        result['sample_attributes'] = frame.drop(columns=['geometry'], errors='ignore').astype(str).to_dict('records')
        for col in ('HYBAS_ID', 'NEXT_DOWN'):
            if col not in frame.columns:
                result['status'] = 'REVIEW_MISSING_TOPOLOGY_COLUMNS'
    except Exception as exc:
        result['status'] = 'ERROR'
        result['error'] = str(exc)
    return result


def find_gdb(root: Path) -> list[Path]:
    base = root / 'raw_data' / 'HydroRIVERS_v10_af'
    if not base.exists():
        return []
    return sorted(p for p in base.rglob('*') if p.is_dir() and p.suffix.lower() == '.gdb')


def inspect_rivers(root: Path) -> dict:
    base = root / 'raw_data' / 'HydroRIVERS_v10_af'
    output = {'type': 'HydroRIVERS', 'directory': str(base), 'status': 'NOT_FOUND', 'geodatabases': [],
              'shapefiles': [], 'layers': [], 'sample_attributes': [], 'columns': []}
    if not base.exists():
        return output
    output['geodatabases'] = [str(p) for p in find_gdb(root)]
    output['shapefiles'] = [str(p) for p in sorted(base.rglob('*.shp'))[:10]]
    files = [Path(p) for p in output['geodatabases']] + [Path(p) for p in output['shapefiles']]
    if not files:
        output['status'] = 'NO_READABLE_SPATIAL_DATA'
        output['example_files'] = [str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()][:20]
        return output
    spatial_path = files[0]
    try:
        import pyogrio
        layers = pyogrio.list_layers(spatial_path)
        output['layers'] = [str(x[0]) for x in layers]
        layer = next((str(x[0]) for x in layers if str(x[1]).lower() in ('linestring', 'multilinestring')), str(layers[0][0]))
        frame = gpd.read_file(spatial_path, layer=layer, rows=3)
        output['status'] = 'OK'
        output['spatial_file'] = str(spatial_path)
        output['selected_layer'] = layer
        output['crs'] = str(frame.crs)
        output['columns'] = list(frame.columns)
        output['sample_attributes'] = frame.drop(columns='geometry', errors='ignore').astype(str).to_dict('records')
    except Exception as exc:
        output['status'] = 'ERROR'
        output['error'] = str(exc)
    return output


def main() -> None:
    root = root_from_script()
    out = root / 'processed_data' / 'multivariate_flood_modeling' / '06e_hydro_data_check'
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for level in range(1, 13):
        matches = choose_files(root, level)
        if not matches:
            item = {'level': level, 'status': 'NOT_FOUND', 'path': None, 'columns': [], 'rows_sampled': 0}
        else:
            item = read_sample(matches[0])
            item['level'] = level
            if len(matches) > 1:
                item['other_matches'] = [str(p) for p in matches[1:]]
        summaries.append(item)
        print(f"HydroBASINS level {level:02d}: {item['status']}")
        if item.get('status') not in ('NOT_FOUND', 'ERROR'):
            print('  CRS:', item.get('crs'))
            print('  Columns:', ', '.join(item['columns']))
            if item.get('missing_companions'):
                print('  Missing companions:', item['missing_companions'])
        if item.get('status') == 'ERROR':
            print('  Error:', item.get('error'))
    rivers = inspect_rivers(root)
    print('\nHydroRIVERS:', rivers['status'])
    if rivers['status'] == 'OK':
        print('  Spatial file:', rivers['spatial_file'])
        print('  Layer:', rivers['selected_layer'])
        print('  CRS:', rivers['crs'])
        print('  Columns:', ', '.join(rivers['columns']))
    elif rivers.get('error'):
        print('  Error:', rivers['error'])

    (out / 'hydro_data_inventory.json').write_text(
        json.dumps({'project_root': str(root), 'hydrobasins': summaries, 'hydrorivers': rivers},
                   ensure_ascii=False, indent=2), encoding='utf-8'
    )
    pd.DataFrame([{k: item.get(k) for k in ('level', 'status', 'path', 'crs', 'rows_sampled')}
                  for item in summaries]).to_csv(out / 'hydrobasins_levels_summary.csv', index=False)
    print('\nReports saved to:', out)
    print('This is a data-structure check, not a validated hydrological connectivity mapping.')


if __name__ == '__main__':
    main()
