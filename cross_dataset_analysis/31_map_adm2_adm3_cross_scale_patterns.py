"""Script 31 — map Script 30 finalized cross-scale patterns.

Merge boundaries using authoritative administrative PCODEs, not names.
No row-order ID fabrication. Missing/ambiguous PCODEs fail with QC evidence.
"""
from pathlib import Path
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path('/Users/shibai/Documents/GitHub/DBL-Group-22')
BASE = ROOT / 'processed_data/cross_dataset_analysis/04_cross_scale'
IN_DIR = BASE / '30_compare_adm2_adm3_patterns/tables'
OUT = BASE / '31_map_adm2_adm3_cross_scale_patterns'
FIG = OUT / 'figures'
TABLE = OUT / 'tables'
for d in (FIG, TABLE):
    d.mkdir(parents=True, exist_ok=True)
BOUND = ROOT / 'raw_data/administrative_boundaries'


def code_series(gdf, level):
    """Find a genuine PCODE-valued boundary field; never manufacture IDs by row order."""
    length = 4 if level == 'ADM2' else 6
    pattern = rf'SS\d{{{length}}}'
    candidates = [
        f'{level}_ID', f'{level}_PCODE', f'{level}Pcode', f'{level}_CODE',
        f'adm{level[-1]}_pcode', f'adm{level[-1]}_id',
        f'ADM{level[-1]}_PCODE', 'PCODE', 'pcode', 'id', 'ID',
    ]
    ordered = []
    for candidate in candidates:
        ordered.extend(c for c in gdf.columns if c.casefold() == candidate.casefold() and c not in ordered)
    # Also allow unusual source column names, but only with full-value PCODE validation.
    ordered.extend(c for c in gdf.columns if c not in ordered and c != 'geometry')
    for c in ordered:
        values = gdf[c].astype('string').str.strip().str.upper()
        if values.notna().all() and values.str.fullmatch(pattern).all() and values.is_unique:
            print(f'Using {level} boundary PCODE column: {c}')
            return values
    raise ValueError(
        f'Could not locate a unique {level} PCODE column matching {pattern} '
        f'in {level} boundary. Available columns: {list(gdf.columns)}. '
        'Check boundary schema instead of matching by inconsistent names.'
    )


def assert_unique(frame, field, label):
    if frame[field].isna().any() or frame[field].duplicated().any():
        raise ValueError(f'{label}: missing or duplicate {field} values')


def load():
    cross = pd.read_csv(IN_DIR / '30_01_adm3_cross_scale_comparison.csv', dtype={'ADM3_ID':'string', 'PARENT_ADM2_ID':'string'})
    summary = pd.read_csv(IN_DIR / '30_02_adm2_cross_scale_summary.csv')
    a3 = gpd.read_file(BOUND / 'ssd_admin3.geojson')
    a2 = gpd.read_file(BOUND / 'ssd_admin2.geojson')
    if a3.crs is None or a2.crs is None:
        raise ValueError('Missing boundary CRS')
    a3 = a3.to_crs('EPSG:4326')
    a2 = a2.to_crs('EPSG:4326')
    a3['_PCODE'] = code_series(a3, 'ADM3')
    a2['_PCODE'] = code_series(a2, 'ADM2')
    cross['_PCODE'] = cross['ADM3_ID'].astype('string').str.strip().str.upper()
    assert_unique(cross, '_PCODE', 'Script 30 ADM3 results')
    assert_unique(a3, '_PCODE', 'ADM3 boundaries')
    assert_unique(a2, '_PCODE', 'ADM2 boundaries')
    if len(cross) != 512 or len(a3) != 512 or len(a2) != 79 or len(summary) != 79:
        raise ValueError(f'Unexpected counts: cross={len(cross)}, boundary ADM3={len(a3)}, boundary ADM2={len(a2)}, summary={len(summary)}')
    # Ensure Script 30 hierarchy agrees with PCODE prefix.
    if 'PARENT_ADM2_ID' not in cross.columns:
        raise ValueError('Script 30 output lacks PARENT_ADM2_ID. Run updated Script 30 first.')
    parent = cross['PARENT_ADM2_ID'].astype('string').str.strip().str.upper()
    if not (parent == cross['_PCODE'].str[:6]).all():
        raise ValueError('Script 30 parent PCODE does not match ADM3 PCODE prefix')
    # Keep boundary geometry and PCODE only: avoid pandas _x/_y suffix collisions
    # with Script 30 analytical columns (especially flag_*).
    a3_geometry = a3[['_PCODE', 'geometry']].copy()
    m3 = a3_geometry.merge(cross, on='_PCODE', how='left', validate='one_to_one', indicator=True)
    expected_flags = [
        'flag_local_expansion_nonexpanding_parent',
        'flag_regional_expansion_local_persistence',
        'flag_regional_persistent_local_expansion',
        'flag_regional_low_local_nonlow_signal',
    ]
    absent = [f for f in expected_flags if f not in m3.columns]
    if absent:
        print('Available Script 30 columns:', list(cross.columns))
        raise ValueError(f'Script 30 input lacks these required fields: {absent}')
    m3.loc[m3['_merge']!='both', ['_PCODE']].to_csv(TABLE / '31_unmatched_adm3_geometries.csv', index=False)
    if (m3['_merge']!='both').any() or len(m3) != len(cross):
        raise ValueError('ADM3 PCODE merge incomplete: see 31_unmatched_adm3_geometries.csv')
    # Script 30 ADM2 summary uses names but no code: extract unique PCODE/name crosswalk.
    parents = cross[['PARENT_ADM2_ID', 'ADM2_NAME_ADM2']].drop_duplicates()
    if parents['PARENT_ADM2_ID'].duplicated().any():
        raise ValueError('Multiple Script 30 ADM2 names per parent PCODE')
    name_to_id = parents.rename(columns={'ADM2_NAME_ADM2':'ADM2_NAME', 'PARENT_ADM2_ID':'_PCODE'})
    if name_to_id['ADM2_NAME'].duplicated().any():
        raise ValueError('Nonunique ADM2 names in Script 30 crosswalk')
    summary = summary.merge(name_to_id, on='ADM2_NAME', how='left', validate='one_to_one')
    if summary['_PCODE'].isna().any():
        summary.loc[summary['_PCODE'].isna()].to_csv(TABLE / '31_unmatched_adm2_summary.csv',index=False)
        raise ValueError('Some ADM2 summary names are missing from Script 30 code crosswalk')
    assert_unique(summary, '_PCODE', 'Script 30 ADM2 summary')
    a2_geometry = a2[['_PCODE', 'geometry']].copy()
    m2 = a2_geometry.merge(summary, on='_PCODE', how='left', validate='one_to_one', indicator=True)
    m2.loc[m2['_merge']!='both', ['_PCODE']].to_csv(TABLE/'31_unmatched_adm2_geometries.csv',index=False)
    if (m2['_merge']!='both').any() or len(m2)!=len(summary):
        raise ValueError('ADM2 PCODE merge incomplete: see QC CSV')
    print(f'ADM3 boundary match: {len(m3)}/{len(cross)}; ADM2 boundary match: {len(m2)}/{len(summary)}')
    print('Script 30 cross-scale flags verified:', ', '.join(expected_flags))
    return m3, m2


def backdrop(ax,a3,a2):
    a3.plot(ax=ax,color='#eaeaea',edgecolor='#f9f9f9',linewidth=.25)
    a2.boundary.plot(ax=ax,color='#505050',linewidth=.5)
    ax.set_axis_off(); ax.set_aspect('equal')


def save(fig,filename):
    fig.savefig(FIG/filename,dpi=300,bbox_inches='tight'); plt.close(fig)


def category_map(a3,a2):
    counts=a3['cross_scale_relationship'].value_counts()
    labels=sorted(counts.index.tolist())
    cmap=plt.get_cmap('tab20',len(labels)); color={c:cmap(i) for i,c in enumerate(labels)}
    fig,ax=plt.subplots(figsize=(14,10)); backdrop(ax,a3,a2)
    for name in labels:
        a3.loc[a3['cross_scale_relationship']==name].plot(ax=ax,color=color[name],edgecolor='white',linewidth=.1)
    a2.boundary.plot(ax=ax,color='#555555',linewidth=.5)
    ax.set_title('ADM2–ADM3 Flood Regime Relationships (Final Typology)')
    ax.legend(handles=[Patch(color=color[n],label=f'{n} (n={counts[n]})') for n in labels],
              bbox_to_anchor=(1.01,1),loc='upper left',fontsize=7,frameon=False)
    save(fig,'31_01_final_cross_scale_relationships.png')


def highlight(a3,a2,flag,title,filename):
    if flag not in a3.columns:
        raise ValueError(f'Missing Script 30 field: {flag}')
    fig,ax=plt.subplots(figsize=(12,9)); backdrop(ax,a3,a2)
    status=a3['ADM3_IS_STATUS'].astype(str).str.lower().isin(['true','1'])
    if status.any(): a3.loc[status].plot(ax=ax,color='#b8b8b8',edgecolor='white',linewidth=.15)
    sub=a3.loc[pd.to_numeric(a3[flag],errors='coerce').eq(1) & ~status]
    if not sub.empty: sub.plot(ax=ax,color='#a71930',edgecolor='white',linewidth=.3)
    a2.boundary.plot(ax=ax,color='#555555',linewidth=.5)
    ax.set_title(f'{title} (n={len(sub)})')
    ax.legend(handles=[Patch(color='#a71930',label='Flagged ADM3'),Patch(color='#eaeaea',label='Other classified ADM3'),Patch(color='#b8b8b8',label='Coverage/no detected record')],loc='lower left')
    save(fig,filename)
    return len(sub)


def entropy_map(a2):
    fig,ax=plt.subplots(figsize=(12,9))
    a2.plot(ax=ax,column='adm3_hydrological_pattern_entropy',cmap='YlOrRd',vmin=0,vmax=1,legend=True,
            edgecolor='white',linewidth=.45,missing_kwds={'color':'#cfcfcf','label':'No hydrological class'})
    ax.set_axis_off();ax.set_title('Within-ADM2 Diversity of ADM3 Flood Regimes (Normalized Entropy)')
    save(fig,'31_06_adm2_hydrological_entropy.png')


def main():
    a3,a2=load()
    category_map(a3,a2)
    cases=[
      ('flag_local_expansion_nonexpanding_parent','Local Expansion with Non-expanding Parent','31_02_local_expansion.png'),
      ('flag_regional_expansion_local_persistence','Regional Expansion + Local Persistence','31_03_regional_expansion_local_persistence.png'),
      ('flag_regional_persistent_local_expansion','Regional Persistence + Local Expansion','31_04_regional_persistence_local_expansion.png'),
      ('flag_regional_low_local_nonlow_signal','Regional Low/Stable + Local Flood Signal','31_05_regional_low_local_signal.png')]
    counts={f:highlight(a3,a2,f,t,n) for f,t,n in cases}
    entropy_map(a2)
    if len(a3) != 512 or len(a2) != 79: raise ValueError('Geometry coverage failed')
    pd.DataFrame([{'flag':f,'count':v} for f,v in counts.items()]).to_csv(TABLE/'31_map_qc.csv',index=False)
    pd.DataFrame(a3.drop(columns='geometry')).to_csv(TABLE/'31_adm3_map_data.csv',index=False)
    print('Script 31 completed:', FIG)

if __name__=='__main__': main()
