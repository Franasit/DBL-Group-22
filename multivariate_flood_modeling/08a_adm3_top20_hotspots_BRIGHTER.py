#!/usr/bin/env python3
"""South Sudan ADM3 flood hotspot map and parent ADM2 summary.

Uses EXISTING annual ADM3 flood indicators, NOT a manually created severity CSV.
This is a historical/descriptive analysis (2000-2025), NOT a forecast.

Default inputs (relative to project root):
  processed_data/multivariate_flood_modeling/02_adm3_panel_dataset/full_analysis_panel.csv
  raw_data/administrative_boundaries/ssd_admin3.geojson
  raw_data/administrative_boundaries/ssd_admin2.geojson

If the full_analysis_panel.csv is instead stored elsewhere, the script also
searches processed_data/ for that basename with suitable ADM3 columns.

Outputs under processed_data/multivariate_flood_modeling/08a_adm3_flood_hotspots/:
  01_adm3_historical_flood_severity.csv
  02_top_adm3_hotspots.csv
  03_top_hotspots_by_parent_adm2.csv
  04_all_high_severity_by_parent_adm2.csv
  05_adm3_hotspot_map.png
  06_methodology.json

Notes:
- 'Flood days' are the counts defined by your previously generated flood
  indicators; observational completeness is NOT independently verified.
- This map colors whole ADM3 polygons. It does NOT render pixel-level flood
  footprints like the VIIRS illustration.
- Severity categories are distribution-relative presentation groups, not
  formally validated disaster-risk thresholds.
"""
from __future__ import annotations

import argparse
import json
import math
import calendar
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

PALETTE = {
    'No recorded flood days': '#F4F0F8',
    'Moderate': '#7EC9F2',
    'High': '#C18AE0',
    'Very High': '#7546B7',
    'No historical data': '#CECECE',
}


def find_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == 'multivariate_flood_modeling' and p.parent.parent.name == 'scripts':
        return p.parents[2]
    return Path.cwd().resolve()


def find_panel(root: Path, explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.is_absolute() else root / candidate
    candidates = [
        root / 'processed_data/multivariate_flood_modeling/02_adm3_panel_dataset/full_analysis_panel.csv',
        root / 'processed_data/multivariate_flood_modeling/02_adm3_panel_dataset.csv',
        root / 'processed_data/cross_dataset_analysis/02_adm3_flood/10_build_adm3_flood_indicators/tables/adm3_annual_flood_indicators_combined.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in (root / 'processed_data').rglob('full_analysis_panel.csv'):
        try:
            cols = pd.read_csv(p, nrows=0).columns
            if 'flood__ADM3_NAME' in cols:
                return p
        except Exception:
            continue
    raise FileNotFoundError('ADM3 annual flood panel not found. Use --panel to provide its path.')


def column(df, options, what, required=True):
    exact = {c.lower(): c for c in df.columns}
    for option in options:
        if option in df.columns:
            return option
        if option.lower() in exact:
            return exact[option.lower()]
    if required:
        raise ValueError(f'Cannot find {what}. Tried {options}. Available: {list(df.columns)}')
    return None


def read_historical(panel: Path, first: int, last: int) -> pd.DataFrame:
    df = pd.read_csv(panel, low_memory=False)
    id_col = column(df, ['adm_id','ADM3_ID','ADM3_PCODE','adm3_pcode'], 'ADM3 code')
    year_col = column(df, ['year', 'Year'], 'year')
    days_col = column(df, ['flood__flood_days', 'flood_days'], 'annual flood days')
    adm3_name = column(df, ['flood__ADM3_NAME','ADM3_NAME','adm3_name'], 'ADM3 name', False)
    adm2_name = column(df, ['flood__ADM2_NAME','ADM2_NAME','adm2_name'], 'parent ADM2 name', False)
    adm2_id = column(df, ['flood__ADM2_ID', 'ADM2_ID','adm2_pcode'], 'parent ADM2 code', False)
    present = column(df, ['target_row_present'], 'target presence flag', False)

    clean = pd.DataFrame({
        'adm3_id': df[id_col].astype('string').str.strip(),
        'year': pd.to_numeric(df[year_col], errors='coerce'),
        'flood_days': pd.to_numeric(df[days_col], errors='coerce'),
        'adm3_name': df[adm3_name].astype('string') if adm3_name else pd.NA,
        'parent_adm2_name': df[adm2_name].astype('string') if adm2_name else pd.NA,
        'parent_adm2_id': df[adm2_id].astype('string') if adm2_id else pd.NA,
    })
    if present:
        valid = df[present].astype(str).str.lower().isin(['true','1','yes'])
        clean.loc[~valid, 'flood_days'] = np.nan
    clean = clean[clean['year'].between(first, last)].copy()
    clean = clean.dropna(subset=['adm3_id','year'])
    if clean.duplicated(['adm3_id','year']).any():
        duplicates = clean.loc[clean.duplicated(['adm3_id','year'], keep=False), ['adm3_id','year']]
        raise ValueError(f'Duplicate ADM3/year entries. Cannot aggregate ambiguously:\n{duplicates.head(10)}')
    if (clean['flood_days'] < 0).any() or (clean['flood_days'] > 366).any():
        raise ValueError('Flood days outside [0,366]; check upstream indicator calculation.')
    names = clean.sort_values('year').groupby('adm3_id', as_index=False).agg(
        adm3_name=('adm3_name', 'last'),
        parent_adm2_name=('parent_adm2_name', 'last'),
        parent_adm2_id=('parent_adm2_id', 'last'),
    )
    stats = clean.groupby('adm3_id', as_index=False).agg(
        mean_annual_flood_days=('flood_days','mean'),
        total_recorded_flood_days=('flood_days','sum'),
        observed_years=('flood_days','count'),
        years_with_recorded_flood_activity=('flood_days', lambda s: int((s > 0).sum())),
    )
    out = stats.merge(names, on='adm3_id', validate='one_to_one')
    out['expected_years'] = last - first + 1
    out['year_coverage_fraction'] = out['observed_years'] / out['expected_years']
    return out


def classify(stats: pd.DataFrame, min_fraction: float) -> tuple[pd.DataFrame, dict]:
    stats = stats.copy()
    stats['eligible'] = stats['year_coverage_fraction'].ge(min_fraction) & stats['observed_years'].gt(0)
    valid = stats.loc[stats['eligible'] & stats['mean_annual_flood_days'].gt(0), 'mean_annual_flood_days']
    if valid.empty:
        raise ValueError('No areas with positive annual flood-day counts and sufficient coverage.')
    # Positive observations only, so zero is not misleadingly called moderate.
    q66, q85 = valid.quantile([0.66, 0.85])
    def level(r):
        if not r['eligible']:
            return 'No historical data'
        x = r['mean_annual_flood_days']
        if x <= 0:
            return 'No recorded flood days'
        if x >= q85:
            return 'Very High'
        if x >= q66:
            return 'High'
        return 'Moderate'
    stats['severity_class'] = stats.apply(level, axis=1)
    return stats, {'positive_days_quantile_66': float(q66), 'positive_days_quantile_85': float(q85),
                   'min_year_coverage_fraction': float(min_fraction), 'classes_are_relative_not_alert_thresholds': True}



def read_monthly_seasonality(path: Path, first: int, last: int, top_months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate historical peak months from the REAL monthly event-date proxy.

    This is NOT reconstructed by dividing annual flood days into months.
    Each month score is the mean across available years of the number of
    dates with at least one recorded flood event in the ADM3.
    A missing observation is not automatically a zero flood month.
    """
    if not path.exists():
        raise FileNotFoundError(
            f'Monthly panel is missing: {path}. Run 05a_build_monthly_panel.py first.'
        )
    x = pd.read_csv(path, usecols=['adm_id', 'year', 'month', 'flood_days', 'target_status'], low_memory=False)
    x = x.rename(columns={'adm_id': 'adm3_id', 'flood_days': 'event_date_proxy'})
    x['adm3_id'] = x['adm3_id'].astype('string').str.strip()
    for c in ['year', 'month', 'event_date_proxy']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    x = x[x['year'].between(first, last) & x['month'].between(1, 12)].copy()
    if x.duplicated(['adm3_id', 'year', 'month']).any():
        raise ValueError('Duplicate ADM3/year/month records in monthly panel.')
    # Do not assume a panel zero means a verified absence of flooding.
    x.loc[x['event_date_proxy'].lt(0) | x['event_date_proxy'].gt(31), 'event_date_proxy'] = np.nan
    x['month'] = x['month'].astype(int)
    monthly = x.groupby(['adm3_id', 'month'], as_index=False).agg(
        mean_recorded_event_days=('event_date_proxy', 'mean'),
        years_with_month_data=('event_date_proxy', 'count'),
        total_recorded_event_days=('event_date_proxy', 'sum'),
    )
    month_names = dict(enumerate(calendar.month_abbr))
    monthly['month_name'] = monthly['month'].map(month_names)
    peaks = []
    for adm3_id, grp in monthly.groupby('adm3_id'):
        positive = grp[grp['years_with_month_data'].gt(0)].sort_values(
            ['mean_recorded_event_days', 'month'], ascending=[False, True]
        )
        top = positive.head(top_months)
        peaks.append({
            'adm3_id': adm3_id,
            'peak_months': ', '.join(top['month_name'].astype(str)),
            'peak_month_mean_event_days': ', '.join(
                f'{r.month_name}: {r.mean_recorded_event_days:.1f}' for r in top.itertuples()
            ),
            'peak_month_observation_note': 'Exploratory event-date proxy, not validated daily flood coverage',
        })
    return pd.DataFrame(peaks), monthly

def read_geo(file: Path, kind: str) -> gpd.GeoDataFrame:
    if not file.exists():
        raise FileNotFoundError(f'Missing {kind} boundary file: {file}')
    g = gpd.read_file(file)
    if g.crs is None:
        raise ValueError(f'{kind} has no declared CRS: {file}')
    code = column(g, ['adm3_pcode','ADM3_PCODE','adm3_id','ADM3_ID','admin3Pcod','adm_id'] if kind=='ADM3'
                  else ['adm2_pcode','ADM2_PCODE','adm2_id','ADM2_ID','admin2Pcod','adm_id'], f'{kind} geographic code')
    g['admin_id'] = g[code].astype('string').str.strip()
    if g['admin_id'].duplicated().any():
        raise ValueError(f'Duplicate {kind} geographic codes in boundary file: {file}')
    return g[['admin_id','geometry']]


def draw_scale(ax, length_km=200):
    # Map projection EPSG:6933 (units: meters).
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    width=x1-x0; height=y1-y0
    if length_km*1000 > width*0.35:
        length_km = 100
    step=length_km*1000/4
    left=x0+width*0.67; bottom=y0+height*0.045
    h=height*0.009
    if left+length_km*1000>x1-width*.01:
        left=x1-width*.03-length_km*1000
    for i in range(4):
        ax.add_patch(Rectangle((left+i*step,bottom), step,h, facecolor='black' if i%2==0 else 'white', edgecolor='black',lw=.8,zorder=15))
    for i in [0,1,2,4]:
        ax.text(left+i*step,bottom+h*2, f'{length_km*i//4}',ha='center',va='bottom',fontsize=8,zorder=16)
    ax.text(left+length_km*1000+width*.01,bottom+h*2,'km',fontsize=8,va='bottom',zorder=16)


def plot_map(gdf: gpd.GeoDataFrame, adm2: gpd.GeoDataFrame, ranked: pd.DataFrame, output: Path,
             first: int,last: int, top_n: int, min_coverage: float):
    # Equal-area projection allows the scale bar to use meters.
    g = gdf.to_crs('EPSG:6933')
    b = adm2.to_crs('EPSG:6933')
    fig = plt.figure(figsize=(18, 10), facecolor='#EAEBEF')
    gs = fig.add_gridspec(1,2, left=.035,right=.98,top=.85,bottom=.09,width_ratios=[3.15,1.72],wspace=.045)
    ax = fig.add_subplot(gs[0,0]); tb = fig.add_subplot(gs[0,1]); ax.set_facecolor('#EAEBEF'); tb.axis('off')
    extent=g.total_bounds
    dx=extent[2]-extent[0]; dy=extent[3]-extent[1]
    ax.set_xlim(extent[0]-dx*.05,extent[2]+dx*.07)
    ax.set_ylim(extent[1]-dy*.07,extent[3]+dy*.10)
    for category in ['No historical data','No recorded flood days','Moderate','High','Very High']:
        subset=g[g['severity_class'].fillna('No historical data').eq(category)]
        if not subset.empty:
            subset.plot(ax=ax,color=PALETTE[category],edgecolor='#F9F9FA',linewidth=.18,zorder=1)
    b.boundary.plot(ax=ax,color='#4A4A4A',linewidth=.65,zorder=3)
    # Black circle marker with white rank number. Number is inside the ADM3 geometry.
    keys=ranked['adm3_id'].astype(str).tolist()
    markers=g[g['adm3_id'].isin(keys)].copy()
    markers['rank'] = markers['adm3_id'].map(dict(zip(ranked['adm3_id'],ranked['rank'])))
    points=markers.geometry.representative_point()
    ax.scatter(points.x,points.y,s=275,c='#121212',edgecolors='white',linewidths=1.3,zorder=8)
    for (_,r), pt in zip(markers.iterrows(),points):
        ax.text(pt.x,pt.y,str(int(r['rank'])),color='white',ha='center',va='center',fontweight='bold',fontsize=9,zorder=9)
    # Legend
    legend=[Patch(facecolor=PALETTE[c],edgecolor='#777',label=c) for c in ['Moderate','High','Very High','No recorded flood days','No historical data']]
    leg=ax.legend(handles=legend,title='Relative Flood Activity',loc='upper left',frameon=True,facecolor='white',edgecolor='#4A4A4A',fontsize=9,title_fontsize=11)
    leg.get_title().set_weight('bold')
    ax.annotate('N',xy=(.94,.95),xytext=(.94,.84),xycoords='axes fraction',textcoords='axes fraction',ha='center',fontsize=14,fontweight='bold',arrowprops=dict(facecolor='black',edgecolor='black',width=4,headwidth=14,headlength=17),zorder=20)
    draw_scale(ax)
    ax.set_axis_off()
    # Right-hand table; numbered labels match map.
    tb.add_patch(Rectangle((0,.915),1,.085,transform=tb.transAxes,color=PALETTE['Very High']))
    tb.text(.5,.958,'Top ADM3 Flood Hotspots',transform=tb.transAxes,ha='center',va='center',color='white',fontsize=15,weight='bold')
    table=ranked[['rank','adm3_name','parent_adm2_name','mean_annual_flood_days','peak_months']].copy()
    table['mean_annual_flood_days']=table['mean_annual_flood_days'].map(lambda x:f'{x:.1f}')
    values=table.fillna('Unknown').astype(str).values.tolist()
    tab=tb.table(cellText=values,colLabels=['#','ADM3','Parent ADM2','Days/yr','Peak months'],
                 colWidths=[.07,.24,.27,.15,.27],cellLoc='center',colLoc='center',bbox=[0,.15,1,.75])
    tab.auto_set_font_size(False);tab.set_fontsize(7.5)
    for (i,j),cell in tab.get_celld().items():
        cell.set_edgecolor('#B2B2B7');cell.set_linewidth(.55)
        if i==0:
            cell.set_facecolor('#E7DEF4');cell.set_text_props(weight='bold')
        elif i%2==0:
            cell.set_facecolor('#F7F6FA')
        if i>0 and j==0:
            cell.set_text_props(weight='bold')
    tb.text(.02,.10,'Numbers on the map match the table.',transform=tb.transAxes,fontsize=9,va='top')
    tb.text(.02,.065,'Peak months: 05a monthly event-date proxy (unverified observation coverage).\nAnnual ranking: separate historical annual indicators.',transform=tb.transAxes,fontsize=7.5,color='#555',va='top')
    fig.text(.04,.955,'South Sudan — Historical Flood Hotspots (ADM3)',fontsize=25,weight='bold',va='top')
    fig.text(.04,.902,f'{first}–{last} | Average annual recorded-activity days + peak months | Top {top_n}',fontsize=13,color='#393939')
    fig.text(.04,.045,f'Source: processed ADM3 annual flood indicators. Areas require ≥{min_coverage:.0%} observed-year coverage.\n'
                     'Caution: observational completeness of original flood-event records is not independently verified.',fontsize=8,color='#444')
    fig.savefig(output,dpi=250,facecolor=fig.get_facecolor(),bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='ADM3 historical flood hotspots map and parent ADM2 concentration table.')
    ap.add_argument('--root',default=str(find_root()))
    ap.add_argument('--panel',default=None)
    ap.add_argument('--monthly-panel',default='processed_data/multivariate_flood_modeling/05a_monthly_panel/adm3_monthly_panel.csv')
    ap.add_argument('--adm3-boundary',default='raw_data/administrative_boundaries/ssd_admin3.geojson')
    ap.add_argument('--adm2-boundary',default='raw_data/administrative_boundaries/ssd_admin2.geojson')
    ap.add_argument('--start-year',type=int,default=2000)
    ap.add_argument('--end-year',type=int,default=2025)
    ap.add_argument('--top-n',type=int,default=20)
    ap.add_argument('--map-top-n',type=int,default=12,help='Number of numbered hotspot markers on map (max 12 recommended).')
    ap.add_argument('--min-coverage',type=float,default=.8)
    ap.add_argument('--output',default='processed_data/multivariate_flood_modeling/08a_adm3_flood_hotspots')
    a=ap.parse_args(); root=Path(a.root).resolve(); output=root/a.output;output.mkdir(parents=True,exist_ok=True)
    if a.start_year>a.end_year or a.top_n<1 or a.map_top_n<1 or a.map_top_n>a.top_n or not (0<a.min_coverage<=1):
        raise ValueError('Invalid years, top-n or min-coverage.')
    panel=find_panel(root,a.panel)
    stats,levels=classify(read_historical(panel,a.start_year,a.end_year),a.min_coverage)
    peaks,monthly_summary=read_monthly_seasonality(root/a.monthly_panel,a.start_year,a.end_year)
    stats=stats.merge(peaks,on='adm3_id',how='left',validate='one_to_one')
    adm3=read_geo(root/a.adm3_boundary,'ADM3'); adm2=read_geo(root/a.adm2_boundary,'ADM2')
    gdf=adm3.merge(stats,left_on='admin_id',right_on='adm3_id',how='left',validate='one_to_one')
    n_matched=int(gdf['mean_annual_flood_days'].notna().sum())
    if n_matched==0:
        raise ValueError('Zero ADM3 IDs matched the flood panel. Check ADM3 code fields before mapping.')
    if n_matched/len(gdf)<.8:
        raise ValueError(f'Only {n_matched}/{len(gdf)} ADM3 boundaries matched. Stopping to avoid a misleading map.')
    ranked=gdf[gdf['eligible'].fillna(False)].sort_values(['mean_annual_flood_days','adm3_id'],ascending=[False,True]).head(a.top_n).copy()
    ranked.insert(0,'rank',range(1,len(ranked)+1))
    map_ranked=ranked.head(a.map_top_n).copy()
    stats.to_csv(output/'01_adm3_historical_flood_severity.csv',index=False)
    ranked[['rank','adm3_id','adm3_name','parent_adm2_id','parent_adm2_name','mean_annual_flood_days','peak_months','peak_month_mean_event_days','total_recorded_flood_days','observed_years','year_coverage_fraction','severity_class']].to_csv(output/'02_top_adm3_hotspots.csv',index=False)
    summary=ranked.groupby(['parent_adm2_id','parent_adm2_name'],dropna=False).agg(top_hotspot_count=('adm3_id','nunique'),average_hotspot_flood_days=('mean_annual_flood_days','mean')).reset_index().sort_values(['top_hotspot_count','average_hotspot_flood_days'],ascending=False)
    summary.to_csv(output/'03_top_hotspots_by_parent_adm2.csv',index=False)
    high=stats[stats['severity_class'].isin(['High','Very High'])].copy()
    high.groupby(['parent_adm2_id','parent_adm2_name'],dropna=False).agg(high_or_very_high_adm3_count=('adm3_id','nunique'),mean_annual_flood_days=('mean_annual_flood_days','mean')).reset_index().sort_values('high_or_very_high_adm3_count',ascending=False).to_csv(output/'04_all_high_severity_by_parent_adm2.csv',index=False)
    monthly_summary.to_csv(output/'07_adm3_monthly_seasonality.csv',index=False)
    plot_map(gdf,adm2,map_ranked,output/'05_adm3_hotspot_map.png',a.start_year,a.end_year,len(map_ranked),a.min_coverage)
    method={'input_panel':str(panel),'adm3_boundary':str(root/a.adm3_boundary),'adm2_boundary':str(root/a.adm2_boundary),'years':[a.start_year,a.end_year], 'ranking_metric':'mean annual recorded flood days', 'matched_adm3':n_matched,'total_adm3_geometries':len(gdf), 'top_n_ranked':len(ranked),'top_n_on_map':len(map_ranked),'thresholds':levels,'source_data_limitation':'Flood mask zero/observation denominator has not been independently verified; descriptive observed records only.','not_a_prediction':True,'monthly_panel':str(root/a.monthly_panel),'peak_month_method':'Top 3 calendar months by mean monthly recorded event-date proxy across available years, not actual flood extent or confirmed flood duration'}
    (output/'06_methodology.json').write_text(json.dumps(method,indent=2,ensure_ascii=False),encoding='utf8')
    print('DONE:',output)
    print(f'Matched ADM3: {n_matched}/{len(gdf)}')
    print(f'Flood severity thresholds: High >= {levels["positive_days_quantile_66"]:.2f} days/year, Very High >= {levels["positive_days_quantile_85"]:.2f} days/year')
    print('Top ADM2 concentration:')
    print(summary[['parent_adm2_name','top_hotspot_count']].to_string(index=False))

if __name__=='__main__':
    main()
