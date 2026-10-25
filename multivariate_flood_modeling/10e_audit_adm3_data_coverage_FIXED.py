#!/usr/bin/env python3
"""10e | South Sudan ADM3 coverage audit and honest historical peak mapping.

Run without parameters from scripts/multivariate_flood_modeling/.
Inputs are the already produced Script 05a monthly panel, Script 10b test
predictions, and official ADM3 administrative GeoJSON. Script 10d outputs are
optional; their peak months are NEVER used to infer observation coverage.

Distinctions made:
  1) row/target presence in the processed MONTHLY PANEL (not satellite coverage),
  2) 10b complete-case/prediction inclusion,
  3) identifiable *recorded-event-date proxy* seasonal peak (2002-2019),
  4) rainfall/runoff/ET0 feature availability,
  5) remotely sensed observation coverage, which remains UNKNOWN unless an
     independently verified observation denominator is provided.

No missing flood product record is assumed to mean observed flood-free time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

BASE_COLORS = {'predicted': '#7546B7', 'not_predicted': '#D8D3DE',
               'peak_unspecified': '#F0E6DC', 'no_panel': '#BAB6BD'}
PEAK_COLORS = {1:'#7EC9F2',2:'#70B5EE',3:'#6399DA',4:'#8FDCD8',
               5:'#70C2B4',6:'#53A99D',7:'#D8BFF1',8:'#C18AE0',
               9:'#9870D3',10:'#B95CA6',11:'#9C3E91',12:'#7546B7'}


def locate_root() -> Path:
    p=Path(__file__).resolve()
    if p.parent.name=='multivariate_flood_modeling' and p.parent.parent.name=='scripts':
        return p.parents[2]
    return Path.cwd()


def load_boundaries(root):
    p=root/'raw_data/administrative_boundaries/ssd_admin3.geojson'
    if not p.exists():
        raise FileNotFoundError(f'ADM3 boundary missing: {p}')
    g=gpd.read_file(p)
    key=next((col for col in ('adm3_pcode','ADM3_PCODE','admin3Pcod','adm_id') if col in g.columns),None)
    if key is None:
        raise ValueError(f'No ADM3 ID in {p.name}; available: {list(g.columns)}')
    if g.crs is None: raise ValueError(f'Boundary has no CRS: {p}')
    g=g.rename(columns={key:'adm_id'}) if key!='adm_id' else g
    g['adm_id']=g.adm_id.astype('string').str.strip()
    if g.adm_id.isna().any() or g.adm_id.duplicated().any():
        raise ValueError('Null/duplicate ADM3 IDs in boundary file')
    return g[['adm_id','geometry']].to_crs('EPSG:6933')


def load_panel(root):
    p=root/'processed_data/multivariate_flood_modeling/05a_monthly_panel/adm3_monthly_panel.csv'
    if not p.exists(): raise FileNotFoundError(f'05a panel missing: {p}')
    x=pd.read_csv(p, dtype={'adm_id':'string'})
    required={'adm_id','year','month','flood_days','rainfall_mm','runoff_mm','et0_mm'}
    if required-set(x): raise ValueError(f'05a missing fields: {required-set(x)}')
    x['adm_id']=x.adm_id.str.strip()
    if x.duplicated(['adm_id','year','month']).any():
        raise ValueError('05a contains duplicated ADM3-year-month records; audit before proceeding')
    return x


def load_prediction_ids(root):
    p=root/'processed_data/multivariate_flood_modeling/10b_monthly_rolling_origin/01_test_predictions_monthly.csv'
    if not p.exists(): raise FileNotFoundError(f'10b predictions missing: {p}')
    x=pd.read_csv(p, dtype={'adm_id':'string'}, usecols=['level','memory','model','strategy','adm_id','year','month'])
    x=x[(x.level=='ADM3') & (x.memory=='C_combined') &
        (x.model=='M4_all_drivers') & (x.strategy=='Monthly Rolling')].copy()
    if x.empty: raise ValueError('No ADM3 M4 / C / Monthly Rolling predictions in 10b CSV')
    x['adm_id']=x.adm_id.str.strip()
    if x.duplicated(['adm_id','year','month']).any():
        raise ValueError('Duplicate ADM3 month predictions in selected 10b subset')
    return x.groupby('adm_id').size().rename('predicted_test_months')


def compute_panel_audit(panel):
    train=panel[panel.year.between(2002,2019)].copy()
    all_records=panel.groupby('adm_id',as_index=False).agg(
        panel_rows=('year','size'),
        target_nonnull_rows=('flood_days',lambda s: int(s.notna().sum())),
        rainfall_nonnull_rows=('rainfall_mm',lambda s: int(s.notna().sum())),
        runoff_nonnull_rows=('runoff_mm',lambda s: int(s.notna().sum())),
        et0_nonnull_rows=('et0_mm',lambda s: int(s.notna().sum())),
    )
    train_counts=train.groupby('adm_id',as_index=False).agg(
        training_rows=('year','size'),
        training_target_rows=('flood_days',lambda s:int(s.notna().sum())),
        training_positive_proxy_months=('flood_days',lambda s:int(s.gt(0).sum())),
        training_rainfall_nonnull=('rainfall_mm',lambda s:int(s.notna().sum())),
        training_runoff_nonnull=('runoff_mm',lambda s:int(s.notna().sum())),
        training_et0_nonnull=('et0_mm',lambda s:int(s.notna().sum())),
    )
    out=all_records.merge(train_counts,on='adm_id',how='left',validate='one_to_one')
    for c in out.columns:
        if c!='adm_id':out[c]=out[c].fillna(0).astype(int)
    return out


def compute_peaks(panel, years=range(2002,2020), min_years=8, relative_tie_tolerance=0.02):
    """Do not assign a peak if no recorded positive activity or near-tied months."""
    hist=panel[panel.year.isin(years)].copy()
    rows=[]
    for adm_id, part in hist.groupby('adm_id',sort=False):
        means=part.groupby('month').flood_days.mean().reindex(range(1,13))
        counts=part.groupby('month').flood_days.count().reindex(range(1,13),fill_value=0)
        eligible=(counts>=min_years) & means.notna()
        record={'adm_id':adm_id,'peak_month':pd.NA,
                'peak_mean_event_dates':np.nan,'peak_margin_event_dates':np.nan,
                'peak_status':'INSUFFICIENT_HISTORY',
                'eligible_calendar_months':int(eligible.sum()),
                'max_monthly_mean_proxy':float(means.max()) if means.notna().any() else np.nan}
        if int(eligible.sum())==12:
            eligible_means=means[eligible]
            highest=float(eligible_means.max())
            if highest<=0:
                record['peak_status']='NO_RECORDED_POSITIVE_ACTIVITY'
                record['peak_mean_event_dates']=0.0
            else:
                ordered=eligible_means.sort_values(ascending=False)
                second=float(ordered.iloc[1]); difference=highest-second
                tol=max(1e-10, highest*relative_tie_tolerance)
                record['peak_mean_event_dates']=highest
                record['peak_margin_event_dates']=difference
                if difference<=tol:
                    record['peak_status']='AMBIGUOUS_OR_TIED_PEAK'
                else:
                    record['peak_status']='IDENTIFIABLE_PROXY_PEAK'
                    record['peak_month']=int(ordered.index[0])
        rows.append(record)
    return pd.DataFrame(rows)


def build_coverage(boundary,panel,pred_counts,min_years,tol):
    x=compute_panel_audit(panel)
    peaks=compute_peaks(panel,min_years=min_years,relative_tie_tolerance=tol)
    d=boundary[['adm_id']].merge(x,on='adm_id',how='left',validate='one_to_one')
    d=d.merge(peaks,on='adm_id',how='left',validate='one_to_one')
    d=d.merge(pred_counts.rename_axis('adm_id').reset_index(),on='adm_id',how='left',validate='one_to_one')
    d['predicted_test_months']=d.predicted_test_months.fillna(0).astype(int)
    d['model_coverage_status']=np.where(d.predicted_test_months.gt(0),'PREDICTION_AVAILABLE','NO_10B_PREDICTION')
    d['historical_proxy_table_status']=np.where(d.training_target_rows.fillna(0).gt(0),'PROXY_VALUES_PRESENT','NO_TRAINING_PROXY_VALUES')
    d['peak_status']=d.peak_status.fillna('NO_TRAINING_PROXY_VALUES')
    # Important: separate 'data rows present' from satellite effective observation coverage.
    d['satellite_observation_coverage']='UNKNOWN_NOT_VERIFIED'
    d['peak_month']=pd.to_numeric(d.peak_month,errors='coerce').astype('Int64')
    expected=18*12
    d['training_complete_monthly_proxy_table']=d.training_target_rows.fillna(0).eq(expected)
    for var in ('rainfall','runoff','et0'):
        d[f'training_{var}_nonmissing_fraction']=d[f'training_{var}_nonnull'].fillna(0)/expected
    d['all_driver_training_fraction']=d[['training_rainfall_nonnull','training_runoff_nonnull','training_et0_nonnull']].fillna(0).min(axis=1)/expected
    return d


def map_layer(ax,g,colors,title, legend_items):
    ax.set_aspect('equal');ax.set_axis_off();ax.set_title(title,fontsize=14,fontweight='bold',pad=12)
    for color,parts in g.groupby(colors,dropna=False):
        parts.plot(ax=ax,color=str(color),edgecolor='#A7A3AD',linewidth=.22,zorder=2)
    g.boundary.plot(ax=ax,color='#C5C1C9',linewidth=.20,zorder=3)
    handles=[patches.Patch(facecolor=c,edgecolor='#99929F',label=label) for c,label in legend_items]
    ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(.03,-.02),fontsize=8.2,
              frameon=True,framealpha=.97,title=None)


def coverage_map(geo,out):
    fig,axes=plt.subplots(1,2,figsize=(16.4,8.4),facecolor='white')
    fig.subplots_adjust(left=.025,right=.985,top=.84,bottom=.14,wspace=.06)
    g=geo.copy()
    g['prediction_color']=g.model_coverage_status.map({'PREDICTION_AVAILABLE':BASE_COLORS['predicted'],
                    'NO_10B_PREDICTION':BASE_COLORS['not_predicted']})
    g['history_color']=g.historical_proxy_table_status.map({'PROXY_VALUES_PRESENT':'#7EC9F2',
                    'NO_TRAINING_PROXY_VALUES':'#D8D3DE'})
    map_layer(axes[0],g,'history_color','Historical proxy data in panel (2002–2019)',
              [('#7EC9F2','Monthly proxy values present'),('#D8D3DE','No historical proxy values')])
    map_layer(axes[1],g,'prediction_color','10b: Monthly Rolling prediction coverage (2023–2025)',
              [(BASE_COLORS['predicted'],'Prediction available'),(BASE_COLORS['not_predicted'],'No 10b model prediction')])
    yes=int(g.model_coverage_status.eq('PREDICTION_AVAILABLE').sum())
    fig.suptitle('South Sudan ADM3 | Data Table vs Prediction Coverage',fontsize=19,fontweight='bold',y=.97)
    fig.text(.5,.073,f'{len(g)} ADM3 in boundary  |  {yes} predicted  |  {len(g)-yes} without 10b prediction',ha='center',fontsize=11,fontweight='bold')
    fig.text(.5,.042,'PROXY TABLE PRESENCE is NOT proof of valid satellite observation. Absence of a prediction is NOT zero flooding.',
             ha='center',fontsize=9.4,color='#5F5268')
    save(fig,out,'01_adm3_historical_vs_prediction_coverage')


def peak_map(geo,out):
    g=geo.copy()
    def color(row):
        if row.peak_status=='IDENTIFIABLE_PROXY_PEAK' and pd.notna(row.peak_month):
            return PEAK_COLORS[int(row.peak_month)]
        return {'NO_RECORDED_POSITIVE_ACTIVITY':'#ECE9EE',
                'AMBIGUOUS_OR_TIED_PEAK':'#EBCFAA',
                'INSUFFICIENT_HISTORY':'#BCB8C1',
                'NO_TRAINING_PROXY_VALUES':'#BCB8C1'}.get(row.peak_status,'#BCB8C1')
    g['peak_color']=g.apply(color,axis=1)
    fig,ax=plt.subplots(figsize=(12.8,10.5),facecolor='white')
    fig.subplots_adjust(left=.035,right=.78,top=.88,bottom=.12)
    for c,parts in g.groupby('peak_color'):
        parts.plot(ax=ax,color=c,edgecolor='#C9C5CD',linewidth=.25)
    ax.set_axis_off();ax.set_aspect('equal')
    h=[patches.Patch(facecolor=PEAK_COLORS[m],edgecolor='white',label=f'{m:02d} — {pd.Timestamp(2020,m,1).strftime("%B")}') for m in range(1,13)]
    h += [patches.Patch(facecolor='#ECE9EE',label='No positive recorded activity'),
          patches.Patch(facecolor='#EBCFAA',label='Tied / indistinct peak'),
          patches.Patch(facecolor='#BCB8C1',label='Insufficient historical table')]
    fig.legend(handles=h,loc='center left',bbox_to_anchor=(.79,.53),fontsize=9.7,
               frameon=False,title='Calendar month / special status',title_fontsize=10.4)
    fig.suptitle('ADM3 | Historical Peak of Recorded Flood Activity Proxy (2002–2019)',
                 fontsize=17,fontweight='bold',y=.96)
    fig.text(.5,.065,'Computed from 05a monthly panel for ALL available ADM3, not only the 10b model cohort.',
             ha='center',fontsize=9.7)
    fig.text(.5,.039,'Satellite observation coverage UNKNOWN; proxy peak is NOT a verified physical flood-season peak.',
             ha='center',fontsize=9.4,color='#5F5268')
    save(fig,out,'02_adm3_proxy_peak_month_with_uncertainty')


def driver_map(geo,out):
    g=geo.copy()
    bins=[-0.01,0.01,.50,.80,.99,1.001]
    labels=['0%','1–50%','51–80%','81–99%','100%']
    cmap={'0%':'#D8D3DE','1–50%':'#E3DAEF','51–80%':'#C18AE0','81–99%':'#9772D4','100%':'#7546B7'}
    g['input_bin']=pd.cut(g.all_driver_training_fraction.fillna(-.1),bins=bins,labels=labels,include_lowest=True)
    g['input_color']=g.input_bin.astype(str).map(cmap).fillna('#B5B0B9')
    fig,ax=plt.subplots(figsize=(11.5,9.6),facecolor='white')
    fig.subplots_adjust(left=.035,right=.79,top=.87,bottom=.12)
    for c,sub in g.groupby('input_color'):
        sub.plot(ax=ax,color=c,edgecolor='#C7C2CB',linewidth=.24)
    ax.set_axis_off();ax.set_aspect('equal')
    h=[patches.Patch(facecolor=cmap[lab],label=lab) for lab in labels]
    h.append(patches.Patch(facecolor='#B5B0B9',label='Not assessable'))
    fig.legend(handles=h,loc='center left',bbox_to_anchor=(.80,.53),frameon=False,title='Minimum non-null share\nacross 3 driver variables')
    fig.suptitle('ADM3 | Training-Period Driver Field Completeness (2002–2019)',fontsize=17,weight='bold',y=.96)
    fig.text(.5,.063,'For each ADM3, minimum non-null share of rainfall, runoff and ET0, based on the 216 monthly panel rows.',ha='center',fontsize=9.8)
    fig.text(.5,.035,'Field completeness is NOT satellite flood observation coverage, and does NOT fully determine 10b eligibility.',ha='center',fontsize=9.2,color='#5F5268')
    save(fig,out,'03_adm3_driver_field_completeness')


def save(fig,out,stem):
    for ext in ('png','pdf'):
        fig.savefig(out/f'{stem}.{ext}',dpi=250,facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=locate_root())
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--min-years',type=int,default=8)
    ap.add_argument('--peak-tie-fraction',type=float,default=.02)
    args=ap.parse_args()
    if not 1<=args.min_years<=18:ap.error('--min-years must be 1–18')
    if not 0<=args.peak_tie_fraction<=1:ap.error('--peak-tie-fraction must be between 0 and 1')
    root=args.root.resolve()
    out=(args.out or root/'processed_data/multivariate_flood_modeling/10e_adm3_data_coverage').resolve()
    out.mkdir(parents=True,exist_ok=True)
    geo=load_boundaries(root)
    panel=load_panel(root)
    counts=load_prediction_ids(root)
    merged=build_coverage(geo,panel,counts,args.min_years,args.peak_tie_fraction)
    not_in_geometry=set(panel.adm_id.dropna())-set(geo.adm_id)
    if not_in_geometry: print('WARNING: panel IDs absent from geometry:',sorted(not_in_geometry)[:15])
    absent_pred=set(counts.index)-set(geo.adm_id)
    if absent_pred: print('WARNING: prediction IDs absent from geometry:',sorted(absent_pred)[:15])
    merged.drop(columns='geometry',errors='ignore').to_csv(out/'00_all_adm3_coverage_audit.csv',index=False)
    merged[merged.model_coverage_status.eq('NO_10B_PREDICTION')].drop(columns='geometry', errors='ignore').to_csv(out/'04_adm3_no_model_prediction.csv',index=False)
    merged[merged.peak_status.ne('IDENTIFIABLE_PROXY_PEAK')].drop(columns='geometry', errors='ignore').to_csv(out/'05_adm3_no_identifiable_proxy_peak.csv',index=False)
    coverage_map(geo.merge(merged.drop(columns='geometry', errors='ignore'),on='adm_id',validate='one_to_one'),out)
    peak_map(geo.merge(merged.drop(columns='geometry', errors='ignore'),on='adm_id',validate='one_to_one'),out)
    driver_map(geo.merge(merged.drop(columns='geometry', errors='ignore'),on='adm_id',validate='one_to_one'),out)
    count=lambda x: {str(k):int(v) for k,v in x.value_counts(dropna=False).items()}
    summary={
        'boundary_adm3_count':len(geo),'panel_adm3_count':int(panel.adm_id.nunique()),
        'model_prediction_status_counts':count(merged.model_coverage_status),
        'historical_proxy_table_status_counts':count(merged.historical_proxy_table_status),
        'peak_status_counts':count(merged.peak_status),
        'satellite_observation_coverage':'UNKNOWN: no verified denominator of valid satellite observation days supplied',
        'peak_period':'2002-2019',
        'peak_definition':'largest historical mean flood-event-date proxy across 12 months, requiring min observation table years per calendar month and no near tie',
        'tie_fraction':args.peak_tie_fraction,
        'validity_notes':[
            'A non-null flood_days=0 is not proof of a verified flood-free observation.',
            'Observation coverage and number of flood-product event dates are different quantities.',
            'Driver field coverage is not flood satellite observation coverage.',
            'Missing model prediction does not imply no flood activity.',
            'Peak-month map describes recorded event-date proxy, not verified physical peak flood season.',
            'Month ties, all-zero proxy and insufficient historical table are excluded from a peak assignment.'
        ]}
    (out/'06_coverage_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved:',out)
    print('ADM3 boundaries:',len(geo),'| monthly-panel regions:',panel.adm_id.nunique())
    print('10b coverage:',summary['model_prediction_status_counts'])
    print('Peak classifications:',summary['peak_status_counts'])
    print('Satellite observation coverage: UNKNOWN (no observation denominator).')

if __name__=='__main__':
    main()
