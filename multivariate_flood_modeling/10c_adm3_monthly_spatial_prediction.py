#!/usr/bin/env python3
"""10c — ADM3-specific monthly flood prediction and spatial error review.

Uses Script 10b TEST PREDICTIONS; does not refit a model. Outputs regional
Actual vs Predicted month series, one selected month map, and 2023–2025
spatial MAE / false-negative / false-positive maps.

The outcome is a RECORDED FLOOD-EVENT-DATE PROXY, not measured flood duration.
Predicted warnings are HIGH-ACTIVITY threshold flags, not calibrated flood
occurrence probabilities. Missing ADM3 predictions remain missing, never 0.

Run without arguments in VS Code. Defaults: M4, C memory, Monthly Rolling,
2024-08; region automatically selected by highest false-negative count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

COL = {'actual':'#7546B7','predicted':'#3A9CD2','negative':'#D77C8F','positive':'#7D49B7'}
METRIC_NOTE = ('Recorded flood-event-date proxy (not verified flood duration); '
               'high-activity warning flags are not flood-occurrence probabilities.')
MODEL_EXPLAIN = {
    'M0_history':'M0 = historical flood activity only',
    'M1_rainfall':'M1 = flood history + rainfall',
    'M2_runoff':'M2 = flood history + runoff',
    'M3_et0':'M3 = flood history + ET0',
    'M4_all_drivers':'M4 = flood history + rainfall + runoff + ET0',
}
MEMORY_EXPLAIN = {
    'A_same_month_previous_years':'A: previous 1 & 2 years, same month',
    'B_recent_months':'B: previous 1, 2 & 3 months',
    'C_combined':'C: recent 1–3 months + same month 1 & 2 years ago',
}


def project_root():
    p = Path(__file__).resolve()
    if p.parent.name == 'multivariate_flood_modeling' and p.parent.parent.name == 'scripts':
        return p.parents[2]
    return Path.cwd()


def read_predictions(path, model, memory, strategy):
    if not path.is_file():
        raise FileNotFoundError(f'10b prediction CSV not found: {path}\nRun 10b first.')
    keep = ['level','memory','model','strategy','adm_id','year','month',
            'actual_event_dates','predicted_event_dates','actual_high_activity',
            'predicted_warning','event_threshold','warning_threshold']
    p = pd.read_csv(path, usecols=keep, dtype={'adm_id':'string'})
    p = p[(p.level=='ADM3') & (p.memory==memory) & (p.model==model)
          & (p.strategy==strategy)].copy()
    if p.empty:
        raise ValueError('No ADM3 predictions match model/memory/strategy; available selections may differ.')
    p['adm_id'] = p['adm_id'].str.strip()
    for c in ['year','month']:
        p[c] = pd.to_numeric(p[c], errors='raise').astype(int)
    if p.duplicated(['adm_id','year','month']).any():
        raise ValueError('Duplicate ADM3-month predictions; check input CSV.')
    if p[['actual_event_dates','predicted_event_dates','actual_high_activity','predicted_warning']].isna().any().any():
        raise ValueError('Missing actuals or predictions in chosen 10b rows.')
    p['date'] = pd.to_datetime(dict(year=p.year, month=p.month, day=1))
    p['absolute_error'] = (p.actual_event_dates - p.predicted_event_dates).abs()
    p['false_negative'] = ((p.actual_high_activity==1) & (p.predicted_warning==0)).astype(int)
    p['false_positive'] = ((p.actual_high_activity==0) & (p.predicted_warning==1)).astype(int)
    return p


def read_boundary(root):
    f = root/'raw_data/administrative_boundaries/ssd_admin3.geojson'
    if not f.is_file():
        raise FileNotFoundError(f'ADM3 boundary missing: {f}')
    g = gpd.read_file(f)
    if g.crs is None:
        raise ValueError('ADM3 geometry missing CRS')
    fields = ('adm3_pcode','ADM3_PCODE','admin3Pcod','adm_id')
    key = next((c for c in fields if c in g.columns), None)
    if key is None:
        raise ValueError(f'ADM3 key not found. Columns: {list(g.columns)}')
    g['adm_id'] = g[key].astype('string').str.strip()
    if g.adm_id.isna().any() or g.adm_id.duplicated().any():
        raise ValueError('ADM3 geometry keys missing or duplicated')
    return g[['adm_id','geometry']].to_crs('EPSG:6933')


def name_table(root):
    f=root/'processed_data/multivariate_flood_modeling/02_adm3_panel_dataset/full_analysis_panel.csv'
    if not f.is_file():
        return pd.DataFrame(columns=['adm_id','adm3_name','parent_adm2'])
    cols=pd.read_csv(f, nrows=0).columns
    wanted=['adm_id','year','flood__ADM3_NAME','flood__ADM2_NAME']
    if not set(wanted).issubset(cols):
        return pd.DataFrame(columns=['adm_id','adm3_name','parent_adm2'])
    x=pd.read_csv(f,usecols=wanted,dtype={'adm_id':'string'})
    x['adm_id']=x.adm_id.str.strip()
    x=x.sort_values('year').drop_duplicates('adm_id',keep='last')
    return x.rename(columns={'flood__ADM3_NAME':'adm3_name',
                             'flood__ADM2_NAME':'parent_adm2'})[['adm_id','adm3_name','parent_adm2']]


def style_map(ax, boundary):
    boundary.boundary.plot(ax=ax, color='#AAA5B5', linewidth=.25, zorder=4)
    ax.set_axis_off(); ax.set_aspect('equal')


def save(fig, out, stem):
    for ext in ('png','pdf'):
        fig.savefig(out/f'{stem}.{ext}', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_month_map(boundary, pred, ym, labels, out):
    year,month=map(int, ym.split('-'))
    this=pred[(pred.year==year)&(pred.month==month)]
    if this.empty:
        raise ValueError(f'No prediction rows for {ym}. Available months: '
                         f'{pred.date.min():%Y-%m} to {pred.date.max():%Y-%m}')
    geo=boundary.merge(this[['adm_id','actual_event_dates','predicted_event_dates',
                             'false_negative','false_positive']],on='adm_id',how='left',validate='one_to_one')
    vmax=max(1.0,float(np.nanpercentile(geo[['actual_event_dates','predicted_event_dates']].values,95)))
    cmap=plt.get_cmap('BuPu'); norm=mcolors.Normalize(0,vmax)
    fig,axes=plt.subplots(1,2,figsize=(16.2,8.8))
    for ax,key,title in zip(axes,['actual_event_dates','predicted_event_dates'],
                            ['Observed activity proxy','Predicted activity proxy']):
        geo.plot(column=key, ax=ax, cmap=cmap, norm=norm, missing_kwds={
            'color':'#ECEAEC','edgecolor':'#CCC8D0','label':'No prediction'},linewidth=0, zorder=2)
        style_map(ax,geo)
        ax.set_title(title, fontsize=14,weight='bold')
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,location='bottom',
                    shrink=.65,aspect=35,pad=.02,extend='max')
    cb.set_label('Recorded flood-event-date proxy in selected month (days)',fontsize=10)
    fig.suptitle(f'ADM3 | Observed vs Predicted Monthly Flood Activity | {ym}',
                 fontsize=19,weight='bold',y=.98)
    fig.text(.5,.07,labels,ha='center',fontsize=10,weight='bold')
    fig.text(.5,.045,'Light grey = no prediction / outside comparable model cohort. '
             'Same colour scale for observed and predicted.',ha='center',fontsize=9)
    fig.text(.5,.02,METRIC_NOTE,ha='center',fontsize=8.5,color='#55505B')
    save(fig,out,f'01_adm3_actual_vs_predicted_{ym}')
    return geo


def plot_spatial_errors(boundary,pred,labels,out):
    summary=(pred.groupby('adm_id',as_index=False)
            .agg(n_months=('date','size'),MAE=('absolute_error','mean'),
                 FN=('false_negative','sum'),FP=('false_positive','sum'),
                 actual_high_months=('actual_high_activity','sum')))
    summary['FN_rate_given_actual_high']=(summary.FN/summary.actual_high_months.replace(0,np.nan))
    geo=boundary.merge(summary,on='adm_id',how='left',validate='one_to_one')
    fig,axes=plt.subplots(1,3,figsize=(21,8.5))
    specs=[('MAE','MAE (event-date proxy days)','BuPu'),
           ('FN','False negatives: count (2023–2025)','OrRd'),
           ('FP','False positives: count (2023–2025)','PuRd')]
    for ax,(key,title,cmap) in zip(axes,specs):
        geo.plot(column=key,ax=ax,cmap=cmap,legend=True,
                 legend_kwds={'shrink':.58,'label':title},
                 missing_kwds={'color':'#ECEAEC'},linewidth=0)
        style_map(ax,geo);ax.set_title(title,fontsize=12,weight='bold')
    fig.suptitle('ADM3 | Spatial Forecast Errors (2023–2025)',fontsize=20,weight='bold',y=.97)
    fig.text(.5,.067,labels,ha='center',fontsize=10,weight='bold')
    fig.text(.5,.04,'FN/FP = incorrect HIGH-ACTIVITY warning flags; counts depend on the number of test months. Grey = no prediction.',
             ha='center',fontsize=9)
    fig.text(.5,.02,METRIC_NOTE,ha='center',fontsize=8.5,color='#55505B')
    save(fig,out,'02_adm3_spatial_errors_2023_2025')
    return summary


def plot_region(pred, names, chosen, labels, out):
    d=pred[pred.adm_id==chosen].sort_values('date')
    if d.empty:
        raise ValueError(f'No prediction for ADM3 {chosen}')
    details=names[names.adm_id==chosen]
    title=chosen if details.empty else f'{details.adm3_name.iloc[0]} ({chosen}), {details.parent_adm2.iloc[0]}'
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(17.5,8.6),sharex=True,
                                gridspec_kw={'height_ratios':[3,1]})
    fig.subplots_adjust(left=.065,right=.975,top=.80,bottom=.15,hspace=.14)
    dates=d.date.to_numpy()
    ax.plot(d.date,d.actual_event_dates,color=COL['actual'],marker='o',ms=4,lw=2,label='Observed proxy')
    ax.plot(d.date,d.predicted_event_dates,color=COL['predicted'],marker='s',ms=3,lw=1.8,label='Predicted proxy')
    ax.set_ylabel('Recorded event-date proxy (days)')
    ax.grid(alpha=.18);ax.legend(frameon=False,ncol=2)
    missed=d[d.false_negative==1]
    extra=d[d.false_positive==1]
    ax2.scatter(missed.date,[1]*len(missed),marker='x',color=COL['negative'],s=70,label='False negative')
    ax2.scatter(extra.date,[0]*len(extra),marker='^',color=COL['positive'],s=60,label='False positive')
    ax2.set_ylim(-.8,1.8);ax2.set_yticks([0,1],['FP','FN'])
    ax2.grid(axis='x',alpha=.17)
    ax2.legend(frameon=False,loc='upper right',ncol=2)
    ax2.set_xticks(d.date)
    ax2.set_xticklabels(d.date.dt.strftime('%Y-%m'),rotation=90,fontsize=8)
    fig.suptitle(f'ADM3 Monthly Forecast vs Observations | {title}',
                 fontsize=17,weight='bold',y=.96)
    fig.text(.5,.88,labels,ha='center',fontsize=10,weight='bold')
    fig.text(.5,.06,'36-month historical BACKTEST (2023–2025), not a live 2026 forecast.',ha='center',fontsize=9)
    fig.text(.5,.032,METRIC_NOTE,ha='center',fontsize=8.6,color='#55505B')
    save(fig,out,f'03_adm3_{chosen}_monthly_series')
    return d


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    root=project_root()
    ap.add_argument('--root',type=Path,default=root)
    ap.add_argument('--model',choices=list(MODEL_EXPLAIN),default='M4_all_drivers')
    ap.add_argument('--memory',choices=list(MEMORY_EXPLAIN),default='C_combined')
    ap.add_argument('--strategy',choices=['Fixed','Monthly Rolling'],default='Monthly Rolling')
    ap.add_argument('--month',default='2024-08',help='Map month YYYY-MM; must be in 2023–2025 test')
    ap.add_argument('--adm3',default=None,help='Specific ADM3 code, e.g. SS020306; default highest FN count')
    ap.add_argument('--output',type=Path,default=None)
    a=ap.parse_args();root=a.root.resolve()
    out=a.output or root/'processed_data/multivariate_flood_modeling/10c_adm3_spatial_prediction'
    out.mkdir(parents=True,exist_ok=True)
    infile=root/'processed_data/multivariate_flood_modeling/10b_monthly_rolling_origin/01_test_predictions_monthly.csv'
    pred=read_predictions(infile,a.model,a.memory,a.strategy)
    boundary=read_boundary(root);names=name_table(root)
    missing=set(pred.adm_id)-set(boundary.adm_id)
    if missing:
        raise ValueError(f'{len(missing)} prediction region IDs absent from map, e.g. {sorted(missing)[:8]}. Fix IDs first.')
    summary=plot_spatial_errors(boundary,pred,MODEL_EXPLAIN[a.model]+' | '+MEMORY_EXPLAIN[a.memory]+' | '+a.strategy,out)
    summary=summary.merge(names,on='adm_id',how='left',validate='one_to_one')
    summary.to_csv(out/'04_adm3_2023_2025_error_metrics.csv',index=False)
    selected=a.adm3 or summary.sort_values(['FN','MAE'],ascending=False).adm_id.iloc[0]
    selected=str(selected)
    labels=MODEL_EXPLAIN[a.model]+' | '+MEMORY_EXPLAIN[a.memory]+' | '+a.strategy
    plot_month_map(boundary,pred,a.month,labels,out)
    chosen=plot_region(pred,names,selected,labels,out)
    chosen.to_csv(out/f'05_adm3_{selected}_monthly_predictions.csv',index=False)
    meta={'prediction_source':str(infile),'model':a.model,'memory':a.memory,
          'strategy':a.strategy,'map_month':a.month,'selected_adm3':selected,
          'predicted_adm3_count':int(pred.adm_id.nunique()),
          'boundary_adm3_count':int(len(boundary)),
          'meaning':'Recorded flood-event-date proxy; not verified flood occurrence probability.',
          'methodology':'Post-hoc spatial evaluation using frozen Script 10b backtest predictions; no training or tuning.'}
    (out/'06_methodology.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'Output: {out}\nPredicted ADM3: {pred.adm_id.nunique()} / {len(boundary)}'
          f'\nDetailed monthly series ADM3: {selected}\nMap month: {a.month}')


if __name__=='__main__':
    main()
