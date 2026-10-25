#!/usr/bin/env python3
"""JBG060 10d — peak-season, ADM3-specific monthly forecast evaluation.

RUN directly from scripts/multivariate_flood_modeling/ (VS Code Run works).
Uses 10b predictions and 05a monthly panels; NEVER refits the model.

Selection protocol (no test-year look-ahead):
  * define national peak *calendar months* using observed proxy in 2002–2019
    for the same ADM3 cohort as the 10b test;
  * test selected seasonal months in 2023–2025, without selecting on their
    2023–2025 observations or prediction performance;
  * independently identify each ADM3's historical peak calendar month using
    2002–2019 observations, with a minimum-history requirement.

IMPORTANT: target is a recorded flood-event-date proxy, NOT verified actual
inundation duration, flood occurrence, severity, or operational flood warnings.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

BG = '#FBFAFD'
BLUE = '#7EC9F2'
VIOLET = '#7546B7'
PINK = '#C18AE0'
TEXT = '#302B39'
MODEL_DESCRIPTIONS = {
    'M0_history': 'M0 = Historical flood activity only',
    'M1_rainfall': 'M1 = Historical flood + Rainfall',
    'M2_runoff': 'M2 = Historical flood + Runoff',
    'M3_et0': 'M3 = Historical flood + ET0',
    'M4_all_drivers': 'M4 = Historical flood + Rainfall + Runoff + ET0',
}
MEMORY_DESCRIPTIONS = {
    'A_same_month_previous_years': 'A = same month 1 & 2 years ago',
    'B_recent_months': 'B = previous 1, 2 & 3 months',
    'C_combined': 'C = recent 1–3 months + same month 1 & 2 years ago',
}
DISCLAIMER = 'Recorded flood-event-date PROXY; not verified inundation duration or calibrated flood-occurrence probability.'


def default_root():
    p = Path(__file__).resolve()
    if p.parent.name == 'multivariate_flood_modeling' and p.parent.parent.name == 'scripts':
        return p.parents[2]
    return Path.cwd()


def save_fig(fig, out, name):
    for ext in ['png','pdf']:
        fig.savefig(out / f'{name}.{ext}', dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def read_predictions(path, model, memory, strategy):
    if not path.exists():
        raise FileNotFoundError(f'10b predictions missing: {path}. Run 10b first.')
    keep = ['level','memory','model','strategy','adm_id','year','month',
            'actual_event_dates','predicted_event_dates','actual_high_activity',
            'predicted_warning']
    df = pd.read_csv(path, usecols=keep, dtype={'adm_id':str})
    df = df[(df.level=='ADM3') & (df.memory==memory) &
            (df.model==model) & (df.strategy==strategy)].copy()
    if df.empty:
        raise ValueError(f'No matching 10b predictions for {model}, {memory}, {strategy}')
    df['adm_id'] = df.adm_id.str.strip()
    if df.duplicated(['adm_id','year','month']).any():
        raise ValueError('Duplicate ADM3-month rows in 10b predictions')
    required=['actual_event_dates','predicted_event_dates','actual_high_activity','predicted_warning']
    if df[required].isna().any().any():
        raise ValueError('Selected test predictions include missing values')
    df['error'] = df.predicted_event_dates - df.actual_event_dates
    df['absolute_error'] = df.error.abs()
    df['FN'] = ((df.actual_high_activity==1)&(df.predicted_warning==0)).astype(int)
    df['FP'] = ((df.actual_high_activity==0)&(df.predicted_warning==1)).astype(int)
    df['TP'] = ((df.actual_high_activity==1)&(df.predicted_warning==1)).astype(int)
    df['TN'] = ((df.actual_high_activity==0)&(df.predicted_warning==0)).astype(int)
    return df


def read_training(panel_path, region_ids):
    if not panel_path.exists():
        raise FileNotFoundError(f'Monthly panel missing: {panel_path}. Run 05a first.')
    names=['adm_id','year','month','flood_days']
    df=pd.read_csv(panel_path, usecols=names, dtype={'adm_id':str})
    df['adm_id'] = df.adm_id.str.strip()
    df=df[df.year.between(2002,2019) & df.adm_id.isin(region_ids)].copy()
    df['flood_days']=pd.to_numeric(df.flood_days,errors='coerce')
    df=df.dropna(subset=['flood_days'])
    if df.duplicated(['adm_id','year','month']).any():
        raise ValueError('Duplicate ADM3-month records in training panel')
    if df.empty: raise ValueError('No usable 2002–2019 training proxy observations')
    return df


def select_peaks(train, n_peaks=3, min_history=8):
    # Equal ADM3 weight, with each historical month retained only when observed.
    mean_month=(train.groupby('month',as_index=False).agg(
        historical_mean_proxy=('flood_days','mean'),
        historical_rows=('flood_days','size'),
        historical_adm3=('adm_id','nunique'))
        .sort_values(['historical_mean_proxy','month'],ascending=[False,True]))
    if len(mean_month) != 12:
        raise ValueError('Training dataset does not contain all 12 calendar months')
    national=list(mean_month.head(n_peaks).month.astype(int))
    by_admin=(train.groupby(['adm_id','month'],as_index=False).agg(
        mean_activity=('flood_days','mean'), n_years=('flood_days','size')))
    by_admin=by_admin[by_admin.n_years>=min_history].copy()
    by_admin=by_admin.sort_values(['adm_id','mean_activity','month'],ascending=[True,False,True])
    local=by_admin.drop_duplicates('adm_id').rename(columns={
        'month':'local_peak_month','mean_activity':'local_peak_mean_activity',
        'n_years':'local_peak_observation_years'})
    return mean_month,national,local[['adm_id','local_peak_month','local_peak_mean_activity','local_peak_observation_years']]


def metrics(df):
    tp=int(df.TP.sum());fp=int(df.FP.sum());fn=int(df.FN.sum());tn=int(df.TN.sum())
    precision=tp/(tp+fp) if tp+fp else np.nan
    recall=tp/(tp+fn) if tp+fn else np.nan
    f1=(2*precision*recall/(precision+recall)) if pd.notna(precision) and pd.notna(recall) and precision+recall else np.nan
    return dict(n_adm3=df.adm_id.nunique(),n_adm3_months=len(df),
                MAE=float(df.absolute_error.mean()),Recall=recall,Precision=precision,
                F1=f1,TP=tp,FP=fp,FN=fn,TN=tn,
                observed_mean=float(df.actual_event_dates.mean()),
                predicted_mean=float(df.predicted_event_dates.mean()))


def seasonal_metrics(test, national):
    x=test.copy();x['season']=np.where(x.month.isin(national),'Historical peak months','Other months')
    result=[]
    for season, group in x.groupby('season'):
        result.append(dict(section='national_training_peak_vs_other',category=season,**metrics(group)))
    x['year_month']=x.year.astype(str)+'-'+x.month.astype(str).str.zfill(2)
    for ym, group in x.groupby('year_month'):
        result.append(dict(section='each_test_month', category=ym,**metrics(group)))
    return pd.DataFrame(result)


def read_geo(root):
    boundary=root/'raw_data/administrative_boundaries/ssd_admin3.geojson'
    if not boundary.exists(): raise FileNotFoundError(f'ADM3 boundary not found: {boundary}')
    g=gpd.read_file(boundary)
    key=next((x for x in ['adm3_pcode','ADM3_PCODE','admin3Pcod','adm_id'] if x in g.columns),None)
    if key is None: raise ValueError(f'ADM3 code column missing from {boundary}; fields: {list(g.columns)}')
    if g.crs is None: raise ValueError('ADM3 layer missing CRS')
    g['adm_id']=g[key].astype(str).str.strip()
    if g.adm_id.duplicated().any(): raise ValueError('Duplicated ADM3 region codes in geometry')
    return g[['adm_id','geometry']].to_crs('EPSG:6933')


def select_showcase(year, national, method):
    # Deterministic from TRAINING-ONLY selection, never test outcome values.
    if method == 'highest': month=national[0]
    else: month=national[min(1,len(national)-1)]
    return f'{year}-{month:02d}'


def map_triptych(geo, df, ym, out, labels, national):
    year,month=map(int,ym.split('-'))
    chosen=df[(df.year==year)&(df.month==month)]
    if chosen.empty: raise ValueError(f'No predictions for {ym}')
    g=geo.merge(chosen[['adm_id','actual_event_dates','predicted_event_dates','error']],
                on='adm_id',how='left',validate='one_to_one')
    m=g[['actual_event_dates','predicted_event_dates']].to_numpy(dtype=float)
    vmax=max(1.0,float(np.nanquantile(m,.97)))
    emax=max(1.0,float(np.nanquantile(np.abs(g['error'].dropna().to_numpy()),.97)))
    norm=Normalize(0,vmax)
    errnorm=TwoSlopeNorm(vmin=-emax,vcenter=0,vmax=emax)
    fig,axs=plt.subplots(1,3,figsize=(21,9.4),facecolor=BG)
    fig.subplots_adjust(left=.025,right=.985,top=.83,bottom=.22,wspace=.045)
    choices=[('actual_event_dates','Observed','BuPu',norm),
             ('predicted_event_dates','Predicted','BuPu',norm),
             ('error','Error: Predicted − Observed','PuOr',errnorm)]
    for ax,(col,title,cmap,nm) in zip(axs,choices):
        g.plot(column=col,ax=ax,cmap=cmap,norm=nm,edgecolor='none',
               missing_kwds={'color':'#EAE8ED'},zorder=2)
        g.boundary.plot(ax=ax,linewidth=.26,color='#B9B3C2',zorder=3)
        ax.set_title(title,fontsize=13,weight='bold',color=TEXT)
        ax.set_axis_off();ax.set_aspect('equal')
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap='BuPu'),ax=axs[:2],orientation='horizontal',shrink=.65,pad=.03)
    cb.set_label('Monthly recorded event-date proxy',fontsize=10)
    cb2=fig.colorbar(ScalarMappable(norm=errnorm,cmap='PuOr'),ax=axs[2],orientation='horizontal',shrink=.65,pad=.03)
    cb2.set_label('Prediction error: negative = underestimated, positive = overestimated',fontsize=9)
    fig.suptitle(f'ADM3 | Peak-Season Monthly Forecast | {ym}\nObserved • Predicted • Prediction Error',
                 fontsize=19,weight='bold',y=.975)
    fig.text(.5,.11,labels,ha='center',fontsize=10.4,weight='bold',color=TEXT)
    fig.text(.5,.083,f'Calendar month selected using 2002–2019 historical data only. Training-derived national peak months: {national}.',
             ha='center',fontsize=9.2,color=TEXT)
    fig.text(.5,.056,'Light grey = outside comparable predicted ADM3 cohort (NOT zero activity). Colours on first two maps share one scale.',
             ha='center',fontsize=9,color=TEXT)
    fig.text(.5,.028,DISCLAIMER,ha='center',fontsize=8.7,color='#5B5665')
    save_fig(fig,out,f'01_peak_season_observed_predicted_error_{ym}')
    return chosen


def plot_seasonality(month_stats, local, out, national):
    t=month_stats.sort_values('month');months=t.month.to_numpy(int)
    colors=[VIOLET if m in national else BLUE for m in months]
    fig,ax=plt.subplots(figsize=(12.6,6.5),facecolor=BG)
    fig.subplots_adjust(left=.085,right=.97,top=.78,bottom=.26)
    bars=ax.bar(months,t.historical_mean_proxy,color=colors,width=.68)
    for bar,value in zip(bars,t.historical_mean_proxy):
        ax.annotate(f'{value:.2f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),
                    xytext=(0,5),textcoords='offset points',ha='center',fontsize=9,weight='bold')
    ax.set_xticks(months);ax.set_xlabel('Calendar month');ax.set_ylabel('Mean recorded event dates per ADM3-month')
    ax.set_title('Historical monthly flood-activity pattern (training years 2002–2019)',fontsize=16,weight='bold')
    ax.set_ylim(0,max(t.historical_mean_proxy.max()*1.23,.1));ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Peak Months Selected Without Looking at the Test Years',fontsize=19,weight='bold',y=.95)
    fig.text(.5,.105,f'Purple = historical national Top {len(national)} calendar months: {national}; blue = other months.',
             ha='center',fontsize=10,weight='bold')
    fig.text(.5,.065,DISCLAIMER,ha='center',fontsize=8.7,color='#5B5665')
    save_fig(fig,out,'02_historical_seasonality_and_selected_peaks')


def plot_peak_comparison(results,out):
    d=results[results.section=='national_training_peak_vs_other'].set_index('category')
    cats=['Historical peak months','Other months']
    if not set(cats).issubset(d.index):return
    fig,axs=plt.subplots(2,2,figsize=(13.5,9.5),facecolor=BG)
    fig.subplots_adjust(left=.09,right=.97,top=.83,bottom=.18,wspace=.26,hspace=.39)
    for ax,(metric,fmt) in zip(axs.ravel(), [('MAE','{:.2f}'),('Recall','{:.1%}'),('F1','{:.3f}'),('FN','{:d}')]):
        vals=[d.loc[c,metric] for c in cats]
        bs=ax.bar(range(2),vals,color=[VIOLET,BLUE],width=.55)
        for b,v in zip(bs,vals):
            label=fmt.format(int(v) if metric=='FN' else v) if np.isfinite(v) else 'N/A'
            if np.isfinite(v): ax.annotate(label,(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,5),
                                          textcoords='offset points',ha='center',weight='bold',fontsize=10)
        ax.set_xticks([0,1],['Peak months','Other months'])
        ax.set_title(metric + (' (higher is better)' if metric in ('Recall','F1') else ' (lower is better)'),weight='bold')
        ax.set_ylim(0,max(.1, np.nanmax(vals)*1.3));ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.suptitle('ADM3 | Historical Peak vs Other Months\nMonthly Rolling prediction, 2023–2025',fontsize=17,weight='bold',y=.97)
    fig.text(.5,.09,'Different numbers of tested ADM3-months across groups; compare FN counts with the test sample sizes and recall.',
             ha='center',fontsize=9.4)
    fig.text(.5,.055,DISCLAIMER,ha='center',fontsize=8.5,color='#5B5665')
    save_fig(fig,out,'03_peak_vs_other_months_performance')


def plot_local_peak(geo,local,out):
    g=geo.merge(local[['adm_id','local_peak_month']],on='adm_id',how='left',validate='one_to_one')
    fig,ax=plt.subplots(figsize=(12.5,9.5),facecolor=BG)
    fig.subplots_adjust(left=.04,right=.93,top=.88,bottom=.14)
    g.plot(column='local_peak_month',ax=ax,cmap='twilight_shifted',vmin=.5,vmax=12.5,
           edgecolor='none',missing_kwds={'color':'#EBE8ED'},linewidth=0)
    g.boundary.plot(ax=ax,color='#ADA8B4',linewidth=.21)
    ax.set_axis_off();ax.set_aspect('equal')
    cb=fig.colorbar(ScalarMappable(norm=Normalize(.5,12.5),cmap='twilight_shifted'),ax=ax,
                    shrink=.6,aspect=30,pad=.03,ticks=range(1,13))
    cb.set_label('Local historical peak calendar month (1=Jan ... 12=Dec)',fontsize=10)
    fig.suptitle('ADM3-Specific Historical Peak Month (2002–2019)',fontsize=18,weight='bold',y=.965)
    fig.text(.5,.07,'Historical peak computed separately for each ADM3 (requires 8 observed years for the selected month).',
             ha='center',fontsize=9)
    fig.text(.5,.04,'Grey = insufficient historical coverage / outside comparable cohort. '+DISCLAIMER,
             ha='center',fontsize=8.5,color='#5B5665')
    save_fig(fig,out,'04_adm3_local_historical_peak_month')


def load_names(root):
    f=root/'processed_data/multivariate_flood_modeling/02_adm3_panel_dataset/full_analysis_panel.csv'
    cols=['adm_id','year','flood__ADM3_NAME','flood__ADM2_NAME']
    if not f.exists():return pd.DataFrame(columns=['adm_id','adm3_name','parent_adm2'])
    try:
        d=pd.read_csv(f,usecols=cols,dtype={'adm_id':str}).sort_values('year').drop_duplicates('adm_id',keep='last')
    except ValueError:
        return pd.DataFrame(columns=['adm_id','adm3_name','parent_adm2'])
    d['adm_id']=d.adm_id.str.strip()
    return d.rename(columns={'flood__ADM3_NAME':'adm3_name','flood__ADM2_NAME':'parent_adm2'})[['adm_id','adm3_name','parent_adm2']]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=default_root())
    p.add_argument('--model',choices=list(MODEL_DESCRIPTIONS),default='M4_all_drivers')
    p.add_argument('--memory',choices=list(MEMORY_DESCRIPTIONS),default='C_combined')
    p.add_argument('--strategy',choices=['Fixed','Monthly Rolling'],default='Monthly Rolling')
    p.add_argument('--peak-count',type=int,default=3,help='National peak calendar months chosen using 2002–2019 only')
    p.add_argument('--map-year',type=int,default=2025,help='Which test year to map for pre-selected peak month')
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    if a.peak_count<1 or a.peak_count>6:p.error('--peak-count must be 1..6')
    if a.map_year not in [2023,2024,2025]:p.error('--map-year must be 2023, 2024 or 2025')
    root=a.root.resolve()
    out=(a.output or root/'processed_data/multivariate_flood_modeling/10d_peak_season_adm3').resolve()
    out.mkdir(parents=True,exist_ok=True)
    test=read_predictions(root/'processed_data/multivariate_flood_modeling/10b_monthly_rolling_origin/01_test_predictions_monthly.csv',a.model,a.memory,a.strategy)
    train=read_training(root/'processed_data/multivariate_flood_modeling/05a_monthly_panel/adm3_monthly_panel.csv',set(test.adm_id))
    season,national,local=select_peaks(train,n_peaks=a.peak_count)
    eval_table=seasonal_metrics(test,national)
    local_eval=test.merge(local[['adm_id','local_peak_month']],on='adm_id',how='left',validate='many_to_one')
    local_eval['is_local_peak_month']=local_eval.month.eq(local_eval.local_peak_month)
    # Per-ADM3 results specifically for their own historical peak month (not the national peak).
    area_rows=[]
    for adm_id,g in local_eval.groupby('adm_id'):
        pk=g[g.is_local_peak_month]
        if pk.empty:continue
        area_rows.append({'adm_id':adm_id,'local_peak_month':int(pk.local_peak_month.iloc[0]),**metrics(pk)})
    region_metrics=pd.DataFrame(area_rows)
    names=load_names(root)
    if not names.empty:
        local=local.merge(names,on='adm_id',how='left',validate='one_to_one')
        if not region_metrics.empty:region_metrics=region_metrics.merge(names,on='adm_id',how='left',validate='one_to_one')
    season.to_csv(out/'01_train_monthly_peak_ranking.csv',index=False)
    local.to_csv(out/'02_adm3_historical_local_peak_months.csv',index=False)
    eval_table.to_csv(out/'03_peak_vs_other_and_monthly_test_metrics.csv',index=False)
    region_metrics.to_csv(out/'04_adm3_local_peak_test_metrics.csv',index=False)
    geo=read_geo(root)
    labels=f'{MODEL_DESCRIPTIONS[a.model]}  |  {MEMORY_DESCRIPTIONS[a.memory]}  |  {a.strategy}'
    ym=select_showcase(a.map_year,national,method='highest')
    selected=map_triptych(geo,test,ym,out,labels,national)
    selected[['adm_id','year','month','actual_event_dates','predicted_event_dates','error','FN','FP']].to_csv(out/f'05_showcase_{ym}_adm3_predictions.csv',index=False)
    plot_seasonality(season,local,out,national)
    plot_peak_comparison(eval_table,out)
    plot_local_peak(geo,local,out)
    metadata={
        'scope':'ADM3-specific monthly historical backtest 2023–2025, not future 2026 forecast',
        'peak_definition':'National top calendar months ranked by mean flood_days across observed ADM3-months in training years 2002–2019 only',
        'local_peak_definition':'Highest historical mean proxy month within each ADM3; minimum 8 observed training years',
        'test_selection':'Map displays top historical national peak calendar month in specified test year; no use of test target for month selection',
        'national_peak_months':national,'showcase_month':ym,'test_adm3':int(test.adm_id.nunique()),
        'mapped_adm3_geometry':int(len(geo)), 'model':labels,
        'important_limitations':[
            DISCLAIMER,
            'Peak defined using recorded proxy; differences in temporal observation coverage may influence peak ranking',
            'Comparisons peak vs other are observational summaries, not evidence of causal season effect',
            'No prediction is generated for ADM3 units outside the existing 10b cohort',
            'Recall/FN/FP refer to HIGH-ACTIVITY warning thresholds, not any-flood occurrence',
            'No model retraining: all forecasts are read unchanged from 10b test CSV',
            'Test years have previously informed project analysis, so not a pristine untouched test set'
        ]
    }
    (out/'06_methodology.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Peak calendar months from training data:',national)
    print('Showcase:',ym)
    print('Predicted ADM3:',test.adm_id.nunique(), 'of',len(geo),'mapped ADM3')
    print(eval_table[eval_table.section=='national_training_peak_vs_other'][['category','n_adm3_months','MAE','Recall','F1','FP','FN']].to_string(index=False))
    print('Saved to:',out)

if __name__=='__main__':
    main()
