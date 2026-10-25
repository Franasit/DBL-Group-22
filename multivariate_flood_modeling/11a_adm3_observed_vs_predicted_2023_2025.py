#!/usr/bin/env python3
"""11a | ADM3 2023–2025 Observed vs Predicted Monthly-Forecast Summaries.

Standalone visualization: reads Script 10b predictions; DOES NOT retrain models.
Each annual map summarizes 12 separate NEXT-MONTH prediction evaluations:
mean monthly recorded flood-event-date proxy over available paired months.
NOT an annual flood duration, annual flood-days forecast, or flood probability.

Outputs one 3x3 poster grid (years x Observed/Predicted/Error), one figure
per year, and a region-level CSV audit. All years share the SAME observed /
predicted color scale and symmetric error scale. Regions with no predictions
are shown in light gray, not zero. Model M4 + memory C + Monthly Rolling
are defaults; use --help for alternatives.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

MODELS = {
    'M0_history': 'M0: Historical flood activity',
    'M1_rainfall': 'M1: Flood history + Rainfall',
    'M2_runoff': 'M2: Flood history + Runoff',
    'M3_et0': 'M3: Flood history + ET0',
    'M4_all_drivers': 'M4: Flood history + Rainfall + Runoff + ET0',
}
MEMORIES = {
    'A_same_month_previous_years': 'A: Previous 1 and 2 years, same month',
    'B_recent_months': 'B: Previous 1–3 months',
    'C_combined': 'C: Previous 1–3 months + same month 1 and 2 years ago',
}
YEARS=(2023,2024,2025)
NO_DATA='#ECEAF0'
PROXY_NOTE='Recorded flood-event-date proxy; NOT verified inundation duration or flood occurrence probability.'


def find_root():
    p=Path(__file__).resolve()
    return p.parents[2] if p.parent.name=='multivariate_flood_modeling' and p.parent.parent.name=='scripts' else Path.cwd()


def load_predictions(f, model, memory, strategy):
    if not f.exists():
        raise FileNotFoundError(f'Missing 10b CSV: {f}. Run 10b first.')
    cols=['level','memory','model','strategy','adm_id','year','month',
          'actual_event_dates','predicted_event_dates']
    p=pd.read_csv(f,usecols=cols,dtype={'adm_id':'string'})
    p=p[(p.level=='ADM3') & (p.model==model) & (p.memory==memory) &
        (p.strategy==strategy) & p.year.isin(YEARS)].copy()
    if p.empty:
        raise ValueError('No matching ADM3 predictions in 10b CSV for selected model / memory / strategy.')
    p['adm_id']=p.adm_id.str.strip()
    if p[['actual_event_dates','predicted_event_dates']].isna().any().any():
        raise ValueError('Missing values among paired observed/predicted records.')
    if p.duplicated(['adm_id','year','month']).any():
        raise ValueError('Duplicate rows for the same ADM3/year/month; verify 10b output.')
    invalid=p[(~p.month.between(1,12)) | (p.actual_event_dates.lt(0)) | (p.predicted_event_dates.lt(0))]
    if len(invalid): raise ValueError(f'Invalid monthly values: {len(invalid)} rows')
    return p


def load_boundary(root):
    path=root/'raw_data/administrative_boundaries/ssd_admin3.geojson'
    if not path.exists(): raise FileNotFoundError(f'ADM3 boundary not found: {path}')
    g=gpd.read_file(path)
    if g.crs is None: raise ValueError('ADM3 boundary is missing a coordinate reference system')
    candidates=['adm3_pcode','ADM3_PCODE','admin3Pcod','adm_id']
    key=next((k for k in candidates if k in g.columns),None)
    if key is None: raise ValueError(f'Could not find ADM3 ID field. Available: {list(g.columns)}')
    g['adm_id']=g[key].astype('string').str.strip()
    if g.adm_id.isna().any() or g.adm_id.duplicated().any():
        raise ValueError('ADM3 boundary contains missing/duplicate admin codes')
    return g[['adm_id','geometry']].to_crs('EPSG:6933')


def summary_by_adm_year(pred, boundary):
    summary=(pred.groupby(['adm_id','year'],as_index=False)
             .agg(months_with_paired_predictions=('month','nunique'),
                  observed_monthly_mean=('actual_event_dates','mean'),
                  predicted_monthly_mean=('predicted_event_dates','mean')))
    summary['signed_error']=summary.predicted_monthly_mean-summary.observed_monthly_mean
    summary['absolute_annual_mean_error']=summary.signed_error.abs()
    # The below metric is mean of monthly absolute errors (not error of annual means).
    monthly=(pred.assign(monthly_absolute_error=lambda d:
                         (d.actual_event_dates-d.predicted_event_dates).abs())
                .groupby(['adm_id','year'],as_index=False)
                .agg(MAE_over_test_months=('monthly_absolute_error','mean')))
    summary=summary.merge(monthly,on=['adm_id','year'],validate='one_to_one')
    valid=set(boundary.adm_id)
    missing=sorted(set(summary.adm_id)-valid)
    if missing: raise ValueError(f'{len(missing)} ADM IDs from predictions not in polygons, e.g. {missing[:6]}')
    return summary


def make_year_maps(boundary, summary):
    maps={}
    for year in YEARS:
        sub=summary.loc[summary.year==year].drop(columns='year')
        geo=boundary.merge(sub,on='adm_id',how='left',validate='one_to_one')
        geo['coverage_status']=np.where(geo.months_with_paired_predictions.isna(),
                                        'NO_MODEL_PREDICTION',
                                        np.where(geo.months_with_paired_predictions.eq(12),
                                                 'ALL_12_MONTHS','PARTIAL_MONTHS'))
        maps[year]=geo
    return maps


def plot_map(ax,geo,key,cmap,norm,title):
    geo.plot(column=key,ax=ax,cmap=cmap,norm=norm,linewidth=0,
             missing_kwds={'color':NO_DATA},zorder=2)
    geo.boundary.plot(ax=ax,color='#B6B2C1',linewidth=.23,zorder=3)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_title(title,fontsize=12,weight='bold',pad=10)


def legend_footer(fig,axes,common_norm,error_norm, model,memory,strategy):
    # A single shared scale for every year and both observed/predicted columns.
    cb1=fig.colorbar(ScalarMappable(cmap='BuPu',norm=common_norm),ax=axes,
                     orientation='horizontal',fraction=.018,pad=.025,shrink=.59,aspect=50)
    cb1.set_label('Observed / predicted mean recorded event dates per test month',fontsize=10)
    # Separate error scale with blue underprediction and violet/red overprediction.
    return cb1


def save(fig,out,stem):
    for ext in ['png','pdf']:
        fig.savefig(out/f'{stem}.{ext}',dpi=280,facecolor='white',bbox_inches='tight')
    plt.close(fig)


def make_poster_grid(maps, obs_norm, err_norm, model, memory, strategy, out):
    fig,axes=plt.subplots(3,3,figsize=(19.5,20.0),facecolor='white')
    fig.subplots_adjust(left=.025,right=.985,top=.90,bottom=.20,hspace=.08,wspace=.08)
    for i,year in enumerate(YEARS):
        g=maps[year]
        n=int(g.months_with_paired_predictions.notna().sum())
        plot_map(axes[i,0],g,'observed_monthly_mean','BuPu',obs_norm,
                 f'{year} — Observed  (n={n} ADM3)')
        plot_map(axes[i,1],g,'predicted_monthly_mean','BuPu',obs_norm,
                 f'{year} — Predicted')
        plot_map(axes[i,2],g,'signed_error','PuOr_r',err_norm,
                 f'{year} — Error (Predicted − Observed)')
    fig.suptitle('ADM3 | Observed vs Predicted Flood Activity (2023–2025)\nAnnual summaries of monthly predictions',
                 y=.975,fontsize=19,weight='bold')
    bar1=fig.add_axes([.12,.155,.31,.014]);
    c1=fig.colorbar(ScalarMappable(norm=obs_norm,cmap='BuPu'),cax=bar1,orientation='horizontal')
    c1.set_label('Observed / predicted: mean event-date proxy per month',fontsize=9)
    bar2=fig.add_axes([.58,.155,.30,.014]);
    c2=fig.colorbar(ScalarMappable(norm=err_norm,cmap='PuOr_r'),cax=bar2,orientation='horizontal')
    c2.set_label('Error: blue-violet = underprediction; orange = overprediction',fontsize=9)
    fig.legend(handles=[Patch(facecolor=NO_DATA,edgecolor='#B6B2C1',label='No model prediction')],
               loc='lower center',bbox_to_anchor=(.50,.124),frameon=False,fontsize=9)
    fig.text(.5,.100,f'{MODELS[model]}  |  {MEMORIES[memory]}  |  {strategy}',
             ha='center',fontsize=10.3,weight='bold')
    fig.text(.5,.079,'Each ADM3 value is the mean of its paired monthly predictions in that year. 12 months expected; see coverage CSV for exceptions.',
             ha='center',fontsize=9.4)
    fig.text(.5,.058,PROXY_NOTE,ha='center',fontsize=9,color='#5E5866')
    fig.text(.5,.038,'Historical backtest (2023–2025), NOT a forecast made for 2026. Grey = no model prediction, not zero flood activity.',
             ha='center',fontsize=8.8,color='#5E5866')
    save(fig,out,'01_adm3_2023_2025_observed_predicted_error_GRID')


def make_year_figure(year,geo,obs_norm,err_norm,model,memory,strategy,out):
    fig,axs=plt.subplots(1,3,figsize=(20,9),facecolor='white')
    fig.subplots_adjust(left=.025,right=.985,top=.84,bottom=.22,wspace=.08)
    for ax,key,title,cmap,norm in zip(axs,
        ['observed_monthly_mean','predicted_monthly_mean','signed_error'],
        ['Observed','Predicted','Prediction error'],
        ['BuPu','BuPu','PuOr_r'],[obs_norm,obs_norm,err_norm]):
        plot_map(ax,geo,key,cmap,norm,title)
    fig.suptitle(f'{year} | ADM3 Observed vs Predicted Flood Activity\nYearly mean of monthly prediction records',
                 fontsize=18,weight='bold',y=.96)
    cax1=fig.add_axes([.12,.165,.31,.023]);
    cb1=fig.colorbar(ScalarMappable(norm=obs_norm,cmap='BuPu'),cax=cax1,orientation='horizontal')
    cb1.set_label('Observed / predicted mean per month',fontsize=9)
    cax2=fig.add_axes([.57,.165,.31,.023]);
    cb2=fig.colorbar(ScalarMappable(norm=err_norm,cmap='PuOr_r'),cax=cax2,orientation='horizontal')
    cb2.set_label('Error = predicted − observed',fontsize=9)
    fig.legend(handles=[Patch(facecolor=NO_DATA,edgecolor='#B6B2C1',label='No model prediction')],
               loc='lower center',bbox_to_anchor=(.50,.135),frameon=False)
    n=int(geo.months_with_paired_predictions.notna().sum())
    fig.text(.5,.103,f'{MODELS[model]}  |  {MEMORIES[memory]}  |  {strategy}  |  {n} ADM3 with predictions',
             ha='center',fontsize=10.1,weight='bold')
    fig.text(.5,.071,PROXY_NOTE,ha='center',fontsize=9,color='#5E5866')
    fig.text(.5,.042,'Annual map = average of monthly predictions, not an independent yearly forecast. Grey = no prediction.',
             ha='center',fontsize=9,color='#5E5866')
    save(fig,out,f'02_adm3_{year}_observed_predicted_error')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=find_root())
    ap.add_argument('--model',choices=list(MODELS),default='M4_all_drivers')
    ap.add_argument('--memory',choices=list(MEMORIES),default='C_combined')
    ap.add_argument('--strategy',choices=['Fixed','Monthly Rolling'],default='Monthly Rolling')
    ap.add_argument('--output',type=Path,default=None)
    args=ap.parse_args()
    root=args.root.resolve()
    out=(args.output if args.output else root/'processed_data/multivariate_flood_modeling/11a_adm3_annual_prediction_comparison').resolve()
    out.mkdir(parents=True,exist_ok=True)
    p=load_predictions(root/'processed_data/multivariate_flood_modeling/10b_monthly_rolling_origin/01_test_predictions_monthly.csv',
                       args.model,args.memory,args.strategy)
    boundary=load_boundary(root)
    summary=summary_by_adm_year(p,boundary)
    maps=make_year_maps(boundary,summary)
    all_vals=np.r_[summary.observed_monthly_mean.to_numpy(),summary.predicted_monthly_mean.to_numpy()]
    vmax=max(1.,float(np.nanmax(all_vals)))
    signed=max(.5,float(np.nanmax(np.abs(summary.signed_error.to_numpy()))))
    obs_norm=colors.Normalize(vmin=0,vmax=vmax)
    err_norm=colors.TwoSlopeNorm(vmin=-signed,vcenter=0,vmax=signed)
    make_poster_grid(maps,obs_norm,err_norm,args.model,args.memory,args.strategy,out)
    for year in YEARS: make_year_figure(year,maps[year],obs_norm,err_norm,args.model,args.memory,args.strategy,out)
    summary.sort_values(['year','adm_id']).to_csv(out/'03_adm3_annual_mean_prediction_metrics.csv',index=False)
    coverage=[]
    for year in YEARS:
        g=maps[year]
        coverage.append(dict(year=year,total_adm3=len(g),with_predictions=int(g.months_with_paired_predictions.notna().sum()),
                             no_prediction=int(g.months_with_paired_predictions.isna().sum()),
                             complete_12_months=int(g.months_with_paired_predictions.eq(12).sum()),
                             partial_months=int(g.months_with_paired_predictions.between(1,11).sum())))
    pd.DataFrame(coverage).to_csv(out/'04_annual_coverage_audit.csv',index=False)
    metadata={'input':'10b monthly test predictions','years':list(YEARS),'model':args.model,'memory':args.memory,
              'strategy':args.strategy,'aggregation':'Arithmetic mean of paired month-level proxy observations and monthly forecasts within each year and ADM3.',
              'error':'Difference between predicted and observed annual means; not the MAE over monthly samples.',
              'both_color_scales':'Shared across all three years; error uses a common symmetric range.',
              'missing':'No model prediction is NaN and gray, never interpreted as zero activity.',
              'limitations':['Historical 2023–2025 backtest, not live future forecast.',
                             'Event-date proxy is not validated flood duration.',
                             'Full satellite observation coverage has not been established.',
                             'Individual ADM3 annual means mask potentially important month-level errors.',
                             'Only ADM3 with paired predictions are evaluated.']}
    (out/'05_methodology.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Saved:',out)
    print(pd.DataFrame(coverage).to_string(index=False))
    print('Figures: one 3x3 poster grid + three annual 3-panel comparisons (PNG and PDF).')

if __name__=='__main__': main()
