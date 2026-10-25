#!/usr/bin/env python3
"""12b: complete English ADM3 prediction poster chart pack.

Requires 12a_generate_prediction_summary_figures.py in the same folder.
Loads original 10b pre-computed class flags; uses 11c's validated spatial metrics.
NO re-thresholding on test data. Run from VS Code without arguments.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sys import path as syspath
syspath.insert(0,str(Path(__file__).resolve().parent))
import importlib
A=importlib.import_module('12a_generate_prediction_summary_figures')

BLUE='#7EC9F2'; PURPLE='#7546B7'; GREEN='#58B36B'; ORANGE='#F28B44'
STRATEGIES={'Fixed':BLUE,'Monthly Rolling':PURPLE}

def monthly_mae(pred):
    q=pred.assign(date=pd.to_datetime(dict(year=pred.year,month=pred.month,day=1)))
    m=q.groupby(['strategy','date'],as_index=False).apply(
        lambda z:pd.Series({'MAE':float(np.abs(z.actual_event_dates-z.predicted_event_dates).mean()),'test_n':len(z)}),include_groups=False)
    return m

def annual_metrics(pred):
    rows=[]
    for (strategy,year),g in pred.groupby(['strategy','year']):
        obs=g.actual_high_activity.to_numpy(int);alert=g.predicted_warning.to_numpy(int)
        tp=int(((obs==1)&(alert==1)).sum());fp=int(((obs==0)&(alert==1)).sum());fn=int(((obs==1)&(alert==0)).sum());tn=int(((obs==0)&(alert==0)).sum())
        prec=tp/(tp+fp) if tp+fp else np.nan;rec=tp/(tp+fn) if tp+fn else np.nan
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
        rows.append({'strategy':strategy,'year':int(year),'Recall':rec,'Precision':prec,'F1':f1,'FP':fp,'FN':fn,'TP':tp,'TN':tn,'test_n':len(g)})
    return pd.DataFrame(rows)

def monthly_top10(pred,top_n=10):
    if top_n<1:raise ValueError('top_n must be >=1')
    rows=[]
    for (strategy,year,month),g in pred.groupby(['strategy','year','month']):
        if g.adm_id.isna().any() or g.adm_id.duplicated().any():raise ValueError('Missing or duplicate ADM IDs in top-N cohort')
        n=min(top_n,len(g))
        observed=set(g.sort_values(['actual_event_dates','adm_id'],ascending=[False,True]).head(n).adm_id)
        predicted=set(g.sort_values(['predicted_event_dates','adm_id'],ascending=[False,True]).head(n).adm_id)
        rows.append({'strategy':strategy,'year':year,'month':month,'date':pd.Timestamp(year=year,month=month,day=1),
                     'top_n':n,'overlap_count':len(observed&predicted),'hit_rate':len(observed&predicted)/n})
    return pd.DataFrame(rows)

def chart_monthly_mae(m,out):
    fig,ax=plt.subplots(figsize=(22,7.7));fig.subplots_adjust(left=.065,right=.985,top=.82,bottom=.23)
    dates=sorted(m.date.unique());x=np.arange(len(dates))
    for strategy,color in STRATEGIES.items():
        z=m[m.strategy==strategy].set_index('date').reindex(dates)
        y=z.MAE.to_numpy(float)
        ax.plot(x,y,'o-',lw=2,ms=5,color=color,label=strategy)
        for xx,yy in zip(x,y):
            ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,9 if strategy=='Fixed' else -15),textcoords='offset points',fontsize=7.3,color=color,ha='center',weight='bold')
    ax.set_xticks(x,[pd.Timestamp(d).strftime('%Y-%m') for d in dates],rotation=90)
    ax.set_xlabel('Month (36 one-month-ahead test origins)');ax.set_ylabel('MAE (recorded event-date proxy)')
    ax.grid(axis='y',ls='--',alpha=.25);ax.legend(ncol=2,frameon=False)
    ymin,ymax=ax.get_ylim();ax.set_ylim(max(0,ymin-(ymax-ymin)*.1),ymax+(ymax-ymin)*.16)
    fig.suptitle('Monthly Prediction Performance | 2023–2025',fontsize=18,weight='bold')
    fig.text(.5,.07,'ADM3 | M4 = flood history + rainfall + runoff + ET0 | C = recent months + same months in preceding years',ha='center',fontsize=10)
    A.save(fig,out,'03_monthly_performance_36months')

def chart_annual(annual,out):
    years=sorted(annual.year.unique());x=np.arange(len(years));fig,axes=plt.subplots(2,2,figsize=(16,10))
    fig.subplots_adjust(left=.08,right=.96,top=.83,bottom=.13,wspace=.28,hspace=.35)
    for ax,metric,percent in zip(axes.flat,['Recall','F1','FP','FN'],[True,True,False,False]):
        for i,(strategy,color) in enumerate(STRATEGIES.items()):
            q=annual[annual.strategy==strategy].set_index('year').reindex(years)
            vals=q[metric].to_numpy(float);bars=ax.bar(x+(-.18 if i==0 else .18),vals,width=.34,color=color,label=strategy)
            for b,v in zip(bars,vals):
                label=f'{v*100:.1f}%' if percent else str(int(v))
                ax.annotate(label,(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,4),textcoords='offset points',ha='center',weight='bold',fontsize=9)
        ax.set_xticks(x,[str(v) for v in years]);ax.set_title(metric+' (%)' if percent else metric+' (ADM3-month count)',weight='bold')
        ax.set_ylim(0,max(1.05,float(annual[metric].max())*1.22)) if percent else ax.set_ylim(0,max(1,float(annual[metric].max())*1.22))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(frameon=False,loc='upper left');fig.suptitle('Annual Performance Summary | Fixed vs Monthly Rolling',weight='bold',fontsize=18)
    fig.text(.5,.055,'ADM3 | M4 + combined month memory (C) | 2023–2025 | Validation-defined warning thresholds',ha='center',fontsize=10)
    A.save(fig,out,'04_annual_performance_summary')

def chart_top10(top,out):
    fig,ax=plt.subplots(figsize=(22,7.2));fig.subplots_adjust(left=.06,right=.985,top=.82,bottom=.25)
    dates=sorted(top.date.unique());x=np.arange(len(dates))
    for strategy,color in STRATEGIES.items():
        z=top[top.strategy==strategy].set_index('date').reindex(dates)
        y=z.hit_rate.to_numpy(float)*100
        ax.plot(x,y,'o-',lw=2,ms=5,color=color,label=strategy)
        for xx,yy in zip(x,y):ax.annotate(f'{yy:.0f}%',(xx,yy),xytext=(0,8 if strategy=='Fixed' else -14),textcoords='offset points',fontsize=7.5,ha='center',color=color)
    ax.set_xticks(x,[pd.Timestamp(d).strftime('%Y-%m') for d in dates],rotation=90);ax.set_ylim(-5,105)
    ax.set_ylabel('Top-10 overlap (%)');ax.set_xlabel('Test month');ax.grid(axis='y',alpha=.25);ax.legend(frameon=False)
    fig.suptitle('Monthly Top-10 ADM3 Hotspot Overlap | 2023–2025',weight='bold',fontsize=18)
    fig.text(.5,.065,'Overlap of the 10 highest predicted and observed ADM3 activity-proxy values; tie-break by ADM3 ID.',ha='center',fontsize=10)
    A.save(fig,out,'05_monthly_top10_hit_rate')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=A.project_root());ap.add_argument('--output',type=Path)
    args=ap.parse_args();root=args.root.resolve();out=(args.output or root/'processed_data/multivariate_flood_modeling/12b_prediction_poster_pack').resolve();out.mkdir(parents=True,exist_ok=True)
    m,a,p=A.load_inputs(root)
    mm=monthly_mae(p);ann=annual_metrics(p);top=monthly_top10(p)
    mm.to_csv(out/'table_monthly_MAE.csv',index=False);ann.to_csv(out/'table_annual_metrics_fixed_rolling.csv',index=False)
    top.to_csv(out/'table_monthly_top10.csv',index=False)
    A.trend(a,out);A.dashboard(m,a,p,out);chart_monthly_mae(mm,out);chart_annual(ann,out);chart_top10(top,out)
    print('12b created 5 English poster figure pairs and CSV tables:',out)

if __name__=='__main__':main()
