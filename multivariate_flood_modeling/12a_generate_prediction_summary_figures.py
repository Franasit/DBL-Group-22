#!/usr/bin/env python3
"""12a: English spatial-reliability trend and KPI summary from *existing* 11c/10b files.

Run from scripts/multivariate_flood_modeling/, no arguments needed.
Never re-estimates event or warning cutoffs from test observations.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE='#4C8BF5'; GREEN='#58B36B'; ORANGE='#F28B44'; PURPLE='#7546B7'
TP_C='#2CB67D'; FP_C='#E09F4F'; FN_C='#D65A9A'; TN_C='#A9C7E2'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def project_root():
    p=Path(__file__).resolve()
    return p.parents[2] if p.parent.name=='multivariate_flood_modeling' and p.parent.parent.name=='scripts' else Path.cwd()

def paths(root):
    base=root/'processed_data'/'multivariate_flood_modeling'
    return (base/'11c_spatial_prediction_reliability',base/'10b_monthly_rolling_origin',base/'12a_prediction_summary_figures')

def load_inputs(root):
    s,b,_=paths(root)
    monthly=s/'01_monthly_spatial_metrics.csv'; annual=s/'02_annual_spatial_metrics.csv'
    prediction=b/'01_test_predictions_monthly.csv'
    for file in (monthly,annual,prediction):
        if not file.is_file():raise FileNotFoundError(f'Missing prerequisite: {file}')
    m=pd.read_csv(monthly); a=pd.read_csv(annual)
    p=pd.read_csv(prediction,dtype={'adm_id':'string'})
    required={'level','memory','model','strategy','adm_id','year','month','actual_event_dates','predicted_event_dates','actual_high_activity','predicted_warning'}
    if not required.issubset(p):raise ValueError(f'10b predictions missing columns: {sorted(required-set(p))}')
    p=p[(p.level=='ADM3')&(p.memory=='C_combined')&(p.model=='M4_all_drivers')&p.strategy.isin(['Fixed','Monthly Rolling'])].copy()
    if p.empty:raise ValueError('No ADM3 M4 C predictions found in 10b CSV')
    if p.duplicated(['strategy','adm_id','year','month']).any():raise ValueError('Duplicate ADM3-month-strategy records')
    for c in ('year','month','actual_event_dates','predicted_event_dates','actual_high_activity','predicted_warning'):
        p[c]=pd.to_numeric(p[c],errors='raise')
    if p[['actual_event_dates','predicted_event_dates','actual_high_activity','predicted_warning']].isna().any().any():
        raise ValueError('Missing evaluated predictions; refusing to silently treat them as zero')
    if not p.actual_high_activity.isin([0,1]).all() or not p.predicted_warning.isin([0,1]).all():
        raise ValueError('Expected pre-computed binary validation-defined labels from 10b')
    # Compare independently computed 11c pooled counts to 10b to detect stale/mismatched datasets.
    q=p[p.strategy=='Monthly Rolling']
    counts={'TP':int(((q.actual_high_activity==1)&(q.predicted_warning==1)).sum()),
            'FP':int(((q.actual_high_activity==0)&(q.predicted_warning==1)).sum()),
            'FN':int(((q.actual_high_activity==1)&(q.predicted_warning==0)).sum()),
            'TN':int(((q.actual_high_activity==0)&(q.predicted_warning==0)).sum())}
    for name,num in counts.items():
        if int(a[name].sum())!=num:
            raise ValueError(f'11c/10b mismatch for {name}: 11c={int(a[name].sum())},10b={num}. Re-run 11c for same M4/C/Monthly Rolling configuration.')
    return m,a,p

def save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(out/f'{stem}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def trend(annual,out):
    annual=annual.sort_values('year')
    fig,ax=plt.subplots(figsize=(12.7,7))
    fig.subplots_adjust(bottom=.21,top=.82,left=.11,right=.95)
    for col,label,color,offset in [('IoU','Spatial IoU',BLUE,-13),('Precision','Precision',GREEN,11),('Recall','Recall',ORANGE,11)]:
        y=annual[col].to_numpy(float)*100
        ax.plot(annual.year,y,marker='o',markersize=9,lw=2.5,color=color,label=label)
        for xx,yy in zip(annual.year,y):ax.annotate(f'{yy:.2f}%',(xx,yy),xytext=(0,offset),textcoords='offset points',color=color,ha='center',weight='bold',fontsize=9)
    ax.set_xticks(annual.year.astype(int));ax.set_ylim(0,105);ax.set_yticks([0,25,50,75,100]);ax.set_yticklabels(['0%','25%','50%','75%','100%']);ax.grid(axis='y',ls='--',alpha=.25)
    ax.set_ylabel('ADM3-month classification metric')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.11),ncol=3,frameon=False)
    fig.suptitle('Spatial Reliability Across Test Years | ADM3 (2023–2025)',weight='bold',fontsize=18,y=.96)
    fig.text(.5,.04,'M4: flood history + rainfall + runoff + ET0 | C: combined month memory | Monthly Rolling',ha='center',fontsize=9)
    save(fig,out,'01_spatial_reliability_3years')

def dashboard(monthly,annual,pred,out):
    q=pred[pred.strategy=='Monthly Rolling']
    TP,FP,FN,TN=[int(annual[c].sum()) for c in ('TP','FP','FN','TN')]
    recall=100*TP/(TP+FN) if TP+FN else np.nan
    iou=100*TP/(TP+FP+FN) if TP+FP+FN else np.nan
    top=100*monthly.top_hit_rate.mean()
    n=q.adm_id.nunique();den=512
    fig=plt.figure(figsize=(14.8,8),facecolor='white')
    fig.patches.append(FancyBboxPatch((.04,.09),.92,.82,boxstyle='round,pad=0.015,rounding_size=.025',transform=fig.transFigure,facecolor='#F8F8F9',edgecolor='#DDDBE1',lw=1.2,zorder=-1))
    fig.text(.08,.85,'ADM3 | M4 + Combined Memory (C) | Monthly Rolling',fontsize=20,weight='bold')
    for x,y,label,value,note in [
        (.08,.75,'Overall Recall',f'{recall:.2f}%','Share of actual high-activity ADM3-months correctly flagged'),
        (.55,.75,'Pooled ADM3-Month IoU',f'{iou:.2f}%','Intersection / union of high-activity ADM3-month labels'),
        (.08,.53,'Mean Monthly Top-10 Hit Rate',f'{top:.2f}%','Predicted top-10 versus observed top-10 each month'),
        (.55,.53,'ADM3 Coverage',f'{n} / {den}',f'{100*n/den:.1f}% of all 512 ADM3 units evaluated')]:
        fig.text(x,y,label,fontsize=12,color='#59565D');fig.text(x,y-.07,value,fontsize=28,weight='bold');fig.text(x,y-.125,note,fontsize=9,color='#66616C')
    fig.add_artist(plt.Line2D([.08,.92],[.33,.33],transform=fig.transFigure,color='#DDD9E0'))
    for i,(label,value,color) in enumerate([('TP | Correct warning',TP,TP_C),('FP | False alarm',FP,FP_C),('FN | Missed warning',FN,FN_C),('TN | Correct negative',TN,TN_C)]):
        x=.08+i*.22
        fig.add_artist(plt.Line2D([x,x+.19],[.285,.285],transform=fig.transFigure,color=color,lw=5,solid_capstyle='round'))
        fig.text(x,.25,label,fontsize=10,color='#555');fig.text(x,.175,f'{value:,}',fontsize=23,weight='bold')
    fig.text(.08,.108,'Unit: ADM3-months (2023–2025). Recorded event-date proxy; NOT pixel-level inundation IoU.',fontsize=9,color='#555')
    save(fig,out,'02_prediction_summary_dashboard')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=project_root());ap.add_argument('--output',type=Path)
    args=ap.parse_args();root=args.root.resolve();out=(args.output or paths(root)[2]).resolve()
    m,a,p=load_inputs(root);trend(a,out);dashboard(m,a,p,out)
    print('12a created 2 English poster figures (PNG/PDF):',out)

if __name__=='__main__':main()
