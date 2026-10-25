#!/usr/bin/env python3
"""12c: final English poster selection + ADM3 spatial agreement map.

Requires both 12a and 12b Python scripts in the same folder. Uses existing
11c and 10b output files. Does not retrain Random Forest or tune thresholds.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import importlib
A=importlib.import_module('12a_generate_prediction_summary_figures')
B=importlib.import_module('12b_prediction_poster_pack')
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

STATUS_COLORS={'TP':'#2CB67D','FP':'#E09F4F','FN':'#D65A9A','TN':'#A9C7E2','No prediction':'#E8E5EB'}

def read_boundaries(root):
    p=root/'raw_data/administrative_boundaries/ssd_admin3.geojson'
    if not p.is_file():raise FileNotFoundError(f'Missing ADM3 boundaries: {p}')
    g=gpd.read_file(p)
    id_col=next((c for c in ('adm_id','adm3_pcode','ADM3_PCODE','admin3Pcod','ADM3_CODE','ADM3_PCode') if c in g),None)
    if id_col is None:raise ValueError('Cannot identify ADM3 ID in geojson. Columns: '+', '.join(g.columns))
    if not g.crs:raise ValueError('Boundary lacks CRS')
    g=g.rename(columns={id_col:'adm_id'}).to_crs('EPSG:6933')
    g['adm_id']=g.adm_id.astype('string').str.strip()
    if g.adm_id.isna().any() or g.adm_id.duplicated().any():raise ValueError('ADM3 boundary has null/duplicate identifiers')
    return g[['adm_id','geometry']].copy()

def agreement_month(pred,year,month):
    q=pred[(pred.strategy=='Monthly Rolling')&(pred.year==year)&(pred.month==month)].copy()
    if q.empty:raise ValueError(f'No model predictions for {year}-{month:02d}')
    actual=q.actual_high_activity.to_numpy(int);forecast=q.predicted_warning.to_numpy(int)
    q['status']=[('TP' if a==1 and b==1 else 'FN' if a==1 else 'FP' if b==1 else 'TN') for a,b in zip(actual,forecast)]
    return q[['adm_id','status']].copy()

def choose_representative_month(monthly):
    # Pick the busiest training-selected season month? Without training data, avoid hindsight-based peak selection.
    # Default fixed December: established seasonal comparison month, not selected on test performance.
    return 2025,12

def plot_agreement(root,pred,monthly,out,year=2025,month=12):
    boundary=read_boundaries(root)
    labels=agreement_month(pred,year,month)
    unknown=set(labels.adm_id)-set(boundary.adm_id)
    if unknown:raise ValueError(f'{len(unknown)} predictions cannot join to map; examples {sorted(unknown)[:5]}')
    g=boundary.merge(labels,on='adm_id',how='left',validate='one_to_one')
    g['status']=g.status.fillna('No prediction')
    fig,ax=plt.subplots(figsize=(14,10.5));fig.subplots_adjust(left=.02,right=.83,top=.88,bottom=.10)
    for status,color in STATUS_COLORS.items():
        sub=g[g.status==status]
        if len(sub):sub.plot(ax=ax,color=color,edgecolor='#B5B1BB',linewidth=.22)
    ax.set_axis_off();ax.set_aspect('equal')
    handles=[Patch(facecolor=color,edgecolor='#B5B1BB',label=f'{name} ({int((g.status==name).sum())})') for name,color in STATUS_COLORS.items()]
    ax.legend(handles=handles,loc='center left',bbox_to_anchor=(1.0,.55),frameon=False,title='ADM3 status',fontsize=10,title_fontsize=12)
    fig.suptitle(f'ADM3 Spatial Agreement | {year}-{month:02d}',weight='bold',fontsize=20,y=.97)
    fig.text(.5,.07,'TP = high activity correctly flagged | FP = false alarm | FN = missed high activity | TN = correct negative',ha='center',fontsize=10)
    fig.text(.5,.043,'High recorded activity is a proxy label, not confirmed flood inundation. Grey = no model prediction.',ha='center',fontsize=9,color='#5E5964')
    A.save(fig,out,'04_spatial_agreement_ADM3_poster')
    return {name:int((g.status==name).sum()) for name in STATUS_COLORS}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=A.project_root());ap.add_argument('--output',type=Path)
    ap.add_argument('--year',type=int,default=2025);ap.add_argument('--month',type=int,default=12)
    args=ap.parse_args()
    if not 1<=args.month<=12:ap.error('month must be 1..12')
    root=args.root.resolve();out=(args.output or root/'processed_data/multivariate_flood_modeling/12c_final_prediction_figures').resolve();out.mkdir(parents=True,exist_ok=True)
    monthly,annual,pred=A.load_inputs(root)
    mm=B.monthly_mae(pred)
    A.trend(annual,out);A.dashboard(monthly,annual,pred,out);B.chart_monthly_mae(mm,out)
    counts=plot_agreement(root,pred,monthly,out,args.year,args.month)
    (out/'README_12c.txt').write_text(
        '12c final poster figure set: English text, 300 dpi PNG and vector PDF.\n'
        '01: Annual ADM3 high-activity label IoU/Precision/Recall.\n'
        '02: Pooled ADM3-month summary.\n'
        '03: Monthly MAE of one-month-ahead predictions (2023-2025).\n'
        f'04: Spatial agreement map for {args.year}-{args.month:02d}. This is an example month, not a model chosen via test optimization.\n'
        'All classification metrics use labels pre-computed by 10b with its validation thresholds.\n'
        'No-prediction districts are excluded from metric denominators and shown in grey on the map.\n'
        'The target is recorded flood-event dates proxy; not a verified inundation extent.\n',encoding='utf-8')
    print('12c finished:',out,'Agreement counts:',counts)
if __name__=='__main__':main()
