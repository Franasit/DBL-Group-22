#!/usr/bin/env python3
"""11c — ADM3 spatial reliability of monthly high-activity warnings.

Input: Script 10b 01_test_predictions_monthly.csv (already predicted months).
No training, no new threshold selection, and no assumption that zero recorded
flood-event dates means an observed dry month. The binary label is specifically
HIGH RECORDED FLOOD ACTIVITY, NOT ANY FLOOD OCCURRENCE.

Outputs: 36-month IoU/time series, yearly summary, Top-10 overlap,
three yearly selected-peak-season spatial agreement maps (TP/FP/FN/TN),
per-ADM3 counts and coverage, plus CSVs and PDF/PNG charts.

Note: IoU is based on ADM3 ADMINISTRATIVE UNIT classification, not satellite
flood-pixel extent or inundated area. Missing predictions stay 'NO PREDICTION'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

MODEL = {
    'M0_history': 'M0: flood history',
    'M1_rainfall': 'M1: flood history + rainfall',
    'M2_runoff': 'M2: flood history + runoff',
    'M3_et0': 'M3: flood history + ET0',
    'M4_all_drivers': 'M4: flood history + rainfall + runoff + ET0',
}
MEMORY = {
    'A_same_month_previous_years': 'A: same month in previous two years',
    'B_recent_months': 'B: previous 1–3 months',
    'C_combined': 'C: previous 1–3 months + same month 1–2 years ago',
}
STATUS_COLORS = {
    'TP': '#3AAE99',
    'FP': '#E5A15C',
    'FN': '#CC5D91',
    'TN': '#D4E6F4',
    'NO_PREDICTION': '#E8E5EB',
}
STATUS_NAMES = {
    'TP': 'Hit (TP)', 'FP': 'False alarm (FP)',
    'FN': 'Miss (FN)', 'TN': 'Correct negative (TN)',
    'NO_PREDICTION': 'No prediction',
}
BLUE = '#7EC9F2'
PURPLE = '#7546B7'
INK = '#342E3B'
NOTE = ('High recorded flood-activity proxy warning; NOT a confirmed flood-occurrence forecast. '
        'IoU counts ADM3 units, NOT inundated-pixel or area overlap.')


def repo_root():
    f = Path(__file__).resolve()
    if f.parent.name == 'multivariate_flood_modeling' and f.parent.parent.name == 'scripts':
        return f.parents[2]
    return Path.cwd()


def read_prediction_csv(path, model, memory, strategy):
    if not path.exists():
        raise FileNotFoundError(f'Missing 10b output: {path}\nRun 10b first.')
    cols = ['level', 'memory', 'model', 'strategy', 'adm_id', 'year', 'month',
            'actual_event_dates', 'predicted_event_dates', 'actual_high_activity',
            'predicted_warning', 'event_threshold', 'warning_threshold']
    data = pd.read_csv(path, usecols=cols, dtype={'adm_id': 'string'})
    p = data.loc[(data.level == 'ADM3') & (data.memory == memory) &
                 (data.model == model) & (data.strategy == strategy)].copy()
    if p.empty:
        raise ValueError('No ADM3 records with the requested model, memory and strategy.')
    p['adm_id'] = p.adm_id.str.strip()
    for col in ['year', 'month', 'actual_high_activity', 'predicted_warning']:
        p[col] = pd.to_numeric(p[col], errors='raise').astype(int)
    if not p.year.between(2023, 2025).all() or not p.month.between(1, 12).all():
        raise ValueError('Expected 10b evaluation period 2023–2025.')
    if p.duplicated(['adm_id', 'year', 'month']).any():
        raise ValueError('Duplicate ADM3/month prediction rows.')
    if p[cols[7:]].isna().any().any():
        raise ValueError('Null outcome, warning or threshold in 10b output.')
    if not p.actual_high_activity.isin((0, 1)).all() or not p.predicted_warning.isin((0, 1)).all():
        raise ValueError('Warning labels must be binary 0/1.')
    # Assert stored warnings match the frozen threshold definitions.
    observed_check = (p.actual_event_dates >= p.event_threshold).astype(int)
    predicted_check = (p.predicted_event_dates >= p.warning_threshold).astype(int)
    if not observed_check.equals(p.actual_high_activity) or not predicted_check.equals(p.predicted_warning):
        raise ValueError('The 10b high-activity flags disagree with their stored thresholds.')
    p['date'] = pd.to_datetime(dict(year=p.year, month=p.month, day=1))
    actual, predicted = p.actual_high_activity, p.predicted_warning
    p['status'] = np.select(
        [(actual == 1) & (predicted == 1), (actual == 0) & (predicted == 1),
         (actual == 1) & (predicted == 0), (actual == 0) & (predicted == 0)],
        ['TP', 'FP', 'FN', 'TN'], default='INVALID')
    return p


def read_boundary(path):
    if not path.exists():
        raise FileNotFoundError(f'ADM3 boundary file missing: {path}')
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError('Boundary CRS is missing.')
    candidates = ('adm_id', 'adm3_pcode', 'ADM3_PCODE', 'admin3Pcod', 'ADM3_CODE', 'ADM3_PCode')
    key = next((c for c in candidates if c in g.columns), None)
    if key is None:
        raise ValueError(f'No ADM3 ID column. Available: {list(g.columns)}')
    g['adm_id'] = g[key].astype('string').str.strip()
    if g.adm_id.isna().any() or g.adm_id.duplicated().any():
        raise ValueError('ADM3 boundary IDs are missing or duplicated.')
    return g[['adm_id', 'geometry']].to_crs('EPSG:6933')


def score(group):
    counts = group.status.value_counts()
    tp, fp, fn, tn = [int(counts.get(k, 0)) for k in ('TP', 'FP', 'FN', 'TN')]
    d_iou, d_prec, d_recall = tp+fp+fn, tp+fp, tp+fn
    precision = tp/d_prec if d_prec else np.nan
    recall = tp/d_recall if d_recall else np.nan
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else np.nan
    return dict(test_n=len(group), TP=tp, FP=fp, FN=fn, TN=tn,
                Precision=precision, Recall=recall, F1=f1,
                IoU=tp/d_iou if d_iou else np.nan,
                observed_positive_n=int(group.actual_high_activity.sum()),
                predicted_positive_n=int(group.predicted_warning.sum()),
                MAE=float((group.predicted_event_dates - group.actual_event_dates).abs().mean()))


def top_overlap(group, n):
    if len(group) < n:
        return dict(top_n=n, top_hit_n=np.nan, top_hit_rate=np.nan, top_jaccard=np.nan)
    # Fixed secondary sort ensures reproducibility when activity values tie.
    obs = set(group.sort_values(['actual_event_dates', 'adm_id'], ascending=[False, True])
              .head(n).adm_id)
    pred = set(group.sort_values(['predicted_event_dates', 'adm_id'], ascending=[False, True])
               .head(n).adm_id)
    hits = len(obs & pred)
    return dict(top_n=n, top_hit_n=hits, top_hit_rate=hits/n,
                top_jaccard=hits/(2*n-hits))


def metrics_tables(p, n):
    monthly = []
    for (year, month), part in p.groupby(['year', 'month'], sort=True):
        monthly.append(dict(year=year, month=month, date=f'{year}-{month:02d}',
                            **score(part), **top_overlap(part, n)))
    bymonth = pd.DataFrame(monthly)
    yearly = []
    for year, part in p.groupby('year', sort=True):
        yearly.append(dict(year=year, **score(part),
                           mean_monthly_IoU=float(bymonth.loc[bymonth.year.eq(year),'IoU'].mean()),
                           mean_monthly_top_hit_rate=float(bymonth.loc[bymonth.year.eq(year),'top_hit_rate'].mean())))
    byyear = pd.DataFrame(yearly)
    byregion = []
    for adm_id, part in p.groupby('adm_id', sort=True):
        byregion.append(dict(adm_id=adm_id, months_with_predictions=len(part), **score(part)))
    return bymonth, byyear, pd.DataFrame(byregion)


def peak_months_from_training(root, prediction, top_count, override=None):
    """Select peak calendar months with TRAINING-PERIOD proxy only, if available.

    Fallback is safe: select one predefined calendar month (October), clearly
    flagged as an illustrative choice, not selected using 2023–2025 outcomes.
    """
    if override:
        return [override], 'User-specified illustration month; no peak claim'
    panel = root/'processed_data/multivariate_flood_modeling/05a_monthly_panel/adm3_monthly_panel.csv'
    if not panel.exists():
        raise FileNotFoundError(f'Training-period monthly panel missing: {panel}. Use --month to pick a month explicitly.')
    needed = ['adm_id', 'year', 'month', 'flood_days']
    raw = pd.read_csv(panel, usecols=needed, dtype={'adm_id':'string'})
    raw['adm_id'] = raw.adm_id.str.strip()
    raw = raw.loc[raw.year.between(2002, 2019) & raw.adm_id.isin(prediction.adm_id.unique())].copy()
    if raw.empty:
        raise ValueError('No 2002–2019 training-period rows found for the predicted ADM3 cohort.')
    # Mean across ADM3-month values (model-eligible cohort), not area-weighted.
    stats = raw.groupby('month').flood_days.agg(['mean', 'count']).reset_index()
    if not set(range(1,13)).issubset(stats.month):
        raise ValueError('Incomplete historical month list, cannot choose historical peaks safely.')
    stats = stats.sort_values(['mean','month'], ascending=[False,True])
    months = stats.head(top_count)['month'].astype(int).tolist()
    return months, 'Selected using 2002–2019 mean recorded proxy over model-eligible ADM3 only; observation completeness unverified'


def save(fig, folder, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(folder/f'{stem}.{ext}', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def style_geo(ax, geo):
    geo.boundary.plot(ax=ax, color='#B9B2C1', linewidth=.23, zorder=4)
    ax.set_aspect('equal'); ax.set_axis_off()


def plot_status_map(boundary, part, label, details, out, stem):
    subset = part[['adm_id', 'status']].copy()
    joined = boundary.merge(subset, on='adm_id', how='left', validate='one_to_one')
    joined['status'] = joined.status.fillna('NO_PREDICTION')
    fig, ax = plt.subplots(figsize=(14.5, 9.5))
    fig.subplots_adjust(left=.03, right=.76, top=.86, bottom=.12)
    for status, color in STATUS_COLORS.items():
        sub = joined[joined.status.eq(status)]
        if not sub.empty:
            sub.plot(ax=ax, color=color, edgecolor='none', zorder=2)
    style_geo(ax, joined)
    legend = [Patch(facecolor=c, edgecolor='#BBB3C3', label=f'{STATUS_NAMES[k]} ({int((joined.status==k).sum())})')
              for k,c in STATUS_COLORS.items()]
    ax.legend(handles=legend, loc='center left', bbox_to_anchor=(1.02,.63), frameon=False,
              title='High-activity classification', fontsize=10, title_fontsize=11)
    fig.suptitle(f'ADM3 | Spatial Agreement of Monthly High-Activity Warnings | {label}',
                 fontsize=18, weight='bold', y=.965)
    fig.text(.5,.08,details,ha='center', fontsize=10, weight='bold', color=INK)
    fig.text(.5,.045,NOTE,ha='center',fontsize=8.5,color='#655D69')
    save(fig,out,stem)


def plot_monthly(bymonth, out):
    fig, axes = plt.subplots(2,1,figsize=(19,10),sharex=True)
    fig.subplots_adjust(left=.065,right=.985,top=.88,bottom=.18,hspace=.28)
    xx = np.arange(len(bymonth))
    axes[0].plot(xx, bymonth.IoU.to_numpy(float), 'o-', color=PURPLE, lw=2.2, ms=5.5)
    axes[0].set_ylabel('IoU (ADM3 unit overlap)', weight='bold')
    axes[0].set_ylim(0,1.09)
    for i,val in enumerate(bymonth.IoU):
        if pd.notna(val):
            axes[0].annotate(f'{val:.2f}',(i,val),xytext=(0,7),textcoords='offset points',
                             ha='center',fontsize=7.6,color=PURPLE)
    axes[1].bar(xx-.18, bymonth.FN, width=.36, color='#CC5D91', label='FN (miss)')
    axes[1].bar(xx+.18, bymonth.FP, width=.36, color='#E5A15C', label='FP (false alarm)')
    axes[1].set_ylabel('ADM3-month count', weight='bold')
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_xticks(xx, bymonth.date, rotation=90)
    for ax in axes:
        ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    fig.suptitle('ADM3 | Monthly Spatial Warning Agreement (2023–2025)\nIoU and Miss / False Alarm Counts',
                 fontsize=18, weight='bold',y=.98)
    fig.text(.5,.065,'IoU = TP / (TP + FP + FN); undefined when no high-activity units exist in either set.',
             ha='center',fontsize=10)
    fig.text(.5,.035,NOTE,ha='center',fontsize=8.5,color='#655D69')
    save(fig,out,'01_monthly_IoU_FN_FP_36months')


def plot_top10(bymonth, out, n):
    fig, ax=plt.subplots(figsize=(18,6.5))
    fig.subplots_adjust(left=.07,right=.985,top=.82,bottom=.25)
    x=np.arange(len(bymonth)); y=bymonth.top_hit_rate.to_numpy(float)
    bars=ax.bar(x,y,color=BLUE,width=.8)
    for b,v in zip(bars,y):
        if np.isfinite(v):
            ax.annotate(f'{v*100:.0f}%', (b.get_x()+b.get_width()/2,v),
                        xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.set_ylim(0,1.13); ax.set_yticks(np.linspace(0,1,6))
    ax.set_ylabel(f'Overlap@{n} (%)',weight='bold')
    ax.set_xticks(x,bymonth.date,rotation=90)
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    fig.suptitle(f'ADM3 | Monthly Top-{n} Recorded-Activity Hotspot Overlap (2023–2025)',
                 fontsize=17,weight='bold',y=.96)
    fig.text(.5,.075,'Rankings use the observed and predicted proxy values, not verified flood area; ADM3 IDs break exact ties.',
             ha='center',fontsize=9.5)
    save(fig,out,'02_monthly_top10_overlap_36months')


def plot_annual(byyear, out):
    fig, axs=plt.subplots(2,2,figsize=(15.8,10))
    fig.subplots_adjust(left=.08,right=.985,top=.85,bottom=.16,hspace=.32,wspace=.25)
    specs=[('IoU','ADM3-unit IoU',False),('Recall','High-activity Recall',True),
           ('Precision','High-activity Precision',True),('mean_monthly_top_hit_rate','Mean monthly Top-10 overlap',True)]
    xx=np.arange(len(byyear)); years=byyear.year.astype(str)
    for ax,(col,title,percent) in zip(axs.flat,specs):
        vals=byyear[col].to_numpy(float)
        bars=ax.bar(xx,vals,color=PURPLE if col=='IoU' else BLUE,width=.65)
        for b,val in zip(bars,vals):
            if np.isfinite(val):
                ax.annotate(f'{val*100:.1f}%' if percent else f'{val:.3f}',
                            (b.get_x()+b.get_width()/2,val),xytext=(0,5),
                            textcoords='offset points',ha='center',fontsize=10,weight='bold')
        ax.set_title(title,weight='bold');ax.set_xticks(xx,years)
        ax.set_ylim(0,1.15);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.suptitle('ADM3 | Annual Spatial Reliability Summary (2023–2025)',fontsize=18,weight='bold',y=.96)
    fig.text(.5,.07,'Annual IoU/Recall/Precision pool ADM3-month cases; Top-10 overlap is the mean of monthly overlap values.',
             ha='center',fontsize=9.5)
    fig.text(.5,.04,NOTE,ha='center',fontsize=8.5,color='#655D69')
    save(fig,out,'03_annual_spatial_reliability_summary')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=repo_root())
    ap.add_argument('--predictions',type=Path,default=None)
    ap.add_argument('--boundary',type=Path,default=None)
    ap.add_argument('--output',type=Path,default=None)
    ap.add_argument('--model',choices=list(MODEL),default='M4_all_drivers')
    ap.add_argument('--memory',choices=list(MEMORY),default='C_combined')
    ap.add_argument('--strategy',choices=['Fixed','Monthly Rolling'],default='Monthly Rolling')
    ap.add_argument('--top-n',type=int,default=10)
    ap.add_argument('--month',type=int,choices=range(1,13),default=None,
                    help='Override peak-season selection with an explicitly named calendar month')
    ap.add_argument('--historical-top',type=int,choices=[1,2,3],default=1,
                    help='Number of historic (2002–2019) highest-proxy calendar months to map in each test year')
    args=ap.parse_args()
    if args.top_n<1: ap.error('--top-n must be >=1')
    root=args.root.expanduser().resolve()
    def path(v, default):
        p=Path(v) if v is not None else Path(default)
        return (p if p.is_absolute() else root/p).resolve()
    predpath=path(args.predictions,'processed_data/multivariate_flood_modeling/10b_monthly_rolling_origin/01_test_predictions_monthly.csv')
    boundarypath=path(args.boundary,'raw_data/administrative_boundaries/ssd_admin3.geojson')
    out=path(args.output,'processed_data/multivariate_flood_modeling/11c_spatial_prediction_reliability')
    out.mkdir(parents=True,exist_ok=True)
    p=read_prediction_csv(predpath,args.model,args.memory,args.strategy)
    g=read_boundary(boundarypath)
    unmatched=set(p.adm_id)-set(g.adm_id)
    if unmatched:
        raise ValueError(f'{len(unmatched)} predicted ADM3 IDs do not match boundary IDs. Examples: {list(sorted(unmatched))[:6]}')
    bymonth,byyear,byregion=metrics_tables(p,args.top_n)
    if len(bymonth)!=36:
        warnings.warn(f'Found {len(bymonth)} test months, expected 36. Some may be missing.')
    byregion=g[['adm_id']].merge(byregion,on='adm_id',how='left',validate='one_to_one')
    byregion['has_prediction']=byregion.months_with_predictions.notna()
    p.to_csv(out/'00_adm3_monthly_classifications.csv',index=False)
    bymonth.to_csv(out/'01_monthly_spatial_metrics.csv',index=False)
    byyear.to_csv(out/'02_annual_spatial_metrics.csv',index=False)
    byregion.to_csv(out/'03_adm3_reliability_coverage.csv',index=False)
    plot_monthly(bymonth,out)
    plot_top10(bymonth,out,args.top_n)
    plot_annual(byyear,out)
    months,selection=peak_months_from_training(root,p,args.historical_top,args.month)
    chosen=[]
    for year in (2023,2024,2025):
        for month in months:
            sub=p[(p.year==year)&(p.month==month)]
            if sub.empty:
                continue
            stats=score(sub)
            label=f'{year}-{month:02d}'
            detail=(f"{MODEL[args.model]} | {MEMORY[args.memory]} | {args.strategy}   "
                    f"IoU {stats['IoU']:.3f} | TP {stats['TP']} | FP {stats['FP']} | FN {stats['FN']} | TN {stats['TN']}")
            plot_status_map(g,sub,label,detail,out,f'04_adm3_spatial_agreement_{label}')
            chosen.append(label)
    methodology={
        'objective':'Evaluate spatial high-activity warning classification across ADM3 administrative units',
        'model':MODEL[args.model], 'memory':MEMORY[args.memory], 'strategy':args.strategy,
        'test_period':'2023-01 to 2025-12',
        'binary_labels':'Use 10b actual_high_activity and predicted_warning, validated against stored frozen thresholds',
        'IoU_definition':'TP/(TP+FP+FN) on ADM3 units; not satellite footprint or inundation-area intersection',
        'IoU_undefined':'NaN if both actual and predicted positive sets empty',
        'top_n':args.top_n,
        'top_n_method':'Sort each month by proxy observed or predicted activity, then deterministic ADM3 ID tie break; use identical available ADM3 cohort',
        'top_n_limitation':'Low-activity/zero ties may reduce substantive meaning in a month with fewer than N active areas',
        'maps_months':chosen,'maps_month_selection':selection,
        'missing_prediction_count':int((~byregion.has_prediction).sum()),
        'not_observed_zero':'No record or zero proxy does not establish an observed flood-free month because effective remote sensing coverage is not known',
        'known_limitations':[
            'Only ADM3 with complete 10b features and predictions are evaluated; nonpredicted ADM3 are grey.',
            'This is historical backtesting, not current/future operational prediction.',
            '2023-2025 results were previously inspected during model development.',
            'Binary high-activity labels are NOT any-flood yes/no labels.',
            'Spatial agreement measures ADM3 classification, NOT location of flooded pixels within ADM3.',
            'Top-N overlap compares a rank ordering of recorded-activity proxy, not true flood hazard footprint.',
            'Remote sensing observation opportunity is not fully validated.',
        ]}
    (out/'04_methodology.json').write_text(json.dumps(methodology,indent=2,ensure_ascii=False),encoding='utf8')
    print('\nOutput:',out)
    print('Selected historically high-proxy months:', months)
    print('Predicted ADM3:',p.adm_id.nunique(),'of boundary ADM3:',len(g))
    print(byyear[['year','test_n','IoU','Precision','Recall','F1','FP','FN','mean_monthly_top_hit_rate']].to_string(index=False))
    print('\nNOTE:',NOTE)


if __name__=='__main__':
    main()
