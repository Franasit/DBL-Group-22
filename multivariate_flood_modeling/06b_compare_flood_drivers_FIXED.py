#!/usr/bin/env python3
"""JBG060 driver ablation: ADM2/ADM3, strategy C, monthly Random Forest.

Forecast t using lag1,lag2,lag12. For each added driver use exactly the
same region-month cohort as its reference baseline to compute delta MAE.
Lake and DFO features are only available if 06a has a REVIEWED mapping.
Uses event-date PROXY targets when inputs originate from 05a --event-proxy.
Not a causal analysis. Units and observation completeness must be documented.

Run with no arguments from any directory (defaults to 05a/06a output).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

SPLITS={'train':(2001,2019),'validation':(2020,2022),'test':(2023,2025)}
LAGS=[12,2,1]
DRIVERS={
    'rainfall':['rainfall_mm'],
    'runoff':['runoff_mm'],
    'et0':['et0_mm'],
    'lake_level':['lake_level_mean'],
    'dfo_discharge':['dfo_discharge_mean'],
}


def load(path):
    df=pd.read_csv(path,dtype={'adm_id':str},low_memory=False)
    required={'adm_id','year','month','flood_days'}
    if not required.issubset(df):raise ValueError(f'{path} missing {sorted(required-set(df))}')
    # pandas 3.x removed errors='ignore'. Convert only known numeric fields;
    # keep administrative IDs, target labels and station IDs as text.
    numeric_cols = ['year', 'month', 'flood_days'] + [
        field for fields in DRIVERS.values() for field in fields
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[['year', 'month']].isna().any().any():
        raise ValueError(f'Invalid year/month values in {path}')
    df['year']=df['year'].astype(int)
    df['month']=df['month'].astype(int)
    if df.duplicated(['adm_id','year','month']).any():raise ValueError('Duplicate ADM-year-month rows')
    return df


def lags(df):
    x=df.copy()
    x['period']=x.year*12+x.month
    x=x.sort_values(['adm_id','period']).reset_index(drop=True)
    src=['flood_days']+[v for vals in DRIVERS.values() for v in vals if v in x.columns]
    for k in LAGS:
        before=x.groupby('adm_id',sort=False)['period'].shift(k)
        valid=before.eq(x.period-k)
        for c in src:
            val=pd.to_numeric(x[c],errors='coerce')
            x[f'{c}_lag{k}']=val.groupby(x.adm_id,sort=False).shift(k).where(valid)
    return x


def fit_compare(df,level,driver,trees,seed):
    baseline=[f'flood_days_lag{k}' for k in LAGS]
    extra=[f'{v}_lag{k}' for v in DRIVERS[driver] for k in LAGS]
    if any(v not in df.columns for v in extra):
        return None,{'level':level,'driver':driver,'status':'NO_FEATURE_COLUMNS','test_n':0}
    cohort=df.dropna(subset=['flood_days']+baseline+extra).copy()
    tr=cohort[cohort.year.between(*SPLITS['train'])]
    va=cohort[cohort.year.between(*SPLITS['validation'])]
    te=cohort[cohort.year.between(*SPLITS['test'])]
    if len(tr)<30 or len(te)<15:
        return None,{'level':level,'driver':driver,'status':'INSUFFICIENT_MAPPED_SAMPLES','train_n':len(tr),'test_n':len(te)}
    ytr=tr.flood_days.astype(float).to_numpy()
    yte=te.flood_days.astype(float).to_numpy()
    days=pd.to_datetime(dict(year=te.year,month=te.month,day=1)).dt.days_in_month.to_numpy()
    rows=[];predictions=[]
    for variant,cols in [('baseline',baseline),('baseline_plus_'+driver,baseline+extra)]:
        model=RandomForestRegressor(n_estimators=trees,min_samples_leaf=3,random_state=seed,n_jobs=-1)
        model.fit(tr[cols],ytr)
        yhat=np.clip(model.predict(te[cols]),0,days)
        scores={'MAE':mean_absolute_error(yte,yhat), 'RMSE':np.sqrt(mean_squared_error(yte,yhat)),
                'R2':r2_score(yte,yhat)}
        rows.append({'level':level,'driver':driver,'variant':variant,'train_n':len(tr),'validation_n':len(va),
                     'test_n':len(te),**{k:float(v) for k,v in scores.items()}})
        for adm,yr,mo,act,pred in zip(te.adm_id,te.year,te.month,yte,yhat):
            predictions.append({'level':level,'driver':driver,'variant':variant,'adm_id':adm,
                                'year':int(yr),'month':int(mo),'actual':float(act),'predicted':float(pred)})
    comparison={
        'level':level,'driver':driver,'status':'OK','train_n':len(tr),'validation_n':len(va),'test_n':len(te),
        'baseline_MAE':rows[0]['MAE'],'with_driver_MAE':rows[1]['MAE'],
        'delta_MAE_baseline_minus_driver':rows[0]['MAE']-rows[1]['MAE'],
        'pct_improvement':100*(rows[0]['MAE']-rows[1]['MAE'])/rows[0]['MAE'] if rows[0]['MAE'] else np.nan,
    }
    return (rows,predictions),comparison


def plot(df,output):
    ok=df[df.status=='OK'].copy()
    if ok.empty:return
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,constrained_layout=True)
    for ax,level in zip(axes,['ADM2','ADM3']):
        d=ok[ok.level==level].sort_values('delta_MAE_baseline_minus_driver')
        ax.barh(d.driver,d.delta_MAE_baseline_minus_driver)
        ax.axvline(0,ls='--',lw=1,c='black')
        ax.set_title(level)
        ax.set_xlabel('MAE reduction versus same-cohort baseline (positive = improvement)')
    fig.suptitle('Incremental Driver Predictive Value | Monthly Strategy C | 2023-2025')
    fig.savefig(output/'driver_incremental_mae.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[2]
    ap.add_argument('--root',type=Path,default=root)
    ap.add_argument('--trees',type=int,default=250)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    root=args.root.resolve()
    out=root/'processed_data/multivariate_flood_modeling/06b_driver_contribution'
    out.mkdir(parents=True,exist_ok=True)
    results=[];scores=[];preds=[];sources={}
    for level in ['ADM2','ADM3']:
        augmented=root/f'processed_data/multivariate_flood_modeling/06a_lake_dfo_monthly/{level.lower()}_monthly_augmented.csv'
        original=root/f'processed_data/multivariate_flood_modeling/05a_monthly_panel/{level.lower()}_monthly_panel.csv'
        path=augmented if augmented.exists() else original
        if not path.exists():raise FileNotFoundError(f'Run 05a first; missing {original}')
        sources[level]=str(path)
        df=lags(load(path))
        for driver in DRIVERS:
            fit,comp=fit_compare(df,level,driver,args.trees,args.seed)
            results.append(comp)
            print(level,driver,comp['status'],f"test_n={comp.get('test_n',0)}")
            if fit:
                score,pred=fit;scores.extend(score);preds.extend(pred)
    table=pd.DataFrame(results)
    table.to_csv(out/'driver_contribution_summary.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'model_metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(out/'test_predictions.csv',index=False)
    plot(table,out)
    note={
        'target':'Monthly flood event-date PROXY unless verified target data provided',
        'strategy':'C: target month predicted from 1, 2, 12 months earlier',
        'test_years':'2023-2025',
        'interpretation':'Delta MAE positive means the driver improved out-of-time predictive error versus a baseline trained on EXACTLY the same sample.',
        'warning':'Comparisons across different drivers may use different ADM/time subsets; only compare paired baseline vs driver within each row. Feature importance does not establish causal effect.',
        'lake_and_dfo':'Only included if reviewed 06a station-to-ADM mapping yields matched and temporally available observations.',
        'sources':sources,
    }
    (out/'methodology.json').write_text(json.dumps(note,indent=2),encoding='utf8')
    print('Output:',out)

if __name__=='__main__':main()
