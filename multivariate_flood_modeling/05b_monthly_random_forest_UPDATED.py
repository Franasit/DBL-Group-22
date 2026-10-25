#!/usr/bin/env python3
"""JBG060 monthly flood forecasting, ADM2 / ADM3, temporal strategies A/B/C.

Input is a REAL region-month table; annual flood days must NOT be divided by 12.
Required columns: adm_id,year,month,flood_days,rainfall_mm,runoff_mm,et0_mm.
One record per region per month; flood_days should represent a scientifically
validated monthly flood-day outcome, not an event-record count. If validity of
non-event observations cannot be established, output is exploratory only.

A = same month one year earlier (lead ~12 months)
B = last month (lead 1 month)
C = same month last year + last month + two months ago (lead 1 month)
M0 = historical flood days; M1 adds hydrology; M2 adds ET0.
For B/C, common complete-case rows ensure matched model comparisons.
Separate models are trained for A/B/C, since lead times differ.

Run:
  python scripts/multivariate_flood_modeling/05b_monthly_random_forest.py
  # Automatically uses files created by 05a

For a synthetic pipeline smoke-test (not a scientific result):
  python scripts/multivariate_flood_modeling/05b_monthly_random_forest.py --demo
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED = ['adm_id', 'year', 'month', 'flood_days', 'rainfall_mm', 'runoff_mm', 'et0_mm']
VARIABLES = ['flood_days','rainfall_mm','runoff_mm','et0_mm']
SPLITS = {'train':(2001,2019), 'validation':(2020,2022), 'test':(2023,2025)}
STRATEGIES = {'A':[12], 'B':[1], 'C':[12,1,2]}

def synthetic(level):
    """Only for testing software, NOT for reporting flood forecasting performance."""
    rng=np.random.default_rng(10 if level=='ADM2' else 11)
    ids=[f'{level}_{j:03d}' for j in range(6)]
    dates=pd.date_range('2000-01-01','2025-12-01',freq='MS')
    rows=[]
    for ix,aid in enumerate(ids):
        previous=12.
        for dt in dates:
            wet=max(0,np.sin((dt.month-5)/12*2*np.pi))
            rainfall=max(0,65+110*wet+rng.normal(0,20))
            runoff=max(0,rainfall*.18+rng.normal(0,5))
            et0=max(0,120-20*wet+rng.normal(0,5))
            flood=np.clip(.4*previous+.17*runoff+ix*1.8+rng.normal(0,4),0,dt.days_in_month)
            rows.append([aid,dt.year,dt.month,flood,rainfall,runoff,et0])
            previous=flood
    return pd.DataFrame(rows,columns=REQUIRED)

def load(path, level):
    df=pd.read_csv(path,low_memory=False)
    missing=set(REQUIRED)-set(df.columns)
    if missing: raise ValueError(f'{level} missing required columns: {sorted(missing)} in {path}')
    df=df[REQUIRED].copy()
    df['adm_id']=df['adm_id'].astype('string').str.strip()
    for c in REQUIRED[1:]: df[c]=pd.to_numeric(df[c],errors='coerce')
    if df[['adm_id','year','month']].isna().any().any():
        raise ValueError(f'{level} has missing/malformed adm_id/year/month keys')
    if ((df.month<1)|(df.month>12)|(df.year<1900)|(df.year>2100)|
        (df.year%1!=0)|(df.month%1!=0)).any():
        raise ValueError(f'{level} year/month outside valid range or fractional')
    df[['year','month']]=df[['year','month']].astype(int)
    if df.duplicated(['adm_id','year','month']).any():
        raise ValueError(f'{level} duplicate region-month rows; deduplicate upstream')
    days=pd.to_datetime(dict(year=df.year,month=df.month,day=1)).dt.days_in_month
    if (df['flood_days'].dropna()<0).any() or (df['flood_days']>days).any():
        raise ValueError(f'{level} flood_days invalid: expected values in 0..days_in_month')
    return df

def build_lags(df):
    x=df.copy()
    x['period']=x.year*12+x.month
    x=x.sort_values(['adm_id','period']).reset_index(drop=True)
    for lag in [1,2,12]:
        prev=x.groupby('adm_id',sort=False)['period'].shift(lag)
        # shift row position is not necessarily shift calendar month!
        valid=prev.eq(x.period-lag)
        for v in VARIABLES:
            x[f'{v}_lag{lag}']=x.groupby('adm_id',sort=False)[v].shift(lag).where(valid)
    return x

def feature_sets(strategy):
    lags=STRATEGIES[strategy]
    return {
      'M0':[f'flood_days_lag{k}' for k in lags],
      'M1':[f'{v}_lag{k}' for k in lags for v in ('flood_days','rainfall_mm','runoff_mm')],
      'M2':[f'{v}_lag{k}' for k in lags for v in VARIABLES]
    }

def score(y,p):
    return {'MAE':float(mean_absolute_error(y,p)),
            'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p)) if len(y)>1 else None}

def run_one(df, level, strategy, trees, seed):
    columns=feature_sets(strategy)
    # Fair comparison within one strategy: M0/M1/M2 evaluated on same M2 cohort.
    x=df.dropna(subset=['flood_days']+columns['M2']).copy()
    if x.empty: raise ValueError(f'{level}/{strategy}: zero complete monthly rows')
    date=x['year'].astype(str)+'-'+x['month'].astype(str).str.zfill(2)
    cohort=[]; predictions=[]; scores=[]; importances=[]
    for split,(lo,hi) in SPLITS.items():
        cohort.append({'level':level,'strategy':strategy,'split':split,'n':int(x.year.between(lo,hi).sum())})
    train=x[x.year.between(*SPLITS['train'])]
    if train.empty: raise ValueError(f'{level}/{strategy} training split is empty')
    baseline_col='flood_days_lag12' if strategy=='A' else 'flood_days_lag1'
    for model_name,cols in columns.items():
        model=RandomForestRegressor(n_estimators=trees,min_samples_leaf=3,n_jobs=-1,random_state=seed)
        model.fit(train[cols],train['flood_days'])
        for col,val in zip(cols,model.feature_importances_):
            importances.append(dict(level=level,strategy=strategy,model=model_name,feature=col,importance=float(val)))
        for split,(lo,hi) in SPLITS.items():
            z=x[x.year.between(lo,hi)]
            if z.empty: continue
            target=z.flood_days.to_numpy(float)
            max_days=pd.to_datetime(dict(year=z.year,month=z.month,day=1)).dt.days_in_month.to_numpy()
            rf=np.clip(model.predict(z[cols]),0,max_days)
            persistence=np.clip(z[baseline_col].to_numpy(float),0,max_days)
            for algorithm,pred in [('Persistence',persistence),('RandomForest',rf)]:
                scores.append(dict(level=level,strategy=strategy,model=model_name,algorithm=algorithm,
                                   split=split,n=len(z),**score(target,pred)))
                for adm,yr,mo,observed,p in zip(z.adm_id,z.year,z.month,target,pred):
                    predictions.append(dict(level=level,strategy=strategy,model=model_name,algorithm=algorithm,
                                            split=split,adm_id=str(adm),year=int(yr),month=int(mo),
                                            actual_flood_days=float(observed),predicted_flood_days=float(p)))
    return cohort,scores,predictions,importances

def plot(out,df,level):
    test=df[(df.level==level)&df.split.eq('test')&df.model.eq('M2')]
    if test.empty: return
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
    for ax,(strategy,title) in zip(axes,[('A','A: same month last year'),('B','B: last month'),('C','C: combined history')]):
        z=test[(test.strategy==strategy)&(test.algorithm=='RandomForest')]
        if len(z):
            ax.scatter(z.actual_flood_days,z.predicted_flood_days,s=8,alpha=.25)
        ax.plot([0,31],[0,31],ls='--',c='gray')
        ax.set(title=title,xlabel='Actual flood days',xlim=(0,31),ylim=(0,31))
    axes[0].set_ylabel('Predicted flood days')
    fig.suptitle(f'{level} monthly flood forecast (test 2023–2025, M2)')
    fig.tight_layout()
    fig.savefig(out/f'{level.lower()}_monthly_test.png',dpi=170)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    default_panels = project_root / 'processed_data/multivariate_flood_modeling/05a_monthly_panel'
    ap.add_argument('--adm2-panel',type=Path,default=default_panels / 'adm2_monthly_panel.csv',help='ADM2 monthly CSV (defaults to 05a output)')
    ap.add_argument('--adm3-panel',type=Path,default=default_panels / 'adm3_monthly_panel.csv',help='ADM3 monthly CSV (defaults to 05a output)')
    ap.add_argument('--output',type=Path,default=project_root / 'processed_data/multivariate_flood_modeling/05b_monthly_random_forest')
    ap.add_argument('--trees',type=int,default=250)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--demo',action='store_true',help='Use SYNTHETIC data for code testing only')
    args=ap.parse_args()
    if not args.demo:
        absent = [str(path) for path in (args.adm2_panel, args.adm3_panel) if not path.is_file()]
        if absent:
            ap.error('Monthly panel files do not exist. Run 05a_build_monthly_panel.py first. Missing: ' + '; '.join(absent))
        print('Using ADM2 panel:', args.adm2_panel)
        print('Using ADM3 panel:', args.adm3_panel)
        print('NOTE: If generated using 05a --event-proxy, results are exploratory event-date proxy forecasts, not verified flood-day predictions.')
    args.output.mkdir(parents=True,exist_ok=True)
    cohort=[];scores=[];pred=[];importance=[]
    for level,path in [('ADM2',args.adm2_panel),('ADM3',args.adm3_panel)]:
        if path is None and not args.demo:continue
        df=synthetic(level) if args.demo else load(path,level)
        x=build_lags(df)
        for strategy in STRATEGIES:
            c,s,p,i=run_one(x,level,strategy,args.trees,args.seed)
            cohort.extend(c);scores.extend(s);pred.extend(p);importance.extend(i)
    score_df=pd.DataFrame(scores)
    preds=pd.DataFrame(pred)
    pd.DataFrame(cohort).to_csv(args.output/'sample_coverage.csv',index=False)
    score_df.to_csv(args.output/'model_metrics.csv',index=False)
    preds.to_csv(args.output/'all_predictions.csv',index=False)
    pd.DataFrame(importance).to_csv(args.output/'feature_importances.csv',index=False)
    for lvl in preds.level.unique(): plot(args.output,preds,lvl)
    details={'dataset':'SYNTHETIC DEMONSTRATION - DO NOT REPORT' if args.demo else 'user-provided regional monthly panels',
             'required_columns':REQUIRED,'feature_sets':{k:feature_sets(k) for k in STRATEGIES},
             'time_splits':SPLITS,'year_ahead_A_vs_month_ahead_BC':'Different lead times; do not interpret as equal-horizon comparison.',
             'methodology_warning':'Event-only parquet records do not validate zero-flood observations or monthly flood-day outcome. Validate the target before interpreting metrics.',
             'cohort':'Within each level and strategy, M0 M1 M2 share complete M2 rows. A B C may have different cohorts.',
             'fit_policy':'No hyperparameter tuning or test-set fitting. Training years only. Validation for later tuning.',
             'prediction_lead':'A ~12 months; B/C 1 month'}
    (args.output/'methodology.json').write_text(json.dumps(details,indent=2),encoding='utf8')
    print('\n'+ ('SYNTHETIC SOFTWARE TEST — NOT SCIENTIFIC RESULTS\n' if args.demo else 'MONTHLY MODEL RESULTS\n'))
    print(score_df[score_df.split=='test'][['level','strategy','model','algorithm','n','MAE','RMSE','R2']].to_string(index=False))
    print('\nOutput:',args.output.resolve())

if __name__=='__main__': main()
