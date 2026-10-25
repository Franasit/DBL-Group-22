#!/usr/bin/env python3
"""JBG060 | Annual ADM2/ADM3 flood-days prediction: persistence vs Random Forest.

Run from DBL-Group-22 root:
    python scripts/multivariate_flood_modeling/05_random_forest.py

Only prior-calendar-year features are used (no current-year climate leakage).
M0 = previous flood days (naive persistence); M1 = M0 + prior rainfall/runoff;
M2 = M1 + prior reference ET0. M1/M2 share the same complete-case rows
within each administrative level, so their differences are comparable.
The baseline M0 is also evaluated on exactly those matched M1/M2 rows.

Inputs: existing 01/02 full_analysis_panel.csv and Script 04 ADM3 hydrology.
Never changes inputs. It does not claim to predict operational flood warnings.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SPECS = {
 'ADM2': {'dir':'01_adm2_panel_dataset', 'rain':'era5_adm2_annual__annual_rainfall_mm',
          'run':'era5_adm2_annual__annual_runoff_mm','et':'et0_adm2_annual__et0_annual_mm'},
 'ADM3': {'dir':'02_adm3_panel_dataset', 'rain':'rainfall_annual_mm',
          'run':'runoff_annual_mm','et':'et0_adm3_annual__et0_annual_mm'}
}

def load_level(root:Path, level:str)->pd.DataFrame:
    spec=SPECS[level]
    path=root/'processed_data'/'multivariate_flood_modeling'/spec['dir']/'full_analysis_panel.csv'
    if not path.is_file(): raise FileNotFoundError(f'Missing {path}; run Script 01/02 first')
    df=pd.read_csv(path,low_memory=False)
    needed=['adm_id','year','target_row_present','flood__flood_days',spec['et']]
    missing=set(needed)-set(df.columns)
    if missing: raise ValueError(f'{level} missing columns: {sorted(missing)}')
    if level=='ADM3':
        hydro=root/'processed_data/multivariate_flood_modeling/04_adm3_rainfall_runoff/adm3_annual_rainfall_runoff.csv'
        if not hydro.is_file(): raise FileNotFoundError(f'Missing {hydro}; run hydrology Script 04')
        h=pd.read_csv(hydro)
        needed_h=['adm_id','year',spec['rain'],spec['run'],'hydrology_qc_status',
                  'rainfall_valid_days','runoff_valid_days']
        if set(needed_h)-set(h.columns): raise ValueError('ADM3 hydrology columns are missing')
        if h.duplicated(['adm_id','year']).any(): raise ValueError('Duplicate ADM3 hydrology keys')
        df=df.merge(h[needed_h],how='left',on=['adm_id','year'],validate='one_to_one')
        # 2025 has 364 valid days; it remains a partial observation, never imputed.
        df['hydro_source_complete']=(df['rainfall_valid_days']>=df.year.map(lambda y: 366 if pd.Timestamp(f'{int(y)}-12-31').is_leap_year else 365)) & (df['runoff_valid_days']>=df.year.map(lambda y: 366 if pd.Timestamp(f'{int(y)}-12-31').is_leap_year else 365))
        df.loc[df['hydrology_qc_status'].ne('PASS'),[spec['rain'],spec['run']]]=np.nan
    else:
        if set([spec['rain'],spec['run']])-set(df.columns): raise ValueError('ADM2 hydrology missing')
        if 'qc__era5_suspect_zero_with_missing_daily' in df:
            flag=df['qc__era5_suspect_zero_with_missing_daily'].fillna(False).astype(bool)
            df.loc[flag,[spec['rain'],spec['run']]]=np.nan
        df['hydro_source_complete']=True # ADM2 source daily coverage not verified here
    df['adm_id']=df['adm_id'].astype('string').str.strip()
    df['year']=pd.to_numeric(df['year'],errors='raise').astype(int)
    if df.duplicated(['adm_id','year']).any(): raise ValueError(f'{level} duplicate region/year keys')
    df=df.sort_values(['adm_id','year']).reset_index(drop=True)
    source={'flood__flood_days':'flood_lag1',spec['rain']:'rainfall_lag1',
            spec['run']:'runoff_lag1',spec['et']:'et0_lag1',
            'hydro_source_complete':'prior_hydrology_complete'}
    previous_year=df.groupby('adm_id',sort=False)['year'].shift(1)
    for source_col,target_col in source.items():
        df[target_col]=df.groupby('adm_id',sort=False)[source_col].shift(1).where(previous_year.eq(df.year-1))
    df['target']=pd.to_numeric(df['flood__flood_days'],errors='coerce')
    present=df['target_row_present'].fillna(False).astype(bool)
    df.loc[~present,'target']=np.nan
    df=df[df.year.between(2001,2025)].copy()
    df['level']=level
    return df[['level','adm_id','year','target','flood_lag1','rainfall_lag1',
               'runoff_lag1','et0_lag1','prior_hydrology_complete']]

def metrics(y,p):
    return {'n':len(y),'MAE':float(mean_absolute_error(y,p)),
            'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p)) if len(y)>1 else None}

def split(x):
    return {'train':x.year.between(2001,2019),
            'validation':x.year.between(2020,2022),
            'test':x.year.between(2023,2025)}

def run(root,output,trees,seed):
    output.mkdir(parents=True,exist_ok=True)
    score_rows=[]; pred_rows=[]; importance_rows=[]; coverage_rows=[]
    for level in SPECS:
        df=load_level(root,level)
        for col in ['target','flood_lag1','rainfall_lag1','runoff_lag1','et0_lag1']:
            coverage_rows.append({'level':level,'feature':col,'available':int(df[col].notna().sum()),
                                  'missing':int(df[col].isna().sum()),'total':len(df)})
        # Shared complete-case cohort for a fair M0/M1/M2 comparison.
        cohort=df.dropna(subset=['target','flood_lag1','rainfall_lag1','runoff_lag1','et0_lag1']).copy()
        if cohort.empty: raise ValueError(f'{level}: no complete cases for M2; inspect data coverage')
        s=split(cohort)
        if min(int(v.sum()) for v in s.values())==0: raise ValueError(f'{level}: empty train/validation/test cohort')
        features={'M0':['flood_lag1'],
                  'M1':['flood_lag1','rainfall_lag1','runoff_lag1'],
                  'M2':['flood_lag1','rainfall_lag1','runoff_lag1','et0_lag1']}
        print(f'{level}: complete-case region-years={len(cohort)}; train={s["train"].sum()}, val={s["validation"].sum()}, test={s["test"].sum()}',flush=True)
        for m,cols in features.items():
            model=RandomForestRegressor(n_estimators=trees,random_state=seed,
                                       min_samples_leaf=3,n_jobs=-1,max_features=1.0)
            model.fit(cohort.loc[s['train'],cols],cohort.loc[s['train'],'target'])
            for feature,value in zip(cols,model.feature_importances_):
                importance_rows.append({'level':level,'set':m,'feature':feature,'importance':float(value)})
            for partition,mask in s.items():
                sub=cohort.loc[mask]
                y=sub.target.to_numpy()
                baseline=np.maximum(0,sub.flood_lag1.to_numpy())
                estimated=np.maximum(0,model.predict(sub[cols]))
                # IMPORTANT: baseline and RF use identical cohort and dates.
                for algorithm,vals in [('Persistence',baseline),('RandomForest',estimated)]:
                    score_rows.append({'level':level,'set':m,'algorithm':algorithm,'split':partition,
                                       **metrics(y,vals)})
                    for row,val in zip(sub.itertuples(index=False),vals):
                        pred_rows.append({'level':level,'set':m,'algorithm':algorithm,
                                          'split':partition,'adm_id':str(row.adm_id),'year':int(row.year),
                                          'actual_flood_days':float(row.target),'predicted_flood_days':float(val),
                                          'prior_hydrology_complete':bool(row.prior_hydrology_complete) if pd.notna(row.prior_hydrology_complete) else None})
        # One figure per level, only test results from M2.
        results=pd.DataFrame(pred_rows)
        sub=results[(results.level==level)&(results['set']=='M2')&(results.split=='test')]
        fig,ax=plt.subplots(figsize=(6,5))
        for name,marker in [('Persistence','o'),('RandomForest','x')]:
            z=sub[sub.algorithm==name]
            ax.scatter(z.actual_flood_days,z.predicted_flood_days,alpha=.25,s=15,marker=marker,label=name)
        mx=max(1,float(sub[['actual_flood_days','predicted_flood_days']].max().max()))
        ax.plot([0,mx],[0,mx],linestyle='--',color='gray')
        ax.set(xlabel='Observed flood days',ylabel='Predicted flood days',title=f'{level}: 2023–2025 held-out test (M2)')
        ax.legend(); fig.tight_layout();fig.savefig(output/f'{level.lower()}_test_actual_vs_predicted.png',dpi=180);plt.close(fig)
    scores=pd.DataFrame(score_rows)
    scores.to_csv(output/'model_metrics.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(output/'all_predictions.csv',index=False)
    pd.DataFrame(importance_rows).to_csv(output/'feature_importances.csv',index=False)
    pd.DataFrame(coverage_rows).to_csv(output/'feature_coverage.csv',index=False)
    test=scores[scores.split=='test'].sort_values(['level','set','algorithm'])
    print('\nHELD-OUT TEST (2023–2025), common complete-case cohort:',flush=True)
    print(test[['level','set','algorithm','n','MAE','RMSE','R2']].to_string(index=False),flush=True)
    details={'prediction_target':'flood days in year t','features':'only year t-1',
        'splits':{'train':[2001,2019],'validation':[2020,2022],'test':[2023,2025]},
        'comparison':'M0/M1/M2 all assessed on common complete-case cohort per ADM level',
        'baseline':'previous-year flood days (no fitted parameters)',
        'missing_values':'no filling; incomplete climate rows excluded on shared cohort',
        '2025_note':'partial 2025 hydrology inputs not used when predicting 2025 (uses 2024). 2025 hydrology would be partial if used for 2026 forecasts.',
        'limits':['Exploratory annual hindcast; flood target observation consistency requires further validation.',
                  'Same geographic units occur across splits, so not spatial holdout.',
                  'RF parameters are fixed; no hyperparameter tuning or rolling-origin yet.',
                  'Do not interpret impurity-based feature importance as causality.'],
        'source_paths':[str(root/'processed_data/multivariate_flood_modeling'/SPECS[l]['dir']/'full_analysis_panel.csv') for l in SPECS]}
    (output/'methodology.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(f'\nOutputs: {output}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--trees',type=int,default=250)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    root=a.root.resolve()
    out=a.output.resolve() if a.output else root/'processed_data/multivariate_flood_modeling/05_random_forest'
    run(root,out,a.trees,a.seed)
if __name__=='__main__':main()
