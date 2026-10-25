#!/usr/bin/env python3
"""JBG060: summarize A/B/C monthly RF forecasts (M0/M1/M2).

Inputs: outputs of 05b_monthly_random_forest.py. Compares B vs C at same
one-month lead and makes clear A is ~12-month ahead and NOT same horizon.
DO NOT describe event-date proxy as validated flood days.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('processed_data/multivariate_flood_modeling/05b_monthly_random_forest'))
    p.add_argument('--output',type=Path,default=Path('processed_data/multivariate_flood_modeling/05c_compare_monthly_predictions'))
    a=p.parse_args()
    data=a.input/'model_metrics.csv'
    if not data.exists():raise FileNotFoundError(f'Run 05b first: {data}')
    x=pd.read_csv(data)
    expected={'level','strategy','model','algorithm','split','MAE','RMSE','R2','n'}
    if not expected.issubset(x):raise ValueError(f'Missing metrics columns: {sorted(expected-set(x))}')
    x=x[(x.split=='test')&(x.algorithm=='RandomForest')].copy()
    if x.empty:raise ValueError('No test RandomForest rows')
    # Plot test MAE for all strategies, visually separating different horizons.
    a.output.mkdir(parents=True,exist_ok=True)
    x['forecast_horizon_months']=x.strategy.map({'A':12,'B':1,'C':1})
    x.sort_values(['level','model','strategy']).to_csv(a.output/'test_metrics_comparison.csv',index=False)
    b=x[x.strategy=='B'].set_index(['level','model'])
    c=x[x.strategy=='C'].set_index(['level','model'])
    matched=b[['MAE','RMSE','R2','n']].join(c[['MAE','RMSE','R2','n']],lsuffix='_B',rsuffix='_C',how='inner')
    # Same forecast horizon and same date boundaries; require same number of tested rows.
    matched['same_test_row_count']=matched.n_B.eq(matched.n_C)
    matched['MAE_improvement_C_over_B']=matched.MAE_B-matched.MAE_C
    matched['RMSE_improvement_C_over_B']=matched.RMSE_B-matched.RMSE_C
    matched.reset_index().to_csv(a.output/'B_vs_C_one_month_ahead.csv',index=False)
    for level in sorted(x.level.unique()):
        z=x[x.level==level]
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for ax,strats,title in [(axs[0],['B','C'],'One-month-ahead comparison'),
                                (axs[1],['A'],'12-month-ahead reference')]:
            sub=z[z.strategy.isin(strats)]
            positions=np.arange(3);models=['M0','M1','M2']
            if len(strats)==2:
                for delta,s in [(-.18,'B'),(.18,'C')]:
                    vals=[sub.loc[(sub.model==m)&(sub.strategy==s),'MAE'] for m in models]
                    ax.bar(positions+delta,[float(v.iloc[0]) if len(v) else np.nan for v in vals],width=.35,label=s)
            else:
                vals=[sub.loc[(sub.model==m)&(sub.strategy=='A'),'MAE'] for m in models]
                ax.bar(positions,[float(v.iloc[0]) if len(v) else np.nan for v in vals],width=.6,label='A')
            ax.set_xticks(positions,models);ax.set_ylabel('Test MAE (days)');ax.set_title(title);ax.legend()
        fig.suptitle(f'{level}: monthly target forecasting — lower MAE is better')
        fig.tight_layout();fig.savefig(a.output/f'{level.lower()}_strategy_mae.png',dpi=180);plt.close(fig)
    method={}
    meta=a.input/'methodology.json'
    if meta.exists():method=json.loads(meta.read_text(encoding='utf-8'))
    caveats=['A forecasts ~12 months ahead; B/C forecast one month ahead. Do not use A vs B/C as an equal-horizon ranking.',
             'B/C may have different complete-case cohorts; count agreement does NOT prove exact row equality.',
             'If 05a used --event-proxy, target is event-recorded dates, NOT validated flooded days.',
             'Metrics on training data or synthetic demo are not scientific evidence.']
    (a.output/'README.txt').write_text('Monthly comparison\n\n'+'\n'.join('- '+t for t in caveats)+
      '\n\n05b provenance:\n'+json.dumps(method,indent=2),encoding='utf-8')
    print('Monthly forecast comparison written to:',a.output.resolve())
    print(x[['level','strategy','model','n','MAE','RMSE','R2']].sort_values(['level','strategy','model']).to_string(index=False))
    print('\nCAUTION: Event proxies are NOT validated flood days; see README.txt.')

if __name__=='__main__':main()
