"""Script 32 — Dartmouth Flood Observatory discharge audit (2000–2025).
Outputs station-level QC, cleaned daily records, monthly/annual aggregates, and
month-of-year standardized discharge anomalies. NOT a causal flood-driver test.
Install: pandas numpy matplotlib openpyxl.
"""
from pathlib import Path
import zipfile, io, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path('/Users/shibai/Documents/GitHub/DBL-Group-22')
INPUT=ROOT/'raw_data'/'darthmouth_flood_observatory'
OUTPUT=ROOT/'processed_data'/'cross_dataset_analysis'/'05_hydrological_drivers'/'32_dartmouth_discharge'
FORMS=OUTPUT/'tables'; FIG=OUTPUT/'figures'
for p in (FORMS,FIG):p.mkdir(parents=True,exist_ok=True)
START='2000-01-01'; END='2025-12-31'


def csv_sources():
    if INPUT.is_dir():
        for p in sorted(INPUT.rglob('*_discharge.csv')):yield p.name,p.read_bytes()
    elif INPUT.with_suffix('.zip').exists():
        with zipfile.ZipFile(INPUT.with_suffix('.zip')) as z:
            for n in sorted(z.namelist()):
                if not n.startswith('__MACOSX/') and n.endswith('_discharge.csv'):yield Path(n).name,z.read(n)
    else:raise FileNotFoundError(f'Unzip dataset to {INPUT}, or put {INPUT.with_suffix(".zip")}')


def station_metadata():
    for p in [INPUT/'information.xlsx', INPUT.with_suffix('.zip')]:
        if p.is_file():
            if p.suffix=='.xlsx': return pd.read_excel(p)
            with zipfile.ZipFile(p) as z:
                matches=[n for n in z.namelist() if n.endswith('/information.xlsx') or n=='information.xlsx']
                if matches:return pd.read_excel(io.BytesIO(z.read(matches[0])))
    return pd.DataFrame()


def main():
    daily=[]; qc=[]
    expected=pd.date_range(START,END,freq='D')
    for filename,content in csv_sources():
        sid=re.search(r'(\d+)_discharge',filename).group(1)
        raw=pd.read_csv(io.BytesIO(content))
        if not {'Date','Discharge (m3/s)'}.issubset(raw.columns):raise ValueError(f'Unexpected columns: {filename} {list(raw)}')
        obs=pd.DataFrame({'date':pd.to_datetime(raw['Date'],errors='coerce'),
                          'discharge_m3s':pd.to_numeric(raw['Discharge (m3/s)'],errors='coerce')})
        bad_date=int(obs.date.isna().sum()); obs=obs.dropna(subset=['date'])
        obs=obs[obs.date.between(START,END)]
        duplicates=int(obs.date.duplicated().sum())
        if duplicates:raise ValueError(f'{sid}: duplicate observation dates: {duplicates}; resolve before aggregation')
        obs=obs.set_index('date').reindex(expected).rename_axis('date').reset_index()
        negative=int(obs.discharge_m3s.lt(0).sum())
        # Negative physical discharge invalid in this unsigned series; zeros preserved but flagged.
        obs.loc[obs.discharge_m3s.lt(0),'discharge_m3s']=np.nan
        qc.append({'station_id':sid,'expected_days':len(expected),'nonmissing_days':int(obs.discharge_m3s.notna().sum()),
                   'missing_days':int(obs.discharge_m3s.isna().sum()),'zero_days':int(obs.discharge_m3s.eq(0).sum()),
                   'negative_removed':negative,'invalid_source_dates':bad_date,'duplicate_dates':duplicates,
                   'coverage_fraction':float(obs.discharge_m3s.notna().mean())})
        obs['station_id']=sid;daily.append(obs)
    if not daily:raise ValueError('No discharge CSV files found')
    d=pd.concat(daily,ignore_index=True)
    d['year']=d.date.dt.year;d['month']=d.date.dt.month
    m=d.groupby(['station_id','year','month'],as_index=False).agg(
        mean_discharge_m3s=('discharge_m3s','mean'),max_discharge_m3s=('discharge_m3s','max'),
        valid_days=('discharge_m3s','count'),days=('date','size'))
    # Strict coverage filter: do not treat sparse monthly averages as complete.
    m['sufficient_coverage']=m.valid_days.div(m.days).ge(.8)
    m.loc[~m.sufficient_coverage,['mean_discharge_m3s','max_discharge_m3s']]=np.nan
    baseline=m[m.year.between(2000,2019)].groupby(['station_id','month']).mean_discharge_m3s.agg(['mean','std','count']).reset_index()
    baseline=baseline.rename(columns={'mean':'baseline_monthly_mean','std':'baseline_monthly_sd','count':'baseline_years'})
    m=m.merge(baseline,on=['station_id','month'],how='left',validate='many_to_one')
    m['monthly_discharge_anomaly_m3s']=m.mean_discharge_m3s-m.baseline_monthly_mean
    m['monthly_discharge_z']=np.where(m.baseline_monthly_sd.gt(0) & m.baseline_years.ge(10),
      m.monthly_discharge_anomaly_m3s/m.baseline_monthly_sd,np.nan)
    m['anomaly_qc_flag']=np.select([m['baseline_years'].fillna(0).lt(10),m['baseline_monthly_sd'].fillna(0).le(0),m['monthly_discharge_z'].abs().gt(5)],['insufficient_baseline','zero_baseline_sd','extreme_z_review'],default='ok')
    annual=d.groupby(['station_id','year'],as_index=False).agg(
      annual_mean_m3s=('discharge_m3s','mean'),valid_days=('discharge_m3s','count'),days=('date','size'))
    annual['coverage_fraction']=annual.valid_days/annual.days
    annual.loc[annual.coverage_fraction.lt(.8),'annual_mean_m3s']=np.nan
    pd.DataFrame(qc).to_csv(FORMS/'32_01_station_qc.csv',index=False)
    d.to_csv(FORMS/'32_02_daily_discharge_2000_2025.csv',index=False)
    m.to_csv(FORMS/'32_03_monthly_discharge_anomalies.csv',index=False)
    m.groupby(['station_id','anomaly_qc_flag'],dropna=False).size().rename('months').reset_index().to_csv(FORMS/'32_06_anomaly_qc_flags.csv',index=False)
    annual.to_csv(FORMS/'32_04_annual_discharge.csv',index=False)
    info=station_metadata()
    if not info.empty:info.to_csv(FORMS/'32_05_station_metadata_original.csv',index=False)
    for sid,g in m.groupby('station_id'):
        fig,ax=plt.subplots(figsize=(11,3.5));times=pd.to_datetime(g[['year','month']].assign(day=1))
        ax.plot(times,g.monthly_discharge_z,linewidth=.9);ax.axhline(0,color='gray',linewidth=.7)
        ax.set_title(f'Discharge monthly standardized anomalies — station {sid}')
        ax.set_ylabel('z-score (2000–2019 monthly baseline)');fig.tight_layout()
        fig.savefig(FIG/f'32_station_{sid}_monthly_anomalies.png',dpi=170);plt.close(fig)
    print(f'Script 32 complete: {len(qc)} stations. Verify river identity and station provenance before interpretation. {OUTPUT}')
if __name__=='__main__':main()
